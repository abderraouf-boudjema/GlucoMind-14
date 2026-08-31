"""ODE integration for the GlucoMind-14 simulator.

Provides `simulate()` for sparse (event-time) output and `simulate_dense()`
for uniform-dt output. Both handle state discontinuities (meals, boluses,
basal injections) and time-varying modulators (stress, exercise).
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
from scipy.integrate import solve_ivp

from .model import (
    N_STATES, Qsto1, Isc1, Ibase1,
    rhs, Parameters,
)
from .events import (
    Event, EVENT_MEAL, EVENT_BOLUS, EVENT_BASAL,
    EVENT_BGM, EVENT_STRESS, EVENT_EXERCISE,
)


class SimulationError(Exception):
    """Raised when the ODE solver fails to converge."""


def _group_events_by_time(events: list[Event]) -> list[tuple[float, list[Event]]]:
    """Group events that occur at the same time (within 1 ns tolerance)."""
    filtered = sorted(
        [e for e in events if e.kind != EVENT_BGM],
        key=lambda e: e.time,
    )
    grouped: list[tuple[float, list[Event]]] = []
    for e in filtered:
        if not grouped or abs(grouped[-1][0] - e.time) > 1e-9:
            grouped.append((e.time, [e]))
        else:
            grouped[-1][1].append(e)
    return grouped


def _make_rhs_wrap(
    rhs_fn: Callable,
    p: Parameters,
    last_cho: float = 0.0,
    stress_level_fn: Optional[Callable] = None,
    exercise_level_fn: Optional[Callable] = None,
) -> Callable:
    """Create a wrapped RHS function compatible with solve_ivp."""
    def wrapped(t: float, y: np.ndarray) -> np.ndarray:
        s = stress_level_fn(t) if stress_level_fn is not None else 0.0
        e = exercise_level_fn(t) if exercise_level_fn is not None else 0.0
        return rhs_fn(t, y, p, last_cho=last_cho, stress_level=s, exercise_level=e)
    return wrapped


def _integrate_interval(
    y0: np.ndarray,
    t0: float,
    t1: float,
    p: Parameters,
    rtol: float,
    atol: float,
    max_step: float,
    last_cho: float = 0.0,
    stress_level_fn: Optional[Callable] = None,
    exercise_level_fn: Optional[Callable] = None,
) -> np.ndarray:
    """Integrate the ODE over [t0, t1] and return the final state.

    Raises:
        SimulationError: If the solver fails or does not reach t1.
    """
    if t1 <= t0:
        return y0.copy()

    rhs_wrap = _make_rhs_wrap(rhs, p, last_cho, stress_level_fn, exercise_level_fn)

    sol = solve_ivp(
        rhs_wrap,
        (t0, t1),
        y0,
        method="RK45",
        rtol=rtol,
        atol=atol,
        max_step=max_step,
        dense_output=True,
    )

    if not sol.success:
        raise SimulationError(
            f"ODE solver failed on [{t0:.2f}, {t1:.2f}]: {sol.message}"
        )

    if not np.all(np.isfinite(sol.y[:, -1])):
        raise SimulationError(
            f"Non-finite state at t={t1:.2f}: {sol.y[:, -1]}"
        )

    return sol.y[:, -1].copy()


def _integrate_interval_dense(
    y0: np.ndarray,
    t0: float,
    t1: float,
    p: Parameters,
    dense_t: np.ndarray,
    dense_y: np.ndarray,
    rtol: float,
    atol: float,
    max_step: float,
    last_cho: float = 0.0,
    stress_level_fn: Optional[Callable] = None,
    exercise_level_fn: Optional[Callable] = None,
) -> None:
    """Integrate and fill in dense output at the times in dense_t."""
    if t1 <= t0 + 1e-12:
        return

    rhs_wrap = _make_rhs_wrap(rhs, p, last_cho, stress_level_fn, exercise_level_fn)

    sol = solve_ivp(
        rhs_wrap,
        (t0, t1),
        y0,
        method="RK45",
        rtol=rtol,
        atol=atol,
        max_step=max_step,
        dense_output=True,
    )

    if not sol.success:
        raise SimulationError(
            f"ODE solver failed on [{t0:.2f}, {t1:.2f}] (dense): {sol.message}"
        )

    mask = (dense_t >= t0) & (dense_t <= t1)
    idxs = np.where(mask)[0]
    for i in idxs:
        dense_y[i] = sol.sol(dense_t[i])


def _apply_jump(y: np.ndarray, ev: Event) -> np.ndarray:
    """Apply state discontinuities for meals, boluses, and basal injections."""
    y = y.copy()
    if ev.kind == EVENT_MEAL:
        y[Qsto1] += ev.amount
    elif ev.kind == EVENT_BOLUS:
        y[Isc1] += ev.amount * U_TO_MU
    elif ev.kind == EVENT_BASAL:
        y[Ibase1] += ev.amount * U_TO_MU
    return y


U_TO_MU = 1000.0


def simulate(
    initial_state: np.ndarray,
    events: list[Event],
    p: Parameters,
    t_span: tuple[float, float] = (0.0, 720.0),
    rtol: float = 1e-8,
    atol: float = 1e-8,
    max_step: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate the glucose-insulin system with sparse (event-time) output.

    Args:
        initial_state: Initial state vector, shape (11,).
        events: List of Event objects defining the protocol.
        p: Model parameters.
        t_span: (t_start, t_end) in hours.
        rtol: Relative tolerance for the ODE solver.
        atol: Absolute tolerance for the ODE solver.
        max_step: Maximum step size for the ODE solver.

    Returns:
        (times, states) where times has shape (N,) and states has shape (N, 11).
        N includes the initial time and each event time.

    Raises:
        ValueError: If initial_state has wrong shape or t_span is invalid.
        SimulationError: If the ODE solver fails.
    """
    initial_state = np.asarray(initial_state, dtype=float)
    if initial_state.shape != (N_STATES,):
        raise ValueError(
            f"initial_state must have shape ({N_STATES},), got {initial_state.shape}"
        )
    if t_span[1] <= t_span[0]:
        raise ValueError(
            f"t_span[1] must be > t_span[0], got {t_span}"
        )

    grouped = _group_events_by_time(events)
    group_times = [g[0] for g in grouped if t_span[0] <= g[0] <= t_span[1]]

    times = [t_span[0]]
    states = [initial_state.copy()]

    y = initial_state.copy()
    t_prev = t_span[0]
    last_cho = 0.0
    stress_active_until = -1.0
    exercise_active_until = -1.0

    def get_stress_level(t: float) -> float:
        return 1.0 if t < stress_active_until else 0.0

    def get_exercise_level(t: float) -> float:
        if t < exercise_active_until:
            remaining = exercise_active_until - t
            decay = np.exp(-2.0 * remaining)
            return min(1.0, decay * 3.0)
        return 0.0

    for ev_time, ev_group in grouped:
        if ev_time < t_span[0]:
            continue
        if ev_time > t_span[1]:
            break
        if ev_time <= t_prev:
            continue

        y = _integrate_interval(
            y, t_prev, ev_time, p, rtol, atol, max_step,
            last_cho=last_cho,
            stress_level_fn=get_stress_level,
            exercise_level_fn=get_exercise_level,
        )
        for ev in ev_group:
            if ev.kind == EVENT_MEAL:
                last_cho = ev.amount
            elif ev.kind == EVENT_STRESS:
                stress_active_until = ev_time + ev.amount
            elif ev.kind == EVENT_EXERCISE:
                exercise_active_until = ev_time + ev.amount + 2.0
            y = _apply_jump(y, ev)
        t_prev = ev_time
        times.append(t_prev)
        states.append(y.copy())

    if t_prev < t_span[1]:
        y = _integrate_interval(
            y, t_prev, t_span[1], p, rtol, atol, max_step,
            last_cho=last_cho,
            stress_level_fn=get_stress_level,
            exercise_level_fn=get_exercise_level,
        )
        times.append(t_span[1])
        states.append(y.copy())

    return np.array(times), np.array(states)


def simulate_dense(
    initial_state: np.ndarray,
    events: list[Event],
    p: Parameters,
    t_span: tuple[float, float] = (0.0, 720.0),
    dt: float = 0.05,
    rtol: float = 1e-8,
    atol: float = 1e-8,
    max_step: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate with uniform-dt dense output.

    Args:
        initial_state: Initial state vector, shape (11,).
        events: List of Event objects defining the protocol.
        p: Model parameters.
        t_span: (t_start, t_end) in hours.
        dt: Time step for dense output (hours).
        rtol: Relative tolerance for the ODE solver.
        atol: Absolute tolerance for the ODE solver.
        max_step: Maximum step size for the ODE solver.

    Returns:
        (dense_t, dense_y) with dense_t shape (N,) and dense_y shape (N, 11).

    Raises:
        ValueError: If initial_state has wrong shape or t_span is invalid.
        SimulationError: If the ODE solver fails.
    """
    initial_state = np.asarray(initial_state, dtype=float)
    if initial_state.shape != (N_STATES,):
        raise ValueError(
            f"initial_state must have shape ({N_STATES},), got {initial_state.shape}"
        )
    if t_span[1] <= t_span[0]:
        raise ValueError(
            f"t_span[1] must be > t_span[0], got {t_span}"
        )
    if dt <= 0:
        raise ValueError(f"dt must be positive, got {dt}")

    grouped = _group_events_by_time(events)

    dense_t = np.arange(t_span[0], t_span[1] + dt, dt)
    dense_y = np.zeros((len(dense_t), N_STATES))

    y = initial_state.copy()
    t_prev = t_span[0]
    last_cho = 0.0
    stress_active_until = -1.0
    exercise_active_until = -1.0

    def get_stress_level(t: float) -> float:
        return 1.0 if t < stress_active_until else 0.0

    def get_exercise_level(t: float) -> float:
        if t < exercise_active_until:
            remaining = exercise_active_until - t
            decay = np.exp(-2.0 * remaining)
            return min(1.0, decay * 3.0)
        return 0.0

    for ev_time, ev_group in grouped:
        if ev_time < t_span[0]:
            continue
        if ev_time > t_span[1]:
            break
        if ev_time <= t_prev:
            continue

        _integrate_interval_dense(
            y, t_prev, ev_time, p, dense_t, dense_y, rtol, atol, max_step,
            last_cho=last_cho,
            stress_level_fn=get_stress_level,
            exercise_level_fn=get_exercise_level,
        )
        y = _integrate_interval(
            y, t_prev, ev_time, p, rtol, atol, max_step,
            last_cho=last_cho,
            stress_level_fn=get_stress_level,
            exercise_level_fn=get_exercise_level,
        )
        for ev in ev_group:
            if ev.kind == EVENT_MEAL:
                last_cho = ev.amount
            elif ev.kind == EVENT_STRESS:
                stress_active_until = ev_time + ev.amount
            elif ev.kind == EVENT_EXERCISE:
                exercise_active_until = ev_time + ev.amount + 2.0
            y = _apply_jump(y, ev)
        t_prev = ev_time

    if t_prev < t_span[1]:
        _integrate_interval_dense(
            y, t_prev, t_span[1], p, dense_t, dense_y, rtol, atol, max_step,
            last_cho=last_cho,
            stress_level_fn=get_stress_level,
            exercise_level_fn=get_exercise_level,
        )

    return dense_t, dense_y


def get_bgm_times(
    events: list[Event],
    t_span: tuple[float, float] = (0.0, 720.0),
) -> np.ndarray:
    """Extract BGM measurement times from the event list."""
    times = [
        e.time for e in events
        if e.kind == EVENT_BGM and t_span[0] <= e.time <= t_span[1]
    ]
    return np.array(times)
