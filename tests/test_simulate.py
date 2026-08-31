"""Tests for src/simulate.py."""

import numpy as np
import pytest

from src.model import Parameters, N_STATES, Gp, Gi
from src.simulate import simulate, simulate_dense, SimulationError, get_bgm_times
from src.events import generate_mdi_schedule, Event, EVENT_MEAL, EVENT_BASAL


def _state0():
    s = np.zeros(N_STATES)
    s[Gp] = 120.0
    s[Gi] = 90.0
    return s


def test_simulate_output_shape():
    p = Parameters()
    days = 1
    t_span = (0.0, days * 24.0)
    events = generate_mdi_schedule(days=days, rng=np.random.default_rng(0))
    t, y = simulate(_state0(), events, p, t_span)
    assert y.ndim == 2
    assert y.shape[1] == N_STATES
    assert len(t) == len(y)
    assert t[0] == 0.0
    assert t[-1] == pytest.approx(t_span[1])


def test_simulate_dense_uniform_timestep():
    p = Parameters()
    days = 1
    t_span = (0.0, days * 24.0)
    events = generate_mdi_schedule(days=days, rng=np.random.default_rng(0))
    dt = 0.1
    t, y = simulate_dense(_state0(), events, p, t_span, dt=dt)
    # Array spacing should be ~dt
    diffs = np.diff(t)
    assert np.allclose(diffs, dt, atol=1e-9)
    assert y.shape[1] == N_STATES


def test_simulate_meal_increases_glucose():
    p = Parameters()
    days = 1
    t_span = (0.0, days * 24.0)
    events = generate_mdi_schedule(days=days, rng=np.random.default_rng(0))
    _, y = simulate_dense(_state0(), events, p, t_span, dt=0.1)
    # With meals injected, glucose should rise above fasting baseline at some point
    assert y[:, Gp].max() >= 120.0 - 1e-6


def test_simulate_reproducible():
    p = Parameters()
    days = 1
    t_span = (0.0, days * 24.0)
    events = generate_mdi_schedule(days=days, rng=np.random.default_rng(7))
    _, y1 = simulate_dense(_state0(), events, p, t_span, dt=0.2)
    _, y2 = simulate_dense(_state0(), events, p, t_span, dt=0.2)
    np.testing.assert_allclose(y1, y2)


def test_get_bgm_times():
    events = generate_mdi_schedule(days=2, rng=np.random.default_rng(0))
    t_span = (0.0, 2 * 24.0)
    bgm = get_bgm_times(events, t_span)
    assert len(bgm) == 12  # 6 per day * 2 days
    assert all(t_span[0] <= tt <= t_span[1] for tt in bgm)


def test_simulate_invalid_state_shape():
    p = Parameters()
    with pytest.raises(ValueError):
        simulate(np.zeros(5), [], p, (0, 24))


def test_simulate_invalid_tspan():
    p = Parameters()
    with pytest.raises(ValueError):
        simulate(_state0(), [], p, (24, 0))


def test_simulate_with_stress_increases_egp():
    """Stress should elevate glucose via reduced insulin effectiveness + EGP boost."""
    p = Parameters()
    t_span = (0.0, 24.0)

    # No stress events
    events_no_stress = generate_mdi_schedule(
        days=1, stress_probability=0.0, exercise_days_per_week=0,
        rng=np.random.default_rng(1),
    )
    # A stress event starting at 8h, lasting 4h
    events_stress = [e for e in events_no_stress if e.kind != "stress"]
    events_stress.append(Event(8.0, "stress", 4.0))
    events_stress.sort(key=lambda e: e.time)

    t, y_base = simulate_dense(_state0(), events_no_stress, p, t_span, dt=0.1)
    _, y_stress = simulate_dense(_state0(), events_stress, p, t_span, dt=0.1)

    # Mean glucose during stress window should be higher
    mask = (t % 24 >= 8) & (t % 24 <= 12)
    assert y_stress[mask, Gp].mean() > y_base[mask, Gp].mean()
