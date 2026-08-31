"""Fisher Information Matrix and identifiability analysis.

Computes sensitivity trajectories, the observed FIM, and cumulative FIM
rank analysis to assess parameter identifiability from BGM data.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.interpolate import interp1d

from .model import Parameters, Gp
from .simulate import simulate_dense, get_bgm_times
from .events import Event
from .observe import bgm_noise_sd


EPS = 1e-6


def compute_sensitivity_trajectory(
    initial_state: np.ndarray,
    events: list[Event],
    p: Parameters,
    t_span: tuple[float, float],
    param_idx: int,
    dt: float = 0.05,
    rtol: float = 1e-8,
    atol: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the finite-difference sensitivity of Gp to a single parameter.

    Uses forward finite differences with adaptive step size.

    Args:
        initial_state: Initial state vector.
        events: Simulation events.
        p: Nominal parameters.
        t_span: Simulation time span.
        param_idx: Index into the parameter array.
        dt: Dense output time step.
        rtol: ODE solver relative tolerance.
        atol: ODE solver absolute tolerance.

    Returns:
        (time, base_trajectory, sensitivity) arrays.
    """
    base_t, base_y = simulate_dense(
        initial_state, events, p, t_span, dt, rtol, atol,
    )

    arr = p.as_array()
    h = max(EPS, EPS * abs(arr[param_idx]))
    arr[param_idx] += h
    p_plus = Parameters.from_array(arr)

    plus_t, plus_y = simulate_dense(
        initial_state, events, p_plus, t_span, dt, rtol, atol,
    )

    min_len = min(len(base_t), len(plus_t))
    sensitivity = (plus_y[:min_len, Gp] - base_y[:min_len, Gp]) / h
    return base_t[:min_len], base_y[:min_len], sensitivity


def compute_fim(
    initial_state: np.ndarray,
    events: list[Event],
    p: Parameters,
    param_indices: list[int],
    t_span: tuple[float, float],
    rtol: float = 1e-8,
    atol: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the observed Fisher Information Matrix at BGM time points.

    Args:
        initial_state: Initial state vector.
        events: Simulation events.
        p: Model parameters.
        param_indices: Indices of parameters to include in the FIM.
        t_span: Simulation time span.
        rtol: ODE solver relative tolerance.
        atol: ODE solver absolute tolerance.

    Returns:
        (fim, bgm_times) where fim is (n_params, n_params) and bgm_times
        is the array of BGM measurement times used.
    """
    bgm_times = get_bgm_times(events, t_span)
    if len(bgm_times) == 0:
        return np.zeros((len(param_indices), len(param_indices))), bgm_times

    t_sim, y_sim = simulate_dense(initial_state, events, p, t_span, rtol=rtol, atol=atol)

    interp_fn = interp1d(
        t_sim, y_sim[:, Gp], kind="linear",
        bounds_error=False, fill_value="extrapolate",
    )
    g_p_bgm = interp_fn(bgm_times)

    n_params = len(param_indices)
    n_obs = len(bgm_times)
    sensitivity_matrix = np.zeros((n_obs, n_params))

    for j, pidx in enumerate(param_indices):
        _, _, sens = compute_sensitivity_trajectory(
            initial_state, events, p, t_span, pidx,
            rtol=rtol, atol=atol,
        )
        sens_interp = interp1d(
            t_sim[:len(sens)], sens, kind="linear",
            bounds_error=False, fill_value="extrapolate",
        )
        sensitivity_matrix[:, j] = sens_interp(bgm_times)

    fim = np.zeros((n_params, n_params))
    for k in range(n_obs):
        sd = bgm_noise_sd(max(g_p_bgm[k], 0.0), p)
        sk = sensitivity_matrix[k, :]
        fim += np.outer(sk, sk) / (sd ** 2)

    return fim, bgm_times


def compute_fim_over_days(
    initial_state: np.ndarray,
    events: list[Event],
    p: Parameters,
    param_indices: list[int],
    max_days: int = 30,
    rtol: float = 1e-6,
    atol: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute cumulative FIM rank over increasing study duration.

    For each day d in [1, max_days], computes the FIM for the first d days
    and reports its effective rank via SVD.

    Args:
        initial_state: Initial state vector.
        events: Simulation events.
        p: Model parameters.
        param_indices: Parameter indices to include.
        max_days: Maximum number of days to analyze.
        rtol: ODE solver relative tolerance.
        atol: ODE solver absolute tolerance.

    Returns:
        (days_array, ranks_array) both of length max_days.
    """
    ranks = []
    cumulative_fim: Optional[np.ndarray] = None

    for day in range(1, max_days + 1):
        t_span = (0.0, day * 24.0)
        fim, _ = compute_fim(
            initial_state, events, p, param_indices, t_span, rtol, atol,
        )

        if cumulative_fim is None:
            cumulative_fim = fim
        else:
            cumulative_fim += fim

        s = np.linalg.svd(cumulative_fim, compute_uv=False)
        tol = 1e-6 * s[0] if s[0] > 0 else 1e-6
        rank = int(np.sum(s > tol))
        ranks.append(rank)

    return np.arange(1, max_days + 1), np.array(ranks)
