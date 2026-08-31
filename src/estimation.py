"""Maximum a posteriori (MAP) parameter estimation.

Estimates a subset of model parameters from BGM observations using L-BFGS-B
optimization with lognormal priors. Also provides profile likelihood
computation for identifiability analysis.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.optimize import minimize, OptimizeResult
from scipy.interpolate import interp1d

from .model import Parameters, PARAM_NAMES
from .simulate import simulate_dense, SimulationError
from .events import Event
from .observe import compute_log_likelihood


def make_log_prior(
    p_nominal: Parameters,
    param_cv: dict[str, float],
) -> Callable[[Parameters], float]:
    """Create a lognormal log-prior function.

    Args:
        p_nominal: Nominal parameter values (prior means).
        param_cv: Coefficient of variation for each parameter.

    Returns:
        A function that computes the log-prior for a Parameters object.
    """
    def log_prior(p: Parameters) -> float:
        lp = 0.0
        for name in PARAM_NAMES:
            if name not in param_cv:
                continue
            mu = np.log(getattr(p_nominal, name))
            sigma = np.sqrt(np.log(1.0 + param_cv[name] ** 2))
            val = getattr(p, name)
            if val <= 0:
                return -np.inf
            lp -= 0.5 * ((np.log(val) - mu) / sigma) ** 2
            lp -= np.log(val)
        return lp
    return log_prior


def _build_params(
    base: Parameters,
    fixed: dict[str, float],
    estimable_names: list[str],
    x: np.ndarray,
) -> Parameters:
    """Build a Parameters object from fixed values, estimable values, and base defaults."""
    kwargs: dict[str, float] = {}
    for name in PARAM_NAMES:
        if name in fixed:
            kwargs[name] = fixed[name]
        elif name in estimable_names:
            idx = estimable_names.index(name)
            kwargs[name] = float(x[idx])
        else:
            kwargs[name] = getattr(base, name)
    return Parameters(**kwargs)


def estimate_map(
    observations: np.ndarray,
    bgm_times: np.ndarray,
    events: list[Event],
    initial_state: np.ndarray,
    p_guess: Parameters,
    p_nominal: Parameters,
    param_cv: dict[str, float],
    fixed_params: dict[str, float],
    estimable_names: list[str],
    t_span: tuple[float, float],
    method: str = "L-BFGS-B",
    rtol: float = 1e-6,
    atol: float = 1e-6,
    max_iter: int = 100,
) -> tuple[Parameters, dict]:
    """Estimate parameters via maximum a posteriori (MAP) optimization.

    Args:
        observations: BGM observations (mg/dL).
        bgm_times: BGM measurement times (hours).
        events: Simulation events.
        initial_state: Initial state vector, shape (11,).
        p_guess: Initial parameter guess.
        p_nominal: Nominal parameters (for prior centering).
        param_cv: CV for each parameter (for prior width).
        fixed_params: Parameters to hold fixed during optimization.
        estimable_names: Names of parameters to estimate.
        t_span: (t_start, t_end) in hours.
        method: Optimization method.
        rtol: ODE solver relative tolerance.
        atol: ODE solver absolute tolerance.
        max_iter: Maximum optimizer iterations.

    Returns:
        (optimized_parameters, info_dict) where info_dict contains
        success, n_iter, fun, message.
    """
    log_prior_fn = make_log_prior(p_nominal, param_cv)

    def objective(x: np.ndarray) -> float:
        p = _build_params(p_guess, fixed_params, estimable_names, x)
        try:
            times, states = simulate_dense(
                initial_state, events, p, t_span,
                rtol=rtol, atol=atol,
            )
            interp_fn = interp1d(
                times, states[:, 0], kind="linear",
                bounds_error=False, fill_value="extrapolate",
            )
            predictions = interp_fn(bgm_times)
            ll = compute_log_likelihood(observations, predictions, p)
            lp = log_prior_fn(p)
            if not np.isfinite(ll + lp):
                return 1e12
            return -(ll + lp)
        except SimulationError:
            return 1e12
        except Exception as e:
            return 1e12

    x0 = np.array([getattr(p_guess, name) for name in estimable_names])
    bounds = [(1e-6, None) for _ in estimable_names]

    result: OptimizeResult = minimize(
        objective, x0, method=method, bounds=bounds,
        options={"maxiter": max_iter, "ftol": 1e-12},
    )

    p_opt = _build_params(p_guess, fixed_params, estimable_names, result.x)
    return p_opt, {
        "success": bool(result.success),
        "n_iter": int(result.nit),
        "fun": float(result.fun),
        "message": str(result.message),
    }


def compute_profile_likelihood(
    observations: np.ndarray,
    bgm_times: np.ndarray,
    events: list[Event],
    initial_state: np.ndarray,
    p_opt: Parameters,
    param_name: str,
    estimable_names: list[str],
    fixed_params: dict[str, float],
    p_nominal: Parameters,
    param_cv: dict[str, float],
    t_span: tuple[float, float],
    n_points: int = 20,
    scale_range: tuple[float, float] = (0.5, 2.0),
) -> tuple[np.ndarray, np.ndarray]:
    """Compute profile likelihood for a single parameter.

    Fixes the target parameter at a grid of values and optimizes all other
    estimable parameters, recording the optimal log-likelihood at each point.

    Args:
        observations: BGM observations.
        bgm_times: BGM measurement times.
        events: Simulation events.
        initial_state: Initial state vector.
        p_opt: Optimized parameter set (starting point).
        param_name: Name of the parameter to profile.
        estimable_names: All estimable parameter names.
        fixed_params: Parameters held fixed.
        p_nominal: Nominal parameters for the prior.
        param_cv: CV for the prior.
        t_span: Simulation time span.
        n_points: Number of grid points for the profile.
        scale_range: (low, high) multiplier range around the optimal value.

    Returns:
        (parameter_values, log_likelihoods) arrays of length n_points.
    """
    log_prior_fn = make_log_prior(p_nominal, param_cv)

    opt_val = getattr(p_opt, param_name)
    values = np.linspace(opt_val * scale_range[0], opt_val * scale_range[1], n_points)
    profile = np.full(n_points, -np.inf)

    other_estimable = [n for n in estimable_names if n != param_name]

    for i, val in enumerate(values):
        fixed_i = dict(fixed_params)
        fixed_i[param_name] = val

        def obj(x: np.ndarray) -> float:
            p = _build_params(p_opt, fixed_i, other_estimable, x)
            try:
                times, states = simulate_dense(initial_state, events, p, t_span)
                interp_fn = interp1d(
                    times, states[:, 0], kind="linear",
                    bounds_error=False, fill_value="extrapolate",
                )
                preds = interp_fn(bgm_times)
                ll = compute_log_likelihood(observations, preds, p)
                lp = log_prior_fn(p)
                if not np.isfinite(ll + lp):
                    return 1e12
                return -(ll + lp)
            except Exception:
                return 1e12

        if other_estimable:
            x0 = np.array([getattr(p_opt, n) for n in other_estimable])
            res = minimize(obj, x0, method="L-BFGS-B", options={"maxiter": 50})
            profile[i] = -res.fun
        else:
            profile[i] = -obj(np.array([]))

    return values, profile
