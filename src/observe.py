"""Observation model for blood glucose monitoring (BGM).

Implements heteroscedastic Gaussian noise on plasma glucose observations,
consistent with real-world BGM meter accuracy.
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import interp1d

from .model import Parameters, Gp


def bgm_noise_sd(g_p: np.ndarray, p: Parameters) -> np.ndarray:
    """Compute the observation noise standard deviation at a given glucose level.

    Uses a linear heteroscedastic model: SD = alpha + beta * Gp.
    This reflects the fact that BGM meters are less accurate at high glucose.

    Accepts a scalar or array of glucose concentrations.

    Args:
        g_p: Plasma glucose concentration(s) (mg/dL).
        p: Model parameters (uses alpha and beta).

    Returns:
        Standard deviation(s) of the BGM noise (mg/dL).
    """
    return p.alpha + p.beta * np.maximum(np.asarray(g_p), 0.0)


def sample_bgm_observations(
    t_bgm: np.ndarray,
    t_sim: np.ndarray,
    y_sim: np.ndarray,
    p: Parameters,
    seed: int = 42,
) -> np.ndarray:
    """Generate synthetic BGM observations from a simulated trajectory.

    Interpolates the true plasma glucose at BGM times and adds
    heteroscedastic Gaussian noise. Observations are clamped to >= 0.

    Args:
        t_bgm: BGM measurement times (hours).
        t_sim: Simulation time points (hours).
        y_sim: Simulation states, shape (N, 11).
        p: Model parameters.
        seed: Random seed for reproducibility.

    Returns:
        Noisy BGM observations (mg/dL), shape (len(t_bgm),).
    """
    if len(t_bgm) == 0:
        return np.array([])

    rng = np.random.default_rng(seed)
    interp = interp1d(
        t_sim, y_sim[:, Gp], kind="linear",
        bounds_error=False, fill_value="extrapolate",
    )
    g_p_true = interp(t_bgm)
    sds = bgm_noise_sd(np.maximum(g_p_true, 0.0), p)
    noise = rng.normal(0.0, sds)
    return np.maximum(g_p_true + noise, 0.0)


def compute_log_likelihood(
    observations: np.ndarray,
    predictions: np.ndarray,
    p: Parameters,
) -> float:
    """Compute the Gaussian log-likelihood of observations given predictions.

    Uses the heteroscedastic noise model from bgm_noise_sd().

    Args:
        observations: Observed BGM values (mg/dL).
        predictions: Predicted plasma glucose at observation times (mg/dL).
        p: Model parameters.

    Returns:
        Log-likelihood value (scalar).
    """
    if len(observations) == 0:
        return 0.0

    sds = bgm_noise_sd(np.maximum(predictions, 0.0), p)
    resid = observations - predictions
    n = len(observations)

    # Guard against zero or negative SDs
    sds = np.maximum(sds, 1e-10)

    ll = -0.5 * n * np.log(2.0 * np.pi)
    ll -= np.sum(np.log(sds))
    ll -= 0.5 * np.sum((resid / sds) ** 2)
    return float(ll)
