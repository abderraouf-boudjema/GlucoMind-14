"""Virtual patient population generation.

Samples physiologically realistic parameter sets using lognormal distributions
with an insulin sensitivity (IS) factor that couples k_u, k_cl, k_x, and k_EGP
to preserve inter-parameter correlations observed in clinical data.
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np

from .model import Parameters, PARAM_NAMES


# Log-normal means (log of nominal values)
PARAM_LOG_MEAN: dict[str, float] = {
    "k_gp": 1.7918, "k_g2": -0.6931, "k_g3": -1.6094,
    "V_G": 0.6313, "V_Gi": 2.4849,
    "U_b": 0.0, "k_u": -2.5257, "K_m": 4.4998,
    "EGP_0": 4.0943, "k_EGP": -4.1997,
    "k_gri": -0.2231, "k_min": -1.6094, "k_max": 1.0986,
    "a": -2.9957, "b": -0.1054, "c": -2.9957, "d": -2.3026,
    "beta_meal": 1.7918,
    "k_d": 1.3863, "k_a1": 1.0986, "k_a2": -0.9163,
    "k_b1": -2.9957, "k_b2": -3.2189, "k_cl": 2.0794, "k_x": -0.6931,
    "V_I": 2.4849,
    "alpha": 0.6931, "beta": -2.9957,
    "k_ge": -3.9120, "circ_amp": -1.8971,
}

# Coefficients of variation for inter-patient variability
PARAM_CV: dict[str, float] = {
    "k_gp": 0.3, "k_g2": 0.3, "k_g3": 0.3, "V_G": 0.2, "V_Gi": 0.2,
    "U_b": 0.3, "k_u": 0.5, "K_m": 0.2,
    "EGP_0": 0.4, "k_EGP": 0.4,
    "k_gri": 0.4, "k_min": 0.4, "k_max": 0.4,
    "a": 0.3, "b": 0.25, "c": 0.3, "d": 0.25, "beta_meal": 0.4,
    "k_d": 0.3, "k_a1": 0.3, "k_a2": 0.3,
    "k_b1": 0.3, "k_b2": 0.3, "k_cl": 0.4, "k_x": 0.4, "V_I": 0.2,
    "alpha": 0.1, "beta": 0.1,
    "k_ge": 0.4, "circ_amp": 0.3,
}

# Physiological bounds
PARAM_LOWER: dict[str, float] = {
    "k_gp": 0.5, "k_g2": 0.1, "k_g3": 0.02, "V_G": 0.5, "V_Gi": 3.0,
    "U_b": 0.1, "k_u": 0.005, "K_m": 30.0,
    "EGP_0": 10.0, "k_EGP": 0.002,
    "k_gri": 0.05, "k_min": 0.02, "k_max": 0.3,
    "a": 0.01, "b": 0.5, "c": 0.01, "d": 0.05, "beta_meal": 0.5,
    "k_d": 0.5, "k_a1": 0.3, "k_a2": 0.05,
    "k_b1": 0.01, "k_b2": 0.01, "k_cl": 1.0, "k_x": 0.05, "V_I": 2.0,
    "alpha": 0.5, "beta": 0.01,
    "k_ge": 0.002, "circ_amp": 0.02,
}

PARAM_UPPER: dict[str, float] = {
    "k_gp": 20.0, "k_g2": 3.0, "k_g3": 2.0, "V_G": 5.0, "V_Gi": 30.0,
    "U_b": 10.0, "k_u": 0.5, "K_m": 300.0,
    "EGP_0": 200.0, "k_EGP": 0.05,
    "k_gri": 3.0, "k_min": 1.0, "k_max": 10.0,
    "a": 0.2, "b": 1.0, "c": 0.2, "d": 0.5, "beta_meal": 20.0,
    "k_d": 10.0, "k_a1": 10.0, "k_a2": 2.0,
    "k_b1": 0.3, "k_b2": 0.3, "k_cl": 20.0, "k_x": 2.0, "V_I": 25.0,
    "alpha": 10.0, "beta": 0.2,
    "k_ge": 0.2, "circ_amp": 0.5,
}


def sample_virtual_patient(
    rng: np.random.Generator,
    p_nominal: Optional[Parameters] = None,
) -> tuple[Parameters, list[str]]:
    """Sample a single virtual patient's parameters.

    Parameters are drawn from lognormal distributions centered on the nominal
    values, with an insulin sensitivity (IS) factor that couples four
    correlated parameters:
        - k_u  (renal clearance)      ↑ with IS
        - k_cl (plasma insulin clear) ↓ with IS
        - k_x  (remote action)        ↓ with sqrt(IS)
        - k_EGP (EGP suppression)     ↑ with IS

    Args:
        rng: NumPy random generator.
        p_nominal: Nominal parameter set (unused, kept for API consistency).

    Returns:
        (parameters, warnings) where warnings lists any clipped parameters.
    """
    kwargs: dict[str, float] = {}
    clip_warnings: list[str] = []

    is_factor = float(np.clip(rng.lognormal(0.0, 0.4), 0.3, 3.0))

    for name in PARAM_NAMES:
        mu = PARAM_LOG_MEAN[name]
        cv = PARAM_CV[name]
        sigma = np.sqrt(np.log(1.0 + cv ** 2))
        val = float(rng.lognormal(mu, sigma))

        lo = PARAM_LOWER[name]
        hi = PARAM_UPPER[name]
        if val < lo:
            clip_warnings.append(f"{name}: {val:.4f} < lower bound {lo}, clipped")
            val = lo
        elif val > hi:
            clip_warnings.append(f"{name}: {val:.4f} > upper bound {hi}, clipped")
            val = hi
        kwargs[name] = val

    # Apply IS factor coupling
    def _clip(name: str, val: float) -> float:
        lo, hi = PARAM_LOWER[name], PARAM_UPPER[name]
        clipped = float(np.clip(val, lo, hi))
        if clipped != val:
            clip_warnings.append(
                f"{name} (IS-coupled): {val:.4f} -> {clipped:.4f}"
            )
        return clipped

    kwargs["k_u"] = _clip("k_u", kwargs["k_u"] * is_factor)
    kwargs["k_cl"] = _clip("k_cl", kwargs["k_cl"] / is_factor)
    kwargs["k_x"] = _clip("k_x", kwargs["k_x"] / np.sqrt(is_factor))
    kwargs["k_EGP"] = _clip("k_EGP", kwargs["k_EGP"] * is_factor)

    return Parameters(**kwargs), clip_warnings


def generate_population(
    n_patients: int,
    p_nominal: Optional[Parameters] = None,
    seed: int = 123,
    warn_on_clip: bool = False,
) -> list[Parameters]:
    """Generate a population of virtual patients.

    Args:
        n_patients: Number of patients to generate.
        p_nominal: Nominal parameter set (unused, kept for API consistency).
        seed: Random seed for reproducibility.
        warn_on_clip: If True, emit warnings when parameters are clipped.

    Returns:
        List of Parameters objects, one per patient.
    """
    if n_patients <= 0:
        raise ValueError(f"n_patients must be positive, got {n_patients}")

    rng = np.random.default_rng(seed)
    patients: list[Parameters] = []
    total_clips = 0

    for _ in range(n_patients):
        p, clips = sample_virtual_patient(rng, p_nominal)
        total_clips += len(clips)
        if warn_on_clip and clips:
            for c in clips:
                warnings.warn(c, UserWarning, stacklevel=2)
        patients.append(p)

    if warn_on_clip and total_clips > 0:
        warnings.warn(
            f"Total parameter clips across {n_patients} patients: {total_clips}",
            UserWarning,
            stacklevel=2,
        )

    return patients
