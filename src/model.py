"""GlucoMind-14: 11-state glucose-insulin ODE model for Type 1 Diabetes.

This module defines the physiological model, parameter space, and right-hand
side of the ODE system used by the simulator. The model is based on the
Dalla Man et al. (2006, 2007) glucose-insulin framework with extensions for
renal clearance, circadian modulation, stress, and exercise.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, fields

N_STATES = 11

# State indices
Gp: int = 0      # Plasma glucose (mg/dL)
Gi: int = 1      # Interstitial glucose (mg/dL)
Qsto1: int = 2   # Stomach content, solid (g)
Qsto2: int = 3   # Stomach content, liquid (g)
Isc1: int = 4     # Subcutaneous insulin compartment 1 (mU)
Isc2: int = 5     # Subcutaneous insulin compartment 2 (mU)
Ibase1: int = 6   # Basal insulin compartment 1 (mU)
Ibase2: int = 7   # Basal insulin compartment 2 (mU)
Ibase: int = 8    # Basal insulin, active (mU)
Ip: int = 9       # Plasma insulin (mU/L)
X: int = 10       # Remote insulin action (dimensionless)

STATE_NAMES: list[str] = [
    "Gp", "Gi", "Qsto1", "Qsto2",
    "Isc1", "Isc2", "Ibase1", "Ibase2", "Ibase",
    "Ip", "X",
]

RENAL_THRESHOLD: float = 180.0
U_TO_MU: float = 1000.0
RENAL_SLOPE: float = 0.3


@dataclass(frozen=False)
class Parameters:
    """Model parameters with physiological units and default (nominal) values.

    All rate constants are in h^{-1} unless otherwise noted. Volume parameters
    are in dL or L. Concentration parameters are in mg/dL or mU/L.
    """

    # Glucose subsystem
    k_gp: float = 6.0       # h^-1, plasma-to-interstitial transport
    k_g2: float = 0.5       # h^-1, glucose disposal (interstitial)
    k_g3: float = 0.2       # h^-1, glucose disposal (interstitial, slow)
    V_G: float = 1.88       # dL/kg, plasma glucose distribution volume
    V_Gi: float = 12.0      # dL, interstitial glucose distribution volume

    # Renal / urinary
    U_b: float = 1.0        # mg/dL/h, basal renal clearance
    k_u: float = 0.08       # dL/(mU*h), renal clearance rate constant
    K_m: float = 90.0       # mg/dL, Michaelis-Menten half-saturation

    # Endogenous glucose production
    EGP_0: float = 60.0     # mg/h, basal EGP rate
    k_EGP: float = 0.015    # L/mU, EGP suppression by insulin action

    # Gastric emptying
    k_gri: float = 0.8      # h^-1, gastric emptying rate (solid->liquid)
    k_min: float = 0.2      # h^-1, minimum liquid emptying rate
    k_max: float = 3.0      # h^-1, maximum liquid emptying rate
    a: float = 0.05         # dimensionless, emptying shape parameter
    b: float = 0.9          # dimensionless, emptying shape parameter
    c: float = 0.05         # dimensionless, emptying shape parameter
    d: float = 0.1          # dimensionless, emptying shape parameter
    beta_meal: float = 6.0  # (mg/dL/h)/g, meal-to-glucose appearance

    # Insulin subsystem
    k_d: float = 4.0        # h^-1, subcutaneous insulin degradation
    k_a1: float = 3.0       # h^-1, subcutaneous -> plasma (bolus)
    k_a2: float = 0.4       # h^-1, basal -> plasma
    k_b1: float = 0.05      # h^-1, basal compartment 1 -> 2
    k_b2: float = 0.04      # h^-1, basal compartment 2 -> active
    k_cl: float = 8.0       # h^-1, plasma insulin clearance
    k_x: float = 0.5        # h^-1, remote insulin action dynamics
    V_I: float = 12.0       # L, plasma insulin distribution volume

    # Insulin-glucose coupling
    alpha: float = 2.0      # mg/dL, BGM noise intercept
    beta: float = 0.05      # dimensionless, BGM noise slope

    # Extensions
    k_ge: float = 0.01      # h^-1, glucose effectiveness (minimal model)
    circ_amp: float = 0.15  # dimensionless, circadian amplitude

    def as_array(self) -> np.ndarray:
        """Serialize parameters to a flat numpy array (30 elements)."""
        return np.array([getattr(self, f.name) for f in fields(self)])

    @classmethod
    def from_array(cls, arr: np.ndarray) -> Parameters:
        """Deserialize parameters from a flat numpy array.

        Supports arrays of length 28 (original) or 30 (with k_ge, circ_amp).
        """
        names = [f.name for f in fields(cls)]
        kwargs = {}
        for i, name in enumerate(names):
            if i < len(arr):
                kwargs[name] = float(arr[i])
        return cls(**kwargs)

    def validate(self) -> list[str]:
        """Return a list of validation warnings for out-of-range parameters."""
        from .population import PARAM_LOWER, PARAM_UPPER
        warnings = []
        for f in fields(self):
            val = getattr(self, f.name)
            lo = PARAM_LOWER.get(f.name)
            hi = PARAM_UPPER.get(f.name)
            if lo is not None and val < lo:
                warnings.append(f"{f.name}={val:.4f} below lower bound {lo}")
            if hi is not None and val > hi:
                warnings.append(f"{f.name}={val:.4f} above upper bound {hi}")
        return warnings


PARAM_NAMES: list[str] = [f.name for f in fields(Parameters)]

PARAM_UNITS: list[str] = [
    "h⁻¹", "h⁻¹", "h⁻¹", "dL/kg", "dL",
    "mg/dL/h", "dL/(mU·h)", "mg/dL",
    "mg/h", "L/mU",
    "h⁻¹", "h⁻¹", "h⁻¹", "—", "—", "—", "—", "(mg/dL/h)/g",
    "h⁻¹", "h⁻¹", "h⁻¹", "h⁻¹", "h⁻¹", "h⁻¹", "h⁻¹", "L",
    "mg/dL", "—",
    "h⁻¹", "—",
]


def circadian_factor(time_of_day_h: float, amp: float) -> float:
    """Compute the circadian modulation factor.

    Peak insulin sensitivity occurs at ~4:00 (trough of cosine).
    Returns values in [1-amp, 1+amp].
    """
    return 1.0 - amp * np.cos(2.0 * np.pi * (time_of_day_h - 4.0) / 24.0)


def renal_switch(g_p: float, p: Parameters) -> float:
    """Renal glucose clearance with smooth threshold at 180 mg/dL.

    Uses logaddexp for numerical stability of the smooth max.
    """
    excess = g_p - RENAL_THRESHOLD
    smooth_excess = np.logaddexp(0.0, excess)
    return -p.k_g2 * g_p - p.k_gp * smooth_excess


def gastric_emptying_rate(q_sto: float, cho: float, p: Parameters) -> float:
    """Compute the gastric emptying rate using the Dalla Man tanh model.

    Args:
        q_sto: Total stomach content (solid + liquid), in grams.
        cho: Carbohydrate amount of the last meal, in grams.
        p: Model parameters.
    """
    arg1 = p.a * (q_sto - p.b * cho)
    arg2 = p.c * (q_sto - p.d * cho)
    return p.k_min + (p.k_max - p.k_min) / 2.0 * (
        np.tanh(arg1) - np.tanh(arg2) + 2.0
    )


def michaelis_menten(
    g_i: float, u_b: float, k_u_eff: float, x_val: float, k_m: float,
) -> float:
    """Michaelis-Menten insulin-mediated glucose uptake."""
    return (u_b + k_u_eff * x_val) * g_i / (g_i + k_m)


def egp_production(egp_0: float, k_egp_eff: float, x_val: float) -> float:
    """Endogenous glucose production, suppressed by remote insulin action.

    Returns max(0, EGP_0 * (1 - k_EGP * X)) to prevent negative production.
    """
    return egp_0 * max(0.0, 1.0 - k_egp_eff * x_val)


def rhs(
    t: float,
    y: np.ndarray,
    p: Parameters,
    last_cho: float = 0.0,
    stress_level: float = 0.0,
    exercise_level: float = 0.0,
) -> np.ndarray:
    """Right-hand side of the 11-state ODE system.

    Args:
        t: Current time in hours.
        y: State vector [Gp, Gi, Qsto1, Qsto2, Isc1, Isc2, Ibase1, Ibase2, Ibase, Ip, X].
        p: Model parameters.
        last_cho: Carbohydrate amount of the most recent meal (g).
        stress_level: Stress level in [0, 1]. Reduces insulin clearance,
            boosts EGP.
        exercise_level: Exercise level in [0, 1]. Increases glucose uptake.

    Returns:
        Time derivative dy/dt, shape (11,).
    """
    dy = np.zeros(N_STATES)

    G_p_val = max(y[Gp], 0.0)
    G_i_val = max(y[Gi], 0.0)
    Q_s1 = max(y[Qsto1], 0.0)
    Q_s2 = max(y[Qsto2], 0.0)
    I_s1 = max(y[Isc1], 0.0)
    I_s2 = max(y[Isc2], 0.0)
    I_b1 = max(y[Ibase1], 0.0)
    I_b2 = max(y[Ibase2], 0.0)
    I_b = max(y[Ibase], 0.0)
    I_p_val = max(y[Ip], 0.0)
    X_val = max(y[X], 0.0)

    t_mod = t % 24.0
    circ = circadian_factor(t_mod, p.circ_amp)
    k_u_eff = p.k_u * circ
    k_EGP_eff = p.k_EGP * circ

    stress_is_effect = stress_level * 0.3
    k_u_eff *= (1.0 - stress_is_effect)
    k_EGP_eff *= (1.0 - stress_is_effect)
    stress_egp_boost = stress_level * p.EGP_0 * 0.15

    exercise_uptake = exercise_level * 0.3 * G_p_val

    Q_sto = Q_s1 + Q_s2
    k_empt = gastric_emptying_rate(Q_sto, last_cho, p)
    R_meal = p.beta_meal * Q_s2
    EGP = egp_production(p.EGP_0, k_EGP_eff, X_val) + stress_egp_boost
    U_gi = michaelis_menten(G_i_val, p.U_b, k_u_eff, X_val, p.K_m)

    dy[Gp] = renal_switch(G_p_val, p) + EGP + R_meal - p.k_ge * G_p_val - exercise_uptake
    dy[Gi] = p.k_g2 * p.V_G / p.V_Gi * G_p_val - p.k_g3 * G_i_val - U_gi
    dy[Qsto1] = -p.k_gri * Q_s1
    dy[Qsto2] = p.k_gri * Q_s1 - k_empt * Q_s2
    dy[Isc1] = -p.k_d * I_s1
    dy[Isc2] = p.k_d * I_s1 - p.k_a1 * I_s2
    dy[Ibase1] = -p.k_b1 * I_b1
    dy[Ibase2] = p.k_b1 * I_b1 - p.k_b2 * I_b2
    dy[Ibase] = p.k_b2 * I_b2 - p.k_a2 * I_b
    dy[Ip] = (p.k_a1 * I_s2 + p.k_a2 * I_b) / p.V_I - p.k_cl * I_p_val
    dy[X] = -p.k_x * X_val + p.k_x * I_p_val

    return dy


def event_gp_180(t: float, y: np.ndarray, p: Parameters, last_cho: float) -> float:
    """Event function: triggers when plasma glucose crosses 180 mg/dL."""
    return y[Gp] - RENAL_THRESHOLD


event_gp_180.terminal = False
event_gp_180.direction = 0
