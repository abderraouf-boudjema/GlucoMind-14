"""GlucoMind-14: Type 1 Diabetes glucose-insulin simulator.

A physiologically grounded ODE model for simulating glucose-insulin dynamics
in Type 1 Diabetes patients under Multiple Daily Injection (MDI) therapy.

Public API:
    - Parameters: Model parameter container
    - simulate / simulate_dense: ODE integration
    - Event / generate_mdi_schedule: Event scheduling
    - sample_bgm_observations: Observation model
    - generate_population: Virtual patient generation
    - time_in_range / mean_glucose / etc.: Glycemic metrics
"""

from .model import (
    Parameters,
    PARAM_NAMES,
    PARAM_UNITS,
    N_STATES,
    Gp, Gi, Qsto1, Qsto2, Isc1, Isc2, Ibase1, Ibase2, Ibase, Ip, X,
    STATE_NAMES,
)
from .simulate import simulate, simulate_dense, SimulationError, get_bgm_times
from .events import Event, generate_mdi_schedule, generate_basal_only_schedule
from .observe import sample_bgm_observations, compute_log_likelihood, bgm_noise_sd
from .population import (
    generate_population,
    PARAM_LOWER,
    PARAM_UPPER,
    PARAM_CV,
)
from .analysis import (
    time_in_range,
    time_below_range,
    time_above_range,
    mean_glucose,
    glucose_std,
    fasting_glucose,
    compute_population_metrics,
)
from .estimation import estimate_map, compute_profile_likelihood
from .identifiability import (
    compute_sensitivity_trajectory,
    compute_fim,
    compute_fim_over_days,
)

__all__ = [
    "Parameters",
    "PARAM_NAMES",
    "PARAM_UNITS",
    "N_STATES",
    "STATE_NAMES",
    "simulate",
    "simulate_dense",
    "SimulationError",
    "get_bgm_times",
    "Event",
    "generate_mdi_schedule",
    "generate_basal_only_schedule",
    "sample_bgm_observations",
    "compute_log_likelihood",
    "bgm_noise_sd",
    "generate_population",
    "PARAM_LOWER",
    "PARAM_UPPER",
    "PARAM_CV",
    "time_in_range",
    "time_below_range",
    "time_above_range",
    "mean_glucose",
    "glucose_std",
    "fasting_glucose",
    "compute_population_metrics",
    "estimate_map",
    "compute_profile_likelihood",
    "compute_sensitivity_trajectory",
    "compute_fim",
    "compute_fim_over_days",
]
