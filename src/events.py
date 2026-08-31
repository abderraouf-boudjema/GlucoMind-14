"""Event generation for the T1D simulator.

Generates realistic MDI (Multiple Daily Injection) schedules with stochastic
variability in meal timing, meal size, carb counting, bolus dosing, stress,
and exercise events.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

EVENT_MEAL = "meal"
EVENT_BOLUS = "bolus"
EVENT_BASAL = "basal"
EVENT_BGM = "bgm"
EVENT_EXERCISE = "exercise"
EVENT_STRESS = "stress"

_VALID_EVENT_KINDS = {EVENT_MEAL, EVENT_BOLUS, EVENT_BASAL, EVENT_BGM,
                      EVENT_EXERCISE, EVENT_STRESS}


@dataclass
class Event:
    """A single simulation event (meal, bolus, basal, BGM, stress, exercise)."""
    time: float
    kind: str
    amount: float

    def __post_init__(self) -> None:
        if self.kind not in _VALID_EVENT_KINDS:
            raise ValueError(
                f"Unknown event kind '{self.kind}'. "
                f"Must be one of: {sorted(_VALID_EVENT_KINDS)}"
            )
        if self.time < 0:
            raise ValueError(f"Event time must be non-negative, got {self.time}")


def _lognormal_sample(
    rng: np.random.Generator, mean: float, cv: float,
) -> float:
    """Draw from a lognormal distribution parameterized by mean and CV.

    Uses the correct parameterization:
        sigma = sqrt(ln(1 + cv^2))
        mu = ln(mean) - sigma^2/2

    This ensures E[X] = mean exactly.
    """
    if cv <= 0:
        return mean
    sigma = np.sqrt(np.log(1.0 + cv ** 2))
    mu = np.log(mean) - 0.5 * sigma ** 2
    return float(rng.lognormal(mu, sigma))


def generate_mdi_schedule(
    days: int = 30,
    meal_times_h: Optional[list[float]] = None,
    carb_amounts_g: Optional[list[float]] = None,
    bolus_units: Optional[list[float]] = None,
    basal_dose_u: float = 12.0,
    basal_time_h: float = 22.0,
    bgm_times_h: Optional[list[float]] = None,
    rng: Optional[np.random.Generator] = None,
    meal_time_sd: float = 0.25,
    meal_size_cv: float = 0.3,
    carb_counting_error_cv: float = 0.2,
    bolus_error_cv: float = 0.1,
    exercise_days_per_week: int = 3,
    exercise_duration_h: float = 0.5,
    stress_probability: float = 0.3,
) -> list[Event]:
    """Generate a multi-day MDI event schedule with realistic variability.

    Args:
        days: Number of simulation days.
        meal_times_h: Nominal meal times within each day (hours).
        carb_amounts_g: Nominal carbohydrate amounts per meal (grams).
        bolus_units: Nominal bolus insulin per meal (units).
        basal_dose_u: Daily basal insulin dose (units).
        basal_time_h: Time of basal injection each day (hours).
        bgm_times_h: Blood glucose monitoring times each day (hours).
        rng: NumPy random generator for reproducibility.
        meal_time_sd: Standard deviation of meal timing noise (hours).
        meal_size_cv: Coefficient of variation for actual meal size.
        carb_counting_error_cv: CV for carb counting error.
        bolus_error_cv: CV for bolus dosing error.
        exercise_days_per_week: Average exercise days per week.
        exercise_duration_h: Nominal exercise duration (hours).
        stress_probability: Probability of a stress event per day.

    Returns:
        Sorted list of Event objects covering the full simulation period.
    """
    if days <= 0:
        raise ValueError(f"days must be positive, got {days}")
    if rng is None:
        rng = np.random.default_rng()

    if meal_times_h is None:
        meal_times_h = [7.0, 12.0, 18.0]
    if carb_amounts_g is None:
        carb_amounts_g = [45.0, 60.0, 55.0]
    if bolus_units is None:
        bolus_units = [4.0, 6.0, 5.0]
    if bgm_times_h is None:
        bgm_times_h = [8.0, 10.0, 12.0, 14.0, 20.0, 22.0]

    if len({len(meal_times_h), len(carb_amounts_g), len(bolus_units)}) != 1:
        raise ValueError(
            "meal_times_h, carb_amounts_g, and bolus_units must have same length"
        )

    carb_ratios = [c / b for c, b in zip(carb_amounts_g, bolus_units)]
    n_meals = len(meal_times_h)

    events: list[Event] = []

    for day in range(days):
        offset = day * 24.0
        daily_stress = rng.uniform() < stress_probability

        for m in range(n_meals):
            t_meal = meal_times_h[m] + rng.normal(0.0, meal_time_sd)
            t_meal = float(np.clip(
                t_meal,
                max(0.0, meal_times_h[m] - 1.0),
                meal_times_h[m] + 1.0,
            ))

            actual_carbs = _lognormal_sample(rng, carb_amounts_g[m], meal_size_cv)
            actual_carbs = max(actual_carbs, 5.0)

            counting_error = _lognormal_sample(rng, 1.0, carb_counting_error_cv)
            estimated_carbs = actual_carbs * counting_error

            computed_bolus = estimated_carbs / carb_ratios[m]

            dose_error = _lognormal_sample(rng, 1.0, bolus_error_cv)
            actual_bolus = computed_bolus * dose_error
            actual_bolus = max(actual_bolus, 0.1)

            events.append(Event(offset + t_meal, EVENT_MEAL, actual_carbs))
            events.append(Event(offset + t_meal, EVENT_BOLUS, actual_bolus))

        actual_basal = _lognormal_sample(rng, basal_dose_u, bolus_error_cv)
        actual_basal = max(actual_basal, 0.5)
        events.append(Event(offset + basal_time_h, EVENT_BASAL, actual_basal))

        for t_bgm in bgm_times_h:
            events.append(Event(offset + t_bgm, EVENT_BGM, 0.0))

        if daily_stress:
            stress_start = rng.uniform(6.0, 20.0)
            stress_duration = rng.uniform(2.0, 8.0)
            events.append(Event(offset + stress_start, EVENT_STRESS, stress_duration))

        if exercise_days_per_week > 0 and rng.uniform() < exercise_days_per_week / 7.0:
            ex_time = offset + rng.uniform(14.0, 18.0)
            ex_duration = _lognormal_sample(rng, exercise_duration_h, 0.2)
            ex_duration = max(ex_duration, 0.25)
            events.append(Event(ex_time, EVENT_EXERCISE, ex_duration))

    events.sort(key=lambda e: e.time)
    return events


def generate_basal_only_schedule(
    days: int = 30,
    basal_dose_u: float = 12.0,
    basal_time_h: float = 22.0,
    bgm_times_h: Optional[list[float]] = None,
) -> list[Event]:
    """Generate a minimal basal-only schedule with BGM readings.

    Args:
        days: Number of simulation days.
        basal_dose_u: Daily basal insulin dose (units).
        basal_time_h: Time of basal injection each day (hours).
        bgm_times_h: Blood glucose monitoring times each day (hours).

    Returns:
        Sorted list of Event objects.
    """
    if bgm_times_h is None:
        bgm_times_h = [8.0, 10.0, 12.0, 14.0, 20.0, 22.0]

    events: list[Event] = []
    for day in range(days):
        offset = day * 24.0
        events.append(Event(offset + basal_time_h, EVENT_BASAL, basal_dose_u))
        for t_bgm in bgm_times_h:
            events.append(Event(offset + t_bgm, EVENT_BGM, 0.0))

    events.sort(key=lambda e: e.time)
    return events


def iterate_event_windows(
    events: list[Event],
    t_start: float = 0.0,
    t_end: float = float("inf"),
):
    """Yield (t_start, t_end, event) tuples for each event window.

    Useful for stepping through the simulation interval-by-interval.
    """
    sorted_events = sorted(events, key=lambda e: e.time)
    t = t_start
    idx = 0
    while idx < len(sorted_events) and sorted_events[idx].time < t_start:
        idx += 1

    while idx < len(sorted_events) and (t_end == float("inf") or sorted_events[idx].time <= t_end):
        next_t = sorted_events[idx].time
        yield t, next_t, sorted_events[idx]
        t = next_t
        idx += 1

    if t < t_end:
        yield t, t_end, None
