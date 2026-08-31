"""Glycemic metrics computation for simulated trajectories.

Provides standard clinical metrics (TIR, TBR, TAR, mean glucose, etc.)
for both individual and population-level analysis.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def time_in_range(
    gp_trace: np.ndarray,
    low: float = 70.0,
    high: float = 180.0,
) -> float:
    """Fraction of time plasma glucose is in [low, high] (mg/dL)."""
    in_range = np.sum((gp_trace >= low) & (gp_trace <= high))
    return float(in_range / len(gp_trace))


def time_below_range(gp_trace: np.ndarray, low: float = 70.0) -> float:
    """Fraction of time plasma glucose is below low (mg/dL)."""
    return float(np.sum(gp_trace < low) / len(gp_trace))


def time_above_range(gp_trace: np.ndarray, high: float = 180.0) -> float:
    """Fraction of time plasma glucose is above high (mg/dL)."""
    return float(np.sum(gp_trace > high) / len(gp_trace))


def mean_glucose(gp_trace: np.ndarray) -> float:
    """Mean plasma glucose (mg/dL)."""
    return float(np.mean(gp_trace))


def glucose_std(gp_trace: np.ndarray) -> float:
    """Standard deviation of plasma glucose (mg/dL)."""
    return float(np.std(gp_trace))


def fasting_glucose(
    gp_trace: np.ndarray,
    t_hours: np.ndarray,
    pre_breakfast_time: float = 7.0,
    window_minutes: float = 15.0,
) -> float:
    """Estimate fasting glucose from a window around the pre-breakfast time.

    Args:
        gp_trace: Plasma glucose trajectory.
        t_hours: Time points in hours.
        pre_breakfast_time: Nominal pre-breakfast time (hours).
        window_minutes: Half-width of the averaging window (minutes).

    Returns:
        Mean glucose in the fasting window.
    """
    window_h = window_minutes / 60.0
    mask = np.abs(t_hours % 24.0 - pre_breakfast_time) <= window_h
    if np.sum(mask) == 0:
        return float(np.mean(gp_trace[:10]))
    return float(np.mean(gp_trace[mask]))


def postprandial_peak(
    gp_trace: np.ndarray,
    t_hours: np.ndarray,
    meal_time: float,
    window_h: float = 3.0,
) -> float:
    """Peak glucose within a postprandial window after a meal.

    Args:
        gp_trace: Plasma glucose trajectory.
        t_hours: Time points in hours.
        meal_time: Time of the meal (hours).
        window_h: Duration of the postprandial window (hours).

    Returns:
        Maximum glucose in the postprandial window.
    """
    mask = (t_hours >= meal_time) & (t_hours <= meal_time + window_h)
    if np.sum(mask) == 0:
        return float(np.max(gp_trace))
    return float(np.max(gp_trace[mask]))


def compute_population_metrics(
    gp_traces: np.ndarray,
    t_hours: np.ndarray,
    meal_times: Optional[list[float]] = None,
) -> dict[str, tuple[float, float, float, float]]:
    """Compute glycemic metrics for a population of simulated patients.

    Args:
        gp_traces: Glucose traces, shape (n_patients, n_timepoints).
        t_hours: Time points in hours.
        meal_times: Nominal meal times for fasting glucose computation.

    Returns:
        Dict mapping metric name to (mean, std, p5, p95) tuple.
    """
    if meal_times is None:
        meal_times = [7.0, 12.0, 18.0]

    metrics: dict[str, list[float]] = {
        "time_in_range": [],
        "time_below": [],
        "time_above": [],
        "mean_glucose": [],
        "glucose_std": [],
        "fasting_glucose": [],
    }

    for i in range(gp_traces.shape[0]):
        trace = gp_traces[i]
        metrics["time_in_range"].append(time_in_range(trace))
        metrics["time_below"].append(time_below_range(trace))
        metrics["time_above"].append(time_above_range(trace))
        metrics["mean_glucose"].append(mean_glucose(trace))
        metrics["glucose_std"].append(glucose_std(trace))
        metrics["fasting_glucose"].append(fasting_glucose(trace, t_hours))

    summary: dict[str, tuple[float, float, float, float]] = {}
    for k, v in metrics.items():
        arr = np.array(v)
        summary[k] = (
            float(np.mean(arr)),
            float(np.std(arr)),
            float(np.percentile(arr, 5)),
            float(np.percentile(arr, 95)),
        )
    return summary
