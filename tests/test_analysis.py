"""Tests for src/analysis.py."""

import numpy as np
import pytest

from src.analysis import (
    time_in_range, time_below_range, time_above_range,
    mean_glucose, glucose_std, fasting_glucose, postprandial_peak,
    compute_population_metrics,
)


def test_tir_in_range():
    trace = np.array([100.0, 100.0, 100.0, 100.0])
    assert time_in_range(trace) == 1.0


def test_tir_out_of_range():
    trace = np.array([200.0, 200.0, 60.0, 60.0])
    assert time_in_range(trace) == 0.0


def test_tir_mixed():
    trace = np.array([100.0, 200.0, 50.0, 100.0])
    assert time_in_range(trace) == 0.5


def test_time_below():
    trace = np.array([50.0, 60.0, 70.0, 80.0])
    assert time_below_range(trace, low=70.0) == 0.5


def test_time_above():
    trace = np.array([200.0, 100.0, 100.0, 250.0])
    assert time_above_range(trace, high=180.0) == 0.5


def test_mean_glucose():
    trace = np.array([100.0, 200.0, 300.0])
    assert mean_glucose(trace) == pytest.approx(200.0)


def test_glucose_std():
    trace = np.array([100.0, 200.0])
    assert glucose_std(trace) == pytest.approx(50.0)


def test_fasting_glucose_basic():
    t = np.array([7.0, 7.0, 12.0, 12.0])
    g = np.array([90.0, 110.0, 200.0, 220.0])
    # Only the 7:00 samples are within the window
    assert fasting_glucose(g, t, pre_breakfast_time=7.0, window_minutes=15.0) == pytest.approx(100.0)


def test_postprandial_peak():
    t = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    g = np.array([100.0, 150.0, 200.0, 180.0, 120.0])
    # meal at t=0, window 0-3h
    assert postprandial_peak(g, t, meal_time=0.0, window_h=3.0) == pytest.approx(200.0)


def test_compute_population_metrics_structure():
    rng = np.random.default_rng(0)
    # 5 patients, 100 timepoints
    traces = rng.uniform(80, 200, size=(5, 100))
    t = np.linspace(0, 24, 100)
    metrics = compute_population_metrics(traces, t)
    for key in ("time_in_range", "time_below", "time_above",
                "mean_glucose", "glucose_std", "fasting_glucose"):
        assert key in metrics
        assert len(metrics[key]) == 4  # (mean, std, p5, p95)
