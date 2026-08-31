"""Tests for src/events.py."""

import numpy as np
import pytest

from src.events import (
    Event, generate_mdi_schedule, generate_basal_only_schedule,
    EVENT_MEAL, EVENT_BOLUS, EVENT_BASAL, EVENT_BGM, EVENT_STRESS, EVENT_EXERCISE,
    _lognormal_sample,
)


def test_event_validation():
    with pytest.raises(ValueError):
        Event(0.0, "not_a_kind", 0.0)
    with pytest.raises(ValueError):
        Event(-1.0, EVENT_MEAL, 10.0)


def test_lognormal_sample_mean():
    rng = np.random.default_rng(0)
    vals = np.array([_lognormal_sample(rng, 45.0, 0.3) for _ in range(20000)])
    # Empirical mean should approximate the requested mean
    assert np.abs(vals.mean() - 45.0) / 45.0 < 0.05


def test_lognormal_sample_constant_cv_zero():
    rng = np.random.default_rng(0)
    assert _lognormal_sample(rng, 10.0, 0.0) == 10.0


def test_generate_mdi_schedule_days():
    events = generate_mdi_schedule(days=3, rng=np.random.default_rng(0))
    times = [e.time for e in events]
    assert max(times) < 3 * 24.0
    assert min(times) >= 0.0
    assert times == sorted(times)


def test_generate_mdi_schedule_contains_meals_and_boluses():
    days = 2
    events = generate_mdi_schedule(days=days, rng=np.random.default_rng(0))
    n_meals = sum(1 for e in events if e.kind == EVENT_MEAL)
    n_boluses = sum(1 for e in events if e.kind == EVENT_BOLUS)
    # 3 meals/day * days, each paired with a bolus
    assert n_meals == 3 * days
    assert n_boluses == 3 * days


def test_generate_mdi_schedule_bgm():
    days = 2
    events = generate_mdi_schedule(days=days, rng=np.random.default_rng(0))
    n_bgm = sum(1 for e in events if e.kind == EVENT_BGM)
    assert n_bgm == 6 * days


def test_generate_mdi_schedule_basal_positive():
    events = generate_mdi_schedule(days=1, rng=np.random.default_rng(0))
    basal = [e for e in events if e.kind == EVENT_BASAL]
    assert len(basal) == 1
    assert basal[0].amount >= 0.5


def test_generate_mdi_schedule_reproducible():
    e1 = generate_mdi_schedule(days=1, rng=np.random.default_rng(42))
    e2 = generate_mdi_schedule(days=1, rng=np.random.default_rng(42))
    assert [(e.time, e.kind, e.amount) for e in e1] == [(e.time, e.kind, e.amount) for e in e2]


def test_generate_basal_only_schedule():
    events = generate_basal_only_schedule(days=1)
    kinds = [e.kind for e in events]
    assert kinds.count(EVENT_BASAL) == 1
    assert kinds.count(EVENT_BGM) == 6


def test_event_input_validation():
    with pytest.raises(ValueError):
        generate_mdi_schedule(days=0)
    with pytest.raises(ValueError):
        generate_mdi_schedule(days=1, meal_times_h=[7, 12], carb_amounts_g=[45, 60, 55])
