"""Tests for src/model.py."""

import numpy as np
import pytest

from src.model import (
    Parameters, N_STATES, Gp, Gi, Qsto1, Qsto2,
    Isc1, Ibase1, Ip, X, STATE_NAMES,
    circadian_factor, gastric_emptying_rate, renal_switch,
    michaelis_menten, egp_production, rhs,
)


def test_parameters_roundtrip():
    p = Parameters()
    arr = p.as_array()
    assert arr.shape == (len(p.as_array()),)
    p2 = Parameters.from_array(arr)
    for name in Parameters.__dataclass_fields__:
        assert getattr(p, name) == pytest.approx(getattr(p2, name), rel=1e-9)


def test_number_of_parameters():
    p = Parameters()
    assert len(p.as_array()) == 30


def test_state_names_match_index():
    assert len(STATE_NAMES) == N_STATES == 11
    # Spot check critical indices
    assert STATE_NAMES[Gp] == "Gp"
    assert STATE_NAMES[Gi] == "Gi"
    assert STATE_NAMES[Qsto1] == "Qsto1"
    assert STATE_NAMES[Ip] == "Ip"
    assert STATE_NAMES[X] == "X"


def test_circadian_factor_range():
    p = Parameters()
    for t in np.linspace(0, 24, 100):
        c = circadian_factor(t, p.circ_amp)
        assert 1.0 - p.circ_amp - 1e-9 <= c <= 1.0 + p.circ_amp + 1e-9


def test_circadian_factor_periodic():
    p = Parameters()
    assert circadian_factor(0, p.circ_amp) == pytest.approx(circadian_factor(24, p.circ_amp))


def test_gastric_emptying_bounds():
    p = Parameters()
    # Empty stomach should give emptying rate near k_min
    rate_empty = gastric_emptying_rate(0.0, 0.0, p)
    # Full stomach should give rate up toward k_max but bounded
    rate_full = gastric_emptying_rate(500.0, 60.0, p)
    assert rate_empty >= 0
    assert 0 <= rate_full <= 1.2 * p.k_max


def test_renal_switch_threshold():
    p = Parameters()
    # Below threshold, no excess renal clearance term is pathological
    low = renal_switch(120.0, p)
    high = renal_switch(220.0, p)
    # More glucose -> more negative (more clearance)
    assert high < low


def test_egp_nonnegative():
    # Very high insulin action should suppress EGP to zero, not negative
    egp = egp_production(60.0, 0.5, 10.0)
    assert egp >= 0.0


def test_michaelis_menten_bounds():
    g = michaelis_menten(90.0, 1.0, 0.08, 1.0, 90.0)
    assert g > 0
    # At very high glucose approaches asymptotic value
    g_inf = michaelis_menten(1e6, 1.0, 0.08, 1.0, 90.0)
    assert g <= g_inf


def test_rhs_shape_and_finite():
    p = Parameters()
    y = np.zeros(N_STATES)
    y[Gp] = 120.0
    y[Gi] = 90.0
    dy = rhs(12.0, y, p)
    assert dy.shape == (N_STATES,)
    assert np.all(np.isfinite(dy))


def test_rhs_meal_changes_state_flow():
    p = Parameters()
    y = np.zeros(N_STATES)
    y[Qsto2] = 60.0  # food in liquid stomach
    dy = rhs(12.0, y, p, last_cho=60.0)
    # Glucose should rise from meal appearance (R_meal = beta_meal * Qsto2)
    assert dy[Gp] > 0
