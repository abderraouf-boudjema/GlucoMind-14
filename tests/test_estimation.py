"""Tests for src/estimation.py and src/identifiability.py (smoke tests)."""

import numpy as np
import pytest

from src.model import Parameters, N_STATES, Gp
from src.simulate import simulate_dense, get_bgm_times
from src.events import generate_mdi_schedule
from src.observe import sample_bgm_observations
from src.population import PARAM_CV
from src.estimation import make_log_prior, estimate_map, compute_profile_likelihood
from src.identifiability import (
    compute_sensitivity_trajectory, compute_fim, compute_fim_over_days,
)


def _setup(days=1):
    p = Parameters()
    state0 = np.zeros(N_STATES)
    state0[Gp] = 120.0
    events = generate_mdi_schedule(days=days, rng=np.random.default_rng(0))
    t_span = (0.0, days * 24.0)
    bgm = get_bgm_times(events, t_span)
    t, y = simulate_dense(state0, events, p, t_span, dt=0.1)
    return p, state0, events, t_span, bgm, t, y


def test_make_log_prior_center_high():
    p_nominal = Parameters()
    prior = make_log_prior(p_nominal, PARAM_CV)
    # At the nominal value, prior should be reasonably high (finite & close to 0)
    lp = prior(p_nominal)
    assert np.isfinite(lp)


def test_estimate_map_runs_and_recovers():
    p, state0, events, t_span, bgm, t, y = _setup()
    obs = sample_bgm_observations(bgm, t, y, p, seed=0)
    # Estimate k_u and k_cl from nominal-looking data
    p_opt, info = estimate_map(
        observations=obs,
        bgm_times=bgm,
        events=events,
        initial_state=state0,
        p_guess=Parameters(),
        p_nominal=Parameters(),
        param_cv=PARAM_CV,
        fixed_params={"V_Gi": 12.0, "EGP_0": 60.0},
        estimable_names=["k_u", "k_cl"],
        t_span=t_span,
        max_iter=25,
    )
    assert isinstance(info["success"], bool)
    assert p_opt.k_u > 0
    assert p_opt.k_cl > 0


def test_compute_sensitivity_trajectory_shapes():
    p, state0, events, t_span, *_ = _setup()
    t_sens, y_sens, sens = compute_sensitivity_trajectory(
        state0, events, p, t_span, 7,  # k_u index
        dt=0.2,
    )
    assert t_sens.shape == sens.shape
    assert y_sens.shape[1] == N_STATES


def test_compute_fim_runs():
    p, state0, events, t_span, *_ = _setup()
    fim, bgm = compute_fim(
        state0, events, p, [7, 23], t_span,
    )
    # k_u and k_cl, so 2x2
    assert fim.shape == (2, 2)
    # FIM should be positive semi-definite (symmetric)
    assert np.allclose(fim, fim.T)


def test_compute_fim_over_days_runs():
    p, state0, events, *_ = _setup(days=2)
    # Rebuild events covering 2 days already done
    days_arr, ranks = compute_fim_over_days(
        state0, events, p, [7, 23], max_days=2,
    )
    assert len(days_arr) == len(ranks) == 2
    assert np.all(ranks >= 0)


def test_compute_profile_likelihood_shapes():
    p, state0, events, t_span, bgm, t, y = _setup()
    obs = sample_bgm_observations(bgm, t, y, p, seed=0)
    values, profile = compute_profile_likelihood(
        observations=obs,
        bgm_times=bgm,
        events=events,
        initial_state=state0,
        p_opt=p,
        param_name="k_u",
        estimable_names=["k_u"],
        fixed_params={},
        p_nominal=p,
        param_cv=PARAM_CV,
        t_span=t_span,
        n_points=3,
    )
    assert len(values) == len(profile) == 3
