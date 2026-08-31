"""Tests for src/observe.py."""

import numpy as np
import pytest

from src.model import Parameters, N_STATES, Gp
from src.observe import bgm_noise_sd, sample_bgm_observations, compute_log_likelihood
from src.simulate import simulate_dense, get_bgm_times
from src.events import generate_mdi_schedule


def _sample_trajectory(days=1, seed=0):
    p = Parameters()
    state0 = np.zeros(N_STATES)
    state0[Gp] = 120.0
    events = generate_mdi_schedule(days=days, rng=np.random.default_rng(seed))
    t_span = (0.0, days * 24.0)
    bgm = get_bgm_times(events, t_span)
    t, y = simulate_dense(state0, events, p, t_span, dt=0.1)
    return t, y, p, bgm


def test_noise_sd_increases_with_glucose():
    p = Parameters()
    assert bgm_noise_sd(100, p) < bgm_noise_sd(300, p)


def test_noise_sd_nonnegative():
    p = Parameters()
    for g in [0, 50, 100, 200]:
        assert bgm_noise_sd(g, p) > 0


def test_sample_bgm_observations_length():
    t, y, p, bgm = _sample_trajectory()
    obs = sample_bgm_observations(bgm, t, y, p, seed=0)
    assert len(obs) == len(bgm)


def test_sample_bgm_observations_reproducible():
    t, y, p, bgm = _sample_trajectory()
    o1 = sample_bgm_observations(bgm, t, y, p, seed=5)
    o2 = sample_bgm_observations(bgm, t, y, p, seed=5)
    np.testing.assert_allclose(o1, o2)


def test_sample_bgm_observations_reasonable_values():
    t, y, p, bgm = _sample_trajectory()
    obs = sample_bgm_observations(bgm, t, y, p, seed=0)
    assert np.all(obs >= 0)
    # Observations should be within a few SDs of true glucose
    interp_true = np.interp(bgm, t, y[:, Gp])
    assert np.all(np.abs(obs - interp_true) < 10 * np.max(
        [bgm_noise_sd(g, p) for g in interp_true]
    ))


def test_compute_log_likelihood_perfect_predictions():
    # Perfect predictions -> high likelihood
    p = Parameters()
    pred = np.array([100.0, 120.0, 140.0])
    ll = compute_log_likelihood(pred, pred, p)
    assert np.isfinite(ll)
    # Should be greater than likelihood for bad predictions
    ll_bad = compute_log_likelihood(pred, pred + 100, p)
    assert ll > ll_bad


def test_compute_log_likelihood_empty():
    p = Parameters()
    assert compute_log_likelihood(np.array([]), np.array([]), p) == 0.0
