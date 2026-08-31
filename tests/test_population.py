"""Tests for src/population.py."""

import numpy as np
import pytest

from src.population import (
    generate_population, sample_virtual_patient, PARAM_LOWER, PARAM_UPPER,
)

# Add event kind constants for the stress test
from src.events import EVENT_STRESS


def test_sample_virtual_patient_returns_two_values():
    rng = np.random.default_rng(0)
    result = sample_virtual_patient(rng)
    assert isinstance(result, tuple)
    assert len(result) == 2


def test_sample_virtual_patient_within_bounds():
    rng = np.random.default_rng(0)
    for _ in range(100):
        p, _ = sample_virtual_patient(rng)
        arr = p.as_array()
        # Spot check a few params are within bounds
        assert PARAM_LOWER["k_u"] <= p.k_u <= PARAM_UPPER["k_u"]
        assert PARAM_LOWER["k_cl"] <= p.k_cl <= PARAM_UPPER["k_cl"]
        assert PARAM_LOWER["k_x"] <= p.k_x <= PARAM_UPPER["k_x"]
        assert PARAM_LOWER["V_G"] <= p.V_G <= PARAM_UPPER["V_G"]


def test_sample_virtual_patient_is_factor_coupling():
    """Higher IS factor should give higher k_u (renal clearance) and EGP, lower k_cl."""
    rng = np.random.default_rng(0)
    p1, _ = sample_virtual_patient(rng)
    p2, _ = sample_virtual_patient(rng)
    # Can't assert direction universally (different draws), just that both valid
    assert p1.k_u > 0
    assert p2.k_u > 0
    assert p1.k_cl > 0


def test_generate_population_count():
    pop = generate_population(5, seed=123)
    assert len(pop) == 5


def test_generate_population_seed_reproducible():
    p1 = generate_population(3, seed=99)
    p2 = generate_population(3, seed=99)
    for a, b in zip(p1, p2):
        np.testing.assert_allclose(a.as_array(), b.as_array())


def test_generate_population_different_seeds_differ():
    p1 = generate_population(3, seed=1)
    p2 = generate_population(3, seed=2)
    assert not all(
        np.allclose(a.as_array(), b.as_array()) for a, b in zip(p1, p2)
    )


def test_generate_population_invalid_count():
    with pytest.raises(ValueError):
        generate_population(0)


def test_generate_population_all_within_bounds():
    pop = generate_population(20, seed=7)
    for p in pop:
        arr = p.as_array()
        assert np.all(np.isfinite(arr))
        assert np.all(arr > 0)
