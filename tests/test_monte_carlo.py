"""Validate the Monte Carlo engine, its error bars, and its variance reduction."""

from __future__ import annotations

import numpy as np
import pytest

from optpricing.contracts import Option
from optpricing.models import black_scholes as bs
from optpricing.models import monte_carlo as mc

HULL = {"S": 42.0, "K": 40.0, "T": 0.5, "r": 0.10, "sigma": 0.20}


@pytest.mark.parametrize("kind", ["call", "put"])
@pytest.mark.parametrize("q", [0.0, 0.03])
def test_price_within_three_standard_errors(kind: str, q: float):
    """A correct estimator lands within 3 SE of the exact price essentially always."""
    opt = Option(**HULL, q=q, kind=kind)
    result = mc.price_with_error(opt, 200_000, seed=12345)
    assert abs(result.price - bs.price(opt)) < 3 * result.std_error


def test_antithetic_reduces_standard_error():
    """Pairing Z with -Z cuts the variance of a monotone payoff."""
    opt = Option(**HULL)
    plain = mc.price_with_error(opt, 200_000, seed=7)
    anti = mc.price_with_error(opt, 200_000, seed=7, antithetic=True)
    assert anti.std_error < plain.std_error


def test_control_variate_reduces_standard_error():
    """The terminal spot has a known mean, so its sampling error can be removed."""
    opt = Option(**HULL)
    plain = mc.price_with_error(opt, 200_000, seed=7)
    controlled = mc.price_with_error(opt, 200_000, seed=7, control_variate=True)
    assert controlled.std_error < plain.std_error


def test_confidence_interval_covers_the_truth():
    """Across repeated runs the 95% interval should contain the exact price ~95% of
    the time. A wrong standard error shows up here and nowhere else."""
    opt = Option(**HULL)
    exact = bs.price(opt)
    runs = 300
    covered = 0
    for seed in range(runs):
        low, high = mc.price_with_error(opt, 4_000, seed=seed).ci95
        covered += low <= exact <= high

    assert 0.90 <= covered / runs <= 0.99


def test_seed_makes_results_reproducible():
    opt = Option(**HULL)
    assert mc.price(opt, 10_000, seed=1) == mc.price(opt, 10_000, seed=1)
    assert mc.price(opt, 10_000, seed=1) != mc.price(opt, 10_000, seed=2)


def test_antithetic_halves_the_sample_count():
    """Each pair collapses to one independent sample."""
    result = mc.price_with_error(Option(**HULL), 10_000, seed=1, antithetic=True)
    assert result.n_samples == 5_000


def test_confidence_interval_is_symmetric_about_the_price():
    result = mc.price_with_error(Option(**HULL), 10_000, seed=1)
    low, high = result.ci95
    assert np.isclose((low + high) / 2, result.price)


def test_rejects_american_style():
    opt = Option(**HULL, style="american")
    with pytest.raises(ValueError, match="European"):
        mc.price_with_error(opt, 1_000)


def test_antithetic_rejects_odd_path_count():
    with pytest.raises(ValueError, match="even"):
        mc.price_with_error(Option(**HULL), 1_001, antithetic=True)
