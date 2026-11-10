"""Cross-check the Greek estimators against the closed form and each other."""

from __future__ import annotations

import math

import pytest

from optpricing import greeks as gk
from optpricing.contracts import Option
from optpricing.models import black_scholes as bs

PARAMS = {"S": 100.0, "K": 95.0, "T": 1.0, "r": 0.05, "sigma": 0.30, "q": 0.02}


@pytest.mark.parametrize("greek", gk.GREEKS)
@pytest.mark.parametrize("kind", ["call", "put"])
def test_finite_difference_matches_analytic(greek: str, kind: str):
    """Two independent routes to the same derivative must agree closely."""
    opt = Option(**PARAMS, kind=kind)
    analytic = bs.greeks(opt)[greek]
    assert gk.finite_difference(opt, greek) == pytest.approx(analytic, rel=1e-5)


def test_call_minus_put_delta_is_the_dividend_discount():
    """Differentiating put-call parity in S gives delta_call - delta_put = exp(-qT)."""
    call = bs.greeks(Option(**PARAMS, kind="call"))
    put = bs.greeks(Option(**PARAMS, kind="put"))
    expected = math.exp(-PARAMS["q"] * PARAMS["T"])
    assert call["delta"] - put["delta"] == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize("greek", ["gamma", "vega"])
def test_gamma_and_vega_do_not_depend_on_option_kind(greek: str):
    """Parity is linear in S and free of sigma, so these match exactly."""
    call = bs.greeks(Option(**PARAMS, kind="call"))[greek]
    put = bs.greeks(Option(**PARAMS, kind="put"))[greek]
    assert call == pytest.approx(put, rel=1e-12)


def test_pathwise_delta_matches_analytic():
    """The pathwise estimator is unbiased, so it converges on the closed form."""
    opt = Option(**PARAMS)
    estimate = gk.pathwise_delta(opt, 400_000, seed=1)
    assert estimate == pytest.approx(bs.greeks(opt)["delta"], abs=0.005)


def test_common_random_numbers_make_bumping_usable():
    """Sharing draws between the two repricings is what makes a bumped MC delta work.

    Without it the simulation noise is divided by the bump size and swamps the
    derivative - here by more than two orders of magnitude.
    """
    opt = Option(**PARAMS)
    exact = bs.greeks(opt)["delta"]

    shared = gk.bumped_delta(opt, 0.01, 400_000, seed=1, common_random=True)
    independent = gk.bumped_delta(opt, 0.01, 400_000, seed=1, common_random=False)

    assert shared == pytest.approx(exact, abs=0.01)
    assert abs(independent - exact) > 100 * abs(shared - exact)


def test_default_step_is_larger_for_second_differences():
    """Gamma trades off at eps^(1/4), delta at eps^(1/3), so gamma wants a bigger bump."""
    opt = Option(**PARAMS)
    assert gk.default_step(opt, "gamma") > gk.default_step(opt, "delta")


def test_analytic_greeks_reject_degenerate_contracts():
    with pytest.raises(ValueError, match="undefined"):
        bs.greeks(Option(**{**PARAMS, "T": 0.0}))


def test_analytic_greeks_reject_american_style():
    with pytest.raises(ValueError, match="European"):
        bs.greeks(Option(**PARAMS, style="american"))


@pytest.mark.parametrize("bad", ["vanna", "", "DELTA"])
def test_unknown_greek_is_rejected(bad: str):
    with pytest.raises(ValueError, match="greek must be"):
        gk.finite_difference(Option(**PARAMS), bad)
