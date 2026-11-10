"""Round-trip the implied-vol solvers and compare their failure modes."""

from __future__ import annotations

import math
from dataclasses import replace

import pytest

from optpricing.contracts import Option
from optpricing.implied_vol import METHODS, implied_vol, no_arbitrage_bounds
from optpricing.market import ExpiryTerms
from optpricing.models import black_scholes as bs

BASE = Option(S=5000.0, K=5000.0, T=0.5, r=0.045, sigma=0.20, q=0.013)


def terms_for(opt: Option) -> ExpiryTerms:
    discount = math.exp(-opt.r * opt.T)
    forward = opt.S * math.exp((opt.r - opt.q) * opt.T)
    return ExpiryTerms("2026-07-01", opt.T, discount, forward, opt.r, opt.q, 1.0, 20)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("true_vol", [0.08, 0.20, 0.55, 1.20])
@pytest.mark.parametrize("kind", ["call", "put"])
def test_solvers_recover_the_volatility_they_were_priced_with(
    method: str, true_vol: float, kind: str
):
    opt = replace(BASE, sigma=true_vol, kind=kind)
    result = implied_vol(bs.price(opt), opt, method)

    assert result.converged
    assert result.sigma == pytest.approx(true_vol, abs=1e-7)


def test_the_two_solvers_agree_where_both_converge():
    opt = replace(BASE, sigma=0.27)
    target = bs.price(opt)

    brent = implied_vol(target, opt, "brent")
    newton = implied_vol(target, opt, "newton")

    assert brent.sigma == pytest.approx(newton.sigma, abs=1e-9)


@pytest.mark.parametrize(
    ("strike", "years", "quote", "reason"),
    [
        (12_000.0, 0.50, 0.05, "left the bracket"),
        (8_000.0, 0.08, 0.05, "vega collapsed"),
    ],
)
def test_newton_fails_in_the_wings_where_brent_does_not(
    strike: float, years: float, quote: float, reason: str
):
    """A far out-of-the-money strike quoted at a nickel - the real market case.

    Its model value is effectively zero, so the implied volatility sits far from
    any sensible starting guess and vega along the way is negligible. Newton
    either steps outside the bracket or divides by a vanishing derivative;
    Brent cannot diverge and returns the root either way.
    """
    opt = replace(BASE, K=strike, T=years)

    brent = implied_vol(quote, opt, "brent")
    newton = implied_vol(quote, opt, "newton")

    assert brent.converged
    assert brent.sigma > 0.0
    assert not newton.converged
    assert newton.reason == reason


def test_brent_uses_more_iterations_than_newton_when_both_work():
    """Brent trades speed for never diverging; that is the whole reason to keep it."""
    opt = replace(BASE, sigma=0.33)
    target = bs.price(opt)

    assert implied_vol(target, opt, "brent").iterations > implied_vol(
        target, opt, "newton"
    ).iterations


def test_price_outside_the_bracket_returns_no_solution():
    opt = replace(BASE)
    unreachable = opt.S * 1.5
    assert not implied_vol(unreachable, opt, "brent").converged


@pytest.mark.parametrize("kind", ["call", "put"])
def test_no_arbitrage_bounds_contain_every_attainable_price(kind: str):
    opt = replace(BASE, kind=kind)
    terms = terms_for(opt)
    low, high = no_arbitrage_bounds(opt, terms)

    for vol in (1e-4, 0.2, 4.0):
        assert low - 1e-9 <= bs.price(replace(opt, sigma=vol)) <= high + 1e-9


def test_unknown_method_is_rejected():
    with pytest.raises(ValueError, match="method must be"):
        implied_vol(100.0, BASE, "bisection")
