"""Inverting a numerical procedure rather than a formula.

Three things have to hold. The inversion must recover a volatility it was given.
It must refuse quotes that no volatility explains. And it must refuse quotes
that *every* low volatility explains equally well - a case with no European
analogue, which is where a naive solver quietly invents a number.
"""

from __future__ import annotations

import itertools

import pytest

from optpricing import implied_vol as iv
from optpricing import pricers
from optpricing.contracts import Option
from optpricing.models import binomial
from optpricing.models import black_scholes as bs

AMERICAN = {"S": 100.0, "K": 100.0, "T": 1.0, "r": 0.06, "style": "american"}


def _american(**overrides) -> Option:
    return Option(**{**AMERICAN, "sigma": 0.28, **overrides})


@pytest.mark.parametrize("kind", ["call", "put"])
@pytest.mark.parametrize("sigma", [0.15, 0.3, 0.6])
def test_round_trip_recovers_the_volatility(kind: str, sigma: float):
    opt = _american(kind=kind, sigma=sigma)
    target = pricers.for_option(opt)(opt)
    result = iv.implied_vol(target, opt)
    assert result.converged
    assert result.sigma == pytest.approx(sigma, abs=1e-6)


@pytest.mark.parametrize("kind", ["call", "put"])
def test_round_trip_survives_a_dividend_schedule(kind: str):
    opt = _american(kind=kind, dividends=((0.3, 1.2), (0.8, 1.2)))
    target = pricers.for_option(opt)(opt)
    assert iv.implied_vol(target, opt).sigma == pytest.approx(0.28, abs=1e-6)


def test_the_european_path_is_untouched():
    """The closed-form inversion must be exactly what it was before the seam."""
    opt = Option(S=100.0, K=105.0, T=0.5, r=0.04, sigma=0.27, q=0.01)
    result = iv.implied_vol(bs.price(opt), opt)
    assert result.converged
    assert result.sigma == pytest.approx(0.27, abs=1e-10)
    assert pricers.for_option(opt) is bs.price


def test_american_bounds_are_tighter_than_the_european_ones():
    """A put may be exercised now, so K is the cap - not K discounted."""
    put = _american(kind="put", K=110.0)
    _, upper = iv.american_bounds(put)
    assert upper == pytest.approx(110.0)

    call = _american(kind="call", K=90.0)
    lower, upper_call = iv.american_bounds(call)
    assert lower >= call.intrinsic()
    assert upper_call == pytest.approx(call.S)


@pytest.mark.parametrize("target, reason", [(9.0, "bounds"), (101.0, "bounds")])
def test_quotes_outside_the_bounds_are_refused(target: float, reason: str):
    """Below intrinsic or above the stock: no volatility produces that price."""
    result = iv.implied_vol(target, _american(kind="call", K=90.0, sigma=0.3))
    assert not result.converged
    assert reason in result.reason


def test_a_quote_at_the_exercise_boundary_is_undetermined_not_solved():
    """The failure mode this whole guard exists for.

    A deep in-the-money American put is worth exactly its intrinsic value at
    every low volatility. Brent, handed that, returns the end of its bracket -
    a number that looks like an implied volatility and is not one.
    """
    put = _american(kind="put", K=120.0, T=0.1, sigma=0.12)
    target = pricers.for_option(put)(put)
    assert target == pytest.approx(put.intrinsic())

    result = iv.implied_vol(target, put)
    assert not result.converged
    assert "undetermined" in result.reason


def test_newton_gives_up_more_often_than_brent_on_american_contracts():
    """A real numerical result, and an extension of the European one.

    Newton needs vega. On a lattice vega is itself a finite difference of a
    numerical price, so it is less reliable than the analytic vega Newton gets
    for free in the European case - and the iteration fails more as a result.
    """
    grid = [
        _american(K=strike, T=years, sigma=vol, kind=kind)
        for strike, years, vol, kind in itertools.product(
            [70.0, 85.0, 100.0, 115.0, 130.0], [0.1, 0.5, 2.0], [0.15, 0.35], ["call", "put"]
        )
    ]
    targets = [binomial.price(opt, 101, "lr") for opt in grid]

    brent = sum(iv.implied_vol(t, o, "brent").converged for t, o in zip(targets, grid, strict=True))
    newton = sum(
        iv.implied_vol(t, o, "newton").converged for t, o in zip(targets, grid, strict=True)
    )
    assert newton < brent


def test_brent_walks_its_bracket_up_when_the_lattice_collapses():
    """Leisen-Reimer cannot be built at a tiny sigma for every contract.

    Rather than pick one floor conservative enough for the worst case, the
    solver raises the floor until the pricer answers. Without that, a deep
    in-the-money short-dated contract raises instead of returning a result.
    """
    opt = _american(kind="call", K=80.0, T=0.1, sigma=0.4)
    target = pricers.for_option(opt)(opt)
    result = iv.implied_vol(target, opt)
    assert result.converged
    assert result.sigma == pytest.approx(0.4, abs=1e-6)
