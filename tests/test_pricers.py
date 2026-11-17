"""The pricer seam: routing, equivalence with the closed form, and accuracy."""

from __future__ import annotations

import itertools

import pytest

from optpricing import pricers
from optpricing.contracts import Option
from optpricing.models import black_scholes as bs

EUROPEAN = Option(S=100.0, K=100.0, T=1.0, r=0.04, sigma=0.25, q=0.015)
AMERICAN = Option(S=100.0, K=100.0, T=1.0, r=0.04, sigma=0.25, style="american")
WITH_DIVIDENDS = Option(S=100.0, K=100.0, T=1.0, r=0.04, sigma=0.25, dividends=((0.5, 2.0),))


def test_closed_form_only_where_it_is_exact():
    assert pricers.has_closed_form(EUROPEAN)
    assert not pricers.has_closed_form(AMERICAN)
    assert not pricers.has_closed_form(WITH_DIVIDENDS)


def test_european_routes_to_black_scholes_itself():
    """Not merely an equal price - the same function, so nothing gets slower."""
    assert pricers.for_option(EUROPEAN) is bs.price


@pytest.mark.parametrize("opt", [AMERICAN, WITH_DIVIDENDS])
def test_anything_else_routes_to_the_lattice(opt: Option):
    assert pricers.for_option(opt) is not bs.price


def test_lattice_rejects_an_even_step_count_for_leisen_reimer():
    with pytest.raises(ValueError, match="odd"):
        pricers.lattice(100, "lr")


def test_lattice_rejects_an_unknown_method():
    with pytest.raises(ValueError, match="method must be one of"):
        pricers.lattice(101, "trinomial")


def test_default_lattice_tracks_the_closed_form_across_the_grid():
    """The default step count has to be good enough to trust, so pin the error.

    Absolute rather than relative, because a relative tolerance on a contract
    worth fractions of a cent measures nothing. A tenth of a cent on a $100
    underlying is well inside any bid-ask spread this will ever meet.
    """
    price = pricers.lattice()
    worst = 0.0
    for spot, strike, years, vol, kind in itertools.product(
        [90.0, 100.0, 110.0], [80.0, 100.0, 120.0], [0.08, 0.5, 2.0], [0.12, 0.25, 0.6],
        ["call", "put"],
    ):
        opt = Option(S=spot, K=strike, T=years, r=0.04, sigma=vol, q=0.015, kind=kind)
        worst = max(worst, abs(price(opt) - bs.price(opt)))
    assert worst < 1e-3


def test_more_steps_converge_quadratically():
    """Leisen-Reimer is O(1/n^2): tripling the steps should cut the error ~9x."""
    opt = Option(S=110.0, K=120.0, T=2.0, r=0.04, sigma=0.6, q=0.015)
    exact = bs.price(opt)
    coarse = abs(pricers.lattice(101)(opt) - exact)
    fine = abs(pricers.lattice(301)(opt) - exact)
    assert fine < coarse / 5


def test_describe_names_the_engine():
    """The UI has to be able to say whether a number is closed form or numerical."""
    assert pricers.describe(pricers.for_option(EUROPEAN)) == "black-scholes"
    assert pricers.describe(pricers.for_option(AMERICAN)) != "black-scholes"
