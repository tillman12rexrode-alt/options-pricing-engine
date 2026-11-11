"""Greeks off a lattice, checked against the one case with an exact answer.

Merton's result gives a free benchmark: an American call on a non-dividend
paying stock is never exercised early, so it is a European call, so its Greeks
are the analytic Black-Scholes Greeks. Any spurious early exercise, or any
error in the bumping, shows up here against an exact reference.
"""

from __future__ import annotations

import itertools

import pytest

from optpricing import greeks, pricers
from optpricing.contracts import Option
from optpricing.models import black_scholes as bs

GRID = list(itertools.product([80.0, 100.0, 120.0], [0.25, 1.0, 2.0], [0.15, 0.3, 0.5]))


@pytest.mark.parametrize("greek", greeks.GREEKS)
def test_american_call_greeks_match_the_analytic_reference(greek: str):
    """Exact where it should be exact, to the lattice's own accuracy."""
    worst = 0.0
    for strike, years, vol in GRID:
        american = Option(
            S=100.0, K=strike, T=years, r=0.04, sigma=vol, kind="call", style="american"
        )
        european = Option(S=100.0, K=strike, T=years, r=0.04, sigma=vol, kind="call")
        reference = bs.greeks(european)[greek]
        if abs(reference) < 1e-6:
            continue
        estimate = greeks.finite_difference(american, greek)
        worst = max(worst, abs(estimate - reference) / abs(reference))
    assert worst < 5e-3


@pytest.mark.parametrize("strike", [90.0, 100.0, 110.0])
def test_american_put_delta_is_more_negative_than_european(strike: float):
    """Early exercise makes the put more sensitive to the stock falling.

    A sign check the European code could never have caught, because before the
    pricer became pluggable this function differentiated Black-Scholes whatever
    the contract said.
    """
    american = Option(S=100.0, K=strike, T=1.0, r=0.08, sigma=0.3, kind="put", style="american")
    european = Option(S=100.0, K=strike, T=1.0, r=0.08, sigma=0.3, kind="put")
    assert greeks.finite_difference(american, "delta") < bs.greeks(european)["delta"]


def test_the_step_size_barely_matters_on_a_lattice():
    """Worth asserting, because the obvious expectation is the opposite.

    A lattice price is not a smooth function of spot, so a tiny bump ought to
    drown delta in discretisation noise. It does not: Leisen-Reimer's error
    varies smoothly enough with spot that it cancels in a central difference.
    The estimate is therefore flat in h across four orders of magnitude, and
    accuracy is set by the step count instead. If this ever starts failing, the
    guidance in finite_difference's docstring needs revisiting.
    """
    opt = Option(S=100.0, K=100.0, T=1.0, r=0.04, sigma=0.3, kind="put", style="american")
    estimates = [greeks.finite_difference(opt, "delta", h=h) for h in (1e-3, 1e-2, 1e-1, 1.0)]
    assert max(estimates) - min(estimates) < 1e-3


def test_more_steps_give_a_better_greek():
    """Accuracy is set by the lattice, so a denser one must help."""
    american = Option(S=100.0, K=110.0, T=1.0, r=0.04, sigma=0.3, kind="call", style="american")
    european = Option(S=100.0, K=110.0, T=1.0, r=0.04, sigma=0.3, kind="call")
    reference = bs.greeks(european)["vega"]

    coarse = greeks.finite_difference(american, "vega", pricer=pricers.lattice(101))
    fine = greeks.finite_difference(american, "vega", pricer=pricers.lattice(401))
    assert abs(fine - reference) < abs(coarse - reference)


def test_estimate_keeps_the_closed_form_on_the_european_path():
    """Nothing that was analytic may quietly become numerical."""
    opt = Option(S=100.0, K=100.0, T=1.0, r=0.04, sigma=0.25, q=0.01)
    for greek in greeks.GREEKS:
        assert greeks.estimate(opt, greek) == bs.greeks(opt)[greek]


def test_estimate_all_covers_every_greek_for_an_american_contract():
    opt = Option(S=100.0, K=100.0, T=1.0, r=0.04, sigma=0.25, kind="put", style="american")
    estimated = greeks.estimate_all(opt)
    assert set(estimated) == set(greeks.GREEKS)
    assert all(abs(value) < 1e6 for value in estimated.values())


def test_greeks_work_with_a_dividend_schedule():
    """Delta must stay inside its theoretical range once cash dividends appear."""
    opt = Option(
        S=100.0, K=100.0, T=1.0, r=0.05, sigma=0.3, kind="call", style="american",
        dividends=((0.4, 1.0), (0.9, 1.0)),
    )
    assert 0.0 < greeks.finite_difference(opt, "delta") < 1.0
    assert greeks.finite_difference(opt, "gamma") > 0.0
