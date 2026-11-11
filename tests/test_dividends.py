"""Discrete cash dividends: the contract, the escrowed lattice, and early exercise.

The point of this file is the asymmetry a continuous yield cannot express. An
American put is exercised early for a reason that varies smoothly with time, so
a yield models it tolerably. An American call is exercised early only in the
instant before an ex-dividend date, which is a dated event - smear the same
cash across the year as a yield and that exercise opportunity disappears.
"""

from __future__ import annotations

import math

import pytest

from optpricing.contracts import Option
from optpricing.models import binomial
from optpricing.models import black_scholes as bs

BASE = {"S": 100.0, "K": 100.0, "T": 1.0, "r": 0.05, "sigma": 0.25}
SCHEDULE = ((0.25, 1.5), (0.75, 1.5))


def test_default_is_no_dividends():
    assert Option(**BASE).dividends == ()
    assert Option(**BASE).pv_dividends() == 0.0


def test_contract_stays_hashable():
    """Frozen with slots is load-bearing: a priced option must not mutate."""
    assert isinstance(hash(Option(**BASE, dividends=SCHEDULE)), int)


def test_present_value_discounts_each_payment():
    opt = Option(**BASE, dividends=SCHEDULE)
    expected = 1.5 * math.exp(-0.05 * 0.25) + 1.5 * math.exp(-0.05 * 0.75)
    assert opt.pv_dividends() == pytest.approx(expected)


def test_dividends_after_expiry_are_ignored():
    """A schedule belongs to the stock, not the contract, so it outlives expiry."""
    opt = Option(**BASE, dividends=((0.5, 2.0), (3.0, 2.0)))
    assert opt.pv_dividends() == pytest.approx(2.0 * math.exp(-0.05 * 0.5))


def test_yield_and_schedule_together_are_rejected():
    """Accepting both would double-count, which produces a plausible wrong price."""
    with pytest.raises(ValueError, match="not both"):
        Option(**BASE, q=0.02, dividends=SCHEDULE)


@pytest.mark.parametrize(
    "bad, message",
    [
        (((-0.1, 1.0),), "time cannot be negative"),
        (((0.5, -1.0),), "amount cannot be negative"),
        (((0.5,),), "time, amount"),
    ],
)
def test_malformed_schedules_are_rejected(bad: tuple, message: str):
    with pytest.raises(ValueError, match=message):
        Option(**BASE, dividends=bad)


@pytest.mark.parametrize("method", binomial.METHODS)
@pytest.mark.parametrize("kind", ["call", "put"])
def test_european_with_dividends_matches_the_escrowed_closed_form(method: str, kind: str):
    """The escrowed method's one exact check.

    A European option on a stock paying known cash is worth exactly a
    Black-Scholes option on the spot less the present value of that cash. If
    the lattice does not reproduce that, the escrow is wrong.
    """
    opt = Option(**BASE, kind=kind, dividends=SCHEDULE)
    escrowed = Option(**{**BASE, "S": BASE["S"] - opt.pv_dividends()}, kind=kind)
    assert binomial.price(opt, 2001, method) == pytest.approx(bs.price(escrowed), abs=2e-3)


@pytest.mark.parametrize("method", binomial.METHODS)
def test_an_empty_schedule_changes_nothing(method: str):
    """Every existing result must be untouched, to the last bit."""
    plain = Option(**BASE, kind="put", style="american")
    explicit = Option(**BASE, kind="put", style="american", dividends=())
    assert binomial.price(plain, 501, method) == binomial.price(explicit, 501, method)


def test_american_call_gains_from_a_dividend():
    """Merton's result has an exception, and this is it.

    Without dividends an American call is never exercised early. With one, it
    may be - and the lattice has to find that, or the early-exercise branch is
    not doing its job on equities.
    """
    schedule = ((0.95, 12.0),)
    american = Option(**BASE, kind="call", style="american", dividends=schedule)
    european = Option(**BASE, kind="call", dividends=schedule)
    premium = binomial.price(american, 1001, "lr") - binomial.price(european, 1001, "lr")
    assert premium > 1.0


def test_a_tiny_dividend_leaves_the_call_premium_near_zero():
    """The other side of the same coin: no dividend worth capturing, no premium."""
    schedule = ((0.95, 0.01),)
    american = Option(**BASE, kind="call", style="american", dividends=schedule)
    european = Option(**BASE, kind="call", dividends=schedule)
    premium = binomial.price(american, 1001, "lr") - binomial.price(european, 1001, "lr")
    assert premium == pytest.approx(0.0, abs=0.01)


def test_dividends_exceeding_the_spot_are_refused():
    with pytest.raises(ValueError, match="escrowed spot"):
        binomial.price(Option(**BASE, dividends=((0.5, 200.0),)), 101, "lr")
