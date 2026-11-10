"""Inject known violations and confirm each check finds exactly its own.

A liquid chain contains no arbitrage, so these tests cannot be written against
real data - a pass there would only prove the functions return nothing. Each
test below builds a consistent chain, breaks one relation deliberately, and
asserts that the matching check fires and the others stay silent.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from optpricing import arbitrage as arb
from optpricing.contracts import Option
from optpricing.market import ExpiryTerms
from optpricing.models import black_scholes as bs
from optpricing.smile import fit_slice

SPOT = 5000.0
RATE, DIVIDEND, VOL = 0.045, 0.012, 0.20
YEARS = 0.5
EXPIRY = "2026-07-01"
HALF_SPREAD = 0.10
STRIKES = [float(k) for k in range(4700, 5301, 50)]


def terms_for(years: float = YEARS, expiry: str = EXPIRY) -> ExpiryTerms:
    discount = math.exp(-RATE * years)
    forward = SPOT * math.exp((RATE - DIVIDEND) * years)
    return ExpiryTerms(expiry, years, discount, forward, RATE, DIVIDEND, 1.0, len(STRIKES))


def consistent_chain(years: float = YEARS, expiry: str = EXPIRY, vol: float = VOL) -> pd.DataFrame:
    rows = []
    for strike in STRIKES:
        for kind in ("call", "put"):
            price = bs.price(
                Option(S=SPOT, K=strike, T=years, r=RATE, sigma=vol, q=DIVIDEND, kind=kind)
            )
            rows.append(
                {
                    "expiry": expiry,
                    "kind": kind,
                    "strike": strike,
                    "bid": price - HALF_SPREAD,
                    "ask": price + HALF_SPREAD,
                    "mid": price,
                }
            )
    return pd.DataFrame(rows)


def test_a_consistent_chain_contains_no_arbitrage():
    chain = consistent_chain()
    found = arb.summarize(
        arb.parity_violations(chain, {EXPIRY: terms_for()}),
        arb.vertical_violations(chain),
        arb.butterfly_violations(chain),
    )
    assert found.empty


def test_parity_violation_is_detected():
    """Mark a call far below its synthetic value and the reversal shows up."""
    chain = consistent_chain()
    target = (chain["kind"] == "call") & (chain["strike"] == 5000.0)
    chain.loc[target, ["bid", "ask"]] -= 25.0

    found = arb.parity_violations(chain, {EXPIRY: terms_for()})
    assert len(found) == 1
    assert found.iloc[0]["detail"] == "K=5000"
    assert found.iloc[0]["edge"] == pytest.approx(25.0 - 2 * HALF_SPREAD, abs=1e-6)


def test_vertical_violation_is_detected():
    """A lower-strike call cheaper than the next strike up cannot stand."""
    chain = consistent_chain()
    target = (chain["kind"] == "call") & (chain["strike"] == 5000.0)
    chain.loc[target, ["bid", "ask"]] -= 60.0

    found = arb.vertical_violations(chain)
    assert not found.empty
    assert any("5000" in detail for detail in found["detail"])


def test_butterfly_violation_is_detected():
    """Pushing the middle strike up breaks convexity."""
    chain = consistent_chain()
    target = (chain["kind"] == "call") & (chain["strike"] == 5000.0)
    chain.loc[target, ["bid", "ask"]] += 40.0

    found = arb.butterfly_violations(chain)
    assert not found.empty
    assert any("5000" in detail for detail in found["detail"])


def test_fees_shrink_the_edge_and_can_remove_it():
    """The honest question is never whether an edge exists, but whether it survives."""
    chain = consistent_chain()
    target = (chain["kind"] == "call") & (chain["strike"] == 5000.0)
    chain.loc[target, ["bid", "ask"]] -= 0.45

    free = arb.parity_violations(chain, {EXPIRY: terms_for()}, fee_per_contract=0.0)
    charged = arb.parity_violations(chain, {EXPIRY: terms_for()}, fee_per_contract=0.65)

    assert len(free) == 1
    assert charged.empty


def test_mid_prices_would_manufacture_violations_that_executable_prices_do_not():
    """Why the checks use bid and ask: a mid-price test flags the spread itself."""
    chain = consistent_chain()
    target = (chain["kind"] == "call") & (chain["strike"] == 5000.0)
    chain.loc[target, ["bid", "ask"]] -= 0.15

    assert arb.parity_violations(chain, {EXPIRY: terms_for()}).empty

    at_mid = chain.assign(bid=chain["mid"] - 0.15 * (chain["kind"] == "call"),
                          ask=chain["mid"] - 0.15 * (chain["kind"] == "call"))
    assert not arb.parity_violations(at_mid, {EXPIRY: terms_for()}).empty


def fit_for(years: float, vol: float):
    terms = terms_for(years, "x")
    grid = np.array([math.log(k / terms.forward) for k in STRIKES])
    return fit_slice(grid, np.full(grid.size, vol), years)


def test_calendar_violation_is_detected_when_variance_falls_with_maturity():
    near = fit_for(0.25, 0.30)
    far = fit_for(1.00, 0.10)   # far total variance 0.01 < near 0.0225

    found = arb.calendar_violations({"near": near, "far": far})
    assert len(found) == 1
    assert found.iloc[0]["check"] == "calendar"


def test_calendar_check_is_silent_when_variance_rises_properly():
    assert arb.calendar_violations(
        {"near": fit_for(0.25, 0.20), "far": fit_for(1.00, 0.20)}
    ).empty


def test_unreliable_slices_are_excluded_from_the_calendar_check():
    class Unreliable:
        reliable = False
        years = 1.0

    assert arb.calendar_violations(
        {"near": fit_for(0.25, 0.30), "far": Unreliable()}
    ).empty
