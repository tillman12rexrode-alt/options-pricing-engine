"""Check that rate and dividend yield can be recovered from quotes alone."""

from __future__ import annotations

import math
from datetime import date

import pandas as pd
import pytest

from optpricing.contracts import Option
from optpricing.market import clean, implied_terms
from optpricing.models import black_scholes as bs

SPOT = 5000.0
ASOF = date(2026, 1, 1)
EXPIRY = "2026-07-01"
RATE = 0.045
DIVIDEND = 0.013
VOL = 0.20


def synthetic_chain() -> pd.DataFrame:
    """A chain priced from known inputs, so the true answer is known."""
    years = (date.fromisoformat(EXPIRY) - ASOF).days / 365.0
    rows = []
    for strike in range(4600, 5451, 25):
        for kind in ("call", "put"):
            opt = Option(
                S=SPOT, K=float(strike), T=years, r=RATE, sigma=VOL, q=DIVIDEND, kind=kind
            )
            price = bs.price(opt)
            rows.append(
                {
                    "expiry": EXPIRY,
                    "kind": kind,
                    "strike": float(strike),
                    "bid": price - 0.05,
                    "ask": price + 0.05,
                    "mid": price,
                }
            )
    return pd.DataFrame(rows)


def test_parity_regression_recovers_the_rate_and_dividend_yield():
    """C - P = D(F - K) exactly for European options, so the fit is exact."""
    terms = implied_terms(synthetic_chain(), SPOT, ASOF)[EXPIRY]

    assert terms.rate == pytest.approx(RATE, abs=1e-8)
    assert terms.dividend_yield == pytest.approx(DIVIDEND, abs=1e-8)
    assert terms.r_squared == pytest.approx(1.0, abs=1e-12)


def test_implied_forward_matches_the_textbook_formula():
    terms = implied_terms(synthetic_chain(), SPOT, ASOF)[EXPIRY]
    expected = SPOT * math.exp((RATE - DIVIDEND) * terms.years)
    assert terms.forward == pytest.approx(expected, rel=1e-9)


def test_discount_factor_is_consistent_with_the_implied_rate():
    terms = implied_terms(synthetic_chain(), SPOT, ASOF)[EXPIRY]
    assert terms.discount == pytest.approx(math.exp(-terms.rate * terms.years), rel=1e-12)


def test_cleaning_drops_one_sided_and_wide_and_far_quotes():
    frame = pd.DataFrame(
        {
            "expiry": [EXPIRY] * 4,
            "kind": ["call"] * 4,
            "strike": [5000.0, 5000.0, 5000.0, 9000.0],
            "bid": [100.0, 0.0, 1.0, 100.0],
            "ask": [101.0, 5.0, 100.0, 101.0],
        }
    )
    kept, funnel = clean(frame, SPOT)

    assert len(kept) == 1
    assert kept.iloc[0]["strike"] == 5000.0
    assert [n for _, n in funnel] == [4, 3, 3, 2, 1]


def test_expiries_with_too_few_pairs_are_skipped():
    frame = synthetic_chain()
    frame = frame[frame["strike"] < 4700]
    assert implied_terms(frame, SPOT, ASOF) == {}
