"""The scanner must find a planted anomaly and refuse to invent others."""

from __future__ import annotations

import math
from datetime import date

import numpy as np
import pandas as pd
import pytest

from optpricing.contracts import Option
from optpricing.models import black_scholes as bs
from optpricing.scanner import analyze, find_outliers

SPOT = 5000.0
ASOF = date(2026, 1, 1)
RATE, DIVIDEND = 0.045, 0.012
#: Year fractions must match how the library measures them, or the implied rate
#: recovered from the chain will not match the one used to build it.
EXPIRIES = {
    expiry: (date.fromisoformat(expiry) - ASOF).days / 365.0
    for expiry in ("2026-04-01", "2026-07-01", "2027-01-01")
}
STRIKES = [float(k) for k in range(4200, 5801, 25)]


def smooth_vol(strike: float, forward: float) -> float:
    """A gentle downward skew - the shape a real index smile has."""
    k = math.log(strike / forward)
    return 0.20 - 0.35 * k + 0.45 * k**2


def synthetic_chain(half_spread: float = 0.08) -> pd.DataFrame:
    rows = []
    for expiry, years in EXPIRIES.items():
        forward = SPOT * math.exp((RATE - DIVIDEND) * years)
        for strike in STRIKES:
            for kind in ("call", "put"):
                vol = smooth_vol(strike, forward)
                price = bs.price(
                    Option(S=SPOT, K=strike, T=years, r=RATE, sigma=vol,
                           q=DIVIDEND, kind=kind)
                )
                if price < 0.05:
                    continue
                rows.append({
                    "expiry": expiry, "kind": kind, "strike": strike,
                    "bid": price - half_spread, "ask": price + half_spread,
                    "volume": 100.0, "open_interest": 500.0,
                })
    return pd.DataFrame(rows)


def test_a_clean_surface_produces_no_outliers():
    """Quotes generated from a smooth curve should all sit on that curve."""
    analysis = analyze(synthetic_chain(), SPOT, ASOF)

    assert len(analysis.reliable_expiries) == len(EXPIRIES)
    assert find_outliers(analysis).empty


def test_implied_terms_are_recovered_from_the_synthetic_chain():
    analysis = analyze(synthetic_chain(), SPOT, ASOF)
    for term in analysis.terms.values():
        assert term.rate == pytest.approx(RATE, abs=1e-6)
        assert term.dividend_yield == pytest.approx(DIVIDEND, abs=1e-6)


def test_a_planted_mispricing_is_found_and_explained():
    """Reprice one strike at a clearly different volatility and it should surface."""
    chain = synthetic_chain()
    expiry, years = "2026-07-01", EXPIRIES["2026-07-01"]
    forward = SPOT * math.exp((RATE - DIVIDEND) * years)
    strike = 5200.0

    bumped = bs.price(
        Option(S=SPOT, K=strike, T=years, r=RATE,
               sigma=smooth_vol(strike, forward) + 0.04, q=DIVIDEND, kind="call")
    )
    target = (chain["expiry"] == expiry) & (chain["strike"] == strike) & (chain["kind"] == "call")
    chain.loc[target, "bid"] = bumped - 0.08
    chain.loc[target, "ask"] = bumped + 0.08

    hits = find_outliers(analyze(chain, SPOT, ASOF), min_vol_points=1.0)

    assert len(hits) == 1
    found = hits.iloc[0]
    assert found["strike"] == strike
    assert found["verdict"] == "rich"
    assert "rich versus its own smile" in found["reason"]
    assert "x the quote's spread" in found["reason"]


def _plant(chain: pd.DataFrame, extra_vol: float, half_spread: float) -> pd.DataFrame:
    """Reprice one call at a different volatility, with a chosen quote width."""
    expiry, years = "2026-07-01", EXPIRIES["2026-07-01"]
    forward = SPOT * math.exp((RATE - DIVIDEND) * years)
    strike = 5200.0

    bumped = bs.price(
        Option(S=SPOT, K=strike, T=years, r=RATE,
               sigma=smooth_vol(strike, forward) + extra_vol, q=DIVIDEND, kind="call")
    )
    target = (chain["expiry"] == expiry) & (chain["strike"] == strike) & (chain["kind"] == "call")
    chain.loc[target, "bid"] = bumped - half_spread
    chain.loc[target, "ask"] = bumped + half_spread
    return chain


def test_the_same_gap_is_reported_on_a_tight_quote_and_not_on_a_wide_one():
    """The guard that stops the tool reporting the bid-ask spread as a signal.

    An identical half-point volatility difference is evidence when the quote is
    tight and meaningless when the quote is wide enough to contain it. Nothing
    about the contract changes between these two cases except its spread.
    """
    tight = analyze(_plant(synthetic_chain(0.08), 0.005, 0.08), SPOT, ASOF)
    wide = analyze(_plant(synthetic_chain(0.08), 0.005, 12.0), SPOT, ASOF, max_rel_spread=0.95)

    reported = find_outliers(tight, min_vol_points=0.4, min_spread_multiple=1.5)
    suppressed = find_outliers(wide, min_vol_points=0.4, min_spread_multiple=1.5)

    assert not reported.empty
    assert reported.iloc[0]["strike"] == 5200.0
    assert suppressed.empty


def test_illiquid_quotes_are_excluded_by_open_interest():
    chain = synthetic_chain()
    chain["open_interest"] = 5.0
    analysis = analyze(chain, SPOT, ASOF)
    assert find_outliers(analysis, min_open_interest=50.0).empty


def test_unreliable_expiries_are_never_scanned():
    """A slice whose smile did not fit contributes no candidates at all."""
    chain = synthetic_chain()
    rng = np.random.default_rng(0)
    noisy = chain["expiry"] == "2026-04-01"
    jitter = rng.normal(0.0, 6.0, int(noisy.sum()))
    chain.loc[noisy, "bid"] = (chain.loc[noisy, "bid"] + jitter).clip(lower=0.05)
    chain.loc[noisy, "ask"] = chain.loc[noisy, "bid"] + 0.16

    analysis = analyze(chain, SPOT, ASOF)
    hits = find_outliers(analysis)

    assert "2026-04-01" not in analysis.reliable_expiries
    assert hits.empty or "2026-04-01" not in set(hits["expiry"])


def test_fit_report_lists_every_expiry_with_a_verdict():
    report = analyze(synthetic_chain(), SPOT, ASOF).fit_report()

    assert len(report) == len(EXPIRIES)
    assert set(report.columns) >= {"expiry", "days", "quotes", "scannable", "note"}
    assert report["note"].notna().all()
