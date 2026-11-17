"""Model vs market for stock options, run on the snapshots committed in data/equity.

These check what the Stock options page shows: that each contract's American
implied vol reprices its own mid, that it agrees with Cboe's independent
calculation, that the carry implied from calls and puts actually reconciles
them, that a smile is only used where the chain supports one, and that
nothing on the page passes a gap off as a trading signal.
"""

from __future__ import annotations

import dataclasses
import math
from datetime import date
from pathlib import Path

import numpy as np
import pytest

from optpricing import pricers
from optpricing.equity_analysis import analyze_expiry, implied_carry
from optpricing.providers import cboe
from optpricing.providers.fred import flat_curve

DATA = Path(__file__).resolve().parent.parent / "data" / "equity"
DAY = date(2026, 11, 16)
EXPIRY = "2026-12-18"
#: A flat rate stands in for FRED so the tests run offline and stay fixed.
CURVE = flat_curve(0.045, DAY)


def _snapshot(symbol: str) -> cboe.ChainSnapshot:
    return cboe.load_snapshot(cboe.list_snapshots(DATA)[symbol][0])


def _window(snap: cboe.ChainSnapshot, width: float = 0.25) -> list[float]:
    strikes = snap.quotes.loc[snap.quotes["expiry"] == EXPIRY, "strike"].unique()
    return sorted(k for k in strikes if abs(k / snap.spot - 1) <= width)


@pytest.fixture(scope="module")
def aapl():
    snap = _snapshot("AAPL")
    return snap, analyze_expiry(snap, EXPIRY, CURVE, strikes=_window(snap))


@pytest.fixture(scope="module")
def rcat():
    snap = _snapshot("RCAT")
    return snap, analyze_expiry(snap, EXPIRY, CURVE, strikes=_window(snap, 0.6))


def test_each_implied_vol_reprices_its_own_mid(aapl):
    _, analysis = aapl
    solved = analysis.table.dropna(subset=["model_iv"])
    assert len(solved) >= 20
    for row in solved.itertuples():
        option = analysis.terms.option(row.strike, row.kind)
        price = pricers.for_option(option)(dataclasses.replace(option, sigma=row.model_iv))
        assert price == pytest.approx(row.mid, abs=2e-3)


def test_our_vols_agree_with_cboes_independent_calculation(aapl):
    """Cboe computes the same quantity its own way; a large gap means one of us is wrong."""
    _, analysis = aapl
    assert analysis.cboe_gap_points is not None
    assert analysis.cboe_gap_points < 0.3


def test_a_liquid_chain_is_priced_off_its_smile(aapl):
    _, analysis = aapl
    assert analysis.priced
    priced = analysis.table.dropna(subset=["model_price"])
    assert len(priced) == len(analysis.table.dropna(subset=["fitted_iv"]))
    inside = (priced["gap_in_spreads"].abs() <= 1).mean()
    assert inside >= 0.75


def test_a_thin_chain_keeps_its_vols_and_greeks_but_is_not_priced(rcat):
    _, analysis = rcat
    assert not analysis.priced
    assert analysis.table["model_iv"].notna().any()
    assert analysis.table["delta"].notna().any()
    assert analysis.table["model_price"].isna().all()
    assert "needed" in analysis.fit_note or "spread" in analysis.fit_note


def test_the_strike_window_is_respected(aapl):
    snap, analysis = aapl
    assert set(analysis.table["strike"]) <= set(_window(snap))


def test_nothing_is_labelled_cheap_rich_or_a_signal(aapl, rcat):
    """A gap against a model fitted to the same quotes is a measurement, not advice."""
    for _, analysis in (aapl, rcat):
        names = " ".join(analysis.table.columns).lower()
        for word in ("cheap", "rich", "signal", "buy", "sell", "verdict"):
            assert word not in names


def test_greeks_have_the_right_signs(aapl):
    table = aapl[1].table.dropna(subset=["delta"])
    calls, puts = table[table["kind"] == "call"], table[table["kind"] == "put"]
    assert calls["delta"].between(0, 1).all()
    assert puts["delta"].between(-1, 0).all()
    assert (table["gamma"] > 0).all()
    assert (table["vega_pt"] > 0).all()
    assert (calls["rho_pt"] > 0).all()
    assert (puts["rho_pt"] < 0).all()
    assert (calls["theta_day"] < 0).all()


def test_vega_is_per_vol_point_and_theta_per_day(aapl):
    """Check the units by bumping the inputs and repricing."""
    analysis = aapl[1]
    row = analysis.table.dropna(subset=["vega_pt"]).iloc[0]
    option = dataclasses.replace(analysis.terms.option(row.strike, row.kind), sigma=row.model_iv)
    pricer = pricers.for_option(option)
    up = pricer(dataclasses.replace(option, sigma=row.model_iv + 0.005))
    down = pricer(dataclasses.replace(option, sigma=row.model_iv - 0.005))
    assert (up - down) == pytest.approx(row.vega_pt, rel=0.02)
    later = pricer(dataclasses.replace(option, T=option.T - 1 / 365))
    assert (later - pricer(option)) == pytest.approx(row.theta_day, rel=0.1, abs=2e-3)


def test_the_implied_carry_makes_calls_and_puts_agree(aapl, rcat):
    """RCAT at the Treasury rate splits by several vol points; at its own carry it does not."""
    for _, analysis in (aapl, rcat):
        assert analysis.implied_rate is not None
        assert analysis.terms.rate == pytest.approx(analysis.implied_rate)
        assert abs(analysis.parity_gap_used) < 0.05
    rcat_analysis = rcat[1]
    assert abs(rcat_analysis.parity_gap_treasury) > 2
    assert rcat_analysis.implied_rate < 0  # costly to borrow, so carry sits below zero


def test_the_treasury_rate_can_still_be_asked_for(aapl):
    snap, implied = aapl
    analysis = analyze_expiry(snap, EXPIRY, CURVE, strikes=_window(snap), carry="treasury")
    assert analysis.implied_rate is None
    assert analysis.terms.rate == pytest.approx(0.045)
    assert analysis.treasury_rate == pytest.approx(0.045)
    assert implied.treasury_rate == pytest.approx(0.045)
    assert analysis.rate_source == "Treasury curve"


def test_no_puts_means_no_implied_carry():
    snap = _snapshot("AAPL")
    calls_only = dataclasses.replace(snap, quotes=snap.quotes[snap.quotes["kind"] == "call"])
    rate, pairs = implied_carry(calls_only, EXPIRY, snap.spot)
    assert rate is None
    assert pairs == 0
    analysis = analyze_expiry(calls_only, EXPIRY, CURVE, strikes=_window(snap))
    assert analysis.rate_source == "Treasury curve"
    assert math.isnan(analysis.parity_gap_treasury)


def test_an_ask_below_the_no_arbitrage_floor_is_named_not_solved():
    """A deep in-the-money call offered under intrinsic is a bad quote, not a vol."""
    snap = _snapshot("AAPL")
    quotes = snap.quotes.copy()
    strike = 250.0
    row = (quotes["expiry"] == EXPIRY) & (quotes["kind"] == "call") & (quotes["strike"] == strike)
    assert row.sum() == 1
    intrinsic = snap.spot - strike
    quotes.loc[row, ["bid", "ask"]] = [intrinsic - 3.0, intrinsic - 2.0]
    broken = dataclasses.replace(snap, quotes=quotes)
    analysis = analyze_expiry(broken, EXPIRY, CURVE, strikes=[strike], carry="treasury")
    call = analysis.table[analysis.table["kind"] == "call"].iloc[0]
    assert np.isnan(call["model_iv"])
    assert "floor" in call["iv_note"]
