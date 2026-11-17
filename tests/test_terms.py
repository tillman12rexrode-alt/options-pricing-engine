"""Equity terms: the curve and the dividend schedule in place of put-call parity."""

from __future__ import annotations

import math
from datetime import date

import pytest

from optpricing import pricers
from optpricing.models import binomial
from optpricing.providers.fred import YieldCurve, flat_curve
from optpricing.terms import equity_terms

ASOF = date(2026, 11, 16)
FLAT = flat_curve(0.04, ASOF)


def test_a_non_payer_has_the_textbook_forward():
    terms = equity_terms("TSLA", 400.0, ASOF, "2027-06-18", FLAT)
    assert terms.dividends == ()
    assert terms.forward == pytest.approx(400.0 * math.exp(0.04 * terms.years))


def test_a_payer_forward_is_reduced_by_its_dividends():
    terms = equity_terms("AAPL", 270.0, ASOF, "2027-06-18", FLAT)
    expected_pv = sum(a * math.exp(-0.04 * t) for t, a in terms.dividends)
    assert len(terms.dividends) == 2
    assert terms.pv_dividends == pytest.approx(expected_pv)
    assert terms.forward == pytest.approx((270.0 - expected_pv) * math.exp(0.04 * terms.years))


def test_the_rate_comes_from_the_curve_at_the_option_maturity():
    curve = YieldCurve(ASOF, (0.0, 1.0, 2.0), (0.03, 0.04, 0.05), "test")
    terms = equity_terms("TSLA", 400.0, ASOF, "2027-11-16", curve)
    assert terms.rate == pytest.approx(curve.rate(terms.years))
    assert terms.discount == pytest.approx(math.exp(-terms.rate * terms.years))


def test_the_option_it_builds_is_american_and_carries_the_dividends():
    terms = equity_terms("NVDA", 180.0, ASOF, "2027-03-19", FLAT)
    opt = terms.option(180.0, "call", sigma=0.45)
    assert opt.style == "american"
    assert opt.dividends == terms.dividends
    assert opt.q == 0.0
    assert not pricers.has_closed_form(opt)


def test_a_non_payer_american_call_prices_as_european():
    """Merton, end to end through the new terms: no dividend, no early exercise."""
    terms = equity_terms("RCAT", 12.0, ASOF, "2027-01-15", FLAT)
    american = binomial.price(terms.option(12.0, "call", 0.9), 501, "lr")
    european = binomial.price(terms.option(12.0, "call", 0.9, style="european"), 501, "lr")
    assert american == pytest.approx(european, abs=1e-9)


def test_terms_report_where_their_inputs_came_from():
    row = equity_terms("AAPL", 270.0, ASOF, "2027-01-15", FLAT).as_row()
    assert row["curve_source"] == "flat assumption"
    assert row["dividends_stale"] is False


def test_expiry_must_be_after_the_valuation_date():
    with pytest.raises(ValueError, match="not after"):
        equity_terms("TSLA", 400.0, ASOF, "2026-11-16", FLAT)


def test_unknown_tickers_are_refused_here_too():
    with pytest.raises(KeyError):
        equity_terms("MSFT", 500.0, ASOF, "2027-06-18", FLAT)
