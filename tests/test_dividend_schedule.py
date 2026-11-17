"""Projecting ex-dividend dates, and refusing to guess for unknown tickers."""

from __future__ import annotations

from datetime import date

import pytest

from optpricing import dividends as divs

ASOF = date(2026, 11, 16)


def test_every_initial_ticker_has_a_sourced_entry():
    for symbol in ("AAPL", "NVDA", "TSLA", "KTOS", "RCAT"):
        policy = divs.POLICIES[symbol]
        assert policy.source, f"{symbol} has no source recorded"
        assert not policy.is_stale(ASOF)


def test_non_payers_have_no_schedule():
    for symbol in ("TSLA", "KTOS", "RCAT"):
        assert divs.schedule(symbol, ASOF, date(2027, 12, 31)) == ()


def test_an_unknown_ticker_raises_rather_than_assuming_zero():
    """Pricing a payer as a non-payer gives a wrong price with no error."""
    with pytest.raises(divs.UnknownTickerError, match="rather than assuming"):
        divs.schedule("MSFT", ASOF, date(2027, 6, 18))


def test_lookup_ignores_case():
    assert divs.schedule("aapl", ASOF, date(2027, 6, 18)) == divs.schedule(
        "AAPL", ASOF, date(2027, 6, 18)
    )


def test_quarterly_projection_from_the_last_ex_date():
    """AAPL went ex on 12 Nov 2026, so the next ones fall a quarter apart."""
    found = divs.payments("AAPL", ASOF, date(2027, 8, 31))
    assert [p.ex_date for p in found] == [
        date(2027, 2, 12), date(2027, 5, 12), date(2027, 8, 12),
    ]
    assert all(p.amount == 0.27 for p in found)


def test_projected_payments_are_labelled_as_projections():
    found = divs.payments("AAPL", ASOF, date(2027, 6, 1))
    assert all(p.projected for p in found)


def test_a_declared_upcoming_dividend_is_not_called_a_projection():
    """Before an announced ex-date, the first payment is the company's own date."""
    found = divs.payments("AAPL", date(2026, 11, 1), date(2027, 1, 1))
    assert found[0].ex_date == date(2026, 11, 12)
    assert not found[0].projected


def test_an_ex_date_on_the_valuation_date_is_excluded():
    """On its ex-date the stock already trades without the dividend."""
    found = divs.payments("AAPL", date(2026, 11, 12), date(2027, 1, 1))
    assert date(2026, 11, 12) not in [p.ex_date for p in found]


def test_schedule_times_use_the_library_day_count():
    """Calendar days over 365, the same clock market.py uses for maturities."""
    (when, amount), *_ = divs.schedule("NVDA", ASOF, date(2027, 3, 1))
    assert when == pytest.approx((date(2026, 12, 10) - ASOF).days / 365.0)
    assert amount == 0.25


def test_month_arithmetic_clamps_to_short_months():
    assert divs._add_months(date(2026, 11, 30), 3) == date(2027, 2, 28)
    assert divs._add_months(date(2027, 11, 30), 3) == date(2028, 2, 29)


def test_an_old_entry_is_flagged_stale():
    policy = divs.POLICIES["AAPL"]
    assert policy.is_stale(date(2027, 6, 1))
