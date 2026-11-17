"""Cboe chains for stocks: which price, which rows, and what counts as missing.

The fixtures are real Cboe responses for AAPL and RCAT captured after the close
on 16 Nov 2026, trimmed to a few expiries. They were taken out of hours on
purpose: that is when the after-hours price and the option quotes disagree.
"""

from __future__ import annotations

import copy
import json
import math
from datetime import UTC, datetime
from pathlib import Path

import pytest

from optpricing.providers import cboe

FIXTURES = Path(__file__).parent / "fixtures"


def _load(symbol: str) -> dict:
    return json.loads((FIXTURES / f"cboe_{symbol.lower()}_2026-11-16.json").read_text())


@pytest.fixture
def aapl() -> dict:
    return _load("AAPL")


@pytest.fixture
def rcat() -> dict:
    return _load("RCAT")


# ------------------------------------------------------------------ the spot price


def test_after_the_close_the_spot_is_the_close_not_the_after_hours_price(aapl):
    """AAPL's options implied a $352.72 forward against a $352.78 close and a
    $353.61 after-hours price. The spot has to be the price the quotes reflect."""
    snap = cboe.parse_payload(aapl, "AAPL")
    assert snap.spot == pytest.approx(352.78)
    assert snap.spot_basis == "regular-session close"
    assert snap.underlying["current_price"] == pytest.approx(353.611)


def test_the_gap_that_motivates_the_rule_is_large_on_a_small_cap(rcat):
    """RCAT closed at $3.53 and traded at $2.89 after hours: an 18% difference."""
    snap = cboe.parse_payload(rcat, "RCAT")
    assert snap.spot == pytest.approx(3.53)
    assert snap.underlying["current_price"] / snap.spot - 1 < -0.15


def test_during_the_session_the_spot_is_the_last_trade(aapl):
    live = copy.deepcopy(aapl)
    live["timestamp"] = "2026-11-16 15:30:00"  # 10:30 in New York, a Monday
    snap = cboe.parse_payload(live, "AAPL")
    assert snap.spot == pytest.approx(353.611)
    assert snap.spot_basis == "last trade"


def test_a_missing_close_falls_back_to_the_previous_close(aapl):
    aapl["data"]["close"] = 0
    snap = cboe.parse_payload(aapl, "AAPL")
    assert snap.spot == pytest.approx(356.27)
    assert snap.spot_basis == "previous close"


def test_no_usable_price_at_all_is_an_error(aapl):
    for key in ("current_price", "close", "prev_day_close"):
        aapl["data"][key] = None
    with pytest.raises(ValueError, match="share price"):
        cboe.parse_payload(aapl, "AAPL")


@pytest.mark.parametrize(
    "stamp, expected",
    [
        ("2026-11-16 15:30:00", True),   # Monday 10:30 ET
        ("2026-11-16 14:29:00", False),  # Monday 09:29 ET, before the open
        ("2026-11-16 21:00:00", False),  # Monday 16:00 ET, the close itself
        ("2026-11-14 16:00:00", False),  # Saturday
    ],
)
def test_regular_session_hours(stamp: str, expected: bool):
    moment = datetime.fromisoformat(stamp).replace(tzinfo=UTC)
    assert cboe.in_regular_session(moment) is expected


# ------------------------------------------------------------------- which rows


def test_contracts_expiring_on_the_session_day_are_removed_and_counted(aapl):
    snap = cboe.parse_payload(aapl, "AAPL")
    assert "2026-11-16" not in snap.expiries
    assert snap.dropped["expired or expiring today"] > 0
    assert min(snap.expiries) > snap.trading_date.isoformat()


def test_adjusted_contracts_are_removed_and_counted(aapl):
    """A modified root such as AAPL1 delivers something other than 100 shares."""
    adjusted = dict(aapl["data"]["options"][-1], option="AAPL1261218C00300000")
    aapl["data"]["options"].append(adjusted)
    snap = cboe.parse_payload(aapl, "AAPL")
    assert snap.dropped["non-standard deliverable"] == 1
    assert set(snap.quotes["root"]) == {"AAPL"}


def test_every_contract_is_for_one_hundred_shares(aapl):
    assert set(cboe.parse_payload(aapl, "AAPL").quotes["multiplier"]) == {100}


def test_a_zero_implied_vol_from_cboe_is_missing_not_zero(aapl):
    snap = cboe.parse_payload(aapl, "AAPL")
    raw_zeros = sum(1 for q in aapl["data"]["options"] if (q.get("iv") or 0) == 0)
    assert raw_zeros > 0
    assert snap.quotes["cboe_iv"].isna().sum() > 0
    assert (snap.quotes["cboe_iv"].dropna() > 0).all()


def test_iv30_arrives_in_percent_and_is_stored_as_a_decimal(aapl):
    snap = cboe.parse_payload(aapl, "AAPL")
    assert snap.underlying["iv30"] == pytest.approx(aapl["data"]["iv30"] / 100)


def test_an_empty_chain_is_an_unknown_symbol_and_still_a_value_error(aapl):
    aapl["data"]["options"] = []
    with pytest.raises(cboe.UnknownSymbolError) as caught:
        cboe.parse_payload(aapl, "AAPL")
    assert isinstance(caught.value, ValueError)


# ------------------------------------------------------- the index path is unchanged


def _index_payload() -> dict:
    return {
        "timestamp": "2026-11-17 02:00:00",
        "symbol": "_SPX",
        "data": {
            "current_price": 7500.0, "close": 7490.0,
            "options": [
                {"option": "SPXW261116C07500000", "bid": 1.0, "ask": 1.2, "iv": 0.0},
                {"option": "SPXW261218C07500000", "bid": 90.0, "ask": 91.0, "iv": 0.15},
            ],
        },
    }


def test_index_spot_rule_and_rows_are_exactly_as_before():
    """Index results are published; nothing about how they are parsed may move."""
    snap = cboe.parse_payload(_index_payload(), "SPX")
    assert snap.kind == "index"
    assert snap.spot == 7500.0
    assert len(snap.quotes) == 2
    assert snap.quotes["cboe_iv"].iloc[0] == 0.0
    assert "spot_basis" not in snap.to_frame().columns
    assert "multiplier" not in snap.quotes.columns


# ------------------------------------------------------------------------ tickers


@pytest.mark.parametrize("typed, expected", [("aapl", "AAPL"), (" $nvda ", "NVDA"), ("^SPX", "SPX")])
def test_tickers_are_normalised(typed: str, expected: str):
    assert cboe.normalize(typed) == expected


@pytest.mark.parametrize("typed", ["", "APPLE1", "TOOLONG", "BRK.B", "12"])
def test_impossible_tickers_are_refused_before_any_network_call(typed: str):
    with pytest.raises(cboe.InvalidTickerError):
        cboe.normalize(typed)


def test_index_and_stock_urls_differ_by_one_underscore(monkeypatch, aapl):
    pytest.importorskip("requests")
    requested = []

    class Response:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return aapl

    def fake_get(url, **_):
        requested.append(url)
        return Response()

    import requests

    monkeypatch.setattr(requests, "get", fake_get)
    cboe.fetch_payload("aapl")
    cboe.fetch_payload("SPX")
    assert requested[0].endswith("/options/AAPL.json")
    assert requested[1].endswith("/options/_SPX.json")


@pytest.mark.parametrize("status", [403, 404])
def test_a_ticker_cboe_does_not_list_says_so(monkeypatch, status: int):
    pytest.importorskip("requests")

    class Response:
        status_code = status

    import requests

    monkeypatch.setattr(requests, "get", lambda url, **_: Response())
    with pytest.raises(cboe.UnknownSymbolError, match="no listed options for ZZZZ"):
        cboe.fetch_chain("ZZZZ")


# -------------------------------------------------------------------- storage


def test_a_saved_snapshot_reads_back_identically(tmp_path, aapl):
    path = cboe.save_payload(aapl, tmp_path)
    original = cboe.parse_payload(aapl, "AAPL")
    reloaded = cboe.load_snapshot(path)
    assert reloaded.spot == original.spot
    assert reloaded.spot_basis == original.spot_basis
    assert reloaded.quotes.equals(original.quotes)


def test_snapshots_are_named_by_the_new_york_session_not_the_utc_date(tmp_path, aapl):
    """Stamped 02:02 UTC on 17 Nov, which is 9:02 pm on the 16th in New York."""
    assert cboe.save_payload(aapl, tmp_path).name == "aapl_2026-11-16.json.gz"


def test_snapshots_are_listed_by_symbol(tmp_path, aapl, rcat):
    cboe.save_payload(aapl, tmp_path)
    cboe.save_payload(rcat, tmp_path)
    assert set(cboe.list_snapshots(tmp_path)) == {"AAPL", "RCAT"}


def test_index_payloads_name_their_symbol_with_an_underscore():
    assert cboe.payload_symbol(_index_payload()) == "SPX"


# -------------------------------------------------------------------- freshness


def test_a_snapshot_is_never_stale_outside_the_session(aapl):
    snap = cboe.parse_payload(aapl, "AAPL")
    saturday = datetime(2026, 11, 21, 15, 0, tzinfo=UTC)
    assert not snap.is_stale(saturday)


def test_an_old_snapshot_during_the_session_is_stale(aapl):
    snap = cboe.parse_payload(aapl, "AAPL")
    next_morning = datetime(2026, 11, 17, 16, 0, tzinfo=UTC)
    assert snap.is_stale(next_morning)
    assert math.isclose(snap.age_minutes(next_morning), 13 * 60 + 57, abs_tol=1)


def test_the_old_import_path_still_works():
    from optpricing.data import cboe as old

    assert old.fetch_chain is cboe.fetch_chain
    assert old.INDEX_SYMBOLS is cboe.INDEX_SYMBOLS
