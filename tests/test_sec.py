"""SEC EDGAR: identifying ourselves, trimming payloads, and storing snapshots.

Every test runs offline. The payloads below are small synthetic stand-ins
shaped like EDGAR's responses; the real ones are exercised through the
committed snapshots in tests/test_fundamentals.py.
"""

from __future__ import annotations

import urllib.error
from datetime import date

import pytest

from optpricing.providers import sec

AGENT = "Test Runner test@example.com"

TICKERS = {"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}}

SUBMISSIONS = {
    "cik": "0000320193", "name": "Apple Inc.", "tickers": ["AAPL"], "exchanges": ["Nasdaq"],
    "sic": "3571", "sicDescription": "Electronic Computers", "fiscalYearEnd": "0925",
    "category": "Large accelerated filer", "addresses": {"business": {"city": "Cupertino"}},
    "filings": {"recent": {
        "form": ["8-K", "10-K", "10-Q", "4"],
        "filingDate": ["2026-11-10", "2026-11-03", "2026-07-31", "2026-07-01"],
        "reportDate": ["2026-11-10", "2026-09-26", "2026-06-27", ""],
        "accessionNumber": ["a-1", "0000320193-26-000026", "0000320193-26-000020", "a-4"],
        "primaryDocument": ["x.htm", "aapl-20260926.htm", "aapl-20260627.htm", "y.xml"],
    }},
}

FACTS = {
    "cik": 320193,
    "facts": {
        "us-gaap": {
            "NetIncomeLoss": {"label": "Net income", "units": {"USD": [
                {"start": "2024-09-29", "end": "2025-09-27", "val": 112010000000, "form": "10-K",
                 "filed": "2025-10-31", "accn": "k25", "fy": 2025, "fp": "FY", "frame": "CY2025"},
                {"start": "2024-09-29", "end": "2025-09-27", "val": 1, "form": "8-K",
                 "filed": "2025-10-30", "accn": "x"},
            ]}},
            "SomethingElse": {"label": "Unused", "units": {"USD": [
                {"end": "2025-09-27", "val": 5, "form": "10-K", "filed": "2025-10-31"},
            ]}},
        },
    },
}


def _fake(responses: dict, seen: list | None = None):
    def get_json(url: str, user_agent: str, timeout: float) -> dict:
        if seen is not None:
            seen.append((url, user_agent))
        for fragment, body in responses.items():
            if fragment in url:
                return body
        raise urllib.error.HTTPError(url, 404, "Not Found", None, None)

    return get_json


EDGAR = {"company_tickers": TICKERS, "submissions": SUBMISSIONS, "companyfacts": FACTS}


def test_a_user_agent_is_required(monkeypatch):
    monkeypatch.delenv("SEC_USER_AGENT", raising=False)
    with pytest.raises(sec.MissingUserAgentError, match="name and email"):
        sec.resolve_user_agent()


def test_the_user_agent_must_carry_an_email():
    with pytest.raises(sec.MissingUserAgentError, match="email"):
        sec.resolve_user_agent("Just A Name")


def test_the_user_agent_comes_from_the_environment(monkeypatch):
    monkeypatch.setenv("SEC_USER_AGENT", f"  {AGENT} ")
    assert sec.resolve_user_agent() == AGENT


def test_every_request_says_who_is_asking():
    seen: list = []
    sec.fetch_snapshot("AAPL", AGENT, today=date(2026, 11, 18), get_json=_fake(EDGAR, seen))
    assert [url.split("/")[-1] for url, _ in seen] == [
        "company_tickers.json", "CIK0000320193.json", "CIK0000320193.json",
    ]
    assert all(agent == AGENT for _, agent in seen)


def test_requests_are_spaced_out():
    """The SEC's limit is ten a second; the client keeps well under it."""
    waits: list[float] = []
    client = sec._Client(AGENT, _fake(EDGAR), 5.0, sleep=waits.append)
    for _ in range(3):
        client(sec.TICKERS_URL)
    assert len(waits) == 2
    assert all(0 < w <= sec.MIN_INTERVAL for w in waits)


def test_a_known_cik_skips_the_ticker_lookup():
    seen: list = []
    sec.fetch_snapshot("AAPL", AGENT, cik=320193, get_json=_fake(EDGAR, seen))
    assert not any("company_tickers" in url for url, _ in seen)


def test_an_unknown_ticker_is_explained():
    with pytest.raises(sec.UnknownCompanyError, match="ETFs"):
        sec.lookup_cik("SPY", TICKERS)


def test_a_refusal_mentions_the_user_agent():
    def forbidden(url, agent, timeout):
        raise urllib.error.HTTPError(url, 403, "Forbidden", None, None)

    with pytest.raises(sec.SECError, match="User-Agent"):
        sec.fetch_snapshot("AAPL", AGENT, cik=320193, get_json=forbidden)


def test_only_periodic_filings_are_kept_newest_first():
    profile, filings = sec.trim_submissions(SUBMISSIONS)
    assert [f["form"] for f in filings] == ["10-K", "10-Q"]
    assert profile["sicDescription"] == "Electronic Computers"
    assert "addresses" not in profile  # nothing personal or unused is stored


def test_facts_are_trimmed_to_what_is_read():
    trimmed = sec.trim_facts(FACTS, [("us-gaap", "NetIncomeLoss"), ("us-gaap", "Missing")])
    assert list(trimmed["us-gaap"]) == ["NetIncomeLoss"]
    rows = trimmed["us-gaap"]["NetIncomeLoss"]["units"]["USD"]
    assert [r["form"] for r in rows] == ["10-K"]  # the 8-K figure is dropped
    assert "frame" not in rows[0]


def test_a_snapshot_survives_a_round_trip(tmp_path):
    snapshot = sec.build_snapshot("aapl", SUBMISSIONS, FACTS, date(2026, 11, 18))
    path = sec.save_snapshot(snapshot, tmp_path)
    assert path.name == "aapl_2026-11-18.json.gz"
    loaded = sec.load_snapshot(path)
    assert loaded == snapshot
    assert loaded.cik == 320193 and loaded.symbol == "AAPL"
    assert sec.list_snapshots(tmp_path) == {"AAPL": [path]}


def test_saving_the_same_snapshot_twice_gives_identical_bytes(tmp_path):
    """So re-fetching unchanged data never shows up as a change in git."""
    snapshot = sec.build_snapshot("AAPL", SUBMISSIONS, FACTS, date(2026, 11, 18))
    first = sec.save_snapshot(snapshot, tmp_path / "a").read_bytes()
    second = sec.save_snapshot(snapshot, tmp_path / "b").read_bytes()
    assert first == second


def test_filing_links_point_at_the_document():
    snapshot = sec.build_snapshot("AAPL", SUBMISSIONS, FACTS, date(2026, 11, 18))
    latest = snapshot.latest_filing("10-K")
    assert sec.filing_url(snapshot.cik, latest) == (
        "https://www.sec.gov/Archives/edgar/data/320193/000032019326000026/aapl-20260926.htm"
    )
