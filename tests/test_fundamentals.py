"""Reported financials, read from the SEC snapshots committed in data/fundamentals.

The real-data tests pin figures to the companies' own published results, so a
change in how facts are chosen or combined shows up as a wrong number for a
named company. The synthetic tests cover the cases those five filers happen
not to exercise.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from optpricing import fundamentals as fun
from optpricing.providers import sec

DATA = Path(__file__).resolve().parent.parent / "data" / "fundamentals"


@pytest.fixture(scope="module")
def companies() -> dict[str, fun.Fundamentals]:
    return {
        symbol: fun.analyze(sec.load_snapshot(paths[-1]))
        for symbol, paths in sec.list_snapshots(DATA).items()
    }


# ----------------------------------------------------------------- real filings


def test_apples_published_annual_results(companies):
    """Apple FY2025 (to 27 Sep 2025): revenue $416.161B, net income $112.010B, EPS $7.46."""
    aapl = companies["AAPL"]
    fy25 = next(y for y in aapl.years if y.year == 2025)
    assert aapl.annual["revenue"][fy25] == 416_161_000_000
    assert aapl.annual["net_income"][fy25] == 112_010_000_000
    assert aapl.annual["eps_diluted"][fy25] == pytest.approx(7.46)


def test_the_revenue_tag_change_is_followed(companies):
    """NVIDIA moved from RevenueFromContract... to Revenues in 2022; the newer tag wins."""
    assert companies["NVDA"].concepts["revenue"] == "Revenues"
    assert companies["AAPL"].concepts["revenue"] == "RevenueFromContractWithCustomerExcludingAssessedTax"


def test_twelve_months_after_a_10k_is_that_fiscal_year(companies):
    aapl = companies["AAPL"]
    assert aapl.latest_end == aapl.years[-1]
    for key in ("revenue", "net_income", "operating_cash_flow"):
        assert aapl.ttm[key] == aapl.annual[key][aapl.years[-1]]


def test_twelve_months_mid_year_is_ytd_plus_last_year_minus_prior_ytd():
    """Tesla to 30 Sep 2026, rebuilt by hand from the snapshot's own facts."""
    snapshot = sec.load_snapshot(sec.list_snapshots(DATA)["TSLA"][-1])
    rows = snapshot.facts["us-gaap"]["RevenueFromContractWithCustomerExcludingAssessedTax"]["units"]["USD"]

    def value(start: str, end: str) -> float:
        return max((r for r in rows if r["start"] == start and r["end"] == end), key=lambda r: r["filed"])["val"]

    expected = (
        value("2026-01-01", "2026-09-30") + value("2025-01-01", "2025-12-31")
        - value("2025-01-01", "2025-09-30")
    )
    assert fun.analyze(snapshot).ttm["revenue"] == expected


def test_a_flat_year_to_date_is_not_mistaken_for_a_bug(companies):
    """Kratos earned $16.1M in the first nine months of both 2025 and 2026, so its
    twelve months to September 2026 equal its 2025 full year exactly."""
    ktos = companies["KTOS"]
    assert ktos.ttm["net_income"] == ktos.annual["net_income"][ktos.years[-1]] == 22_000_000


def test_a_changed_fiscal_year_leaves_no_false_comparison(companies):
    """Red Cat moved its year-end from April to December in 2024. The eight-month
    transition period is not a year, and there is no prior twelve months to compare."""
    rcat = companies["RCAT"]
    assert [y.month for y in rcat.years][-2:] == [4, 12]
    assert rcat.ttm["revenue"] is not None
    assert rcat.ttm_prior["revenue"] is None
    assert rcat.growth("revenue") is None


def test_a_fallback_tag_fills_a_gap(companies):
    """Red Cat tags operating cash flow only as 'continuing operations' in some years."""
    assert companies["RCAT"].ttm["operating_cash_flow"] is not None


def test_valuation_multiples(companies):
    aapl = companies["AAPL"]
    v = aapl.valuation(352.78)
    assert v["market_cap"] == pytest.approx(352.78 * aapl.shares[0])
    assert v["pe"] == pytest.approx(v["market_cap"] / aapl.ttm["net_income"])
    assert v["ps"] == pytest.approx(v["market_cap"] / aapl.ttm["revenue"])


def test_no_pe_on_a_loss(companies):
    rcat = companies["RCAT"]
    assert rcat.ttm["net_income"] < 0
    assert rcat.valuation(3.53)["pe"] is None


def test_balance_sheets_balance(companies):
    for f in companies.values():
        assets, liabilities, equity = (f.balances[k] for k in ("assets", "liabilities", "equity"))
        assert assets[1] == liabilities[1] == equity[1]
        assert liabilities[0] + equity[0] == pytest.approx(assets[0], rel=0.02)


def test_every_company_has_a_profile_and_filings(companies):
    for f in companies.values():
        assert f.industry and f.exchange == "Nasdaq"
        assert f.shares and f.shares[0] > 0
        assert any(x["form"] == "10-Q" for x in f.filings)
        assert not f.missing


# --------------------------------------------------------------- synthetic cases


def _snapshot(us_gaap: dict, dei: dict | None = None) -> sec.FundamentalsSnapshot:
    facts = {"us-gaap": {k: {"units": {"USD": v}} for k, v in us_gaap.items()}}
    if dei:
        facts["dei"] = {k: {"units": {"shares": v}} for k, v in dei.items()}
    return sec.FundamentalsSnapshot("TEST", 1, date(2026, 1, 1), {"name": "Test"}, [], facts)


def _fact(start, end, val, filed, form="10-K", accn="a"):
    row = {"end": end, "val": val, "filed": filed, "form": form, "accn": accn}
    if start:
        row["start"] = start
    return row


def test_a_restatement_replaces_the_original():
    f = fun.analyze(_snapshot({"Revenues": [
        _fact("2024-01-01", "2024-12-31", 100, "2025-02-01"),
        _fact("2024-01-01", "2024-12-31", 90, "2026-02-01"),  # restated in next year's 10-K
        _fact("2025-01-01", "2025-12-31", 120, "2026-02-01"),
    ]}))
    assert f.annual["revenue"] == {date(2024, 12, 31): 90, date(2025, 12, 31): 120}
    assert f.growth("revenue") == pytest.approx(120 / 90 - 1)


def test_share_classes_are_added_together():
    f = fun.analyze(_snapshot(
        {"Revenues": [_fact("2025-01-01", "2025-12-31", 1, "2026-02-01")]},
        {"EntityCommonStockSharesOutstanding": [
            _fact(None, "2026-01-20", 600, "2026-02-01", accn="k"),
            _fact(None, "2026-01-20", 400, "2026-02-01", accn="k"),
            _fact(None, "2025-01-20", 999, "2025-02-01", accn="old"),
        ]},
    ))
    assert f.shares == (1000, date(2026, 1, 20))


def test_missing_pieces_give_none_not_an_error():
    f = fun.analyze(_snapshot({"Revenues": [_fact("2025-01-01", "2025-06-30", 50, "2025-08-01", "10-Q")]}))
    assert f.ttm["revenue"] is None  # half a year, and no full year to build from
    assert f.shares is None
    assert f.valuation(10.0) == {"market_cap": None, "pe": None, "ps": None, "pb": None, "fcf_yield": None}
    assert "Net income" in f.missing


def test_labels():
    assert fun.fiscal_label(date(2026, 9, 26)) == "FY Sep 2026"
