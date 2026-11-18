"""A company's reported numbers, read from its XBRL facts the way an analyst would.

Three problems make raw companyfacts unusable as-is, and each is handled here:

1. The same quantity goes by different names. Apple's revenue is tagged
   `Revenues` until 2018 and `RevenueFromContractWithCustomerExcludingAssessedTax`
   after; NVIDIA did the reverse in 2022. For each metric this module takes the
   candidate concept with the most recent data, and fills older periods from the
   other candidates.
2. Every number is filed several times: once in its own report, then again as
   a comparative in later reports, sometimes restated. The most recently filed
   value for each period wins.
3. 10-Qs report fiscal-year-to-date figures, so there is usually no fourth
   quarter on file. The trailing twelve months are built from what is:
   year-to-date + the last full year - the same year-to-date a year earlier.

What this module does not do: adjust for one-off items, convert currencies,
or reconcile segment data. Every figure is as the company tagged it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date

from optpricing.providers.sec import FundamentalsSnapshot

__all__ = [
    "ANNUAL_DAYS",
    "CONCEPTS",
    "METRICS",
    "Fundamentals",
    "Metric",
    "analyze",
    "fiscal_label",
]


@dataclass(frozen=True)
class Metric:
    key: str
    label: str
    kind: str  # "duration" (a flow over a period) or "instant" (a balance on a day)
    candidates: tuple[str, ...]
    unit: str = "USD"
    taxonomy: str = "us-gaap"


METRICS: tuple[Metric, ...] = (
    Metric("revenue", "Revenue", "duration", (
        "RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues",
        "SalesRevenueNet", "RevenueFromContractWithCustomerIncludingAssessedTax",
    )),
    Metric("gross_profit", "Gross profit", "duration", ("GrossProfit",)),
    Metric("operating_income", "Operating income", "duration", ("OperatingIncomeLoss",)),
    Metric("net_income", "Net income", "duration", ("NetIncomeLoss", "ProfitLoss")),
    Metric("eps_diluted", "Diluted EPS", "duration", (
        "EarningsPerShareDiluted", "EarningsPerShareBasicAndDiluted",
    ), unit="USD/shares"),
    Metric("operating_cash_flow", "Operating cash flow", "duration", (
        "NetCashProvidedByUsedInOperatingActivities",
        "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
    )),
    Metric("capex", "Capital expenditure", "duration", (
        "PaymentsToAcquirePropertyPlantAndEquipment", "PaymentsToAcquireProductiveAssets",
    )),
    Metric("cash", "Cash and equivalents", "instant", (
        "CashAndCashEquivalentsAtCarryingValue", "Cash",
        "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
    )),
    Metric("assets", "Total assets", "instant", ("Assets",)),
    Metric("liabilities", "Total liabilities", "instant", ("Liabilities",)),
    Metric("equity", "Shareholders' equity", "instant", (
        "StockholdersEquity",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    )),
)

SHARES = Metric(
    "shares", "Shares outstanding", "instant",
    ("EntityCommonStockSharesOutstanding",), unit="shares", taxonomy="dei",
)

#: Every (taxonomy, concept) a snapshot needs to keep.
CONCEPTS: tuple[tuple[str, str], ...] = tuple(
    (m.taxonomy, c) for m in (*METRICS, SHARES) for c in m.candidates
)

#: A fiscal year is 52 or 53 weeks, or a calendar year.
ANNUAL_DAYS = (350, 380)

#: Fiscal periods ending this close together are the same period.
_SAME_DAY = 7

#: Years of annual history shown.
YEARS = 4


def fiscal_label(end: date) -> str:
    """FY Sep 2026: the month a fiscal year ends identifies it unambiguously."""
    return f"FY {end:%b} {end.year}"


# ---------------------------------------------------------------- reading facts


def _days(start: date | None, end: date) -> int:
    return 0 if start is None else (end - start).days + 1


@dataclass(frozen=True)
class _Fact:
    start: date | None
    end: date
    value: float
    filed: str
    form: str
    accession: str

    @property
    def days(self) -> int:
        return _days(self.start, self.end)


def _rows(snapshot: FundamentalsSnapshot, metric: Metric, concept: str) -> list[_Fact]:
    units = snapshot.facts.get(metric.taxonomy, {}).get(concept, {}).get("units", {})
    rows = []
    for row in units.get(metric.unit, []):
        start = date.fromisoformat(row["start"]) if row.get("start") else None
        if (metric.kind == "duration") != (start is not None):
            continue
        rows.append(_Fact(
            start, date.fromisoformat(row["end"]), float(row["val"]),
            row.get("filed", ""), row.get("form", ""), row.get("accn", ""),
        ))
    return rows


def _latest_filed(rows: list[_Fact]) -> dict[tuple, _Fact]:
    """One fact per period: the most recently filed, originals before amendments on a tie."""
    best: dict[tuple, _Fact] = {}
    for fact in rows:
        key = (fact.start, fact.end)
        current = best.get(key)
        rank = (fact.filed, not fact.form.endswith("/A"))
        if current is None or rank > (current.filed, not current.form.endswith("/A")):
            best[key] = fact
    return best


def _series(snapshot: FundamentalsSnapshot, metric: Metric) -> tuple[dict[tuple, _Fact], str | None]:
    """Every period the metric is known for, and the concept it mainly comes from.

    The main concept is whichever candidate has the most recent data; older
    periods it lacks are filled from the other candidates, in priority order.
    """
    by_concept = {c: _latest_filed(_rows(snapshot, metric, c)) for c in metric.candidates}
    present = [c for c in metric.candidates if by_concept[c]]
    if not present:
        return {}, None
    main = max(present, key=lambda c: (max(k[1] for k in by_concept[c]), -present.index(c)))
    merged: dict[tuple, _Fact] = {}
    for concept in [main, *(c for c in present if c != main)]:
        for key, fact in by_concept[concept].items():
            merged.setdefault(key, fact)
    return merged, main


def _near(a: date, b: date) -> bool:
    return abs((a - b).days) <= _SAME_DAY


def _is_annual(fact: _Fact) -> bool:
    return ANNUAL_DAYS[0] <= fact.days <= ANNUAL_DAYS[1]


def _find(series: dict[tuple, _Fact], end: date, days: int | None = None) -> _Fact | None:
    """The fact ending near `end`; with `days`, of about that length."""
    matches = [
        f for f in series.values()
        if _near(f.end, end) and (days is None or abs(f.days - days) <= 10)
    ]
    return max(matches, key=lambda f: f.filed) if matches else None


def _ttm(series: dict[tuple, _Fact], end: date) -> float | None:
    """Twelve months to `end`: a full year if one ends there, else YTD + last FY - prior YTD."""
    annual = [f for f in series.values() if _near(f.end, end) and _is_annual(f)]
    if annual:
        return max(annual, key=lambda f: f.filed).value
    ytd = [f for f in series.values() if _near(f.end, end) and 60 <= f.days < ANNUAL_DAYS[0]]
    if not ytd:
        return None
    current = max(ytd, key=lambda f: f.days)
    prior_year = [
        f for f in series.values()
        if _is_annual(f) and abs((current.start - f.end).days - 1) <= _SAME_DAY
    ]
    prior_ytd = _find(series, _year_before(current.end), current.days)
    if not prior_year or prior_ytd is None:
        return None
    return current.value + max(prior_year, key=lambda f: f.filed).value - prior_ytd.value


def _year_before(day: date) -> date:
    try:
        return day.replace(year=day.year - 1)
    except ValueError:  # 29 February
        return day.replace(year=day.year - 1, day=28)


# ------------------------------------------------------------------ the result


@dataclass(frozen=True)
class Fundamentals:
    """What a company reported, summarised: profile, history, the latest twelve months."""

    symbol: str
    name: str
    cik: int
    industry: str | None
    exchange: str | None
    fiscal_year_end: str | None
    filer_category: str | None
    fetched: date
    #: Fiscal-year ends shown in `annual`, oldest first.
    years: tuple[date, ...]
    #: metric key -> {fiscal-year end -> value}
    annual: dict[str, dict[date, float]]
    #: The latest period end the company has reported, and twelve months to it.
    latest_end: date | None
    ttm: dict[str, float | None]
    #: The same twelve-month measure a year earlier, for growth.
    ttm_prior: dict[str, float | None]
    #: metric key -> (value, as of) for balances, from the latest report.
    balances: dict[str, tuple[float, date] | None]
    shares: tuple[float, date] | None
    concepts: dict[str, str | None]
    missing: tuple[str, ...] = field(default=())
    #: Recent 10-K and 10-Q filings, newest first, as listed by EDGAR.
    filings: tuple[dict, ...] = field(default=(), compare=False)

    # ----------------------------------------------------------- derived figures

    @property
    def free_cash_flow(self) -> float | None:
        cfo, capex = self.ttm.get("operating_cash_flow"), self.ttm.get("capex")
        return None if cfo is None or capex is None else cfo - capex

    def margin(self, key: str) -> float | None:
        top, value = self.ttm.get("revenue"), self.ttm.get(key)
        return None if not top or value is None else value / top

    def growth(self, key: str) -> float | None:
        now, before = self.ttm.get(key), self.ttm_prior.get(key)
        if now is None or before is None or before <= 0:
            return None
        return now / before - 1

    def market_cap(self, price: float) -> float | None:
        return None if self.shares is None else price * self.shares[0]

    def valuation(self, price: float) -> dict[str, float | None]:
        """Price multiples at `price`. None where a multiple has no meaning."""
        cap = self.market_cap(price)

        def ratio(denominator: float | None) -> float | None:
            if cap is None or denominator is None or denominator <= 0:
                return None
            return cap / denominator

        equity = self.balances.get("equity")
        return {
            "market_cap": cap,
            "pe": ratio(self.ttm.get("net_income")),
            "ps": ratio(self.ttm.get("revenue")),
            "pb": ratio(equity[0] if equity else None),
            "fcf_yield": None if cap is None or self.free_cash_flow is None else self.free_cash_flow / cap,
        }


def analyze(snapshot: FundamentalsSnapshot) -> Fundamentals:
    """Summarise a snapshot. Never raises for missing data: it reports it as missing."""
    series, concepts, missing = {}, {}, []
    for metric in METRICS:
        series[metric.key], concepts[metric.key] = _series(snapshot, metric)
        if concepts[metric.key] is None:
            missing.append(metric.label)

    # Fiscal-year ends: from revenue, else net income, the two every filer reports.
    anchor = series["revenue"] or series["net_income"]
    year_ends = sorted({f.end for f in anchor.values() if _is_annual(f)})
    years = tuple(year_ends[-YEARS:])

    annual: dict[str, dict[date, float]] = {}
    for metric in METRICS:
        values = {}
        for end in years:
            if metric.kind == "duration":
                fact = next(
                    (f for f in series[metric.key].values() if _near(f.end, end) and _is_annual(f)),
                    None,
                )
            else:
                fact = _find(series[metric.key], end)
            if fact is not None:
                values[end] = fact.value
        annual[metric.key] = values

    durations = [f.end for f in anchor.values() if f.days >= 60]
    latest_end = max(durations) if durations else None
    ttm, ttm_prior = {}, {}
    for metric in METRICS:
        if metric.kind != "duration":
            continue
        ttm[metric.key] = _ttm(series[metric.key], latest_end) if latest_end else None
        ttm_prior[metric.key] = (
            _ttm(series[metric.key], _year_before(latest_end)) if latest_end else None
        )

    balances = {}
    for metric in METRICS:
        if metric.kind != "instant":
            continue
        facts = series[metric.key]
        if not facts:
            balances[metric.key] = None
            continue
        newest = max(facts.values(), key=lambda f: (f.end, f.filed))
        balances[metric.key] = (newest.value, newest.end)

    profile = snapshot.profile
    exchanges = [e for e in profile.get("exchanges") or [] if e]
    return Fundamentals(
        symbol=snapshot.symbol,
        name=snapshot.name,
        cik=snapshot.cik,
        industry=profile.get("sicDescription") or None,
        exchange=exchanges[0] if exchanges else None,
        fiscal_year_end=_month_day(profile.get("fiscalYearEnd")),
        filer_category=(profile.get("category") or "").replace("<br>", ", ") or None,
        fetched=snapshot.fetched,
        years=years,
        annual=annual,
        latest_end=latest_end,
        ttm=ttm,
        ttm_prior=ttm_prior,
        balances=balances,
        shares=_shares(snapshot),
        concepts=concepts,
        missing=tuple(missing),
        filings=tuple(snapshot.filings),
    )


def _shares(snapshot: FundamentalsSnapshot) -> tuple[float, date] | None:
    """Shares outstanding from the latest cover page, summed across share classes."""
    rows = _rows(snapshot, SHARES, SHARES.candidates[0])
    if not rows:
        return None
    newest = max(rows, key=lambda f: (f.end, f.filed))
    same = [f for f in rows if f.end == newest.end and f.accession == newest.accession]
    total = math.fsum(f.value for f in same)
    return total, newest.end


def _month_day(mmdd: str | None) -> str | None:
    """'0925' -> 'Sep 25'."""
    if not mmdd or len(mmdd) != 4 or not mmdd.isdigit():
        return None
    return f"{date(2000, int(mmdd[:2]), int(mmdd[2:])):%b} {int(mmdd[2:])}"
