"""Dividend schedules for single-name equities, hand-maintained and sourced.

Pricing an American call on a dividend payer needs the dates, not just the
amount, because early exercise happens in the instant before an ex-date. No
free, licensable feed supplies forward ex-dates for this project, so the table
below is maintained by hand: one row per ticker, each with the source it came
from and the day it was checked.

A company announces only its next dividend. Every ex-date beyond that is a
projection at the company's usual cadence from the last one it declared, and
real dates drift by a week or so either way. That error matters only for a
contract expiring close to an ex-date, and it is the reason each projected
payment is labelled as such.

An unknown ticker raises rather than defaulting to no dividend. Pricing a payer
as though it paid nothing produces a plausible wrong price with no error, which
is the failure this module is built to prevent.
"""

from __future__ import annotations

import calendar
from dataclasses import dataclass
from datetime import date

from optpricing.contracts import DividendSchedule

__all__ = [
    "POLICIES",
    "STALE_AFTER_DAYS",
    "DividendPolicy",
    "Payment",
    "UnknownTickerError",
    "payments",
    "schedule",
]

#: A table entry older than this should be re-checked before it is trusted.
STALE_AFTER_DAYS = 120


class UnknownTickerError(KeyError):
    """No dividend policy is recorded for this ticker."""


@dataclass(frozen=True)
class DividendPolicy:
    """What a company pays, how often, and how we know."""

    symbol: str
    last_ex_date: date | None
    amount: float
    per_year: int
    verified: date
    source: str

    @property
    def pays(self) -> bool:
        return self.per_year > 0 and self.amount > 0 and self.last_ex_date is not None

    def is_stale(self, asof: date) -> bool:
        return (asof - self.verified).days > STALE_AFTER_DAYS


@dataclass(frozen=True)
class Payment:
    ex_date: date
    amount: float
    #: False only for a dividend the company has already declared.
    projected: bool


#: Verified 2026-11-16. Update a row whenever a company declares a new dividend.
POLICIES: dict[str, DividendPolicy] = {
    policy.symbol: policy
    for policy in (
        DividendPolicy(
            "AAPL", date(2026, 11, 12), 0.27, 4, date(2026, 11, 16),
            "https://stockanalysis.com/stocks/aapl/dividend/",
        ),
        # Raised from $0.01 to $0.25 a quarter with the June 2026 ex-date.
        DividendPolicy(
            "NVDA", date(2026, 9, 10), 0.25, 4, date(2026, 11, 16),
            "https://stockanalysis.com/stocks/nvda/dividend/ and "
            "https://www.koyfin.com/company/nvda/dividends/",
        ),
        DividendPolicy("TSLA", None, 0.0, 0, date(2026, 11, 16), "no cash dividend paid"),
        DividendPolicy(
            "KTOS", None, 0.0, 0, date(2026, 11, 16),
            "https://walnutinvest.com/stocks/ktos/dividend",
        ),
        DividendPolicy(
            "RCAT", None, 0.0, 0, date(2026, 11, 16),
            "https://walnutinvest.com/stocks/rcat/dividend",
        ),
    )
}


def _add_months(day: date, months: int) -> date:
    """Calendar month arithmetic, clamping to the end of a shorter month."""
    total = day.year * 12 + (day.month - 1) + months
    year, month = divmod(total, 12)
    month += 1
    return date(year, month, min(day.day, calendar.monthrange(year, month)[1]))


def _policy(symbol: str, policies: dict[str, DividendPolicy] | None) -> DividendPolicy:
    table = POLICIES if policies is None else policies
    try:
        return table[symbol.upper()]
    except KeyError:
        raise UnknownTickerError(
            f"No dividend policy recorded for {symbol.upper()}. Add a sourced row to "
            "optpricing.dividends.POLICIES rather than assuming it pays nothing."
        ) from None


def payments(
    symbol: str,
    asof: date,
    until: date,
    *,
    policies: dict[str, DividendPolicy] | None = None,
) -> list[Payment]:
    """Ex-dividend dates strictly after `asof`, up to and including `until`.

    An ex-date on `asof` itself is excluded: the stock already trades without
    that dividend, so the spot price has absorbed it.
    """
    policy = _policy(symbol, policies)
    if not policy.pays:
        return []

    step = 12 // policy.per_year
    out: list[Payment] = []
    n = 0
    while True:
        ex_date = _add_months(policy.last_ex_date, n * step)
        if ex_date > until:
            break
        if ex_date > asof:
            out.append(Payment(ex_date, policy.amount, projected=n > 0))
        n += 1
    return out


def schedule(
    symbol: str,
    asof: date,
    expiry: date,
    *,
    policies: dict[str, DividendPolicy] | None = None,
) -> DividendSchedule:
    """The (years from asof, amount) pairs an Option takes, through `expiry`.

    Years use calendar days over 365, the convention market.py already uses, so
    dividend times and option maturities are measured on the same clock.
    """
    return tuple(
        ((p.ex_date - asof).days / 365.0, p.amount)
        for p in payments(symbol, asof, expiry, policies=policies)
    )
