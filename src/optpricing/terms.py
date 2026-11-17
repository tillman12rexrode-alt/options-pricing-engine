"""Discounting and forwards for equity options, without put-call parity.

Index options get their terms from market.implied_terms, which reads the rate
and dividend yield off the chain by regressing the call-put spread on strike.
That regression rests on European parity being an equality. For American equity
options it is an inequality, so the same method would return numbers that look
precise and are biased by the early-exercise premium.

So equities take their two inputs from outside the chain instead: the rate from
the Treasury curve, and the dividends from a dated schedule. The forward then
follows by construction, F = (S - PV(dividends)) * exp(rT), rather than being
estimated.

This is a separate module from market.py on purpose. The index method works,
is tested, and produced R-squared above 0.99999 on SPX; it stays exactly as it
was, and the two methods meet only at the Option they hand to a pricer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date

from optpricing import dividends as divs
from optpricing.contracts import DividendSchedule, ExerciseStyle, Option, OptionKind
from optpricing.providers.fred import YieldCurve

__all__ = ["EquityTerms", "equity_terms"]


@dataclass(frozen=True)
class EquityTerms:
    """Everything an equity option on one expiry needs besides strike and volatility."""

    symbol: str
    expiry: str
    asof: date
    years: float
    spot: float
    rate: float
    discount: float
    forward: float
    dividends: DividendSchedule
    curve_source: str
    curve_asof: date
    dividends_stale: bool

    @property
    def pv_dividends(self) -> float:
        return sum(amount * math.exp(-self.rate * when) for when, amount in self.dividends)

    def option(
        self,
        strike: float,
        kind: OptionKind,
        sigma: float = 0.2,
        style: ExerciseStyle = "american",
    ) -> Option:
        """A contract on this expiry, ready for any pricer, implied vol or Greek."""
        return Option(
            S=self.spot,
            K=strike,
            T=self.years,
            r=self.rate,
            sigma=sigma,
            kind=kind,
            style=style,
            dividends=self.dividends,
        )

    def as_row(self) -> dict:
        return {
            "symbol": self.symbol,
            "expiry": self.expiry,
            "years": self.years,
            "rate": self.rate,
            "discount": self.discount,
            "forward": self.forward,
            "n_dividends": len(self.dividends),
            "pv_dividends": self.pv_dividends,
            "curve_source": self.curve_source,
            "curve_asof": self.curve_asof.isoformat(),
            "dividends_stale": self.dividends_stale,
        }


def equity_terms(
    symbol: str,
    spot: float,
    asof: date,
    expiry: date | str,
    curve: YieldCurve,
    *,
    policies: dict[str, divs.DividendPolicy] | None = None,
) -> EquityTerms:
    """Rate from the curve, dividends from the schedule, forward by construction."""
    if spot <= 0:
        raise ValueError(f"spot must be positive, got {spot}")
    expiry_date = date.fromisoformat(expiry) if isinstance(expiry, str) else expiry
    years = (expiry_date - asof).days / 365.0
    if years <= 0:
        raise ValueError(f"expiry {expiry_date} is not after {asof}")

    rate = curve.rate(years)
    discount = math.exp(-rate * years)
    schedule = divs.schedule(symbol, asof, expiry_date, policies=policies)
    pv = sum(amount * math.exp(-rate * when) for when, amount in schedule)
    if pv >= spot:
        raise ValueError(f"dividends worth {pv:.4f} exceed the spot of {spot}")

    table = divs.POLICIES if policies is None else policies
    return EquityTerms(
        symbol=symbol.upper(),
        expiry=expiry_date.isoformat(),
        asof=asof,
        years=years,
        spot=spot,
        rate=rate,
        discount=discount,
        forward=(spot - pv) / discount,
        dividends=schedule,
        curve_source=curve.source,
        curve_asof=curve.asof,
        dividends_stale=table[symbol.upper()].is_stale(asof),
    )
