"""Turn a raw option-chain snapshot into something a pricer can be pointed at.

Two jobs live here. The first is cleaning: quoted chains contain stale prints,
one-sided markets and strikes nobody trades, and using them unfiltered produces
a volatility surface made largely of noise.

The second is extracting the rate and dividend yield, which are not given in the
chain. Rather than fetching a Treasury curve and a dividend estimate from
elsewhere - and risking a mismatch that shows up as a fake skew - both are
implied from the quotes themselves. Put-call parity says

    C - P = D * (F - K)

for European options, so regressing the call-put spread on the strike gives the
discount factor as minus the slope and the forward as the intercept over it.
The fit quality is also a data-integrity check: a chain that does not produce a
near-perfect straight line here has something wrong with it.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

__all__ = ["ExpiryTerms", "clean", "implied_terms", "load_snapshot"]


@dataclass(frozen=True)
class ExpiryTerms:
    """Discounting and forward pricing implied by one expiry's own quotes."""

    expiry: str
    years: float
    discount: float
    forward: float
    rate: float
    dividend_yield: float
    r_squared: float
    n_pairs: int

    def as_row(self) -> dict:
        return {
            "expiry": self.expiry,
            "years": self.years,
            "discount": self.discount,
            "forward": self.forward,
            "rate": self.rate,
            "dividend_yield": self.dividend_yield,
            "r_squared": self.r_squared,
            "n_pairs": self.n_pairs,
        }


def load_snapshot(path: str | Path) -> tuple[pd.DataFrame, float, date]:
    """Read a snapshot written by analysis/fetch_chain.py."""
    frame = pd.read_csv(path)
    spot = float(frame["spot"].iloc[0])
    asof = date.fromisoformat(str(frame["asof"].iloc[0])[:10])
    return frame, spot, asof


def clean(
    frame: pd.DataFrame,
    spot: float,
    *,
    max_rel_spread: float = 0.20,
    moneyness: tuple[float, float] = (0.80, 1.20),
) -> tuple[pd.DataFrame, list[tuple[str, int]]]:
    """Filter to quotes worth believing, and report the funnel.

    The mid of a two-sided market is the usable price; the last trade is not,
    because on an untraded strike it can be hours or days stale.
    """
    funnel = [("raw quotes", len(frame))]
    out = frame.copy()

    out = out[out["bid"].fillna(0) > 0]
    funnel.append(("two-sided (bid > 0)", len(out)))

    out = out[out["ask"] > out["bid"]]
    funnel.append(("ask above bid", len(out)))

    out["mid"] = (out["bid"] + out["ask"]) / 2.0
    out["rel_spread"] = (out["ask"] - out["bid"]) / out["mid"]
    out = out[out["rel_spread"] <= max_rel_spread]
    funnel.append((f"spread <= {max_rel_spread:.0%} of mid", len(out)))

    out["moneyness"] = out["strike"] / spot
    out = out[out["moneyness"].between(*moneyness)]
    funnel.append((f"moneyness in [{moneyness[0]:.2f}, {moneyness[1]:.2f}]", len(out)))

    return out.reset_index(drop=True), funnel


def _fit_parity(strikes: np.ndarray, spread: np.ndarray) -> tuple[float, float, float]:
    """Regress C - P on K. Returns (discount, forward, r_squared)."""
    slope, intercept = np.polyfit(strikes, spread, 1)
    predicted = slope * strikes + intercept
    total = ((spread - spread.mean()) ** 2).sum()
    r_squared = 1.0 - ((spread - predicted) ** 2).sum() / total if total > 0 else float("nan")

    discount = -float(slope)
    forward = float(intercept) / discount
    return discount, forward, float(r_squared)


def implied_terms(
    frame: pd.DataFrame,
    spot: float,
    asof: date,
    *,
    fit_moneyness: tuple[float, float] = (0.95, 1.05),
    min_pairs: int = 8,
) -> dict[str, ExpiryTerms]:
    """Imply the discount factor and forward for each expiry from its own quotes.

    Only strikes near the money are used for the fit: both legs are liquid
    there, so the call-put spread is measured cleanly. Far from the money one
    leg is nearly worthless and its quote is mostly spread.
    """
    terms: dict[str, ExpiryTerms] = {}

    for expiry, group in frame.groupby("expiry"):
        years = (date.fromisoformat(str(expiry)) - asof).days / 365.0
        if years <= 0:
            continue

        paired = group.pivot_table(index="strike", columns="kind", values="mid").dropna()
        if not {"call", "put"} <= set(paired.columns):
            continue

        near = paired[np.logical_and(
            paired.index / spot >= fit_moneyness[0],
            paired.index / spot <= fit_moneyness[1],
        )]
        if len(near) < min_pairs:
            continue

        strikes = near.index.to_numpy(dtype=float)
        discount, forward, r_squared = _fit_parity(
            strikes, (near["call"] - near["put"]).to_numpy(dtype=float)
        )
        if not (0.0 < discount <= 1.05) or forward <= 0:
            continue

        rate = -np.log(discount) / years
        dividend_yield = rate - np.log(forward / spot) / years

        terms[str(expiry)] = ExpiryTerms(
            expiry=str(expiry),
            years=years,
            discount=discount,
            forward=forward,
            rate=float(rate),
            dividend_yield=float(dividend_yield),
            r_squared=r_squared,
            n_pairs=len(near),
        )

    return terms
