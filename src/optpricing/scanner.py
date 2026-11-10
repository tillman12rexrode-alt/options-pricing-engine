"""Find quotes that disagree with their own smile, and say why.

The claim a scanner can honestly make is narrow: this contract is priced
differently from the contracts either side of it on the same underlying and the
same expiry. That is a statement about internal consistency, not a forecast,
and it is worth surfacing because the explanation is usually interesting - a
stale quote, a dividend nobody modelled, or occasionally a real dislocation.

Two guards keep it from inventing signal. A slice whose smile did not fit is
not scanned at all, because every quote on a badly fitted curve looks like an
outlier. And a residual only counts if it is larger than the quote's own
bid-ask spread expressed in volatility points - otherwise the "anomaly" is the
spread, which is present on every quote in the market.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from optpricing.contracts import Option
from optpricing.implied_vol import solve_chain
from optpricing.market import ExpiryTerms, clean, implied_terms
from optpricing.models import black_scholes as bs
from optpricing.smile import SliceFit, fit_slice

__all__ = ["SurfaceAnalysis", "analyze", "find_outliers"]


@dataclass(frozen=True)
class SurfaceAnalysis:
    """Everything derived from one chain snapshot."""

    spot: float
    asof: date
    quotes: pd.DataFrame
    terms: dict[str, ExpiryTerms]
    fits: dict[str, SliceFit]
    funnel: list[tuple[str, int]]

    @property
    def reliable_expiries(self) -> list[str]:
        return [e for e, fit in self.fits.items() if fit.reliable]

    def fit_report(self) -> pd.DataFrame:
        """One row per expiry: did its smile fit, and if not why not."""
        rows = [
            {
                "expiry": expiry,
                "days": round(fit.years * 365),
                "quotes": fit.n_quotes,
                "rmse_vol_points": fit.rmse_vol_points,
                "worst_bias": fit.worst_bucket_bias,
                "scannable": fit.reliable,
                "note": fit.reason,
            }
            for expiry, fit in sorted(self.fits.items(), key=lambda kv: kv[1].years)
        ]
        return pd.DataFrame(rows)


def _spread_in_vol_points(row: pd.Series, term: ExpiryTerms, spot: float) -> float:
    """The quoted spread, converted from dollars into volatility points.

    Vega is the exchange rate: a contract with low vega needs a large move in
    volatility to shift its price by the width of its own spread, so its
    implied volatility is correspondingly ill-determined.
    """
    opt = Option(
        S=spot,
        K=float(row["strike"]),
        T=term.years,
        r=term.rate,
        sigma=max(float(row["iv_brent"]), 1e-4),
        q=term.dividend_yield,
        kind=str(row["kind"]),
    )
    vega = bs.greeks(opt)["vega"]
    if vega <= 1e-8:
        return float("inf")
    return float(row["ask"] - row["bid"]) / vega * 100.0


def analyze(
    frame: pd.DataFrame,
    spot: float,
    asof: date,
    *,
    max_rel_spread: float = 0.20,
    moneyness: tuple[float, float] = (0.80, 1.20),
) -> SurfaceAnalysis:
    """Clean a chain, imply its terms, solve its volatilities and fit its smiles."""
    cleaned, funnel = clean(frame, spot, max_rel_spread=max_rel_spread, moneyness=moneyness)
    terms = implied_terms(cleaned, spot, asof)
    solved, solver_funnel = solve_chain(cleaned, terms, spot, methods=("brent",))

    fits: dict[str, SliceFit] = {}
    pieces = []
    for expiry, group in solved.groupby("expiry"):
        term = terms[str(expiry)]
        group = group.sort_values("log_moneyness").copy()

        weights = 1.0 / np.maximum(group["rel_spread"].to_numpy(), 1e-3)
        fit = fit_slice(
            group["log_moneyness"].to_numpy(),
            group["iv_brent"].to_numpy(),
            term.years,
            weights,
        )
        fits[str(expiry)] = fit

        if fit.reliable:
            group["fitted_iv"] = fit.implied_vol(group["log_moneyness"].to_numpy())
        else:
            group["fitted_iv"] = np.nan

        group["residual_points"] = (group["iv_brent"] - group["fitted_iv"]) * 100.0
        group["spread_vol_points"] = [
            _spread_in_vol_points(row, term, spot) for _, row in group.iterrows()
        ]
        pieces.append(group)

    quotes = pd.concat(pieces, ignore_index=True) if pieces else solved
    return SurfaceAnalysis(spot, asof, quotes, terms, fits, funnel + solver_funnel)


def _describe(row: pd.Series) -> str:
    direction = "rich" if row["residual_points"] > 0 else "cheap"
    multiple = row["residual_points"] / row["spread_vol_points"]
    return (
        f"{abs(row['residual_points']):.2f} vol points {direction} versus its own smile "
        f"({row['iv_brent']:.1%} against a fitted {row['fitted_iv']:.1%}). "
        f"That gap is {abs(multiple):.1f}x the quote's spread in volatility terms "
        f"({row['spread_vol_points']:.2f} points). "
        f"Open interest {row['open_interest']:,.0f}, volume {row['volume']:,.0f}, "
        f"spread {row['rel_spread']:.1%} of mid."
    )


def find_outliers(
    analysis: SurfaceAnalysis,
    *,
    min_vol_points: float = 1.0,
    min_spread_multiple: float = 1.5,
    min_open_interest: float = 50.0,
) -> pd.DataFrame:
    """Quotes far enough off their fitted smile to be worth a second look.

    `min_spread_multiple` is the guard that matters. A residual smaller than
    the spread is not evidence of anything; requiring it to be a multiple of
    the spread is what stops the output being a list of illiquid strikes.
    """
    usable = analysis.quotes[
        analysis.quotes["expiry"].isin(analysis.reliable_expiries)
    ].copy()
    if usable.empty:
        return pd.DataFrame()

    usable = usable[np.isfinite(usable["residual_points"])]
    usable = usable[np.isfinite(usable["spread_vol_points"])]
    usable["spread_multiple"] = (
        usable["residual_points"].abs() / usable["spread_vol_points"]
    )

    hits = usable[
        (usable["residual_points"].abs() >= min_vol_points)
        & (usable["spread_multiple"] >= min_spread_multiple)
        & (usable["open_interest"].fillna(0) >= min_open_interest)
    ].copy()
    if hits.empty:
        return pd.DataFrame()

    hits["verdict"] = np.where(hits["residual_points"] > 0, "rich", "cheap")
    hits["reason"] = [_describe(row) for _, row in hits.iterrows()]

    columns = [
        "expiry", "kind", "strike", "bid", "ask", "mid", "iv_brent", "fitted_iv",
        "residual_points", "spread_vol_points", "spread_multiple", "open_interest",
        "volume", "verdict", "reason",
    ]
    return (
        hits[columns]
        .sort_values("residual_points", key=np.abs, ascending=False)
        .reset_index(drop=True)
    )
