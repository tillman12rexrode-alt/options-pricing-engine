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

__all__ = [
    "SurfaceAnalysis",
    "TradeMechanics",
    "analyze",
    "find_outliers",
    "rank_candidates",
    "trade_mechanics",
]


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


@dataclass(frozen=True)
class TradeMechanics:
    """What the trade on a flagged quote would be, and what it would cost.

    This is arithmetic, not advice. It answers "if this gap is real and it
    closes, does what is left after frictions exceed zero" - which on retail
    fee structures is usually no, and that is the useful part.
    """

    legs: pd.DataFrame
    edge_points: float
    gross: float
    entry_cost: float
    exit_cost: float
    commissions: float
    net: float
    breakeven_points: float
    theta_per_day: float
    multiplier: int

    @property
    def survives_costs(self) -> bool:
        return self.net > 0.0

    @property
    def fraction_of_gap_needed(self) -> float:
        """How much of the gap must close just to break even."""
        if self.edge_points <= 0:
            return float("inf")
        return self.breakeven_points / self.edge_points

    def verdict(self) -> str:
        if not np.isfinite(self.fraction_of_gap_needed):
            return "No measurable gap to trade."
        if self.fraction_of_gap_needed >= 1.0:
            return (
                f"Frictions exceed the whole gap. The position would need "
                f"{self.fraction_of_gap_needed:.1f}x more convergence than exists "
                f"just to break even, so there is nothing here to capture."
            )
        return (
            f"{self.fraction_of_gap_needed:.0%} of the gap must close just to cover "
            f"frictions. Full convergence leaves ${self.net:,.2f} per package, before "
            f"slippage and before the possibility that the quote was simply stale."
        )


def trade_mechanics(
    analysis: SurfaceAnalysis,
    flagged: pd.Series,
    *,
    fee_per_contract: float = 0.65,
    multiplier: int = 100,
) -> TradeMechanics | None:
    """Build the vega- and delta-neutral package around a flagged contract.

    The flagged leg is traded against its two nearest strikes in whatever ratio
    makes the package neutral to both volatility level and index direction, so
    the only thing left is whether the gap itself closes. Trading the flagged
    contract outright instead would be a bet on the index, which is not what
    the scanner found.
    """
    expiry = str(flagged["expiry"])
    term = analysis.terms[expiry]
    same = analysis.quotes[
        (analysis.quotes["expiry"] == expiry) & (analysis.quotes["kind"] == flagged["kind"])
    ].sort_values("strike").reset_index(drop=True)

    position = same.index[same["strike"] == flagged["strike"]]
    if len(position) != 1:
        return None
    centre = int(position[0])
    if centre == 0 or centre == len(same) - 1:
        return None  # no neighbours on one side

    def describe(record: pd.Series) -> tuple[Option, float, float, float]:
        opt = Option(
            S=analysis.spot, K=float(record["strike"]), T=term.years, r=term.rate,
            sigma=max(float(record["iv_brent"]), 1e-4), q=term.dividend_yield,
            kind=str(record["kind"]),
        )
        greeks = bs.greeks(opt)
        spread = (float(record["ask"]) - float(record["bid"])) * multiplier
        return opt, greeks["vega"] / 100.0 * multiplier, greeks["delta"], spread

    rich, vega0, delta0, spread0 = describe(same.loc[centre])
    low, vega1, delta1, spread1 = describe(same.loc[centre - 1])
    high, vega2, delta2, spread2 = describe(same.loc[centre + 1])

    # Solve for the neighbour quantities that zero out both vega and delta.
    matrix = np.array([[vega1, vega2], [delta1, delta2]])
    if abs(np.linalg.det(matrix)) < 1e-12:
        return None
    quantity_low, quantity_high = np.linalg.solve(matrix, np.array([vega0, delta0]))

    rich_is_rich = float(flagged["residual_points"]) > 0
    sign = -1.0 if rich_is_rich else 1.0  # sell what is rich, buy what is cheap

    legs = pd.DataFrame([
        {"strike": rich.K, "action": "sell" if rich_is_rich else "buy",
         "quantity": abs(sign) * 1.0, "bid": float(same.loc[centre, "bid"]),
         "ask": float(same.loc[centre, "ask"]), "vega_per_point": vega0, "delta": delta0},
        {"strike": low.K, "action": "buy" if rich_is_rich else "sell",
         "quantity": abs(quantity_low), "bid": float(same.loc[centre - 1, "bid"]),
         "ask": float(same.loc[centre - 1, "ask"]), "vega_per_point": vega1, "delta": delta1},
        {"strike": high.K, "action": "buy" if rich_is_rich else "sell",
         "quantity": abs(quantity_high), "bid": float(same.loc[centre + 1, "bid"]),
         "ask": float(same.loc[centre + 1, "ask"]), "vega_per_point": vega2, "delta": delta2},
    ])

    contracts = 1.0 + abs(quantity_low) + abs(quantity_high)
    entry = 0.5 * (spread0 + abs(quantity_low) * spread1 + abs(quantity_high) * spread2)
    commissions = 2.0 * contracts * fee_per_contract

    edge_points = abs(float(flagged["residual_points"]))
    gross = edge_points * vega0
    net = gross - 2.0 * entry - commissions
    breakeven = (2.0 * entry + commissions) / vega0 if vega0 > 0 else float("inf")

    theta = (
        sign * bs.greeks(rich)["theta"]
        - sign * quantity_low * bs.greeks(low)["theta"]
        - sign * quantity_high * bs.greeks(high)["theta"]
    ) / 365.0 * multiplier

    return TradeMechanics(
        legs=legs, edge_points=edge_points, gross=gross, entry_cost=entry,
        exit_cost=entry, commissions=commissions, net=net,
        breakeven_points=breakeven, theta_per_day=float(theta), multiplier=multiplier,
    )


def rank_candidates(
    analysis: SurfaceAnalysis,
    hits: pd.DataFrame,
    *,
    fee_per_contract: float = 0.65,
    multiplier: int = 100,
) -> pd.DataFrame:
    """Order flagged quotes by what the model says is left after frictions.

    This is the closest thing to a "best bet" the model can honestly produce.
    It ranks by net profit assuming the gap closes completely, which is the
    optimistic end of the range, and reports how much of the gap must close
    merely to break even. A candidate can be top of this list and still be a
    bad trade - most often because the quote was stale and there was never a
    gap to close - so the columns that matter are `net` and `breakeven_frac`,
    not the ordering.
    """
    rows = []
    for _, flagged in hits.iterrows():
        mechanics = trade_mechanics(
            analysis, flagged, fee_per_contract=fee_per_contract, multiplier=multiplier
        )
        if mechanics is None:
            rows.append({
                "expiry": flagged["expiry"], "strike": flagged["strike"],
                "kind": flagged["kind"], "verdict": flagged["verdict"],
                "gap_points": abs(float(flagged["residual_points"])),
                "gross": np.nan, "costs": np.nan, "net": np.nan,
                "breakeven_frac": np.nan, "theta_per_day": np.nan,
                "note": "No neighbouring strike on one side, so the position cannot be hedged.",
            })
            continue

        rows.append({
            "expiry": flagged["expiry"], "strike": flagged["strike"],
            "kind": flagged["kind"], "verdict": flagged["verdict"],
            "gap_points": mechanics.edge_points,
            "gross": mechanics.gross,
            "costs": mechanics.entry_cost + mechanics.exit_cost + mechanics.commissions,
            "net": mechanics.net,
            "breakeven_frac": mechanics.fraction_of_gap_needed,
            "theta_per_day": mechanics.theta_per_day,
            "note": mechanics.verdict(),
        })

    table = pd.DataFrame(rows)
    if table.empty:
        return table
    return table.sort_values("net", ascending=False, na_position="last").reset_index(drop=True)
