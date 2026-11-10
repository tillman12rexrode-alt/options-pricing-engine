"""Where Black-Scholes breaks, measured on a real SPX chain.

Everything up to here showed the engines are correct. This shows the model is
wrong, and says by how much. Black-Scholes assumes one volatility governs every
strike on an underlying. Invert it against the market and that assumption fails
visibly: implied volatility is a smile in strike and a curve in maturity, not a
number.

The test: take each expiry's at-the-money implied volatility, reprice that whole
expiry with it, and measure the error.

Run from the repository root:

    python analysis/volatility_smile.py
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotstyle as ps

from optpricing.contracts import Option
from optpricing.implied_vol import solve_chain
from optpricing.market import clean, implied_terms, load_snapshot
from optpricing.models import black_scholes as bs

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
FIGURES = ROOT / "figures"
RESULTS = ROOT / "analysis" / "results"

EXCHANGE = ZoneInfo("America/New_York")

#: Moneyness bands for the mispricing breakdown, as K/F.
BUCKETS = [
    ("deep OTM put", 0.80, 0.90),
    ("OTM put", 0.90, 0.97),
    ("at the money", 0.97, 1.03),
    ("OTM call", 1.03, 1.10),
    ("deep OTM call", 1.10, 1.20),
]


def trading_date(stamp: str):
    """CBOE stamps in UTC; a 00:31 UTC snapshot is the previous New York session."""
    utc = datetime.fromisoformat(str(stamp)).replace(tzinfo=ZoneInfo("UTC"))
    return utc.astimezone(EXCHANGE).date()


def latest_snapshot() -> Path:
    files = sorted(DATA.glob("*_chain_*.csv"))
    if not files:
        raise SystemExit("No snapshot in data/. Run: python analysis/fetch_chain.py")
    return files[-1]


def atm_vol(group: pd.DataFrame) -> float:
    """Implied volatility at the forward, interpolated from the nearest strikes."""
    ordered = group.sort_values("log_moneyness")
    return float(np.interp(0.0, ordered["log_moneyness"], ordered["iv_brent"]))


def repricing_errors(solved: pd.DataFrame, terms: dict, spot: float) -> pd.DataFrame:
    """Reprice every quote with its expiry's single at-the-money volatility."""
    rows = []
    for expiry, group in solved.groupby("expiry"):
        term = terms[str(expiry)]
        flat = atm_vol(group)

        for record in group.to_dict("records"):
            opt = Option(
                S=spot,
                K=float(record["strike"]),
                T=term.years,
                r=term.rate,
                sigma=flat,
                q=term.dividend_yield,
                kind=str(record["kind"]),
            )
            model = bs.price(opt)
            market = float(record["mid"])
            rows.append(
                {
                    "expiry": str(expiry),
                    "days": round(term.years * 365),
                    "kind": record["kind"],
                    "strike": opt.K,
                    "k_over_f": opt.K / term.forward,
                    "log_moneyness": record["log_moneyness"],
                    "market": market,
                    "model": model,
                    "flat_vol": flat,
                    "market_iv": record["iv_brent"],
                    "price_error": model - market,
                    "pct_error": (model - market) / market,
                    "vol_points": record["iv_brent"] - flat,
                }
            )
    return pd.DataFrame(rows)


def figure_parity_fit(cleaned: pd.DataFrame, terms: dict, spot: float, expiry: str) -> None:
    """The regression the forward comes from - and a data-integrity check."""
    term = terms[expiry]
    paired = (
        cleaned[cleaned["expiry"] == expiry]
        .pivot_table(index="strike", columns="kind", values="mid")
        .dropna()
    )
    strikes = paired.index.to_numpy(dtype=float)
    spread = (paired["call"] - paired["put"]).to_numpy(dtype=float)
    fitted = term.discount * (term.forward - strikes)

    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    ax.plot(strikes, spread, "o", color=ps.BLUE, markersize=4, label="Observed C - P")
    ax.plot(strikes, fitted, color=ps.ORANGE, linewidth=2, label="Parity fit  D(F - K)")
    ax.axvline(term.forward, color=ps.MUTED, linestyle=":", linewidth=1.4)
    ax.annotate(
        f"implied forward {term.forward:,.0f}\nspot {spot:,.0f}",
        xy=(term.forward, spread.max() * 0.65),
        xytext=(10, 0),
        textcoords="offset points",
        color=ps.MUTED,
        fontsize=9,
    )

    ax.set_xlabel("Strike")
    ax.set_ylabel("Call minus put, mid prices")
    ax.set_title(
        f"Put-call parity holds to R² = {term.r_squared:.6f}  ({expiry})", loc="left", pad=12
    )
    ax.grid(True, alpha=0.6)
    ax.legend(frameon=False, loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURES / "parity_fit.png", bbox_inches="tight")
    plt.close(fig)


def figure_smiles(errors: pd.DataFrame) -> None:
    """One panel per maturity: implied volatility against moneyness."""
    expiries = sorted(errors["expiry"].unique(), key=lambda e: errors.loc[errors["expiry"] == e, "days"].iloc[0])
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 6.4), sharex=True, sharey=True)

    for ax, expiry in zip(axes.ravel(), expiries, strict=False):
        group = errors[errors["expiry"] == expiry].sort_values("k_over_f")
        days = int(group["days"].iloc[0])
        flat = float(group["flat_vol"].iloc[0])

        ax.plot(group["k_over_f"], group["market_iv"], color=ps.BLUE, linewidth=1.8)
        ax.axhline(flat, color=ps.MUTED, linestyle="--", linewidth=1.2)
        ax.set_title(f"{days}d   ATM {flat:.1%}", loc="left", fontsize=10, pad=6)
        ax.grid(True, alpha=0.6)

    for ax in axes[-1]:
        ax.set_xlabel("Strike / forward")
    for ax in axes[:, 0]:
        ax.set_ylabel("Implied volatility")

    fig.suptitle(
        "Implied volatility is not one number: it smiles, and the smile flattens with maturity",
        x=0.012,
        ha="left",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(FIGURES / "volatility_smiles.png", bbox_inches="tight")
    plt.close(fig)


def figure_term_structure(errors: pd.DataFrame) -> None:
    """At-the-money implied volatility against maturity."""
    atm = errors.groupby("expiry").agg(days=("days", "first"), vol=("flat_vol", "first"))
    atm = atm.sort_values("days")

    fig, ax = plt.subplots(figsize=(7.6, 4.3))
    ax.plot(atm["days"], atm["vol"], color=ps.BLUE, marker="o", markersize=5)
    for days, vol in atm.itertuples(index=False):
        ax.annotate(
            f"{vol:.1%}",
            xy=(days, vol),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            fontsize=9,
            color=ps.MUTED,
        )

    ax.set_xscale("log")
    ax.set_xticks(atm["days"])
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_xlabel("Days to expiry")
    ax.set_ylabel("At-the-money implied volatility")
    ax.set_title("Term structure of at-the-money volatility", loc="left", pad=12)
    ax.grid(True, which="both", alpha=0.6)
    ax.margins(y=0.22)
    fig.tight_layout()
    fig.savefig(FIGURES / "term_structure.png", bbox_inches="tight")
    plt.close(fig)


def bucket_table(errors: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label, low, high in BUCKETS:
        band = errors[errors["k_over_f"].between(low, high, inclusive="left")]
        if band.empty:
            continue
        rows.append(
            {
                "bucket": label,
                "k_over_f": f"[{low:.2f}, {high:.2f})",
                "quotes": len(band),
                "median_abs_pct": float(band["pct_error"].abs().median()),
                "median_vol_points": float(band["vol_points"].median()),
                "max_abs_pct": float(band["pct_error"].abs().max()),
            }
        )
    return pd.DataFrame(rows)


def figure_mispricing(table: pd.DataFrame) -> None:
    """How badly a single volatility prices each part of the chain."""
    fig, ax = plt.subplots(figsize=(7.6, 4.3))
    positions = np.arange(len(table))
    ax.bar(
        positions,
        100 * table["median_abs_pct"],
        color=ps.BLUE,
        width=0.6,
    )
    for x, value in zip(positions, 100 * table["median_abs_pct"], strict=False):
        ax.annotate(
            f"{value:.0f}%",
            xy=(x, value),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            fontsize=9,
            color=ps.MUTED,
        )

    ax.set_xticks(positions)
    ax.set_xticklabels([f"{r.bucket}\n{r.k_over_f}" for r in table.itertuples()], fontsize=9)
    ax.set_ylabel("Median absolute pricing error (%)")
    ax.set_title(
        "Pricing the chain with one volatility: error by moneyness", loc="left", pad=12
    )
    ax.grid(True, axis="y", alpha=0.6)
    ax.margins(y=0.18)
    fig.tight_layout()
    fig.savefig(FIGURES / "flat_vol_mispricing.png", bbox_inches="tight")
    plt.close(fig)


def cross_checks(solved: pd.DataFrame, terms: dict) -> pd.DataFrame:
    """Two checks on the implied volatilities that use nothing from this project.

    Call- and put-implied volatility at the same strike must agree, because
    parity is model-free - they only do if the forward backed out of the parity
    regression is right. And CBOE publishes its own implied volatilities in the
    same feed, computed independently, so the difference is a check against an
    outside party rather than against ourselves.
    """
    rows = []

    paired = solved.pivot_table(
        index=["expiry", "strike"], columns="kind", values="iv_brent"
    ).dropna()
    if not paired.empty:
        gap = (paired["call"] - paired["put"]).abs() * 10_000  # basis points
        rows.append({
            "check": "call-implied vs put-implied vol, same strike",
            "quotes": len(paired),
            "median_bp": float(gap.median()),
            "mean_bp": float(gap.mean()),
            "p95_bp": float(gap.quantile(0.95)),
        })

    published = solved[solved["cboe_iv"].fillna(0) > 0]
    if not published.empty:
        gap = (published["iv_brent"] - published["cboe_iv"]).abs() * 10_000
        rows.append({
            "check": "our implied vol vs CBOE's published implied vol",
            "quotes": len(published),
            "median_bp": float(gap.median()),
            "mean_bp": float(gap.mean()),
            "p95_bp": float(gap.quantile(0.95)),
        })

    return pd.DataFrame(rows)


def main() -> None:
    ps.apply()
    FIGURES.mkdir(exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)

    path = latest_snapshot()
    frame, spot, _ = load_snapshot(path)
    asof = trading_date(frame["asof"].iloc[0])
    print(f"snapshot      {path.name}")
    print(f"trading date  {asof}   index {spot:,.2f}\n")

    cleaned, funnel = clean(frame, spot)
    terms = implied_terms(cleaned, spot, asof)
    solved, solver_funnel = solve_chain(cleaned, terms, spot)

    print("cleaning funnel:")
    for label, count in funnel + solver_funnel:
        print(f"  {label:<36}{count:>7}")

    print("\nimplied by put-call parity:")
    print(f"  {'expiry':<12}{'days':>5}{'rate':>8}{'div yld':>9}{'forward':>11}{'R²':>11}")
    term_rows = []
    for term in sorted(terms.values(), key=lambda t: t.years):
        print(
            f"  {term.expiry:<12}{round(term.years * 365):>5}{100 * term.rate:>7.2f}%"
            f"{100 * term.dividend_yield:>8.2f}%{term.forward:>11,.2f}{term.r_squared:>11.6f}"
        )
        term_rows.append(term.as_row())
    pd.DataFrame(term_rows).to_csv(RESULTS / "implied_terms.csv", index=False)

    checks = cross_checks(solved, terms)
    if not checks.empty:
        print("\ncross-checks against things this project did not produce:")
        print(f"  {'check':<48}{'quotes':>8}{'median':>10}{'mean':>9}{'p95':>8}")
        for row in checks.itertuples(index=False):
            print(
                f"  {row.check:<48}{row.quotes:>8,}{row.median_bp:>9.1f}bp"
                f"{row.mean_bp:>8.1f}bp{row.p95_bp:>7.0f}bp"
            )
        checks.to_csv(RESULTS / "cross_checks.csv", index=False)

    errors = repricing_errors(solved, terms, spot)
    errors.to_csv(RESULTS / "flat_vol_errors.csv", index=False)

    table = bucket_table(errors)
    print("\npricing the chain with one at-the-money volatility:")
    print(f"  {'bucket':<16}{'K/F':<16}{'quotes':>7}{'median err':>12}{'vol points':>12}{'worst':>9}")
    for row in table.itertuples(index=False):
        print(
            f"  {row.bucket:<16}{row.k_over_f:<16}{row.quotes:>7}"
            f"{100 * row.median_abs_pct:>11.1f}%{100 * row.median_vol_points:>11.2f}"
            f"{100 * row.max_abs_pct:>8.0f}%"
        )
    table.to_csv(RESULTS / "flat_vol_mispricing.csv", index=False)

    nearest = min(terms.values(), key=lambda t: t.years).expiry
    figure_parity_fit(cleaned, terms, spot, nearest)
    figure_smiles(errors)
    figure_term_structure(errors)
    figure_mispricing(table)

    put_wing = table.loc[table["bucket"] == "deep OTM put", "median_vol_points"]
    call_wing = table.loc[table["bucket"] == "deep OTM call", "median_vol_points"]
    atm = table.loc[table["bucket"] == "at the money", "median_vol_points"]
    if not (put_wing.empty or call_wing.empty or atm.empty):
        print(
            f"\nHeadline: one volatility fits the money to {100 * float(atm.iloc[0]):+.2f} "
            f"vol points, but the market charges {100 * float(put_wing.iloc[0]):+.1f} points "
            f"more in the deep put wing and {100 * float(call_wing.iloc[0]):+.1f} in the "
            f"call wing. That spread is the smile."
        )
    print(f"\nFigures written to {FIGURES}")


if __name__ == "__main__":
    main()
