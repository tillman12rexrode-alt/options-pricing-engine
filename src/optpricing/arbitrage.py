"""No-arbitrage checks on a quoted chain.

These are the only claims in the whole project that need no model at all. A
volatility is an opinion; a butterfly priced below zero is a contradiction. If
the quotes say you can be paid to hold a position that cannot lose, either the
quotes are stale or someone is about to be very happy.

Every test here is run on *executable* prices - you buy at the ask and sell at
the bid, never at the mid. A violation measured on mid prices is usually just
the spread, and reporting those would be the single easiest way to make this
tool worthless.

Vertical, butterfly and calendar relations hold for American exercise as well
as European. Put-call parity does not: early exercise turns the equality into
a pair of inequalities, so that check is restricted to European chains.
"""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pandas as pd

from optpricing.market import ExpiryTerms

__all__ = [
    "butterfly_violations",
    "calendar_violations",
    "parity_violations",
    "summarize",
    "vertical_violations",
]

#: Edges smaller than this are rounding, not opportunity.
MIN_EDGE = 0.01


def _legs(group: pd.DataFrame, kind: str) -> pd.DataFrame:
    legs = group[group["kind"] == kind].set_index("strike").sort_index()
    return legs[~legs.index.duplicated(keep="first")]


def parity_violations(
    frame: pd.DataFrame, terms: dict[str, ExpiryTerms], *, fee_per_contract: float = 0.0
) -> pd.DataFrame:
    """Where C - P strays outside the spread around D(F - K). European only.

    Two trades. A conversion buys the call, sells the put and sells the forward;
    a reversal does the opposite. Both lock in the difference at inception, so
    any gap beyond the quoted spread is free money if it is real.
    """
    rows = []
    for expiry, group in frame.groupby("expiry"):
        term = terms.get(str(expiry))
        if term is None:
            continue

        calls, puts = _legs(group, "call"), _legs(group, "put")
        for strike in calls.index.intersection(puts.index):
            call, put = calls.loc[strike], puts.loc[strike]
            synthetic = term.discount * (term.forward - float(strike))

            conversion = synthetic - (float(call["ask"]) - float(put["bid"]))
            reversal = (float(call["bid"]) - float(put["ask"])) - synthetic
            edge = max(conversion, reversal) - 2.0 * fee_per_contract

            if edge > MIN_EDGE:
                rows.append(
                    {
                        "check": "put-call parity",
                        "expiry": str(expiry),
                        "detail": f"K={strike:g}",
                        "trade": "conversion" if conversion >= reversal else "reversal",
                        "edge": edge,
                    }
                )
    return pd.DataFrame(rows)


def vertical_violations(
    frame: pd.DataFrame, *, fee_per_contract: float = 0.0
) -> pd.DataFrame:
    """A lower-strike call may not cost less than a higher-strike one.

    Buying the cheaper strike and selling the dearer one for a credit gives a
    position whose payoff is never negative, which cannot be right.
    """
    rows = []
    for expiry, group in frame.groupby("expiry"):
        for kind in ("call", "put"):
            legs = _legs(group, kind)
            strikes = legs.index.to_numpy(dtype=float)

            for i in range(len(strikes) - 1):
                low, high = strikes[i], strikes[i + 1]
                if kind == "call":
                    cost = float(legs.loc[low, "ask"]) - float(legs.loc[high, "bid"])
                else:
                    cost = float(legs.loc[high, "ask"]) - float(legs.loc[low, "bid"])

                edge = -cost - 2.0 * fee_per_contract
                if edge > MIN_EDGE:
                    rows.append(
                        {
                            "check": "vertical spread",
                            "expiry": str(expiry),
                            "detail": f"{kind} {low:g}/{high:g}",
                            "trade": "credit spread with non-negative payoff",
                            "edge": edge,
                        }
                    )
    return pd.DataFrame(rows)


def butterfly_violations(
    frame: pd.DataFrame, *, fee_per_contract: float = 0.0, tolerance: float = 1e-9
) -> pd.DataFrame:
    """Option prices must be convex in strike, so a butterfly cannot cost less than nothing."""
    rows = []
    for expiry, group in frame.groupby("expiry"):
        for kind in ("call", "put"):
            legs = _legs(group, kind)
            strikes = legs.index.to_numpy(dtype=float)

            for i in range(len(strikes) - 2):
                low, mid, high = strikes[i], strikes[i + 1], strikes[i + 2]
                if abs((mid - low) - (high - mid)) > tolerance * max(1.0, mid):
                    continue  # unevenly spaced; the simple convexity test does not apply

                cost = (
                    float(legs.loc[low, "ask"])
                    + float(legs.loc[high, "ask"])
                    - 2.0 * float(legs.loc[mid, "bid"])
                )
                edge = -cost - 4.0 * fee_per_contract
                if edge > MIN_EDGE:
                    rows.append(
                        {
                            "check": "butterfly",
                            "expiry": str(expiry),
                            "detail": f"{kind} {low:g}/{mid:g}/{high:g}",
                            "trade": "butterfly for a credit",
                            "edge": edge,
                        }
                    )
    return pd.DataFrame(rows)


def calendar_violations(
    fits: dict[str, object], *, grid: np.ndarray | None = None, tolerance: float = 1e-6
) -> pd.DataFrame:
    """Total implied variance must not fall as maturity rises, at equal moneyness.

    Unlike the other three this is checked on the fitted smiles rather than on
    quotes, because it compares like-for-like moneyness across expiries that
    rarely share strikes. It therefore inherits the fits' reliability, and
    unreliable slices are skipped.
    """
    usable = {
        expiry: fit
        for expiry, fit in fits.items()
        if getattr(fit, "reliable", False)
    }
    if len(usable) < 2:
        return pd.DataFrame()

    if grid is None:
        grid = np.linspace(-0.15, 0.10, 26)

    ordered = sorted(usable.items(), key=lambda item: item[1].years)
    rows = []
    for (near_name, near), (far_name, far) in pairwise(ordered):
        near_w = near.implied_vol(grid) ** 2 * near.years
        far_w = far.implied_vol(grid) ** 2 * far.years
        shortfall = near_w - far_w

        worst = int(np.argmax(shortfall))
        if shortfall[worst] > tolerance:
            rows.append(
                {
                    "check": "calendar",
                    "expiry": f"{near_name} vs {far_name}",
                    "detail": f"k={grid[worst]:+.3f}",
                    "trade": "near-dated total variance exceeds far-dated",
                    "edge": float(shortfall[worst]),
                }
            )
    return pd.DataFrame(rows)


def summarize(*tables: pd.DataFrame) -> pd.DataFrame:
    """Stack the violation tables, worst edge first."""
    present = [table for table in tables if table is not None and not table.empty]
    if not present:
        return pd.DataFrame(columns=["check", "expiry", "detail", "trade", "edge"])
    return (
        pd.concat(present, ignore_index=True)
        .sort_values("edge", ascending=False)
        .reset_index(drop=True)
    )
