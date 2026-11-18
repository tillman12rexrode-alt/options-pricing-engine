"""Small rendering helpers shared by every page."""

from __future__ import annotations

import math

import pandas as pd
import streamlit as st

__all__ = ["LABELS", "money", "show", "text"]

#: Column names people read, rather than the ones the library uses internally.
LABELS = {
    "expiry": "Expiry", "days": "Days", "quotes": "Quotes", "kind": "Type",
    "strike": "Strike", "bid": "Bid", "ask": "Ask", "mid": "Mid",
    "iv_brent": "Implied vol", "fitted_iv": "Fitted vol",
    "residual_points": "Gap (vol pts)", "spread_multiple": "Gap / spread",
    "spread_vol_points": "Spread (vol pts)", "open_interest": "Open interest",
    "volume": "Volume", "verdict": "Verdict", "rmse_vol_points": "RMSE (vol pts)",
    "worst_bias": "Worst bias", "scannable": "Scannable", "note": "Note",
    "check": "Check", "detail": "Contract", "trade": "Trade", "edge": "Edge ($)",
    "stage": "Stage", "gap_points": "Gap (vol pts)", "gross": "Gross ($)",
    "costs": "Frictions ($)", "net": "Net ($)", "breakeven_frac": "Gap needed",
    "theta_per_day": "Theta ($/day)", "action": "Action", "quantity": "Contracts",
    "vega_per_point": "Vega ($/pt)", "delta": "Delta",
}


def show(frame: pd.DataFrame, formats: dict | None = None) -> None:
    """Render a table with readable headers."""
    renamed = frame.rename(columns=LABELS)
    styled = renamed.style.format(na_rep="n/a")
    if formats:
        styled = styled.format(
            {LABELS.get(k, k): v for k, v in formats.items()}, na_rep="n/a"
        )
    st.dataframe(styled, width="stretch", hide_index=True)


def text(value, pattern: str) -> str:
    """Format a number, or say plainly that there isn't one.

    Streamlit draws a missing number as "None" whatever formatting it is
    given, so tables that can have gaps are formatted to text first.
    """
    if value is None or (isinstance(value, float) and math.isnan(value)) or value is pd.NA:
        return "n/a"
    return pattern.format(value)


def money(value, digits: int = 1) -> str:
    """$478.8B, $73.7M, -$121.9M, $0.91. Big numbers people can read at a glance."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
    sign = "-" if value < 0 else ""
    size = abs(value)
    for scale, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")):
        if size >= scale:
            return f"{sign}${size / scale:,.{digits}f}{suffix}"
    return f"{sign}${size:,.2f}"
