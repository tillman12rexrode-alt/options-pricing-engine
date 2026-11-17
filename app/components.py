"""Small rendering helpers shared by every page."""

from __future__ import annotations

import pandas as pd
import streamlit as st

__all__ = ["LABELS", "show"]

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
