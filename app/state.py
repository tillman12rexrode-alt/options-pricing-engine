"""What the visitor is looking at, shared by every page.

One place reads and writes the selected ticker, so pages cannot drift apart,
and changing it clears anything that belonged to the old one. An expiry from
AAPL means nothing on RCAT, and keeping it would quietly mix two companies.
"""

from __future__ import annotations

import os

import streamlit as st

from optpricing.providers.cboe import INDEX_SYMBOLS

DEFAULT_TICKER = "AAPL"

#: Display names for the tickers this project follows. Anything else shows its symbol.
NAMES = {
    "AAPL": "Apple",
    "NVDA": "NVIDIA",
    "TSLA": "Tesla",
    "KTOS": "Kratos Defense & Security Solutions",
    "RCAT": "Red Cat Holdings",
    **INDEX_SYMBOLS,
}

#: Session keys that only make sense for the ticker they were chosen with.
_PER_TICKER = ("expiry",)


def current_ticker() -> str:
    return st.session_state.setdefault("ticker", DEFAULT_TICKER)


def set_ticker(symbol: str) -> None:
    if symbol != st.session_state.get("ticker"):
        st.session_state["ticker"] = symbol
        for key in _PER_TICKER:
            st.session_state.pop(key, None)


def display_name(symbol: str) -> str:
    return NAMES.get(symbol.upper(), symbol.upper())


def live_data_enabled() -> bool:
    """Live Cboe fetches are for running the app on your own machine.

    The public deployment serves bundled snapshots only. Turn live data on
    locally with `LIVE_DATA = true` in .streamlit/secrets.toml, or by setting
    OPTPRICING_LIVE_DATA=1 in the environment.
    """
    if os.environ.get("OPTPRICING_LIVE_DATA", "").strip().lower() in ("1", "true", "yes"):
        return True
    try:
        return bool(st.secrets.get("LIVE_DATA", False))
    except Exception:  # no secrets file exists at all, which is the public default
        return False
