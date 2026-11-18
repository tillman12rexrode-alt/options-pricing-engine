"""What the visitor is looking at, shared by every page.

One place reads and writes the selected ticker, so pages cannot drift apart,
and changing it clears anything that belonged to the old one. An expiry from
AAPL means nothing on RCAT, and keeping it would quietly mix two companies.
"""

from __future__ import annotations

import os

import streamlit as st

from optpricing.providers import cboe
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


def secret(name: str) -> str | None:
    """A setting from the environment or .streamlit/secrets.toml, never from code."""
    value = os.environ.get(name)
    if value:
        return value.strip()
    try:
        value = st.secrets.get(name)
    except Exception:  # no secrets file exists at all, which is the public default
        return None
    return str(value).strip() if value else None


# ------------------------------------------------------------------ the pickers
#
# Each page draws its own Ticker widget, and Streamlit treats widgets on
# different pages as different widgets even under the same key: a page you
# switch to starts from its default. So every page keys its picker by page,
# writes the shared ticker into it before drawing it, and records a pick in a
# change callback. Callbacks run before the page does, so by the time the
# widget is seeded the shared ticker already holds the visitor's new pick.


def ticker_select(symbols: list[str], page: str) -> str:
    """A ticker dropdown in the sidebar that every page shares."""
    key = f"ticker_select_{page}"
    if current_ticker() not in symbols:
        set_ticker(symbols[0])
    st.session_state[key] = current_ticker()
    st.sidebar.selectbox(
        "Ticker", symbols, key=key,
        format_func=lambda s: f"{s}  ({display_name(s)})",
        on_change=lambda: set_ticker(st.session_state[key]),
    )
    return current_ticker()


def ticker_input(page: str) -> str | None:
    """A free-text ticker box in the sidebar, for live data. None while the entry is invalid."""
    key, error_key = f"ticker_input_{page}", "ticker_error"

    def changed() -> None:
        st.session_state[error_key] = None
        try:
            set_ticker(cboe.normalize(st.session_state[key]))
        except cboe.InvalidTickerError as error:
            st.session_state[error_key] = str(error)

    if not st.session_state.get(error_key):
        st.session_state[key] = current_ticker()
    st.sidebar.text_input("Ticker", key=key, max_chars=5, placeholder="AAPL", on_change=changed)
    if st.session_state.get(error_key):
        st.sidebar.error(st.session_state[error_key])
        return None
    return current_ticker()
