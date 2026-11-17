"""Stock options: one ticker's chain, with its source and its age always on screen.

Everything shown here is market-observed: Cboe's quotes and Cboe's own implied
volatilities. Model prices, American implied volatilities and Greeks are built
on top of this page in the next phase, and will sit in their own clearly
labelled columns so a reader never mistakes a model estimate for a quote.
"""

from __future__ import annotations

import math
from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd
import requests
import state
import streamlit as st

from optpricing.providers import cboe

ROOT = Path(__file__).resolve().parent.parent
EQUITY = ROOT / "data" / "equity"

BUNDLED = "Saved snapshot"
LIVE = "Live Cboe (15-min delayed)"

#: Strikes shown by default, as a fraction of the share price either side.
DEFAULT_WINDOW = 0.25

#: ...but never fewer strikes than this, taken nearest the share price.
MIN_STRIKES = 10


# --------------------------------------------------------------------------- data


@st.cache_data(show_spinner=False)
def load_bundled(path: str) -> cboe.ChainSnapshot:
    return cboe.load_snapshot(Path(path))


@st.cache_data(ttl=900, show_spinner=False)
def load_live(symbol: str) -> cboe.ChainSnapshot:
    """Cached for the feed's own delay: fetching sooner returns the same quotes."""
    return cboe.fetch_chain(symbol)


# ----------------------------------------------------------------------- sidebar


def _seed(key: str, value: str) -> None:
    """Give a widget its starting value through session state, once.

    Passing a starting value to the widget itself (index=, value=) looks
    equivalent and is not: Streamlit treats a changed default as a brand-new
    widget and silently discards what the visitor just picked.
    """
    if key not in st.session_state:
        st.session_state[key] = value


def _pick_bundled(bundled: dict[str, list[Path]]) -> Path:
    symbols = sorted(bundled)
    _seed("ticker_picker", state.current_ticker())
    if st.session_state["ticker_picker"] not in symbols:
        st.session_state["ticker_picker"] = symbols[0]
    chosen = st.sidebar.selectbox(
        "Ticker",
        symbols,
        key="ticker_picker",
        format_func=lambda s: f"{s}  ({state.display_name(s)})",
    )
    state.set_ticker(chosen)

    # Keyed by ticker. Every ticker's list shows the same dates, so without a
    # key Streamlit treats them as one widget and keeps the previous ticker's
    # file selected after a switch.
    files = {p.name.split("_")[1].removesuffix(".json.gz"): p for p in reversed(bundled[chosen])}
    session = st.sidebar.selectbox(
        "Snapshot",
        list(files),
        key=f"snapshot_{chosen}",
        disabled=len(files) == 1,
    )
    return files[session]


def _pick_live() -> str | None:
    _seed("ticker_typed", state.current_ticker())
    typed = st.sidebar.text_input("Ticker", key="ticker_typed", max_chars=5, placeholder="AAPL")
    try:
        symbol = cboe.normalize(typed)
    except cboe.InvalidTickerError as error:
        st.sidebar.error(str(error))
        return None
    state.set_ticker(symbol)
    return symbol


def sidebar() -> tuple[str, str] | None:
    """(mode, path or symbol), or None when there is nothing to show."""
    st.sidebar.title("Stock options")
    st.sidebar.caption(
        "American-style options on single stocks, quoted by Cboe. The page "
        "measures every quote against the price its market was actually trading at."
    )

    bundled = cboe.list_snapshots(EQUITY)
    modes = ([BUNDLED] if bundled else []) + ([LIVE] if state.live_data_enabled() else [])
    if not modes:
        return None

    mode = st.sidebar.radio("Data", modes) if len(modes) > 1 else modes[0]
    if mode == BUNDLED:
        return mode, str(_pick_bundled(bundled))
    symbol = _pick_live()
    return (mode, symbol) if symbol else None


# ------------------------------------------------------------------------- header


def _day(when: date) -> str:
    """Nov 16, 2026. Built by hand: the %-d strftime code fails on Windows."""
    return f"{when:%b} {when.day}, {when.year}"


def _eastern(moment: datetime) -> str:
    local = moment.astimezone(cboe.EXCHANGE)
    return f"{_day(local)} at {local.hour % 12 or 12}:{local:%M %p} ET"


def header(snap: cboe.ChainSnapshot, mode: str) -> None:
    st.title(snap.symbol)
    st.caption(state.display_name(snap.symbol))

    u = snap.underlying
    change, pct = u.get("price_change"), u.get("price_change_percent")
    iv30 = u.get("iv30")

    with st.container(key="headline"):
        a, b, c, d = st.columns(4)
        a.metric(
            "Share price",
            f"${snap.spot:,.2f}",
            delta=f"{change:+.2f} ({pct:+.2f}%)" if change is not None and pct is not None else None,
        )
        b.metric("Session", snap.trading_date.isoformat())
        c.metric("Cboe 30-day implied vol", f"{iv30:.1%}" if iv30 else "n/a")
        d.metric("Contracts listed", f"{len(snap.quotes):,}")

    basis = f"the {snap.spot_basis}" if snap.spot_basis != "last trade" else "the last trade"
    note = (
        f"Options are measured against {basis}. "
        f"Quotes as of {_eastern(snap.asof)}. Source: Cboe, data delayed {snap.delay_minutes} minutes."
    )
    after_hours = u.get("current_price")
    if snap.spot_basis == "regular-session close" and after_hours and not math.isclose(
        float(after_hours), snap.spot, rel_tol=1e-4
    ):
        note += (
            f" After-hours trading is at ${float(after_hours):,.2f}, but option quotes stopped "
            "at the close, so the close is the price they reflect."
        )
    st.caption(note)

    if mode == BUNDLED:
        age_days = (datetime.now(UTC).astimezone(cboe.EXCHANGE).date() - snap.trading_date).days
        if age_days > 0:
            st.info(
                f"This is a saved snapshot from {_day(snap.trading_date)}, "
                f"{age_days} day{'s' if age_days != 1 else ''} old. Prices are not current."
            )
    elif snap.is_stale():
        st.warning(
            f"These quotes are {snap.age_minutes():.0f} minutes old during market hours, "
            f"more than the feed's {snap.delay_minutes}-minute delay explains."
        )


# -------------------------------------------------------------------------- chain


def _days(expiry: str, session: date) -> int:
    return (date.fromisoformat(expiry) - session).days


def pick_expiry(snap: cboe.ChainSnapshot) -> str:
    expiries = snap.expiries
    if st.session_state.get("expiry") not in expiries:
        st.session_state.pop("expiry", None)
        st.session_state["expiry"] = min(
            expiries, key=lambda e: abs(_days(e, snap.trading_date) - 30)
        )
    return st.selectbox(
        "Expiry",
        expiries,
        key="expiry",
        format_func=lambda e: f"{e}  ({_days(e, snap.trading_date)} days)",
    )


_SIDE = {"bid": "bid", "ask": "ask", "volume": "vol.", "open_interest": "OI", "cboe_iv": "IV"}


def chain_frame(snap: cboe.ChainSnapshot, expiry: str, window: float | None) -> pd.DataFrame:
    """Calls on the left, puts on the right, one row per strike. Numeric."""
    quotes = snap.quotes[snap.quotes["expiry"] == expiry]
    if window is not None:
        # A percentage window alone fails on a cheap stock: at $3.53 with $1
        # strikes, 25% either side catches two. So never show fewer than the
        # strikes nearest the share price.
        strikes = pd.Series(sorted(quotes["strike"].unique()))
        nearest = set(strikes.iloc[(strikes - snap.spot).abs().argsort()[:MIN_STRIKES]])
        low, high = snap.spot * (1 - window), snap.spot * (1 + window)
        quotes = quotes[quotes["strike"].between(low, high) | quotes["strike"].isin(nearest)]

    sides = []
    for kind, label in (("call", "Call"), ("put", "Put")):
        side = quotes[quotes["kind"] == kind].set_index("strike")[list(_SIDE)]
        side.columns = [f"{label} {name}" for name in _SIDE.values()]
        sides.append(side)

    table = sides[0].join(sides[1], how="outer").sort_index().reset_index()
    table = table.rename(columns={"strike": "Strike"})
    calls = [c for c in table.columns if c.startswith("Call")]
    puts = [c for c in table.columns if c.startswith("Put")]
    return table[[*calls, "Strike", *puts]]


def _text(value: float, pattern: str) -> str:
    """Format a number, or say plainly that there isn't one."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
    return pattern.format(value)


def show_chain(table: pd.DataFrame, spot: float) -> None:
    """In-the-money cells tinted, the strike column emphasised, gaps written as n/a.

    Cells are formatted to text here rather than by Streamlit, because Streamlit
    draws a missing number as "None" whatever formatting it is given.
    """
    calls = [c for c in table.columns if c.startswith("Call")]
    puts = [c for c in table.columns if c.startswith("Put")]
    call_itm = (table["Strike"] < spot).to_numpy()
    put_itm = (table["Strike"] > spot).to_numpy()

    patterns = {"bid": "{:,.2f}", "ask": "{:,.2f}", "vol.": "{:,.0f}", "OI": "{:,.0f}", "IV": "{:.1%}"}
    shown = pd.DataFrame({
        col: [
            _text(v, "{:,.2f}" if col == "Strike" else patterns[col.split(" ", 1)[1]])
            for v in table[col]
        ]
        for col in table.columns
    })

    # Strong enough to read on the dark theme as well as the light one.
    tint = "background-color: rgba(52, 179, 109, 0.22)"

    def shade(column: pd.Series) -> list[str]:
        if column.name in calls:
            mask = call_itm
        elif column.name in puts:
            mask = put_itm
        else:
            return [""] * len(column)
        return [tint if itm else "" for itm in mask]

    styled = shown.style.apply(shade, axis=0).set_properties(
        subset=["Strike"], **{"font-weight": "600"}
    )
    numeric = st.column_config.TextColumn(width="small")
    st.dataframe(
        styled,
        width="stretch",
        hide_index=True,
        height=min(38 + 35 * len(table), 640),
        column_config={col: numeric for col in shown.columns},
    )


def coverage(snap: cboe.ChainSnapshot, expiry: str) -> None:
    quotes = snap.quotes[snap.quotes["expiry"] == expiry]
    bid = quotes["bid"].fillna(0)
    one_sided = int(((bid <= 0) | (quotes["ask"].fillna(0) <= bid)).sum())
    no_iv = int(quotes["cboe_iv"].isna().sum())

    rows = [("Contracts Cboe returned", len(snap.quotes) + sum(snap.dropped.values()))]
    rows += [(f"Removed: {reason}", -count) for reason, count in snap.dropped.items()]
    rows += [("Listed, all expiries", len(snap.quotes)), ("In this expiry", len(quotes))]
    rows += [
        ("  without a two-sided market", one_sided),
        ("  without a Cboe implied vol", no_iv),
    ]
    st.dataframe(
        pd.DataFrame(rows, columns=["", "Contracts"]), hide_index=True, width="content"
    )
    st.caption(
        "A contract without a two-sided market has no bid, or an ask no higher than its "
        "bid. Its last trade can be hours or days old, so it is shown but should not be "
        "treated as a price. Cboe reports an implied vol it could not compute as zero; "
        "those are shown as n/a."
    )


# --------------------------------------------------------------------------- page


def _empty_state() -> None:
    st.title("Stock options")
    st.info(
        "No saved stock snapshots yet. On your own computer, run "
        "`python analysis/fetch_chain.py --symbol AAPL NVDA TSLA KTOS RCAT` and commit "
        "the files it writes to data/equity/. To fetch live instead, add "
        "`LIVE_DATA = true` to .streamlit/secrets.toml."
    )


def render() -> None:
    picked = sidebar()
    if picked is None:
        _empty_state()
        return
    mode, target = picked

    try:
        with st.spinner(f"Fetching {target} from Cboe..." if mode == LIVE else "Loading snapshot..."):
            snap = load_live(target) if mode == LIVE else load_bundled(target)
    except (cboe.UnknownSymbolError, cboe.InvalidTickerError) as error:
        st.title(state.current_ticker())
        st.warning(str(error))
        return
    except requests.RequestException as error:
        st.title(state.current_ticker())
        st.error(f"Could not reach Cboe. {type(error).__name__}. Try again in a minute.")
        return

    header(snap, mode)

    st.subheader("Option chain")
    left, right = st.columns([2, 1], vertical_alignment="bottom")
    with left:
        expiry = pick_expiry(snap)
    with right:
        every = st.toggle("Show every strike", value=False)

    table = chain_frame(snap, expiry, None if every else DEFAULT_WINDOW)
    if table.empty:
        st.info(
            f"No strikes within {DEFAULT_WINDOW:.0%} of the share price for this expiry. "
            "Turn on Show every strike to see the rest."
        )
    else:
        show_chain(table, snap.spot)
        st.caption(
            "Shaded cells are in the money. OI is open interest; IV is Cboe's implied "
            "volatility. Every figure here is Cboe's, unmodified. Each contract is for "
            f"{int(snap.quotes['multiplier'].iloc[0])} shares."
        )

    with st.expander("What was left out, and why"):
        coverage(snap, expiry)
