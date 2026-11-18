"""Stock research: who the company is, what it reported, and what its options price in.

The financials are the company's own numbers from its 10-K and 10-Q filings,
read through SEC EDGAR. The share price and the options figures come from the
same Cboe snapshot the Stock options page uses. Every section says which of
the two it comes from and how old it is, because the two are refreshed on
different schedules: filings quarterly, quotes every fifteen minutes.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import altair as alt
import model_views
import pandas as pd
import state
import stock_options
import streamlit as st
from components import money, text

from optpricing import fundamentals as fun
from optpricing.equity_analysis import term_structure
from optpricing.providers import cboe, fred, sec

ROOT = Path(__file__).resolve().parent.parent
FUNDAMENTALS = ROOT / "data" / "fundamentals"
EQUITY = ROOT / "data" / "equity"

BUNDLED = "Saved snapshot"
LIVE = "Live (SEC EDGAR + Cboe)"

#: Expiries further out than this are left off the options table by default.
TERM_DAYS = 400


# --------------------------------------------------------------------------- data


@st.cache_data(show_spinner=False)
def _bundled_fundamentals(path: str) -> fun.Fundamentals:
    return fun.analyze(sec.load_snapshot(Path(path)))


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def _live_fundamentals(symbol: str) -> fun.Fundamentals:
    """Six hours: filings arrive at most a few times a quarter.

    The User-Agent (the user's email) is read inside, so it never becomes part
    of the cache key.
    """
    return fun.analyze(sec.fetch_snapshot(symbol, state.secret("SEC_USER_AGENT")))


@st.cache_data(show_spinner=False, max_entries=32)
def _term_structure(_snap: cboe.ChainSnapshot, snap_id: str, curve_fields: tuple) -> pd.DataFrame:
    asof_iso, tenors, rates, source = curve_fields
    curve = fred.YieldCurve(date.fromisoformat(asof_iso), tenors, rates, source)
    return term_structure(_snap, curve)


def _latest(paths: dict[str, list[Path]], symbol: str) -> Path | None:
    return paths.get(symbol, [None])[-1]


# ------------------------------------------------------------------------ sidebar


def sidebar() -> tuple[str, str] | None:
    st.sidebar.title("Stock research")
    st.sidebar.caption(
        "A company's own reported numbers, from its SEC filings, next to what its "
        "options say about the months ahead."
    )
    bundled = sec.list_snapshots(FUNDAMENTALS)
    modes = ([BUNDLED] if bundled else []) + ([LIVE] if state.live_data_enabled() else [])
    if not modes:
        return None
    mode = st.sidebar.radio("Data", modes) if len(modes) > 1 else modes[0]

    if mode == BUNDLED:
        return mode, state.ticker_select(sorted(bundled), page="research")
    symbol = state.ticker_input(page="research")
    return (mode, symbol) if symbol else None


def _load(mode: str, symbol: str) -> tuple[fun.Fundamentals | None, cboe.ChainSnapshot | None, str]:
    """The fundamentals and the chain for one ticker, and a note on where they came from."""
    note = ""
    facts = None
    if mode == LIVE:
        if state.secret("SEC_USER_AGENT"):
            try:
                facts = _live_fundamentals(symbol)
            except (sec.SECError, sec.UnknownCompanyError) as error:
                note = str(error)
        else:
            note = (
                "Live filings need SEC_USER_AGENT in .streamlit/secrets.toml: your name and "
                "email, which the SEC asks every request to carry."
            )
    if facts is None:
        path = _latest(sec.list_snapshots(FUNDAMENTALS), symbol)
        if path is not None:
            facts = _bundled_fundamentals(str(path))

    snap = None
    if mode == LIVE:
        try:
            snap = stock_options.load_live(symbol)
        except Exception as error:  # an outage here should not hide the financials
            note += f" Cboe could not be reached ({type(error).__name__})."
    else:
        path = _latest(cboe.list_snapshots(EQUITY), symbol)
        if path is not None:
            snap = stock_options.load_bundled(str(path))
    return facts, snap, note.strip()


# ------------------------------------------------------------------------- header


def _day(when: date) -> str:
    return f"{when:%b} {when.day}, {when.year}"


def header(f: fun.Fundamentals, snap: cboe.ChainSnapshot | None) -> None:
    st.title(f.symbol)
    st.caption(" · ".join(
        part for part in (
            f.name, f.industry, f.exchange,
            f"fiscal year ends {f.fiscal_year_end}" if f.fiscal_year_end else None,
        ) if part
    ))

    price = snap.spot if snap else None
    value = f.valuation(price) if price else {}
    with st.container(key="headline"):
        a, b, c, d = st.columns(4)
        if snap:
            u = snap.underlying
            change, pct = u.get("price_change"), u.get("price_change_percent")
            a.metric(
                "Share price", f"${price:,.2f}",
                delta=f"{change:+.2f} ({pct:+.2f}%)" if change is not None and pct is not None else None,
            )
        else:
            a.metric("Share price", "n/a")
        b.metric(
            "Market cap", money(value.get("market_cap"), 2),
            help="Share price times the shares outstanding on the latest filing's cover page.",
        )
        pe = value.get("pe")
        loss = (f.ttm.get("net_income") or 0) < 0
        c.metric(
            "P/E, last 12 months", "Loss" if pe is None and loss else text(pe, "{:,.1f}x"),
            help="Market cap divided by net income over the last twelve reported months. "
            "Not meaningful when the company lost money.",
        )
        iv30 = snap.underlying.get("iv30") if snap else None
        d.metric(
            "Cboe 30-day IV", text(iv30, "{:.1%}"),
            help="Cboe's own 30-day implied volatility for the stock: the options market's "
            "annualised estimate of how much the price will move.",
        )

    parts = []
    if f.latest_end:
        parts.append(
            f"Financials: the company's filings to {_day(f.latest_end)}, via SEC EDGAR, "
            f"fetched {_day(f.fetched)}."
        )
    if snap:
        parts.append(
            f"Price and options: Cboe, {_day(snap.trading_date)}, the {snap.spot_basis}."
        )
    if f.shares:
        parts.append(f"Shares outstanding {f.shares[0]:,.0f} as of {_day(f.shares[1])}.")
    st.caption(" ".join(parts))


# ------------------------------------------------------------------- financials


def _growth(f: fun.Fundamentals, key: str) -> str | None:
    g = f.growth(key)
    return None if g is None else f"{g:+.1%} YoY"


def last_twelve_months(f: fun.Fundamentals) -> None:
    st.subheader("Last twelve months")
    if f.latest_end is None:
        st.info("No income statement data in this company's filings.")
        return
    st.caption(
        f"The twelve months to {_day(f.latest_end)}. Where that is not a fiscal year-end, "
        "it is built from the filings: year-to-date, plus the last full year, minus the "
        "same months a year earlier."
    )
    with st.container(key="tiles_ttm"):
        a, b, c, d = st.columns(4)
        yoy = "The badge compares with the twelve months to the same date a year earlier."
        a.metric("Revenue", money(f.ttm.get("revenue")), delta=_growth(f, "revenue"), help=yoy)
        b.metric("Net income", money(f.ttm.get("net_income")), delta=_growth(f, "net_income"),
                 delta_color="normal" if (f.ttm.get("net_income") or 0) > 0 else "off", help=yoy)
        c.metric("Gross margin", text(f.margin("gross_profit"), "{:.1%}"))
        d.metric("Operating margin", text(f.margin("operating_income"), "{:.1%}"))
    with st.container(key="tiles_balance"):
        a, b, c, d = st.columns(4)
        a.metric(
            "Free cash flow", money(f.free_cash_flow),
            help="Operating cash flow minus capital expenditure, last twelve months.",
        )
        short = {"cash": "Cash", "liabilities": "Total liabilities", "equity": "Shareholders' equity"}
        for column, key in ((b, "cash"), (c, "liabilities"), (d, "equity")):
            label = short[key]
            balance = f.balances.get(key)
            column.metric(
                label, money(balance[0] if balance else None),
                help=f"On the balance sheet of {_day(balance[1])}." if balance else None,
            )


def annual_table(f: fun.Fundamentals) -> None:
    st.subheader("Annual results")
    if not f.years:
        st.info("No full fiscal years in this company's filings.")
        return
    columns = [fun.fiscal_label(y) for y in f.years]
    rows = []
    for metric in fun.METRICS:
        fmt = (lambda v: text(v, "${:,.2f}")) if metric.key == "eps_diluted" else money
        row = {"": metric.label}
        for year, label in zip(f.years, columns, strict=True):
            row[label] = fmt(f.annual[metric.key].get(year))
        if metric.kind == "duration":
            row["Last 12 months"] = fmt(f.ttm.get(metric.key))
        else:
            balance = f.balances.get(metric.key)
            row["Last 12 months"] = fmt(balance[0] if balance else None)
        rows.append(row)
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", height=38 + 35 * len(rows))
    st.caption(
        "As each company tagged its own filings. A fiscal year is labelled by the month it "
        "ends. Balances in the last column are from the latest balance sheet. Diluted EPS "
        "over the last twelve months is the sum of the reported figures, so share-count "
        "changes within the year make it approximate."
    )


def valuation_table(f: fun.Fundamentals, snap: cboe.ChainSnapshot | None) -> None:
    st.subheader("Valuation")
    if snap is None or f.shares is None:
        st.info("Valuation needs both a share price and a share count; one is missing.")
        return
    v = f.valuation(snap.spot)
    rows = [
        ("Market cap", money(v["market_cap"], 2), "Share price × shares outstanding."),
        ("Price / earnings", text(v["pe"], "{:,.1f}x"), "Market cap ÷ net income, last 12 months."),
        ("Price / sales", text(v["ps"], "{:,.1f}x"), "Market cap ÷ revenue, last 12 months."),
        ("Price / book", text(v["pb"], "{:,.1f}x"), "Market cap ÷ shareholders' equity."),
        ("Free cash flow yield", text(v["fcf_yield"], "{:.1%}"), "Free cash flow ÷ market cap."),
    ]
    st.dataframe(
        pd.DataFrame(rows, columns=["Measure", "Value", "How it is calculated"]),
        hide_index=True, width="stretch",
    )
    st.caption(
        "n/a where a multiple has no meaning, such as a price/earnings ratio on a loss. "
        "No debt-based measures: debt is tagged too inconsistently across these filings "
        "to compare fairly."
    )


# ---------------------------------------------------------------------- options


def options_outlook(snap: cboe.ChainSnapshot | None, historical: bool) -> None:
    st.subheader("What the options price in")
    if snap is None:
        st.info("No option chain saved for this ticker.")
        return
    curve, problem = model_views.load_curve(snap.trading_date if historical else None)
    fields = (curve.asof.isoformat(), curve.tenors, curve.rates, curve.source)
    with st.spinner("Solving at-the-money vols..."):
        ts = _term_structure(snap, f"{snap.symbol}:{snap.trading_date}:{snap.asof}", fields)
    ts = ts[ts["days"] <= TERM_DAYS]
    if ts["straddle"].isna().all():
        st.info("No expiry has a two-sided call and put near the forward.")
        return

    shown = pd.DataFrame({
        "Expiry": ts["expiry"],
        "Days": ts["days"].astype(int).astype(str),
        "Strike": [text(v, "{:,.2f}") for v in ts["strike"]],
        "Straddle": [text(v, "${:,.2f}") for v in ts["straddle"]],
        "Expected move": [
            "n/a" if pd.isna(s) else f"±${s:,.2f} (±{m:.1%})"
            for s, m in zip(ts["straddle"], ts["move_pct"], strict=True)
        ],
        "ATM IV (ours)": [text(v, "{:.1%}") for v in ts["atm_iv"]],
        "ATM IV (Cboe)": [text(v, "{:.1%}") for v in ts["cboe_atm_iv"]],
    })
    st.dataframe(shown, hide_index=True, width="stretch")
    st.caption(
        "Expected move is the at-the-money straddle: what the market charges today for "
        "a call and a put at the strike nearest the forward. It is the price of a move in "
        "either direction, not a forecast of direction, and about 0.8 of one standard "
        "deviation if returns were normal. ATM IV is the average of that call's and put's "
        "American implied vols."
        + (f" Rates: a flat {curve.rates[0]:.2%} because {problem}." if problem else "")
    )

    points = pd.concat([
        pd.DataFrame({"days": ts["days"], "iv": ts["atm_iv"], "series": "Ours"}),
        pd.DataFrame({"days": ts["days"], "iv": ts["cboe_atm_iv"], "series": "Cboe"}),
    ]).dropna()
    color = alt.Color("series:N", scale=alt.Scale(domain=["Ours", "Cboe"]),
                      legend=alt.Legend(title=None, orient="top"))
    tooltip = [alt.Tooltip("series:N", title="Source"), alt.Tooltip("days:Q", title="Days"),
               alt.Tooltip("iv:Q", format=".1%", title="ATM IV")]
    lines = alt.Chart(points).mark_line(point=True).encode(
        x=alt.X("days:Q", title="Days to expiry"),
        y=alt.Y("iv:Q", title="At-the-money implied vol", axis=alt.Axis(format=".0%"),
                scale=alt.Scale(zero=False)),
        color=color, tooltip=tooltip,
    )
    layers = [lines]
    iv30 = snap.underlying.get("iv30")
    if iv30:
        mark = pd.DataFrame({"days": [30], "iv": [iv30], "label": [f"Cboe 30-day {iv30:.1%}"]})
        layers.append(alt.Chart(mark).mark_point(shape="diamond", size=90, filled=True, opacity=0.8)
                      .encode(x="days:Q", y="iv:Q", tooltip=[alt.Tooltip("label:N", title="")]))
    st.altair_chart(alt.layer(*layers).properties(height=320), width="stretch")
    st.caption(
        "The term structure: how much movement the options price in over each horizon, "
        "annualised. A hump at one expiry usually marks a known event inside it, such as "
        "an earnings date."
    )


# ---------------------------------------------------------------------- sources


def filings(f: fun.Fundamentals) -> None:
    st.subheader("Filings")
    shown = []
    for forms in (("10-K", "10-KT"), ("10-Q",)):
        match = max((x for x in f.filings if x["form"] in forms), key=lambda x: x["filed"], default=None)
        if match:
            shown.append(
                f"- [{match['form']} for the period to {_day(date.fromisoformat(match['period']))}]"
                f"({sec.filing_url(f.cik, match)}), filed {_day(date.fromisoformat(match['filed']))}"
            )
    st.markdown("\n".join(shown) if shown else "No 10-K or 10-Q on file.")

    with st.expander("Where each number comes from"):
        rows = [
            (m.label, f.concepts.get(m.key) or "not reported")
            for m in fun.METRICS
        ]
        st.dataframe(pd.DataFrame(rows, columns=["Figure", "XBRL concept"]), hide_index=True, width="stretch")
        st.caption(
            "Companies choose their own tags, and change them. Where a figure has more than "
            "one possible tag, the one with the most recent data is used, and older years "
            "are filled from the others. Each period uses the most recently filed value, "
            "so restatements win over the original."
        )


# ------------------------------------------------------------------------- page


def render() -> None:
    picked = sidebar()
    if picked is None:
        st.title("Stock research")
        st.info(
            "No saved company data yet. On your own computer, run "
            "`python analysis/fetch_fundamentals.py --symbol AAPL` and commit what it "
            "writes to data/fundamentals/."
        )
        return
    mode, symbol = picked
    with st.spinner(f"Loading {symbol}..."):
        f, snap, note = _load(mode, symbol)
    if note:
        st.warning(note)
    if f is None:
        st.title(symbol)
        st.info(f"No SEC filings saved for {symbol}. Funds, ETFs and indices file none.")
        return

    header(f, snap)
    last_twelve_months(f)
    annual_table(f)
    valuation_table(f, snap)
    options_outlook(snap, historical=(mode == BUNDLED))
    filings(f)
