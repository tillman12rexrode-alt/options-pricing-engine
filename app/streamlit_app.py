"""Surface Scanner - a browser app over the optpricing library.

Three questions, three tabs. Which quotes disagree with their own smile; what
is any single contract actually worth; and does the chain contain anything that
is arbitrage regardless of model.

Run it with:

    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from optpricing import arbitrage as arb
from optpricing.contracts import Option
from optpricing.data.cboe import EXCHANGE, INDEX_SYMBOLS, fetch_chain
from optpricing.implied_vol import implied_vol
from optpricing.market import load_snapshot
from optpricing.models import binomial
from optpricing.models import black_scholes as bs
from optpricing.models import monte_carlo as mc
from optpricing.scanner import (
    SurfaceAnalysis,
    analyze,
    find_outliers,
    rank_candidates,
    trade_mechanics,
)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

st.set_page_config(page_title="Surface Scanner", page_icon="📈", layout="wide")

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
    styled = renamed.style.format(na_rep="—")
    if formats:
        styled = styled.format(
            {LABELS.get(k, k): v for k, v in formats.items()}, na_rep="—"
        )
    st.dataframe(styled, width="stretch", hide_index=True)


# --------------------------------------------------------------------------- data


@st.cache_data(ttl=900, show_spinner=False)
def load_live(symbol: str) -> tuple[pd.DataFrame, float, date]:
    snapshot = fetch_chain(symbol)
    return snapshot.to_frame(), snapshot.spot, snapshot.trading_date


@st.cache_data(show_spinner=False)
def load_bundled(path: str) -> tuple[pd.DataFrame, float, date]:
    frame, spot, _ = load_snapshot(path)
    stamp = str(frame["asof"].iloc[0])
    local = datetime.fromisoformat(stamp).replace(tzinfo=UTC).astimezone(EXCHANGE).date()
    return frame, spot, local


@st.cache_data(ttl=900, show_spinner=False)
def run_analysis(
    frame: pd.DataFrame, spot: float, asof: date, max_rel_spread: float
) -> SurfaceAnalysis:
    return analyze(frame, spot, asof, max_rel_spread=max_rel_spread)


# ----------------------------------------------------------------------- sidebar


def sidebar() -> dict:
    st.sidebar.title("Surface Scanner")
    st.sidebar.caption(
        "European, cash-settled index options only. Equity options are American, "
        "which breaks the put-call parity relation this tool derives its forwards from."
    )

    snapshots = sorted(DATA.glob("*_chain_*.csv"))
    sources = ["Live CBOE (15-min delayed)"]
    if snapshots:
        sources.insert(0, "Bundled snapshot")
    source = st.sidebar.radio("Data", sources, index=0)

    symbol, snapshot = "SPX", None
    if source.startswith("Live"):
        symbol = st.sidebar.selectbox(
            "Index",
            sorted(INDEX_SYMBOLS),
            format_func=lambda s: f"{s} — {INDEX_SYMBOLS[s]}",
        )
    else:
        snapshot = st.sidebar.selectbox("Snapshot", snapshots, format_func=lambda p: p.name)

    st.sidebar.divider()
    st.sidebar.subheader("Cleaning")
    max_rel_spread = st.sidebar.slider("Max spread, % of mid", 2, 50, 20, step=1) / 100

    st.sidebar.subheader("What counts as an outlier")
    min_points = st.sidebar.slider("Minimum gap, vol points", 0.25, 5.0, 1.0, step=0.25)
    min_multiple = st.sidebar.slider("Gap as a multiple of the spread", 1.0, 6.0, 1.5, step=0.5)
    min_oi = st.sidebar.number_input("Minimum open interest", 0, 100_000, 50, step=25)

    st.sidebar.divider()
    st.sidebar.subheader("Arbitrage")
    fee = st.sidebar.number_input("Fee per contract ($)", 0.0, 5.0, 0.65, step=0.05)

    return {
        "source": source,
        "symbol": symbol,
        "snapshot": snapshot,
        "max_rel_spread": max_rel_spread,
        "min_points": min_points,
        "min_multiple": min_multiple,
        "min_oi": min_oi,
        "fee": fee,
    }


# -------------------------------------------------------------------------- tabs


def tab_scan(analysis: SurfaceAnalysis, settings: dict) -> None:
    st.subheader("Quotes that disagree with their own smile")
    st.caption(
        "A fitted curve runs through each expiry's implied volatilities. A quote is "
        "flagged only when it sits far off that curve *and* the gap exceeds its own "
        "bid-ask spread expressed in volatility points. Expiries whose curve did not "
        "fit are skipped rather than scanned."
    )

    report = analysis.fit_report()
    scannable = int(report["scannable"].sum())
    st.markdown(f"**Smile fits** — {scannable} of {len(report)} expiries usable")
    show(report, {"rmse_vol_points": "{:.3f}", "worst_bias": "{:.3f}"})

    hits = find_outliers(
        analysis,
        min_vol_points=settings["min_points"],
        min_spread_multiple=settings["min_multiple"],
        min_open_interest=settings["min_oi"],
    )

    st.markdown("**Candidates**")
    if hits.empty:
        st.success(
            "Nothing clears the thresholds. On a liquid chain that is the usual and "
            "correct result — loosen the sliders to see what sits just below the bar."
        )
        return

    show(
        hits[[
            "expiry", "kind", "strike", "bid", "ask", "iv_brent", "fitted_iv",
            "residual_points", "spread_multiple", "open_interest", "verdict",
        ]],
        {
            "strike": "{:,.0f}", "bid": "{:.2f}", "ask": "{:.2f}",
            "iv_brent": "{:.2%}", "fitted_iv": "{:.2%}",
            "residual_points": "{:+.2f}", "spread_multiple": "{:.1f}x",
            "open_interest": "{:,.0f}",
        },
    )

    st.markdown("**Why each one is flagged**")
    for _, row in hits.head(15).iterrows():
        st.markdown(
            f"- **{row['expiry']} {row['strike']:g} {row['kind']}** — {row['reason']}"
        )

    st.divider()
    best_bet(analysis, hits, settings)


def best_bet(analysis: SurfaceAnalysis, hits: pd.DataFrame, settings: dict) -> None:
    """Rank the candidates by what the model says survives frictions."""
    st.subheader("Best bet according to the model")
    st.caption(
        "Each candidate is traded as a package: sell what is rich, buy its neighbours "
        "in whatever ratio makes the position neutral to volatility level and index "
        "direction, so the only thing left is whether the gap closes. Ranked by what "
        "remains after crossing every spread twice and paying commissions. This is "
        "arithmetic on the model, not a recommendation — the model cannot tell you "
        "whether the gap is real or whether the quote was simply stale."
    )

    ranked = rank_candidates(analysis, hits, fee_per_contract=settings["fee"])
    if ranked.empty:
        st.info("Nothing to rank.")
        return

    show(
        ranked[[
            "expiry", "kind", "strike", "verdict", "gap_points",
            "gross", "costs", "net", "breakeven_frac", "theta_per_day",
        ]],
        {
            "strike": "{:,.0f}", "gap_points": "{:.2f}", "gross": "${:,.2f}",
            "costs": "${:,.2f}", "net": "${:,.2f}", "breakeven_frac": "{:.0%}",
            "theta_per_day": "${:,.2f}",
        },
    )

    tradeable = ranked[ranked["net"].notna()]
    if tradeable.empty:
        st.warning(
            "None of the candidates can be hedged — each sits at the edge of its "
            "strike range with no neighbour on one side."
        )
        return

    top = tradeable.iloc[0]
    flagged = hits[
        (hits["expiry"] == top["expiry"])
        & (hits["strike"] == top["strike"])
        & (hits["kind"] == top["kind"])
    ].iloc[0]
    mechanics = trade_mechanics(analysis, flagged, fee_per_contract=settings["fee"])

    st.markdown(
        f"**Top of the list — {top['expiry']} {top['strike']:,.0f} {top['kind']}**"
    )

    a, b, c, d = st.columns(4)
    a.metric("Gross if the gap closes", f"${mechanics.gross:,.2f}")
    b.metric("Frictions", f"-${mechanics.entry_cost + mechanics.exit_cost + mechanics.commissions:,.2f}")
    c.metric("Net", f"${mechanics.net:,.2f}")
    d.metric("Theta", f"${mechanics.theta_per_day:,.2f}/day")

    st.markdown("**The package**")
    show(
        mechanics.legs,
        {
            "strike": "{:,.0f}", "quantity": "{:.2f}", "bid": "{:.2f}",
            "ask": "{:.2f}", "vega_per_point": "${:,.2f}", "delta": "{:+.4f}",
        },
    )

    if mechanics.survives_costs:
        st.info(mechanics.verdict())
    else:
        st.error(mechanics.verdict())

    st.caption(
        "Before acting on any of this: check whether the quote still looks the same "
        "fifteen minutes from now. A gap that disappears on refresh was a stale quote, "
        "and there was never anything to capture."
    )


def tab_inspect(analysis: SurfaceAnalysis, settings: dict) -> None:
    st.subheader("Price any single contract")

    quotes = analysis.quotes
    # Default to an expiry whose smile actually fitted, so the comparison column
    # has something in it on first load.
    choices = sorted(quotes["expiry"].unique())
    default = next((i for i, e in enumerate(choices) if e in analysis.reliable_expiries), 0)

    left, middle, right = st.columns(3)
    expiry = left.selectbox("Expiry", choices, index=default)
    kind = middle.radio("Type", ["call", "put"], horizontal=True)

    slice_ = quotes[(quotes["expiry"] == expiry) & (quotes["kind"] == kind)]
    if slice_.empty:
        st.warning("No quotes for that combination.")
        return
    strike = right.selectbox(
        "Strike", sorted(slice_["strike"].unique()), format_func=lambda k: f"{k:,.0f}"
    )

    row = slice_[slice_["strike"] == strike].iloc[0]
    term = analysis.terms[str(expiry)]
    market_iv = float(row["iv_brent"])

    opt = Option(
        S=analysis.spot, K=float(strike), T=term.years, r=term.rate,
        sigma=market_iv, q=term.dividend_yield, kind=kind,
    )

    st.markdown("**The market**")
    a, b, c, d, e = st.columns(5)
    a.metric("Bid", f"{row['bid']:,.2f}")
    b.metric("Ask", f"{row['ask']:,.2f}")
    c.metric("Mid", f"{row['mid']:,.2f}")
    d.metric("Implied vol", f"{market_iv:.2%}")
    fitted = float(row["fitted_iv"]) if pd.notna(row["fitted_iv"]) else float("nan")
    e.metric(
        "Fitted vol",
        "not fitted" if pd.isna(fitted) else f"{fitted:.2%}",
        None if pd.isna(fitted) else f"{row['residual_points']:+.2f} vol pts",
    )
    if pd.isna(fitted):
        st.caption(
            "This expiry's smile did not fit well enough to scan, so there is nothing "
            "to compare against. The prices and Greeks below are unaffected."
        )

    st.markdown("**Priced three ways at the market's own volatility**")
    lattice_steps = 501
    simulated = mc.price_with_error(opt, 200_000, seed=1, antithetic=True, control_variate=True)
    a, b, c = st.columns(3)
    a.metric("Black-Scholes", f"{bs.price(opt):.3f}")
    b.metric(f"Lattice (LR, {lattice_steps})", f"{binomial.price(opt, lattice_steps, 'lr'):.3f}")
    c.metric("Monte Carlo", f"{simulated.price:.3f}", f"± {1.96 * simulated.std_error:.3f} (95%)")

    st.markdown("**Greeks**")
    greeks = bs.greeks(opt)
    st.dataframe(
        pd.DataFrame([greeks]).rename(columns=str.title).style.format("{:.5f}"),
        width="stretch",
        hide_index=True,
    )

    st.markdown("**Implied by this expiry's own quotes**")
    a, b, c, d = st.columns(4)
    a.metric("Forward", f"{term.forward:,.2f}")
    b.metric("Rate", f"{term.rate:.2%}")
    c.metric("Dividend yield", f"{term.dividend_yield:.2%}")
    d.metric("Parity fit R²", f"{term.r_squared:.6f}")

    st.markdown("**The smile this contract sits on**")
    curve = (
        quotes[quotes["expiry"] == expiry]
        .sort_values("strike")
        .set_index("strike")[["iv_brent", "fitted_iv"]]
        .rename(columns={"iv_brent": "market", "fitted_iv": "fitted"})
    )
    st.line_chart(curve, height=320)

    with st.expander("What would the price be at a different volatility?"):
        guess = st.slider("Volatility", 0.05, 1.00, float(market_iv), step=0.005)
        trial = Option(
            S=analysis.spot, K=float(strike), T=term.years, r=term.rate,
            sigma=guess, q=term.dividend_yield, kind=kind,
        )
        st.metric("Black-Scholes price", f"{bs.price(trial):.3f}",
                  f"{bs.price(trial) - float(row['mid']):+.3f} vs market mid")
        back = implied_vol(float(row["mid"]), opt, "brent")
        st.caption(
            f"Solved back from the market mid: {back.sigma:.4%} in {back.iterations} "
            "Brent iterations."
        )


def tab_arbitrage(analysis: SurfaceAnalysis, settings: dict) -> None:
    st.subheader("Relations that hold regardless of any model")
    st.caption(
        "Every test runs on executable prices — buy at the ask, sell at the bid. "
        "A violation measured on mid prices is usually just the spread."
    )

    fee = settings["fee"]
    quotes = analysis.quotes
    found = arb.summarize(
        arb.parity_violations(quotes, analysis.terms, fee_per_contract=fee),
        arb.vertical_violations(quotes, fee_per_contract=fee),
        arb.butterfly_violations(quotes, fee_per_contract=fee),
        arb.calendar_violations(analysis.fits),
    )

    st.markdown(f"**Checked at ${fee:.2f} per contract**")
    if found.empty:
        st.success(
            "No violations. That is the expected result on a liquid chain, and it is "
            "also a check on the data: a chain riddled with apparent arbitrage usually "
            "means stale quotes rather than free money."
        )
    else:
        show(found, {"edge": "{:.3f}"})
        st.caption(
            "An edge here is the locked-in profit per spread at the quoted sizes, before "
            "slippage, margin and the near-certainty that the quote is stale."
        )

    with st.expander("What is being checked"):
        st.markdown(
            "- **Put-call parity** — C − P must equal D(F − K). European only.\n"
            "- **Vertical spreads** — a lower-strike call cannot cost less than a "
            "higher-strike one.\n"
            "- **Butterflies** — option prices must be convex in strike, so a "
            "butterfly cannot be worth less than nothing.\n"
            "- **Calendars** — total implied variance cannot fall as maturity rises, "
            "compared at equal moneyness on the fitted smiles."
        )


# -------------------------------------------------------------------------- main


def main() -> None:
    settings = sidebar()

    try:
        if settings["source"].startswith("Live"):
            frame, spot, asof = load_live(settings["symbol"])
            label = f"{settings['symbol']} — live, 15-minute delayed"
        else:
            frame, spot, asof = load_bundled(str(settings["snapshot"]))
            label = f"{settings['snapshot'].name} — bundled snapshot"
    except Exception as error:  # noqa: BLE001 - surfaced to the user, not swallowed
        st.error(f"Could not load a chain: {error}")
        st.stop()

    with st.spinner("Cleaning quotes, implying forwards, solving volatilities…"):
        analysis = run_analysis(frame, spot, asof, settings["max_rel_spread"])

    st.title("Surface Scanner")
    a, b, c = st.columns(3)
    a.metric("Underlying", f"{spot:,.2f}")
    b.metric("Session", asof.isoformat())
    c.metric("Quotes solved", f"{len(analysis.quotes):,}")
    st.caption(label)

    with st.expander("How many quotes survived cleaning"):
        show(pd.DataFrame(analysis.funnel, columns=["stage", "quotes"]))

    scan, inspect, arbitrage = st.tabs(["Scan", "Inspect", "Arbitrage"])
    with scan:
        tab_scan(analysis, settings)
    with inspect:
        tab_inspect(analysis, settings)
    with arbitrage:
        tab_arbitrage(analysis, settings)


if __name__ == "__main__":
    main()
