"""The model half of the Stock options page: rates, the analysis, and its views.

Three tabs sit beside the market chain. Model vs market compares each quote
with the price its expiry's smile implies. Greeks gives sensitivities at each
contract's own American implied volatility. Smile draws the curve and the
quotes it was fitted to, with Cboe's own volatilities alongside.

Three kinds of number appear and are never mixed up: market quotes (bid, ask,
mid), model estimates (shaded grey), and Cboe's published figures, which are a
third party's model and are labelled as such. No contract is called cheap or
rich. A gap is a measurement against a model, reported with the bid-ask spread
it has to clear.
"""

from __future__ import annotations

import math
import os
from datetime import UTC, date, datetime

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from components import text

from optpricing import dividends as divs
from optpricing.equity_analysis import ExpiryAnalysis, analyze_expiry
from optpricing.providers import fred
from optpricing.providers.cboe import ChainSnapshot

#: Used only when FRED cannot be reached or no key is configured, and always
#: announced on the page. It is the level SPX's own put-call parity implied for
#: one- to twelve-month expiries in November 2026: 4.40% to 4.92%, in
#: analysis/results/implied_terms.csv.
FALLBACK_RATE = 0.045

#: Model columns are shaded grey so they never read as quotes.
MODEL_TINT = "background-color: rgba(128, 128, 128, 0.12)"


# ------------------------------------------------------------------------- rates


def _fred_key() -> str | None:
    """The FRED key from the environment or Streamlit secrets, never from code."""
    if os.environ.get("OPTPRICING_OFFLINE"):
        return None
    key = os.environ.get("FRED_API_KEY")
    if key:
        return key
    try:
        return st.secrets.get("FRED_API_KEY")
    except Exception:  # no secrets file at all
        return None


def _flat(asof: date) -> tuple:
    curve = fred.flat_curve(FALLBACK_RATE, asof, source="flat assumption")
    return (curve.asof.isoformat(), curve.tenors, curve.rates, curve.source)


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def _curve_fields(asof_iso: str | None) -> tuple[tuple, str | None]:
    """The curve as plain values, plus why it is a fallback if it is one.

    Cached for six hours: FRED publishes once a day, and its limit is per key.
    The key never appears in the arguments, so it never enters the cache.
    """
    asof = date.fromisoformat(asof_iso) if asof_iso else datetime.now(UTC).date()
    key = _fred_key()
    if not key:
        return _flat(asof), "no FRED key is configured"
    try:
        curve = fred.fetch_curve(key, asof=date.fromisoformat(asof_iso) if asof_iso else None)
    except fred.FREDError as error:  # its message never contains the key
        return _flat(asof), f"FRED could not be reached ({error})"
    return (curve.asof.isoformat(), curve.tenors, curve.rates, curve.source), None


def load_curve(asof: date | None) -> tuple[fred.YieldCurve, str | None]:
    fields, problem = _curve_fields(asof.isoformat() if asof else None)
    asof_iso, tenors, rates, source = fields
    return fred.YieldCurve(date.fromisoformat(asof_iso), tenors, rates, source), problem


# ---------------------------------------------------------------------- analysis


@st.cache_data(show_spinner=False, max_entries=64)
def _analysis(
    _snap: ChainSnapshot, snap_id: str, expiry: str, strikes: tuple, curve_fields: tuple
) -> ExpiryAnalysis:
    """Cached per snapshot, expiry, strike window and curve. `_snap` is not hashed."""
    asof_iso, tenors, rates, source = curve_fields
    curve = fred.YieldCurve(date.fromisoformat(asof_iso), tenors, rates, source)
    return analyze_expiry(_snap, expiry, curve, strikes=list(strikes))


def analyze(
    snap: ChainSnapshot, snap_id: str, expiry: str, strikes: list[float], historical: bool
) -> tuple[ExpiryAnalysis, str | None]:
    """Run (or recall) the analysis. A saved snapshot uses its own day's rates."""
    curve, problem = load_curve(snap.trading_date if historical else None)
    fields = (curve.asof.isoformat(), curve.tenors, curve.rates, curve.source)
    return _analysis(snap, snap_id, expiry, tuple(sorted(strikes)), fields), problem


# ------------------------------------------------------------------- assumptions


def _day(when: date) -> str:
    return f"{when:%b} {when.day}, {when.year}"


def assumptions(analysis: ExpiryAnalysis, curve_problem: str | None) -> None:
    """Where the rate and dividends came from, said once above the model tabs."""
    terms = analysis.terms
    treasury = analysis.treasury or terms
    if curve_problem:
        st.warning(
            f"The Treasury reference is a flat {FALLBACK_RATE:.2%} because {curve_problem}. "
            "That is the level SPX's own put-call parity implied in November 2026."
        )
        treasury_line = f"a flat {treasury.rate:.2%} assumption"
    else:
        treasury_line = f"{treasury.rate:.2%} from the FRED curve for {_day(treasury.curve_asof)}"

    if analysis.implied_rate is not None:
        rate_line = (
            f"Rate: {terms.rate:.2%}, the carry this expiry's own options imply: the rate at "
            f"which the {analysis.carry_pairs} call-put pairs nearest the forward agree on "
            f"volatility. The Treasury rate is {treasury_line}; at that rate those calls and "
            f"puts would disagree by {analysis.parity_gap_treasury:+.2f} vol pts."
        )
        if terms.rate < 0:
            rate_line += (
                " A carry below zero is what a stock that is expensive to borrow looks like: "
                "the cost of shorting it is priced into its puts."
            )
    else:
        rate_line = (
            f"Rate: {treasury_line}, continuously compounded to this expiry. Too few "
            "near-the-money call-put pairs to imply the options' own carry."
        )
    paid = divs.payments(terms.symbol, terms.asof, date.fromisoformat(terms.expiry))
    if not paid:
        div_line = "Dividends before expiry: none."
    else:
        listed = ", ".join(
            f"${p.amount:.2f} on {_day(p.ex_date)}{' (projected)' if p.projected else ''}"
            for p in paid
        )
        div_line = f"Dividends before expiry: {listed}."
    if terms.dividends_stale:
        st.warning(
            f"{terms.symbol}'s dividend record has not been checked for over four "
            "months. Its schedule may be out of date."
        )
    st.caption(
        f"{rate_line} {div_line} Forward ${terms.forward:,.2f}. Every model figure "
        "uses an American-exercise lattice with these inputs."
    )


# ------------------------------------------------------------------------ tables


def _render(
    frame: pd.DataFrame, formats: dict[str, str], model_columns: set[str], helps: dict[str, str]
) -> None:
    """Format to text (so missing values read n/a), shade model columns grey."""
    shown = pd.DataFrame({
        column: [text(v, formats[column]) if column in formats else v for v in frame[column]]
        for column in frame.columns
    })

    def shade(column: pd.Series) -> list[str]:
        return [MODEL_TINT if column.name in model_columns else ""] * len(column)

    st.dataframe(
        shown.style.apply(shade, axis=0),
        width="stretch",
        hide_index=True,
        height=min(38 + 35 * len(shown), 560),
        column_config={
            column: st.column_config.TextColumn(help=helps.get(column)) for column in shown.columns
        },
    )


_MARKET_HELP = "Market: Cboe's quote."
_MODEL_HELP = "Model: computed here. Shaded grey."
_CBOE_HELP = "Cboe's published figure: a third party's model, shown as a benchmark."


def model_vs_market(analysis: ExpiryAnalysis) -> None:
    table = analysis.table
    if table.empty:
        st.info("No two-sided quotes in this strike window.")
        return

    a, b, c = st.columns(3)
    gap = analysis.cboe_gap_points
    a.metric(
        "Our IV vs Cboe's",
        "n/a" if gap is None else f"{gap:.2f} vol pts",
        help="Median absolute difference, out of the money. Cboe computes the same "
        "quantity independently, so close agreement checks both.",
    )
    solved = int(table["model_iv"].notna().sum())
    b.metric("Contracts with an implied vol", f"{solved} of {len(table)}")
    if analysis.priced:
        outside = int(table["outside_spread"].fillna(False).sum())
        c.metric(
            "Quotes off the smile by more than their spread",
            f"{outside} of {int(table['outside_spread'].notna().sum())}",
        )
    else:
        c.metric("Priced off the smile", "No")

    if analysis.priced:
        st.caption(f"Smile fit: {analysis.fit_note}.")
    else:
        st.info(
            f"Contracts are not priced off a smile for this expiry: {analysis.fit_note}. "
            "Implied vols and Greeks below are still each contract's own."
        )

    columns = {
        "strike": "Strike", "kind": "Type", "bid": "Bid", "ask": "Ask", "mid": "Mid",
        "model_iv": "Model IV", "cboe_iv": "Cboe IV",
    }
    if analysis.priced:
        columns |= {
            "fitted_iv": "Smile IV", "model_price": "Model price", "gap": "Gap ($)",
            "gap_pct": "Gap (%)", "gap_in_spreads": "Gap / half-spread",
        }
    columns |= {"iv_note": "Note"}
    frame = table[list(columns)].rename(columns=columns)
    formats = {
        "Strike": "{:,.2f}", "Bid": "{:,.2f}", "Ask": "{:,.2f}", "Mid": "{:,.3f}",
        "Model IV": "{:.1%}", "Cboe IV": "{:.1%}", "Smile IV": "{:.1%}",
        "Model price": "{:,.3f}", "Gap ($)": "{:+,.3f}", "Gap (%)": "{:+.1%}",
        "Gap / half-spread": "{:+.2f}",
    }
    model = {"Model IV", "Smile IV", "Model price", "Gap ($)", "Gap (%)", "Gap / half-spread"}
    helps = {name: _MARKET_HELP for name in ("Bid", "Ask", "Mid")}
    helps |= {name: _MODEL_HELP for name in model}
    helps["Cboe IV"] = _CBOE_HELP
    helps["Gap / half-spread"] = (
        "Mid minus model price, in units of half the bid-ask spread. Between -1 and +1 "
        "the model price is inside the market's own quote."
    )
    _render(frame, formats, model, helps)
    st.caption(
        "Grey columns are model estimates. Gap is mid minus model price; a gap inside "
        "half the spread is within the market's own uncertainty. A gap is a measurement "
        "against a model, not a signal: the model is fitted to these same quotes and is "
        "imperfect, and a stale quote looks exactly like a mispriced one."
    )


def greeks_view(analysis: ExpiryAnalysis) -> None:
    table = analysis.table
    if table["model_iv"].notna().sum() == 0:
        st.info("No contract in this window has an implied vol, so there are no Greeks to show.")
        return
    columns = {
        "strike": "Strike", "kind": "Type", "model_iv": "Model IV", "delta": "Delta",
        "cboe_delta": "Cboe delta", "gamma": "Gamma", "theta_day": "Theta ($/day)",
        "vega_pt": "Vega ($/vol pt)", "rho_pt": "Rho ($/rate pt)",
    }
    frame = table[table["model_iv"].notna()][list(columns)].rename(columns=columns)
    formats = {
        "Strike": "{:,.2f}", "Model IV": "{:.1%}", "Delta": "{:+.3f}", "Cboe delta": "{:+.3f}",
        "Gamma": "{:.4f}", "Theta ($/day)": "{:+.4f}", "Vega ($/vol pt)": "{:.4f}",
        "Rho ($/rate pt)": "{:+.4f}",
    }
    model = {"Model IV", "Delta", "Gamma", "Theta ($/day)", "Vega ($/vol pt)", "Rho ($/rate pt)"}
    helps = {name: _MODEL_HELP for name in model}
    helps["Cboe delta"] = _CBOE_HELP
    helps["Theta ($/day)"] = "Change in price over one calendar day, all else fixed."
    helps["Vega ($/vol pt)"] = "Change in price for a one-point rise in implied vol."
    helps["Rho ($/rate pt)"] = "Change in price for a one-point rise in the interest rate."
    _render(frame, formats, model, helps)
    st.caption(
        "Per share, at each contract's own American implied vol, from finite differences "
        "on the lattice. Multiply by 100 for one contract. Cboe's delta is shown as an "
        "independent check."
    )


# ------------------------------------------------------------------------- smile

_OURS, _FITTED, _CBOE = "Our American IV", "Fitted smile", "Cboe IV"


def smile_chart(analysis: ExpiryAnalysis) -> None:
    table = analysis.table
    ours = table.dropna(subset=["model_iv"])
    if ours.empty:
        st.info("No implied vols to draw for this window.")
        return

    points = pd.concat([
        pd.DataFrame({"strike": ours["strike"], "iv": ours["model_iv"], "kind": ours["kind"],
                      "series": _OURS}),
        pd.DataFrame({"strike": table["strike"], "iv": table["cboe_iv"], "kind": table["kind"],
                      "series": _CBOE}).dropna(subset=["iv"]),
    ])
    layers = []
    domain = [_OURS, _CBOE]
    if analysis.fit is not None:
        grid = np.linspace(table["strike"].min(), table["strike"].max(), 200)
        curve = pd.DataFrame({
            "strike": grid,
            "iv": analysis.fit.implied_vol(np.log(grid / analysis.terms.forward)),
            "series": _FITTED,
        })
        domain = [_OURS, _FITTED, _CBOE]
        line = alt.Chart(curve).mark_line(strokeWidth=2.5, strokeDash=[] if analysis.priced else [6, 4])
        layers.append(line.encode(
            x="strike:Q", y="iv:Q", color=alt.Color("series:N", scale=alt.Scale(domain=domain)),
        ))

    color = alt.Color("series:N", scale=alt.Scale(domain=domain), legend=alt.Legend(title=None, orient="top"))
    tooltip = [
        alt.Tooltip("series:N", title="Source"), alt.Tooltip("kind:N", title="Type"),
        alt.Tooltip("strike:Q", format=",.2f", title="Strike"), alt.Tooltip("iv:Q", format=".1%", title="IV"),
    ]
    layers.append(alt.Chart(points[points.series == _CBOE]).mark_point(size=40, filled=False, strokeWidth=1.2)
                  .encode(x="strike:Q", y="iv:Q", color=color, tooltip=tooltip))
    layers.append(alt.Chart(points[points.series == _OURS]).mark_circle(size=55, opacity=0.9)
                  .encode(x="strike:Q", y="iv:Q", color=color, tooltip=tooltip))
    forward = pd.DataFrame({"strike": [analysis.terms.forward], "label": [f"forward ${analysis.terms.forward:,.2f}"]})
    layers.append(alt.Chart(forward).mark_rule(strokeDash=[2, 3], opacity=0.6).encode(x="strike:Q"))
    layers.append(alt.Chart(forward).mark_text(align="left", dx=4, dy=-6, opacity=0.8)
                  .encode(x="strike:Q", y=alt.value(0), text="label:N"))

    chart = alt.layer(*layers).encode(
        x=alt.X("strike:Q", title="Strike ($)", scale=alt.Scale(zero=False)),
        y=alt.Y("iv:Q", title="Implied volatility", axis=alt.Axis(format=".0%"), scale=alt.Scale(zero=False)),
    ).properties(height=380)
    st.altair_chart(chart, width="stretch")
    note = (
        "The fitted smile is solid where it passed the stock-chain test and contracts are "
        "priced off it; dashed where it did not, and is drawn for reference only."
        if analysis.fit is not None else "Too few tight quotes to fit a smile for this expiry."
    )
    st.caption(
        f"{note} Filled points are this project's American implied vols; hollow points "
        "are Cboe's, computed independently."
    )


def finite(value: float) -> bool:
    return value is not None and math.isfinite(value)
