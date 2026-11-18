"""Model against market for one expiry of a stock's option chain.

A model price only means something if it is not the market's own price in
disguise. Solve a contract's implied volatility from its mid, price it back at
that volatility, and you recover the mid exactly: the comparison is circular.
So the model price here comes from a smile fitted through the contract's
neighbours, the same idea the index scanner rests on. A quote is compared with
what the rest of its expiry says it should cost.

Per contract this computes, from the market mid:

- the American implied volatility, by inverting the lattice with Treasury
  rates and the dividend schedule, not Black-Scholes with a yield;
- the volatility the fitted smile assigns that strike, and the American price
  at that volatility;
- the gap between the two, in dollars, as a share of the model price, and as a
  multiple of half the bid-ask spread, since a gap inside the spread is not a
  disagreement anyone can act on;
- the five Greeks at the contract's own implied volatility, in trader units;
- Cboe's published implied volatility and delta, carried alongside as an
  independent third-party benchmark.

The smile is fitted through out-of-the-money quotes only. In-the-money options
on stocks are thinly traded, and an in-the-money American put can sit on its
early-exercise boundary where no volatility is implied at all. Out-of-the-money
quotes carry the information about volatility; the rest are priced off the curve
they define.

Nothing here labels a contract cheap or rich. The gap is a measurement against
a model the project itself shows to be imperfect, so it is reported as a number
with its uncertainty, and the reader decides what it is worth.
"""

from __future__ import annotations

import math
from collections.abc import Collection
from dataclasses import dataclass, replace
from datetime import date

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from optpricing import greeks, pricers
from optpricing.implied_vol import american_bounds, implied_vol
from optpricing.providers.cboe import ChainSnapshot
from optpricing.providers.fred import YieldCurve, flat_curve
from optpricing.smile import SliceFit, fit_slice
from optpricing.terms import EquityTerms, equity_terms

__all__ = ["ExpiryAnalysis", "analyze_expiry", "implied_carry", "term_structure"]

#: Out-of-the-money quotes wider than this, as a share of mid, are left out of
#: the smile fit. Stock chains are wider than SPX's, so this is looser than the
#: index scanner's 20%; the reliability gate in smile.fit_slice still decides.
FIT_MAX_REL_SPREAD = 0.35

#: Below this many usable quotes a five-parameter curve is not worth fitting.
MIN_FIT_QUOTES = 8

#: A smile is trusted for pricing when at least this share of the quotes it was
#: fitted to sit within their own half-spread of it, measured in vol points.
#:
#: The index scanner judges a fit by a fixed RMSE, which suits SPX's tight
#: quotes. Stock chains are wider and noisier, and one stale quote can move an
#: RMSE a long way, so here the test is relative: does the curve agree with the
#: quotes as well as the quotes agree with themselves? On 16 Nov 2026 this
#: passed AAPL (96%) and TSLA (84%) and refused NVDA (54%), whose residuals
#: also cluster by region.
MIN_SHARE_INSIDE_SPREAD = 0.75

#: Nor when residuals in any moneyness band are biased by more than the larger
#: of this and the typical half-spread, in vol points: a wrong shape, not noise.
MIN_BIAS_ALLOWANCE = 0.60

#: Call-put pairs nearest the forward used to imply the market's own carry.
CARRY_PAIRS = 6

#: Fewer pairs than this and the implied carry is not trusted.
MIN_CARRY_PAIRS = 3

#: Search range for the implied carry rate. Wide enough for hard-to-borrow
#: names, whose carry can sit far below the Treasury rate.
CARRY_BRACKET = (-0.05, 0.15)


@dataclass(frozen=True)
class ExpiryAnalysis:
    terms: EquityTerms
    table: pd.DataFrame
    fit: SliceFit | None
    fit_note: str
    funnel: list[tuple[str, int]]
    #: Whether the smile passed the stock-chain test and contracts were priced off it.
    priced: bool = False
    #: The same expiry's terms at the curve supplied, kept for comparison.
    treasury: EquityTerms | None = None
    #: The rate at which near-the-money calls and puts imply the same vol, or None.
    implied_rate: float | None = None
    #: Pairs the implied rate was solved from.
    carry_pairs: int = 0
    #: Median call IV minus put IV at the same strike, in vol points, at each rate.
    parity_gap_treasury: float = float("nan")
    parity_gap_used: float = float("nan")

    @property
    def treasury_rate(self) -> float:
        return self.terms.rate if self.treasury is None else self.treasury.rate

    @property
    def rate_source(self) -> str:
        return "options-implied carry" if self.implied_rate is not None else "Treasury curve"

    @property
    def cboe_gap_points(self) -> float | None:
        """Median |our IV - Cboe's IV| in volatility points, out of the money.

        Cboe computes the same quantity independently, with its own rates,
        dividends and model, so close agreement is evidence both are right.
        """
        both = self.table[self.table["otm"]].dropna(subset=["model_iv", "cboe_iv"])
        if both.empty:
            return None
        return float((both["model_iv"] - both["cboe_iv"]).abs().median() * 100)


def _two_sided(quotes: pd.DataFrame) -> pd.Series:
    bid = pd.to_numeric(quotes["bid"], errors="coerce").fillna(0)
    ask = pd.to_numeric(quotes["ask"], errors="coerce").fillna(0)
    return (bid > 0) & (ask > bid)


def _pairs(snap: ChainSnapshot, expiry: str, forward: float) -> pd.DataFrame:
    """Strikes with a two-sided call and put, nearest the forward first."""
    quotes = snap.quotes[(snap.quotes["expiry"] == expiry) & _two_sided(snap.quotes)]
    mids = quotes.assign(mid=(quotes["bid"] + quotes["ask"]) / 2.0)
    paired = mids.pivot_table(index="strike", columns="kind", values="mid").dropna()
    if not {"call", "put"} <= set(paired.columns):
        return pd.DataFrame(columns=["call", "put"])
    order = np.argsort(np.abs(paired.index.to_numpy() - forward))
    return paired.iloc[order[:CARRY_PAIRS]]


def _parity_gap(snap: ChainSnapshot, expiry: str, rate: float, pairs: pd.DataFrame, policies) -> tuple[float, int]:
    """Median call IV minus put IV across the pairs at a flat `rate`, and pairs used."""
    terms = equity_terms(
        snap.symbol, snap.spot, snap.trading_date, expiry,
        flat_curve(rate, snap.trading_date), policies=policies,
    )
    gaps = []
    for strike, row in pairs.iterrows():
        call = implied_vol(float(row["call"]), terms.option(float(strike), "call"))
        put = implied_vol(float(row["put"]), terms.option(float(strike), "put"))
        if call.converged and put.converged:
            gaps.append(call.sigma - put.sigma)
    if len(gaps) < MIN_CARRY_PAIRS:
        return float("nan"), len(gaps)
    return float(np.median(gaps)) * 100.0, len(gaps)


def implied_carry(
    snap: ChainSnapshot, expiry: str, forward: float, policies=None
) -> tuple[float | None, int]:
    """The rate at which near-the-money calls and puts imply the same volatility.

    For index options the library reads the rate off put-call parity, which is
    an equality for European exercise. For American options it is not, so
    this asks the same question in a form that survives early exercise: on the
    lattice, with the dividend schedule, what rate makes a call and a put at
    the same strike agree on volatility?

    It is rarely the Treasury rate. From the six pairs nearest the forward on
    16 Nov 2026: AAPL December 4.34%, AAPL January 4.59%, NVDA December
    5.08%, TSLA December 5.23%, and RCAT December -2.89%, a negative carry of
    the kind a hard-to-borrow stock shows. The options market prices its own
    carry per underlying and per expiry, folding in borrow costs and any
    dividend expectations that differ from the schedule supplied. Pricing at
    the Treasury rate instead splits the smile into a call branch and a put
    branch, which is how this was found.

    Returns (rate, pairs used), or (None, pairs) when there are too few pairs
    or no rate in `CARRY_BRACKET` closes the gap.
    """
    pairs = _pairs(snap, expiry, forward)
    if len(pairs) < MIN_CARRY_PAIRS:
        return None, len(pairs)

    low, high = CARRY_BRACKET
    try:
        g_low, n = _parity_gap(snap, expiry, low, pairs, policies)
        g_high, _ = _parity_gap(snap, expiry, high, pairs, policies)
    except ValueError:
        return None, 0
    if not (math.isfinite(g_low) and math.isfinite(g_high)) or g_low * g_high > 0:
        return None, n

    def gap(rate: float) -> float:
        value, _ = _parity_gap(snap, expiry, rate, pairs, policies)
        if not math.isfinite(value):
            # Too few pairs solve at this rate. Treating that as "no gap"
            # would invent a root, so give up instead.
            raise ValueError("pairs stopped solving inside the bracket")
        return value

    try:
        rate = brentq(gap, low, high, xtol=1e-5)
    except ValueError:
        return None, n
    return float(rate), n


def analyze_expiry(
    snap: ChainSnapshot,
    expiry: str,
    curve: YieldCurve,
    *,
    strikes: Collection[float] | None = None,
    policies=None,
    carry: str = "implied",
) -> ExpiryAnalysis:
    """Everything the Model vs market view shows, for one expiry.

    `strikes` restricts the work to the strikes on screen. Each contract costs
    an implied-vol solve and five Greeks on a lattice, so analysing a whole
    NVDA expiry takes several seconds where the visible window takes one.
    """
    treasury = equity_terms(
        snap.symbol, snap.spot, snap.trading_date, expiry, curve, policies=policies
    )
    implied, n_pairs = (None, 0)
    if carry == "implied":
        implied, n_pairs = implied_carry(snap, expiry, treasury.forward, policies)
    terms = treasury if implied is None else equity_terms(
        snap.symbol, snap.spot, snap.trading_date, expiry,
        flat_curve(implied, snap.trading_date, source="options-implied carry"),
        policies=policies,
    )
    pairs = _pairs(snap, expiry, terms.forward)
    gap_treasury, _ = _parity_gap(snap, expiry, treasury.rate, pairs, policies)
    gap_used = gap_treasury if implied is None else _parity_gap(snap, expiry, implied, pairs, policies)[0]

    quotes = snap.quotes[snap.quotes["expiry"] == expiry]
    funnel = [("listed on this expiry", len(quotes))]
    if strikes is not None:
        quotes = quotes[quotes["strike"].isin(set(strikes))]
        funnel.append(("in the strike window", len(quotes)))

    usable = quotes[_two_sided(quotes)].copy()
    funnel.append(("with a two-sided market", len(usable)))

    usable["mid"] = (usable["bid"] + usable["ask"]) / 2.0
    usable["half_spread"] = (usable["ask"] - usable["bid"]) / 2.0
    usable["rel_spread"] = 2.0 * usable["half_spread"] / usable["mid"]
    usable["log_moneyness"] = np.log(usable["strike"] / terms.forward)
    usable["otm"] = np.where(
        usable["kind"] == "call",
        usable["strike"] >= terms.forward,
        usable["strike"] < terms.forward,
    )

    rows = [_solve(record, terms) for record in usable.sort_values(["strike", "kind"]).to_dict("records")]
    table = pd.DataFrame(rows)
    if table.empty:
        table = usable.assign(model_iv=np.nan, iv_note="")
    funnel.append(("with an American implied vol", int(table["model_iv"].notna().sum())))

    _add_greeks(table, terms)
    table["spread_vol_pts"] = table["half_spread"] / table["vega_pt"].where(table["vega_pt"] > 1e-8)

    fit, fit_note, reliable = _fit(table, terms)
    _price_off_the_smile(table, terms, fit if reliable else None)

    return ExpiryAnalysis(
        terms, table.reset_index(drop=True), fit, fit_note, funnel, reliable,
        treasury=treasury, implied_rate=implied, carry_pairs=n_pairs,
        parity_gap_treasury=gap_treasury, parity_gap_used=gap_used,
    )


#: What the solver's reasons mean to someone reading the table.
_PLAIN = {
    "no sign change in bracket": "time value too small to imply a volatility",
    "degenerate input": "no price to solve from",
    "pricer could not evaluate any lower bracket": "time value too small to imply a volatility",
}


def _solve(record: dict, terms: EquityTerms) -> dict:
    """The contract's American implied vol, or a plain reason there is none.

    The no-arbitrage floor is checked against the quote before solving. A mid
    a few cents under the floor is the common case deep in the money, where
    a contract has no time value left and the quote straddles intrinsic; that
    is not an arbitrage, and it is not a failure, so it is said plainly. Only
    an ask below the floor, which could be bought and exercised for a profit,
    is called an arbitrage.
    """
    opt = terms.option(float(record["strike"]), str(record["kind"]))
    floor, _ = american_bounds(opt)
    record["model_iv"] = float("nan")
    if float(record["ask"]) < floor - 1e-9:
        record["iv_note"] = "ask below the no-arbitrage floor"
    elif float(record["mid"]) <= floor + 1e-9:
        record["iv_note"] = "no time value at the mid"
    else:
        solved = implied_vol(float(record["mid"]), opt)
        if solved.converged:
            record["model_iv"] = solved.sigma
            record["iv_note"] = ""
        else:
            record["iv_note"] = _PLAIN.get(solved.reason, solved.reason)
    return record


def _fit(table: pd.DataFrame, terms: EquityTerms) -> tuple[SliceFit | None, str, bool]:
    """Fit the smile through tight out-of-the-money quotes, and judge it relatively."""
    candidates = table[
        table["otm"]
        & table["model_iv"].notna()
        & table["spread_vol_pts"].notna()
        & (table["rel_spread"] <= FIT_MAX_REL_SPREAD)
    ]
    if len(candidates) < MIN_FIT_QUOTES:
        return None, (
            f"only {len(candidates)} out-of-the-money quotes are tight enough to fit a "
            f"smile through; {MIN_FIT_QUOTES} are needed"
        ), False

    k = candidates["log_moneyness"].to_numpy()
    fit = fit_slice(
        k,
        candidates["model_iv"].to_numpy(),
        terms.years,
        1.0 / np.maximum(candidates["rel_spread"].to_numpy(), 1e-3),
    )
    residual = np.abs(fit.implied_vol(k) - candidates["model_iv"].to_numpy()) * 100.0
    allowance = candidates["spread_vol_pts"].to_numpy()
    inside = float(np.mean(residual <= allowance))
    bias_limit = max(MIN_BIAS_ALLOWANCE, float(np.median(allowance)))

    if inside < MIN_SHARE_INSIDE_SPREAD:
        return fit, (
            f"only {inside:.0%} of the {len(candidates)} quotes it was fitted to sit within "
            f"their own spread of the curve; {MIN_SHARE_INSIDE_SPREAD:.0%} are needed"
        ), False
    if fit.worst_bucket_bias > bias_limit:
        return fit, (
            f"the curve is off by {fit.worst_bucket_bias:.2f} vol points in one region of "
            "strikes, more than the quotes' own spread explains"
        ), False
    return fit, (
        f"{inside:.0%} of the {len(candidates)} quotes it was fitted to sit within their "
        "own bid-ask spread of the curve"
    ), True


def _price_off_the_smile(table: pd.DataFrame, terms: EquityTerms, fit: SliceFit | None) -> None:
    nan = np.full(len(table), np.nan)
    if fit is None or not fit.reliable:
        for column in ("fitted_iv", "model_price", "gap", "gap_pct", "gap_in_spreads"):
            table[column] = nan
        table["outside_spread"] = pd.array([pd.NA] * len(table), dtype="boolean")
        return

    table["fitted_iv"] = fit.implied_vol(table["log_moneyness"].to_numpy())
    pricer = pricers.lattice()
    table["model_price"] = [
        pricer(terms.option(float(k), str(kind), sigma=float(vol)))
        for k, kind, vol in zip(table["strike"], table["kind"], table["fitted_iv"], strict=True)
    ]
    table["gap"] = table["mid"] - table["model_price"]
    table["gap_pct"] = table["gap"] / table["model_price"].where(table["model_price"] > 0)
    table["gap_in_spreads"] = table["gap"] / table["half_spread"]
    table["outside_spread"] = (table["gap"].abs() > table["half_spread"]).astype("boolean")


def _add_greeks(table: pd.DataFrame, terms: EquityTerms) -> None:
    """At each contract's own implied vol, in the units a trader reads.

    Delta and gamma per share; theta in dollars per calendar day; vega in
    dollars per volatility point; rho in dollars per percentage point of rate.
    All per share: multiply by the contract multiplier for one contract.
    """
    out = {name: [] for name in ("delta", "gamma", "theta_day", "vega_pt", "rho_pt")}
    for k, kind, vol in zip(table["strike"], table["kind"], table["model_iv"], strict=True):
        if not math.isfinite(vol):
            for values in out.values():
                values.append(float("nan"))
            continue
        g = greeks.estimate_all(replace(terms.option(float(k), str(kind)), sigma=float(vol)))
        out["delta"].append(g["delta"])
        out["gamma"].append(g["gamma"])
        out["theta_day"].append(g["theta"] / 365.0)
        out["vega_pt"].append(g["vega"] / 100.0)
        out["rho_pt"].append(g["rho"] / 100.0)
    for name, values in out.items():
        table[name] = values


def term_structure(
    snap: ChainSnapshot, curve: YieldCurve, *, policies=None, max_days: int | None = None
) -> pd.DataFrame:
    """At-the-money volatility and the move the options price in, expiry by expiry.

    For each expiry, the strike nearest the forward with a two-sided call and
    put. The straddle (call mid + put mid) is what the market charges for a
    move in either direction by expiry, the usual "expected move". For a
    lognormal stock it is about 0.8 of one standard deviation; it is reported
    as quoted, not rescaled.

    The at-the-money vol is the average of the call's and the put's American
    implied vols at that strike. Averaging the two cancels most of the carry
    effect that splits them (see `implied_carry`), so the Treasury curve is
    good enough here and no carry is solved per expiry.
    """
    rows = []
    for expiry in snap.expiries:
        days = (date.fromisoformat(expiry) - snap.trading_date).days
        if days <= 0 or (max_days is not None and days > max_days):
            continue
        try:
            terms = equity_terms(
                snap.symbol, snap.spot, snap.trading_date, expiry, curve, policies=policies
            )
        except ValueError:
            continue
        pairs = _pairs(snap, expiry, terms.forward)
        row = {"expiry": expiry, "days": days, "forward": terms.forward}
        if pairs.empty:
            rows.append(row)
            continue
        strike = float(pairs.index[0])
        call, put = float(pairs.iloc[0]["call"]), float(pairs.iloc[0]["put"])
        quotes = snap.quotes[(snap.quotes["expiry"] == expiry) & (snap.quotes["strike"] == strike)]
        cboe = pd.to_numeric(quotes["cboe_iv"], errors="coerce")
        vols = [implied_vol(price, terms.option(strike, kind)) for price, kind in ((call, "call"), (put, "put"))]
        solved = [v.sigma for v in vols if v.converged]
        row |= {
            "strike": strike,
            "straddle": call + put,
            "move_pct": (call + put) / snap.spot,
            "atm_iv": float(np.mean(solved)) if len(solved) == 2 else np.nan,
            "cboe_atm_iv": float(cboe.mean()) if cboe.notna().all() and len(cboe) == 2 else np.nan,
        }
        rows.append(row)
    columns = ["expiry", "days", "forward", "strike", "straddle", "move_pct", "atm_iv", "cboe_atm_iv"]
    return pd.DataFrame(rows, columns=columns)
