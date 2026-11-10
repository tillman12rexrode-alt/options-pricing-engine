"""Back out the volatility the market is charging, by inverting the pricer.

Two solvers, because they fail differently. Brent's method needs only a bracket
containing a sign change and cannot diverge, so it is the one to trust.
Newton-Raphson uses vega and converges in a handful of steps when it works, but
vega goes to zero far from the money, and dividing by a vanishing derivative
sends the iteration anywhere. Running both and reporting where Newton gives up
is a small numerical result in its own right.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from optpricing.contracts import Option
from optpricing.market import ExpiryTerms
from optpricing.models import black_scholes as bs

__all__ = ["METHODS", "IVResult", "implied_vol", "no_arbitrage_bounds", "solve_chain"]

METHODS = ("brent", "newton")

VOL_FLOOR = 1e-6
VOL_CEILING = 5.0


@dataclass(frozen=True)
class IVResult:
    """An implied volatility together with how hard it was to find."""

    sigma: float
    iterations: int
    converged: bool
    reason: str = ""


def no_arbitrage_bounds(opt: Option, terms: ExpiryTerms) -> tuple[float, float]:
    """Prices outside these cannot come from any volatility at all."""
    discount, forward = terms.discount, terms.forward
    if opt.is_call:
        return discount * max(forward - opt.K, 0.0), discount * forward
    return discount * max(opt.K - forward, 0.0), discount * opt.K


def _price_at(opt: Option, sigma: float) -> float:
    return bs.price(replace(opt, sigma=sigma))


def _brent(opt: Option, target: float) -> IVResult:
    low = _price_at(opt, VOL_FLOOR) - target
    high = _price_at(opt, VOL_CEILING) - target
    if low * high > 0:
        return IVResult(float("nan"), 0, False, "no sign change in bracket")

    sigma, result = brentq(
        lambda s: _price_at(opt, s) - target,
        VOL_FLOOR,
        VOL_CEILING,
        xtol=1e-10,
        full_output=True,
    )
    return IVResult(float(sigma), int(result.iterations), bool(result.converged))


def _newton(opt: Option, target: float, *, start: float = 0.20, max_iter: int = 50) -> IVResult:
    sigma = start
    for step in range(1, max_iter + 1):
        bumped = replace(opt, sigma=sigma)
        gap = bs.price(bumped) - target
        if abs(gap) < 1e-10:
            return IVResult(sigma, step, True)

        vega = bs.greeks(bumped)["vega"]
        if abs(vega) < 1e-10:
            return IVResult(float("nan"), step, False, "vega collapsed")

        sigma -= gap / vega
        if not VOL_FLOOR < sigma < VOL_CEILING:
            return IVResult(float("nan"), step, False, "left the bracket")

    return IVResult(float("nan"), max_iter, False, "iteration limit")


def implied_vol(target: float, opt: Option, method: str = "brent") -> IVResult:
    """Volatility that reprices `opt` at `target` under Black-Scholes."""
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")
    if target <= 0 or opt.T <= 0:
        return IVResult(float("nan"), 0, False, "degenerate input")

    return _brent(opt, target) if method == "brent" else _newton(opt, target)


def solve_chain(
    frame: pd.DataFrame,
    terms: dict[str, ExpiryTerms],
    spot: float,
    *,
    methods: tuple[str, ...] = METHODS,
) -> tuple[pd.DataFrame, list[tuple[str, int]]]:
    """Attach an implied volatility to every quote, by each requested method."""
    rows, dropped_bounds = [], 0

    for record in frame.to_dict("records"):
        term = terms.get(str(record["expiry"]))
        if term is None:
            continue

        opt = Option(
            S=spot,
            K=float(record["strike"]),
            T=term.years,
            r=term.rate,
            sigma=0.20,
            q=term.dividend_yield,
            kind=str(record["kind"]),
        )
        target = float(record["mid"])

        low, high = no_arbitrage_bounds(opt, term)
        if not low - 1e-9 <= target <= high + 1e-9:
            dropped_bounds += 1
            continue

        row = dict(record)
        row["years"] = term.years
        row["forward"] = term.forward
        row["log_moneyness"] = math.log(opt.K / term.forward)
        for method in methods:
            result = implied_vol(target, opt, method)
            row[f"iv_{method}"] = result.sigma
            row[f"iters_{method}"] = result.iterations
            row[f"ok_{method}"] = result.converged
        rows.append(row)

    solved = pd.DataFrame(rows)
    funnel = [("inside no-arbitrage bounds", len(solved))]
    if dropped_bounds:
        funnel.insert(0, ("violated no-arbitrage bounds", -dropped_bounds))

    primary = f"iv_{methods[0]}"
    if primary in solved:
        solved = solved[np.isfinite(solved[primary])]
        funnel.append((f"{methods[0]} solver converged", len(solved)))

    return solved.reset_index(drop=True), funnel
