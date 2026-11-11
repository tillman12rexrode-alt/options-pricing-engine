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

from optpricing import greeks, pricers
from optpricing.contracts import Option
from optpricing.market import ExpiryTerms
from optpricing.pricers import Pricer

__all__ = [
    "METHODS",
    "IVResult",
    "american_bounds",
    "implied_vol",
    "no_arbitrage_bounds",
    "solve_chain",
]

METHODS = ("brent", "newton")

VOL_FLOOR = 1e-6
VOL_CEILING = 5.0

#: A lattice cannot be built at an arbitrarily small volatility - Leisen-Reimer's
#: inversion saturates and the tree collapses - so the bracket starts higher when
#: the pricer is numerical. Nothing is lost: a quote implying less than a tenth of
#: a volatility point is sitting on its intrinsic value, and is reported as
#: unsolvable rather than as a precise number nobody should believe.
LATTICE_VOL_FLOOR = 1e-3

#: How close a quote may sit to a bound before its volatility is called
#: undetermined rather than solved. Loose enough to cover a lattice's own
#: discretisation error, which is around 1e-4 of spot at the default step count.
_BOUNDARY_TOLERANCE = 1e-6


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


def american_bounds(opt: Option) -> tuple[float, float]:
    """Model-free bounds on an American option, needing no forward or discount.

    The European bounds above are derived from put-call parity, which holds as
    an equality only for European exercise - so on an equity chain they are not
    merely imprecise, they are the wrong constraint. These hold for any option
    that may be exercised early, under any model:

    - a call is worth at least its intrinsic value and at most the stock;
    - a put is worth at least its intrinsic value and at most the strike,
      undiscounted, because the holder may take K at any moment.

    The European floor is kept as well, computed on the spot net of dividends,
    since an American option is worth at least its European twin.
    """
    spot_net = opt.S - opt.pv_dividends()
    disc_r = math.exp(-opt.r * opt.T)
    disc_q = math.exp(-opt.q * opt.T)

    if opt.is_call:
        european_floor = max(spot_net * disc_q - opt.K * disc_r, 0.0)
        return max(opt.intrinsic(), european_floor), opt.S

    european_floor = max(opt.K * disc_r - spot_net * disc_q, 0.0)
    return max(opt.intrinsic(), european_floor), opt.K


def _price_at(opt: Option, sigma: float, price: Pricer) -> float:
    return price(replace(opt, sigma=sigma))


def _floor_ladder(opt: Option) -> tuple[float, ...]:
    """Candidate lower ends for the bracket, lowest first.

    A closed form evaluates anywhere, so the first rung is always taken. A
    lattice does not: at a small enough sigma * sqrt(T) the tree collapses, and
    how small depends on the contract - a deep in-the-money short-dated call
    gives up far earlier than an at-the-money one. Rather than guess a floor
    that works for every contract, and lose resolution on the ones that would
    have tolerated a lower one, the solver walks up until the pricer answers.
    """
    if pricers.has_closed_form(opt):
        return (VOL_FLOOR,)
    return (LATTICE_VOL_FLOOR, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1)


def _lowest_workable(opt: Option, price: Pricer) -> tuple[float, float] | None:
    """The lowest bracket end this pricer can actually evaluate, and its price."""
    for floor in _floor_ladder(opt):
        try:
            return floor, _price_at(opt, floor, price)
        except ValueError:
            continue
    return None


def _brent(opt: Option, target: float, price: Pricer) -> IVResult:
    bottom = _lowest_workable(opt, price)
    if bottom is None:
        return IVResult(float("nan"), 0, False, "pricer could not evaluate any lower bracket")

    floor, floor_price = bottom
    low = floor_price - target
    high = _price_at(opt, VOL_CEILING, price) - target

    # A flat bottom, not a root. If the price at the lowest volatility the
    # pricer can reach already equals the target, then a whole range of
    # volatilities reprices this quote and brentq would hand back the bracket
    # endpoint as though it meant something. Saying so is the honest answer.
    if abs(low) <= _BOUNDARY_TOLERANCE:
        return IVResult(float("nan"), 0, False, "price is flat in volatility here")

    if low * high > 0:
        return IVResult(float("nan"), 0, False, "no sign change in bracket")

    sigma, result = brentq(
        lambda s: _price_at(opt, s, price) - target,
        floor,
        VOL_CEILING,
        xtol=1e-10,
        full_output=True,
    )
    return IVResult(float(sigma), int(result.iterations), bool(result.converged))


def _newton(
    opt: Option,
    target: float,
    price: Pricer,
    *,
    start: float = 0.20,
    max_iter: int = 50,
) -> IVResult:
    # Tolerance has to follow the pricer. A lattice carries its own
    # discretisation error, so demanding the closed form's 1e-10 agreement on a
    # numerical price asks the iteration to chase noise it cannot remove.
    tolerance = 1e-10 if pricers.has_closed_form(opt) else 1e-7
    floor = _floor_ladder(opt)[0]

    sigma = start
    for step in range(1, max_iter + 1):
        bumped = replace(opt, sigma=sigma)
        try:
            gap = price(bumped) - target
        except ValueError:
            # The lattice collapsed at the volatility Newton wandered to. Brent
            # recovers from this by walking its bracket; Newton has no bracket.
            return IVResult(float("nan"), step, False, "pricer failed at the iterate")
        if abs(gap) < tolerance:
            return IVResult(sigma, step, True)

        # Analytic where one exists, a bumped lattice otherwise - which is
        # noisier, and is why Newton gives up more often on American contracts.
        vega = greeks.estimate(bumped, "vega")
        if abs(vega) < 1e-10:
            return IVResult(float("nan"), step, False, "vega collapsed")

        sigma -= gap / vega
        if not floor < sigma < VOL_CEILING:
            return IVResult(float("nan"), step, False, "left the bracket")

    return IVResult(float("nan"), max_iter, False, "iteration limit")


def implied_vol(
    target: float, opt: Option, method: str = "brent", pricer: Pricer | None = None
) -> IVResult:
    """Volatility that reprices `opt` at `target`.

    The engine follows the contract: Black-Scholes for European exercise with no
    dividend schedule, a Leisen-Reimer lattice otherwise. Inverting the closed
    form on an American quote would return a volatility biased low by the
    early-exercise premium, worst exactly where that premium is largest - in
    the money, on puts - so the choice is not a refinement.

    The cost is real: each candidate volatility is a full lattice solve rather
    than one evaluation of a formula, so an American inversion runs on the order
    of a hundred times slower than a European one.
    """
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")
    if target <= 0 or opt.T <= 0:
        return IVResult(float("nan"), 0, False, "degenerate input")

    price = pricers.for_option(opt) if pricer is None else pricer

    if opt.style == "american":
        low, high = american_bounds(opt)
        if not low - 1e-9 <= target <= high + 1e-9:
            return IVResult(float("nan"), 0, False, "outside American no-arbitrage bounds")

        # Sitting on the lower bound means immediate exercise is optimal, which
        # a deep in-the-money American put routinely is. Such a contract is
        # worth its intrinsic value at every low volatility, so no volatility
        # is implied by the quote - there is a range of them, and reporting any
        # single one would invent precision the market never expressed. This
        # has no European analogue; it is a property of early exercise.
        if target <= low + _BOUNDARY_TOLERANCE:
            return IVResult(
                float("nan"), 0, False, "at the early-exercise boundary; volatility undetermined"
            )

    return _brent(opt, target, price) if method == "brent" else _newton(opt, target, price)


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
