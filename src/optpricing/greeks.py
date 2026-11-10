"""Greek estimators: finite difference, and Monte Carlo pathwise derivatives.

The analytic Greeks live in `optpricing.models.black_scholes.greeks` and are
the reference. This module holds the numerical estimators that have to be used
when no closed form exists, and that are cross-checked against it.
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np

from optpricing.contracts import Option
from optpricing.models import black_scholes as bs
from optpricing.models import monte_carlo as mc

__all__ = ["GREEKS", "bumped_delta", "default_step", "finite_difference", "pathwise_delta"]

GREEKS = ("delta", "gamma", "vega", "theta", "rho")

#: The contract field each Greek differentiates with respect to.
_FIELD = {"delta": "S", "gamma": "S", "vega": "sigma", "theta": "T", "rho": "r"}

_EPS = float(np.finfo(float).eps)


def default_step(opt: Option, greek: str) -> float:
    """Step size that balances truncation error against floating-point cancellation.

    A central first difference has truncation error O(h^2) and roundoff error
    O(eps/h), which trade off at h ~ eps^(1/3) times the scale of the variable.
    A second difference trades O(h^2) against O(eps/h^2), giving eps^(1/4).
    Going smaller than this makes the estimate worse, not better.
    """
    if greek not in GREEKS:
        raise ValueError(f"greek must be one of {GREEKS}, got {greek!r}")

    scale = {
        "delta": opt.S,
        "gamma": opt.S,
        "vega": max(opt.sigma, 0.01),
        "theta": max(opt.T, 1e-3),
        "rho": 1.0,
    }[greek]
    exponent = 0.25 if greek == "gamma" else 1.0 / 3.0
    return scale * _EPS**exponent


def finite_difference(opt: Option, greek: str, h: float | None = None) -> float:
    """Estimate a Greek by bumping the relevant input and repricing.

    Matches the sign conventions of the analytic Greeks: theta is reported as
    the value lost per year, so it negates the derivative with respect to T.
    """
    if greek not in GREEKS:
        raise ValueError(f"greek must be one of {GREEKS}, got {greek!r}")

    field = _FIELD[greek]
    step = default_step(opt, greek) if h is None else h
    base = getattr(opt, field)
    up = replace(opt, **{field: base + step})
    down = replace(opt, **{field: base - step})

    if greek == "gamma":
        return (bs.price(up) - 2.0 * bs.price(opt) + bs.price(down)) / step**2

    derivative = (bs.price(up) - bs.price(down)) / (2.0 * step)
    return -derivative if greek == "theta" else derivative


def pathwise_delta(opt: Option, n_paths: int = 100_000, *, seed: int | None = None) -> float:
    """Monte Carlo delta by differentiating the payoff along each path.

    For a European call, dC/dS0 = exp(-rT) * 1{S_T > K} * S_T / S0. The estimator
    is unbiased and far lower variance than bumping, but it needs a payoff that
    is differentiable in the parameter - which is why it gives delta and not
    gamma, the indicator function not being twice differentiable.
    """
    rng = np.random.default_rng(seed)
    spot = mc.terminal_spot(opt, rng.standard_normal(n_paths))
    discount = math.exp(-opt.r * opt.T)

    in_the_money = spot > opt.K if opt.is_call else spot < opt.K
    sign = 1.0 if opt.is_call else -1.0
    return float(sign * discount * np.mean(in_the_money * spot / opt.S))


def bumped_delta(
    opt: Option,
    h: float = 0.01,
    n_paths: int = 100_000,
    *,
    seed: int | None = None,
    common_random: bool = True,
) -> float:
    """Monte Carlo delta by central bumping.

    With `common_random` the two repricings reuse the same normal draws, so the
    simulation noise cancels in the difference. Without it, that noise scales
    as 1/h and swamps the derivative entirely - which is the point of the
    comparison in `analysis/greeks_study.py`.
    """
    rng = np.random.default_rng(seed)
    z_up = rng.standard_normal(n_paths)
    z_down = z_up if common_random else rng.standard_normal(n_paths)

    up = replace(opt, S=opt.S + h)
    down = replace(opt, S=opt.S - h)
    discount = math.exp(-opt.r * opt.T)

    price_up = discount * float(np.mean(mc.payoff(up, mc.terminal_spot(up, z_up))))
    price_down = discount * float(np.mean(mc.payoff(down, mc.terminal_spot(down, z_down))))
    return (price_up - price_down) / (2.0 * h)
