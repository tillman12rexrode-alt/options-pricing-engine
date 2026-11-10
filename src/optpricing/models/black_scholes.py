"""Black-Scholes-Merton closed-form pricing for European options.

Reference: Hull, *Options, Futures, and Other Derivatives*, the
Black-Scholes-Merton chapter. This module is the reference implementation
that the lattice and Monte Carlo engines are validated against, so it
deliberately shares no code with them.
"""

from __future__ import annotations

import math

from scipy.stats import norm

from optpricing.contracts import Option

__all__ = ["d1_d2", "greeks", "price"]


def d1_d2(opt: Option) -> tuple[float, float]:
    """Return the Black-Scholes d1 and d2 terms.

    Undefined when sigma * sqrt(T) is zero; `price` handles that case
    separately rather than letting it divide by zero here.
    """
    vol_sqrt_t = opt.sigma * math.sqrt(opt.T)
    if vol_sqrt_t == 0:
        raise ValueError("d1 and d2 are undefined when sigma * sqrt(T) == 0")

    d1 = (
        math.log(opt.S / opt.K) + (opt.r - opt.q + 0.5 * opt.sigma**2) * opt.T
    ) / vol_sqrt_t
    d2 = d1 - vol_sqrt_t
    return d1, d2


def price(opt: Option) -> float:
    """Price a European option under Black-Scholes-Merton with dividend yield q."""
    if opt.style != "european":
        raise ValueError("Black-Scholes prices European options only; use the lattice")

    # Degenerate cases: no time left, or no uncertainty. Both collapse to the
    # discounted payoff on a forward price that is known with certainty, which
    # is also the limit of the general formula as sigma * sqrt(T) -> 0.
    if opt.T == 0 or opt.sigma == 0:
        forward = opt.S * math.exp((opt.r - opt.q) * opt.T)
        payoff = opt.intrinsic(forward)
        return math.exp(-opt.r * opt.T) * payoff

    d1, d2 = d1_d2(opt)
    discounted_spot = opt.S * math.exp(-opt.q * opt.T)
    discounted_strike = opt.K * math.exp(-opt.r * opt.T)

    if opt.is_call:
        return discounted_spot * norm.cdf(d1) - discounted_strike * norm.cdf(d2)
    return discounted_strike * norm.cdf(-d2) - discounted_spot * norm.cdf(-d1)


def greeks(opt: Option) -> dict[str, float]:
    """Analytic Black-Scholes-Merton sensitivities.

    Conventions: delta and gamma are per unit of spot, vega is per 1.00 of
    volatility, theta is per year and signed as the value lost by the passage
    of time, and rho is per 1.00 of the interest rate.
    """
    if opt.style != "european":
        raise ValueError("Analytic Greeks are European only; use finite differences")
    if opt.T == 0 or opt.sigma == 0:
        raise ValueError("Greeks are undefined when sigma * sqrt(T) == 0")

    d1, d2 = d1_d2(opt)
    sqrt_t = math.sqrt(opt.T)
    disc_q = math.exp(-opt.q * opt.T)
    disc_r = math.exp(-opt.r * opt.T)
    pdf_d1 = norm.pdf(d1)

    gamma = disc_q * pdf_d1 / (opt.S * opt.sigma * sqrt_t)
    vega = opt.S * disc_q * pdf_d1 * sqrt_t
    decay = -opt.S * disc_q * pdf_d1 * opt.sigma / (2.0 * sqrt_t)

    if opt.is_call:
        delta = disc_q * norm.cdf(d1)
        theta = decay - opt.r * opt.K * disc_r * norm.cdf(d2) + opt.q * opt.S * disc_q * norm.cdf(d1)
        rho = opt.K * opt.T * disc_r * norm.cdf(d2)
    else:
        delta = -disc_q * norm.cdf(-d1)
        theta = (
            decay
            + opt.r * opt.K * disc_r * norm.cdf(-d2)
            - opt.q * opt.S * disc_q * norm.cdf(-d1)
        )
        rho = -opt.K * opt.T * disc_r * norm.cdf(-d2)

    return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta, "rho": rho}
