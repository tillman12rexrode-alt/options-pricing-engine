"""Binomial lattice pricing for vanilla options.

Three parameterizations of the tree are available. They differ only in how the
up move, down move and risk-neutral probability are chosen, and they converge
to the same price at different rates:

``crr``
    Cox-Ross-Rubinstein. The standard choice, O(1/n), with a characteristic
    sawtooth as the strike shifts between terminal nodes.
``jr``
    Jarrow-Rudd. Equal risk-neutral probabilities with the drift carried in the
    moves; also O(1/n), and useful as a second opinion on whether an artifact
    is real or specific to CRR.
``lr``
    Leisen-Reimer. Uses Peizer-Pratt inversion to place the strike at the
    centre of the terminal distribution, which removes the oscillation and
    gives O(1/n^2) convergence. Requires an odd number of steps.
"""

from __future__ import annotations

import math

import numpy as np

from optpricing.contracts import Option

__all__ = ["METHODS", "price"]

METHODS = ("crr", "jr", "lr")


def _payoff(opt: Option, spot: np.ndarray) -> np.ndarray:
    """Exercise value at each node price in `spot`."""
    if opt.is_call:
        return np.maximum(spot - opt.K, 0.0)
    return np.maximum(opt.K - spot, 0.0)


def _crr(opt: Option, dt: float) -> tuple[float, float, float]:
    u = math.exp(opt.sigma * math.sqrt(dt))
    d = 1.0 / u
    p = (math.exp((opt.r - opt.q) * dt) - d) / (u - d)
    return u, d, p


def _jarrow_rudd(opt: Option, dt: float) -> tuple[float, float, float]:
    drift = (opt.r - opt.q - 0.5 * opt.sigma**2) * dt
    spread = opt.sigma * math.sqrt(dt)
    return math.exp(drift + spread), math.exp(drift - spread), 0.5


def _peizer_pratt(z: float, n: int) -> float:
    """Peizer-Pratt inversion (method 2): a normal CDF approximated on a lattice."""
    a = z / (n + 1.0 / 3.0 + 0.1 / (n + 1.0))
    inner = 1.0 - math.exp(-(a**2) * (n + 1.0 / 6.0))
    return 0.5 + math.copysign(0.5 * math.sqrt(inner), z)


def _leisen_reimer(opt: Option, dt: float, n_steps: int) -> tuple[float, float, float]:
    vol_sqrt_t = opt.sigma * math.sqrt(opt.T)
    d1 = (math.log(opt.S / opt.K) + (opt.r - opt.q + 0.5 * opt.sigma**2) * opt.T) / vol_sqrt_t
    d2 = d1 - vol_sqrt_t

    p = _peizer_pratt(d2, n_steps)
    p_star = _peizer_pratt(d1, n_steps)

    growth = math.exp((opt.r - opt.q) * dt)
    u = growth * p_star / p
    d = (growth - p * u) / (1.0 - p)
    return u, d, p


def price(opt: Option, n_steps: int = 500, method: str = "crr") -> float:
    """Price a European or American option on a binomial lattice.

    `method` selects the tree parameterization: "crr", "jr" or "lr".
    """
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")
    if n_steps < 1:
        raise ValueError(f"n_steps must be at least 1, got {n_steps}")
    if method == "lr" and n_steps % 2 == 0:
        raise ValueError(f"Leisen-Reimer requires an odd n_steps, got {n_steps}")

    dt = opt.T / n_steps
    discount = math.exp(-opt.r * dt)

    if method == "crr":
        u, d, p = _crr(opt, dt)
    elif method == "jr":
        u, d, p = _jarrow_rudd(opt, dt)
    else:
        u, d, p = _leisen_reimer(opt, dt, n_steps)

    if not 0.0 < p < 1.0:
        raise ValueError(
            f"risk-neutral probability p={p:.4f} is outside (0, 1); "
            "the lattice is arbitrageable - increase n_steps"
        )

    # Terminal layer: j counts up-moves, so node j has j ups and (n_steps - j) downs.
    j = np.arange(n_steps + 1)
    values = _payoff(opt, opt.S * u**j * d ** (n_steps - j))

    american = opt.style == "american"

    # Backward induction: each pass collapses the layer by one node.
    for i in range(n_steps - 1, -1, -1):
        values = discount * (p * values[1:] + (1 - p) * values[:-1])

        if american:
            # Spot prices at level i - note the exponents use i, not n_steps.
            j = np.arange(i + 1)
            values = np.maximum(values, _payoff(opt, opt.S * u**j * d ** (i - j)))

    return float(values[0])
