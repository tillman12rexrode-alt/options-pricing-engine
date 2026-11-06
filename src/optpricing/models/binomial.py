"""Binomial lattice pricing for vanilla options (Cox-Ross-Rubinstein)."""

from __future__ import annotations

import math

import numpy as np

from optpricing.contracts import Option

__all__ = ["price"]


def _payoff(opt: Option, spot: np.ndarray) -> np.ndarray:
    """Exercise value at each node price in `spot`."""
    if opt.is_call:
        return np.maximum(spot - opt.K, 0.0)
    return np.maximum(opt.K - spot, 0.0)


def price(opt: Option, n_steps: int = 500) -> float:
    """Price a European or American option on a CRR binomial lattice."""
    dt = opt.T / n_steps
    u = math.exp(opt.sigma * math.sqrt(dt))
    d = 1.0 / u
    p = (math.exp((opt.r - opt.q) * dt) - d) / (u - d)
    discount = math.exp(-opt.r * dt)

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
