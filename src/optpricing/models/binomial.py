"""Binomial lattice pricing for vanilla options (Cox-Ross-Rubinstein)."""

from __future__ import annotations

import math

import numpy as np

from optpricing.contracts import Option

__all__ = ["price"]


def price(opt: Option, n_steps: int = 500) -> float:
    """Price a European option on a CRR binomial    dt = opt.T / n_steps
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
    spot = opt.S * u**j * d ** (n_steps - j)
    if opt.is_call:
        values = np.maximum(spot - opt.K, 0.0)
    else:
        values = np.maximum(opt.K - spot, 0.0)

    # Backward induction: each pass collapses the layer by one node.
    for _ in range(n_steps):
        values = discount * (p * values[1:] + (1 - p) * values[:-1])

    return float(values[0]) lattice."""
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
    spot = opt.S * u**j * d ** (n_steps - j)
    if opt.is_call:
        values = np.maximum(spot - opt.K, 0.0)
    else:
        values = np.maximum(opt.K - spot, 0.0)

    # Backward induction: each pass collapses the layer by one node.
    for _ in range(n_steps):
        values = discount * (p * values[1:] + (1 - p) * values[:-1])

    return float(values[0])