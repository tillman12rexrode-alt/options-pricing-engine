"""Monte Carlo pricing for European options under geometric Brownian motion.

For a European payoff the terminal price can be sampled in a single step, so
this estimator carries no time-discretization error at all. Every bit of its
error is sampling error, which is why it falls as 1/sqrt(n) and no faster.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from optpricing.contracts import Option

__all__ = ["MCResult", "payoff", "price", "price_with_error", "terminal_spot"]

_Z95 = 1.959963984540054  # two-sided 95% normal quantile


@dataclass(frozen=True, slots=True)
class MCResult:
    """A Monte Carlo price together with the uncertainty attached to it."""

    price: float
    std_error: float
    n_samples: int

    @property
    def ci95(self) -> tuple[float, float]:
        """Two-sided 95% confidence interval for the price."""
        half = _Z95 * self.std_error
        return self.price - half, self.price + half

    def __str__(self) -> str:
        low, high = self.ci95
        return f"{self.price:.4f} +/- {_Z95 * self.std_error:.4f} (95% CI [{low:.4f}, {high:.4f}])"


def terminal_spot(opt: Option, z: np.ndarray) -> np.ndarray:
    """Exact one-step sample of S_T under GBM for the standard normals in `z`."""
    drift = (opt.r - opt.q - 0.5 * opt.sigma**2) * opt.T
    diffusion = opt.sigma * math.sqrt(opt.T) * z
    return opt.S * np.exp(drift + diffusion)


def payoff(opt: Option, spot: np.ndarray) -> np.ndarray:
    """Exercise value at each simulated terminal price."""
    if opt.is_call:
        return np.maximum(spot - opt.K, 0.0)
    return np.maximum(opt.K - spot, 0.0)


def price_with_error(
    opt: Option,
    n_paths: int = 100_000,
    *,
    seed: int | None = None,
    antithetic: bool = False,
    control_variate: bool = False,
) -> MCResult:
    """Price a European option by simulation, returning the standard error too.

    With `antithetic`, each draw Z is paired with -Z and the pair is averaged
    into one sample, so `n_paths` paths yield `n_paths // 2` samples. With
    `control_variate`, the terminal spot is used as a control: its expectation
    S0*exp((r-q)T) is known exactly, so its simulation error can be subtracted
    off using the variance-minimizing coefficient.
    """
    if opt.style != "european":
        raise ValueError("Monte Carlo here prices European options only; use the lattice")
    if n_paths < 2:
        raise ValueError(f"need at least 2 paths, got {n_paths}")
    if antithetic and n_paths % 2:
        raise ValueError(f"antithetic sampling needs an even n_paths, got {n_paths}")

    rng = np.random.default_rng(seed)
    discount = math.exp(-opt.r * opt.T)

    if antithetic:
        z = rng.standard_normal(n_paths // 2)
        spot = terminal_spot(opt, np.concatenate([z, -z]))
        # Average each (Z, -Z) pair into one sample, so the samples stay independent.
        samples = discount * payoff(opt, spot).reshape(2, -1).mean(axis=0)
        control = spot.reshape(2, -1).mean(axis=0)
    else:
        spot = terminal_spot(opt, rng.standard_normal(n_paths))
        samples = discount * payoff(opt, spot)
        control = spot

    if control_variate:
        expected_control = opt.S * math.exp((opt.r - opt.q) * opt.T)
        var_control = control.var(ddof=1)
        if var_control > 0:
            beta = np.cov(samples, control, ddof=1)[0, 1] / var_control
            samples = samples - beta * (control - expected_control)

    n = samples.size
    return MCResult(
        price=float(samples.mean()),
        std_error=float(samples.std(ddof=1) / math.sqrt(n)),
        n_samples=n,
    )


def price(opt: Option, n_paths: int = 100_000, **kwargs) -> float:
    """Monte Carlo price alone, matching the interface of the other engines."""
    return price_with_error(opt, n_paths, **kwargs).price
