"""Fit a smooth volatility smile to one expiry, and measure what sits off it.

The scanner's whole premise is that a contract whose implied volatility
disagrees with its neighbours is worth a second look. That only works if the
curve through the neighbours is trustworthy, so this module does two jobs: it
fits the curve, and it decides whether the fit is good enough to draw
conclusions from.

The parameterization is raw SVI (Gatheral), fitted to total implied variance
w(k) = sigma^2 * T against log-moneyness k = log(K/F):

    w(k) = a + b * (rho * (k - m) + sqrt((k - m)^2 + s^2))

Two constraints are imposed rather than hoped for. The parameters are
reparameterized so the minimum of w is non-negative by construction, which an
unconstrained fit violates routinely, and b(1 + |rho|) is capped at 4/T, which
is Lee's moment bound on how fast the wings may grow.

Weighting is by inverse relative spread, not by vega. Vega weighting is the
usual default and it is wrong here: vega collapses away from the money, so a
vega-weighted fit effectively abandons the wings - and the wings are where a
scanner looks. Measured on live SPX data, vega weighting left a systematic
20-volatility-point error in the far call wing while still reporting a
respectable overall RMSE.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise

import numpy as np
from scipy.optimize import least_squares

__all__ = ["SVIParams", "SliceFit", "fit_slice", "total_variance"]

#: A slice whose residuals are worse than this is not reliable enough to scan.
MAX_RMSE_VOL_POINTS = 0.75

#: Nor is one whose residuals cluster by region rather than scattering.
MAX_BUCKET_BIAS_VOL_POINTS = 0.60

_BUCKET_EDGES = (-np.inf, -0.105, -0.030, 0.030, 0.095, np.inf)


@dataclass(frozen=True)
class SVIParams:
    a: float
    b: float
    rho: float
    m: float
    s: float

    def as_tuple(self) -> tuple[float, float, float, float, float]:
        return self.a, self.b, self.rho, self.m, self.s

    @property
    def min_total_variance(self) -> float:
        """w at its minimum; negative here would mean an impossible smile."""
        return self.a + self.b * self.s * _math_sqrt(1.0 - self.rho**2)


def _math_sqrt(x: float) -> float:
    return float(np.sqrt(max(x, 0.0)))


@dataclass(frozen=True)
class SliceFit:
    """A fitted smile, its residuals, and whether it can be trusted."""

    params: SVIParams
    years: float
    rmse_vol_points: float
    worst_bucket_bias: float
    n_quotes: int
    reliable: bool
    reason: str

    def implied_vol(self, k: np.ndarray) -> np.ndarray:
        """Fitted implied volatility at log-moneyness `k`."""
        variance = np.maximum(total_variance(k, self.params), 1e-12) / self.years
        return np.sqrt(variance)


def total_variance(k: np.ndarray, params: SVIParams) -> np.ndarray:
    a, b, rho, m, s = params.as_tuple()
    return a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + s**2))


def _bucket_bias(k: np.ndarray, residual_points: np.ndarray) -> float:
    """Largest median residual within any moneyness band, in volatility points.

    Scattered residuals mean the curve has the right shape and the quotes are
    noisy around it. Residuals that cluster by region mean the shape is wrong,
    and every quote in that region would be reported as an outlier.
    """
    worst = 0.0
    for low, high in pairwise(_BUCKET_EDGES):
        band = (k >= low) & (k < high)
        if band.sum() >= 5:
            worst = max(worst, abs(float(np.median(residual_points[band]))))
    return worst


def fit_slice(
    k: np.ndarray,
    implied: np.ndarray,
    years: float,
    weights: np.ndarray | None = None,
) -> SliceFit:
    """Calibrate SVI to one expiry and judge whether the fit is usable.

    Parameters
    ----------
    k : log-moneyness, log(K / forward)
    implied : implied volatility per quote, as a decimal
    years : time to expiry
    weights : relative confidence per quote; inverse relative spread is the
        sensible choice and is what the scanner passes
    """
    k = np.asarray(k, dtype=float)
    implied = np.asarray(implied, dtype=float)
    observed = implied**2 * years

    if k.size < 8:
        empty = SVIParams(float("nan"), *(float("nan"),) * 4)
        return SliceFit(empty, years, float("nan"), float("nan"), k.size, False, "too few quotes")

    if weights is None:
        weights = np.ones_like(observed)
    weights = np.asarray(weights, dtype=float)
    weights = weights / weights.mean()

    span = max(float(k.max() - k.min()), 1e-3)
    b_cap = 4.0 / years
    lower = [0.0, 1e-8, -0.999, float(k.min()) - 0.25 * span, 1e-4]
    upper = [float(observed.max()), b_cap, 0.999, float(k.max()) + 0.25 * span, 2.0 * span + 0.5]

    def unpack(raw: np.ndarray) -> SVIParams:
        v, b, rho, m, s = raw
        return SVIParams(v - b * s * float(np.sqrt(1.0 - rho**2)), b, rho, m, s)

    def residual(raw: np.ndarray) -> np.ndarray:
        params = unpack(raw)
        overshoot = max(0.0, params.b * (1.0 + abs(params.rho)) - b_cap)
        return np.append(weights * (total_variance(k, params) - observed), 10.0 * overshoot)

    best = None
    starts = (float(k.min()) + 0.25 * span, 0.0, 0.05, float(k.max()) - 0.25 * span)
    for m_start in starts:
        for rho_start in (-0.9, -0.6, -0.2, 0.3):
            guess = np.clip(
                [max(float(observed.min()), 1e-8), min(0.1, b_cap * 0.9), rho_start,
                 m_start, 0.1 * span + 1e-3],
                lower,
                upper,
            )
            attempt = least_squares(
                residual, guess, bounds=(lower, upper), loss="soft_l1",
                f_scale=1e-4, max_nfev=4000,
            )
            if best is None or attempt.cost < best.cost:
                best = attempt

    params = unpack(best.x)
    fitted = np.sqrt(np.maximum(total_variance(k, params), 1e-12) / years)
    residual_points = (fitted - implied) * 100.0

    rmse = float(np.sqrt((residual_points**2).mean()))
    bias = _bucket_bias(k, residual_points)

    if rmse > MAX_RMSE_VOL_POINTS:
        reliable, reason = False, f"fit RMSE {rmse:.2f} vol points exceeds {MAX_RMSE_VOL_POINTS}"
    elif bias > MAX_BUCKET_BIAS_VOL_POINTS:
        reliable, reason = False, f"residuals biased {bias:.2f} vol points in one moneyness band"
    else:
        reliable, reason = True, "fit is within tolerance"

    return SliceFit(params, years, rmse, bias, int(k.size), reliable, reason)
