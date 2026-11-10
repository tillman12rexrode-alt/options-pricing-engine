"""Check the smile fit recovers known curves and refuses untrustworthy ones."""

from __future__ import annotations

import numpy as np
import pytest

from optpricing.smile import MAX_RMSE_VOL_POINTS, SVIParams, fit_slice, total_variance

YEARS = 0.25
TRUE = SVIParams(a=0.004, b=0.08, rho=-0.70, m=0.02, s=0.12)
GRID = np.linspace(-0.25, 0.20, 60)


def vols_from(params: SVIParams, k: np.ndarray = GRID, years: float = YEARS) -> np.ndarray:
    return np.sqrt(total_variance(k, params) / years)


def test_fit_recovers_a_noise_free_curve():
    """Given a curve that is exactly SVI, the fit should reproduce it."""
    fitted = fit_slice(GRID, vols_from(TRUE), YEARS)

    assert fitted.reliable
    assert fitted.rmse_vol_points < 0.01
    assert np.allclose(fitted.implied_vol(GRID), vols_from(TRUE), atol=1e-4)


def test_fitted_variance_can_never_go_negative():
    """The reparameterization makes this true by construction, not by luck."""
    fitted = fit_slice(GRID, vols_from(TRUE), YEARS)
    wide = np.linspace(-2.0, 2.0, 400)

    assert fitted.params.min_total_variance >= 0.0
    assert np.all(total_variance(wide, fitted.params) >= -1e-12)


def test_wing_growth_respects_lees_moment_bound():
    """b(1 + |rho|) <= 4/T, or the wings imply infinite moments."""
    fitted = fit_slice(GRID, vols_from(TRUE), YEARS)
    assert fitted.params.b * (1 + abs(fitted.params.rho)) <= 4.0 / YEARS + 1e-6


def test_random_noise_still_fits_and_stays_reliable():
    """Scattered quote noise is what a real chain looks like; it should pass."""
    rng = np.random.default_rng(0)
    noisy = vols_from(TRUE) + rng.normal(0.0, 0.0015, GRID.size)
    fitted = fit_slice(GRID, noisy, YEARS)

    assert fitted.reliable
    assert fitted.rmse_vol_points < MAX_RMSE_VOL_POINTS


def test_a_shape_the_curve_cannot_take_is_reported_unreliable():
    """A kink in one wing biases residuals by region - exactly what the gate catches."""
    distorted = vols_from(TRUE).copy()
    distorted[GRID > 0.095] += 0.08
    fitted = fit_slice(GRID, distorted, YEARS)

    assert not fitted.reliable
    assert "biased" in fitted.reason or "RMSE" in fitted.reason


def test_a_small_shape_error_is_tolerated_on_purpose():
    """The gate is not meant to be hair-trigger.

    A distortion well below the outlier threshold the scanner uses would make
    the fit slightly wrong without making it misleading, and rejecting those
    slices would throw away most of a real chain. This pins that tolerance
    down rather than leaving it implicit.
    """
    distorted = vols_from(TRUE).copy()
    distorted[GRID > 0.095] += 0.03
    fitted = fit_slice(GRID, distorted, YEARS)

    assert fitted.reliable
    assert fitted.worst_bucket_bias < 0.60


def test_too_few_quotes_is_reported_rather_than_fitted():
    k = np.linspace(-0.05, 0.05, 5)
    fitted = fit_slice(k, vols_from(TRUE, k), YEARS)

    assert not fitted.reliable
    assert fitted.reason == "too few quotes"


def test_weights_shift_the_fit_toward_the_trusted_quotes():
    """Down-weighting a corrupted region should stop it dragging the curve."""
    corrupted = vols_from(TRUE).copy()
    corrupted[:10] += 0.05

    flat = fit_slice(GRID, corrupted, YEARS)
    weights = np.ones_like(GRID)
    weights[:10] = 0.01
    weighted = fit_slice(GRID, corrupted, YEARS, weights)

    clean_region = slice(10, None)
    truth = vols_from(TRUE)[clean_region]
    assert np.abs(weighted.implied_vol(GRID)[clean_region] - truth).mean() < np.abs(
        flat.implied_vol(GRID)[clean_region] - truth
    ).mean()


@pytest.mark.parametrize("rho", [-0.9, -0.4, 0.0, 0.4])
def test_fit_works_across_skew_regimes(rho: float):
    params = SVIParams(a=0.004, b=0.08, rho=rho, m=0.0, s=0.12)
    fitted = fit_slice(GRID, vols_from(params), YEARS)
    assert fitted.reliable
