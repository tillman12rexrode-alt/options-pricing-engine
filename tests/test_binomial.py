"""Validate the binomial lattice against Black-Scholes and option theory."""

from __future__ import annotations

import pytest

from optpricing.contracts import Option
from optpricing.models import binomial
from optpricing.models import black_scholes as bs


@pytest.mark.parametrize("kind", ["call", "put"])
@pytest.mark.parametrize("q", [0.0, 0.03])
def test_european_converges_to_black_scholes(kind: str, q: float):
    """With enough steps the lattice must reproduce the closed form."""
    opt = Option(S=42.0, K=40.0, T=0.5, r=0.10, sigma=0.20, q=q, kind=kind)
    assert binomial.price(opt, n_steps=2000) == pytest.approx(bs.price(opt), abs=0.01)


def test_american_call_equals_european_without_dividends():
    """Merton: it is never optimal to exercise an American call early when q = 0.

    The sharpest available test of the early-exercise branch - any spurious
    exercise shows up here as a price above the European value.
    """
    params = {"S": 100.0, "K": 95.0, "T": 1.0, "r": 0.05, "sigma": 0.30, "q": 0.0}
    american = binomial.price(Option(**params, kind="call", style="american"), 1000)
    european = binomial.price(Option(**params, kind="call", style="european"), 1000)
    assert american == pytest.approx(european, abs=1e-10)


def test_american_put_is_worth_more_than_european():
    """Early exercise has strictly positive value for a put."""
    params = {"S": 90.0, "K": 100.0, "T": 1.0, "r": 0.08, "sigma": 0.25}
    american = binomial.price(Option(**params, kind="put", style="american"), 1000)
    european = binomial.price(Option(**params, kind="put", style="european"), 1000)
    assert american > european + 1e-6


def test_early_exercise_premium_grows_with_rate():
    """A higher rate makes exercising a put sooner more attractive."""

    def premium(r: float) -> float:
        params = {"S": 90.0, "K": 100.0, "T": 1.0, "r": r, "sigma": 0.25, "kind": "put"}
        return binomial.price(Option(**params, style="american"), 1000) - binomial.price(
            Option(**params, style="european"), 1000
        )

    assert premium(0.10) > premium(0.02)


def test_rejects_arbitrageable_lattice():
    """Too few steps for the drift leaves p outside (0, 1)."""
    opt = Option(S=100.0, K=100.0, T=1.0, r=1.0, sigma=0.05)
    with pytest.raises(ValueError, match="arbitrageable"):
        binomial.price(opt, n_steps=1)
