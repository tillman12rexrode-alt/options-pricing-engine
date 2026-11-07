"""Validate the binomial lattices against Black-Scholes and option theory."""

from __future__ import annotations

import pytest

from optpricing.contracts import Option
from optpricing.models import binomial
from optpricing.models import black_scholes as bs

PARAMS = {"S": 42.0, "K": 40.0, "T": 0.5, "r": 0.10, "sigma": 0.20}


@pytest.mark.parametrize("method", binomial.METHODS)
@pytest.mark.parametrize("kind", ["call", "put"])
@pytest.mark.parametrize("q", [0.0, 0.03])
def test_european_converges_to_black_scholes(method: str, kind: str, q: float):
    """With enough steps every parameterization must reproduce the closed form."""
    opt = Option(**PARAMS, q=q, kind=kind)
    assert binomial.price(opt, 2001, method) == pytest.approx(bs.price(opt), abs=0.01)


def test_leisen_reimer_is_far_more_accurate_than_crr():
    """LR converges as 1/n^2 against CRR's 1/n, so at equal steps it wins easily."""
    opt = Option(S=100.0, K=95.0, T=1.0, r=0.05, sigma=0.30, q=0.02)
    exact = bs.price(opt)
    crr_error = abs(binomial.price(opt, 201, "crr") - exact)
    lr_error = abs(binomial.price(opt, 201, "lr") - exact)
    assert lr_error < crr_error / 50


@pytest.mark.parametrize("method", binomial.METHODS)
def test_american_call_equals_european_without_dividends(method: str):
    """Merton: it is never optimal to exercise an American call early when q = 0.

    The sharpest available test of the early-exercise branch - any spurious
    exercise shows up here as a price above the European value.
    """
    params = {"S": 100.0, "K": 95.0, "T": 1.0, "r": 0.05, "sigma": 0.30, "q": 0.0}
    american = binomial.price(Option(**params, kind="call", style="american"), 501, method)
    european = binomial.price(Option(**params, kind="call", style="european"), 501, method)
    assert american == pytest.approx(european, abs=1e-10)


@pytest.mark.parametrize("method", binomial.METHODS)
def test_american_put_is_worth_more_than_european(method: str):
    """Early exercise has strictly positive value for a put."""
    params = {"S": 90.0, "K": 100.0, "T": 1.0, "r": 0.08, "sigma": 0.25}
    american = binomial.price(Option(**params, kind="put", style="american"), 501, method)
    european = binomial.price(Option(**params, kind="put", style="european"), 501, method)
    assert american > european + 1e-6


def test_early_exercise_premium_grows_with_rate():
    """A higher rate makes exercising a put sooner more attractive."""

    def premium(r: float) -> float:
        params = {"S": 90.0, "K": 100.0, "T": 1.0, "r": r, "sigma": 0.25, "kind": "put"}
        return binomial.price(Option(**params, style="american"), 501) - binomial.price(
            Option(**params, style="european"), 501
        )

    assert premium(0.10) > premium(0.02)


def test_rejects_arbitrageable_lattice():
    """Too few steps for the drift leaves p outside (0, 1)."""
    opt = Option(S=100.0, K=100.0, T=1.0, r=1.0, sigma=0.05)
    with pytest.raises(ValueError, match="arbitrageable"):
        binomial.price(opt, 1)


def test_leisen_reimer_rejects_even_steps():
    """Peizer-Pratt centres the strike only for an odd number of steps."""
    with pytest.raises(ValueError, match="odd"):
        binomial.price(Option(**PARAMS), 200, "lr")


def test_rejects_unknown_method():
    with pytest.raises(ValueError, match="method must be"):
        binomial.price(Option(**PARAMS), 101, "trinomial")
