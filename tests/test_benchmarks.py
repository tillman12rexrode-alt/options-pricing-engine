"""Validate the closed-form pricer against published values and no-arbitrage law."""

from __future__ import annotations

import math

import pytest

from optpricing.contracts import Option
from optpricing.models import black_scholes as bs

# Hull, *Options, Futures, and Other Derivatives*, Black-Scholes-Merton chapter,
# worked example: S = 42, K = 40, r = 10%, sigma = 20%, T = 0.5y, no dividends.
# Published values: call 4.76, put 0.81.
HULL = {"S": 42.0, "K": 40.0, "T": 0.5, "r": 0.10, "sigma": 0.20, "q": 0.0}


def test_hull_call_benchmark():
    assert bs.price(Option(**HULL, kind="call")) == pytest.approx(4.76, abs=5e-3)


def test_hull_put_benchmark():
    assert bs.price(Option(**HULL, kind="put")) == pytest.approx(0.81, abs=5e-3)


@pytest.mark.parametrize("S", [80.0, 100.0, 125.0])
@pytest.mark.parametrize("T", [0.08, 1.0, 3.0])
@pytest.mark.parametrize("sigma", [0.10, 0.35])
@pytest.mark.parametrize("q", [0.0, 0.03])
def test_put_call_parity(S: float, T: float, sigma: float, q: float):
    """C - P = S*exp(-qT) - K*exp(-rT).

    Parity follows from no-arbitrage alone, so it must hold to near machine
    precision for any correct European pricer, under any inputs.
    """
    K, r = 100.0, 0.05
    shared = {"S": S, "K": K, "T": T, "r": r, "sigma": sigma, "q": q}

    call = bs.price(Option(**shared, kind="call"))
    put = bs.price(Option(**shared, kind="put"))
    expected = S * math.exp(-q * T) - K * math.exp(-r * T)

    assert call - put == pytest.approx(expected, abs=1e-10)


def test_zero_time_to_expiry_is_intrinsic():
    opt = Option(S=110.0, K=100.0, T=0.0, r=0.05, sigma=0.20, kind="call")
    assert bs.price(opt) == pytest.approx(10.0)


def test_zero_volatility_is_discounted_forward_payoff():
    opt = Option(S=100.0, K=100.0, T=1.0, r=0.05, sigma=0.0, kind="call")
    expected = math.exp(-0.05) * (100.0 * math.exp(0.05) - 100.0)
    assert bs.price(opt) == pytest.approx(expected)


def test_american_style_is_rejected():
    opt = Option(S=100.0, K=100.0, T=1.0, r=0.05, sigma=0.20, style="american")
    with pytest.raises(ValueError, match="European"):
        bs.price(opt)


@pytest.mark.parametrize(
    "bad",
    [
        {"S": -100.0},
        {"K": 0.0},
        {"T": -1.0},
        {"sigma": -0.2},
        {"kind": "straddle"},
    ],
)
def test_invalid_contracts_are_rejected(bad: dict):
    params = {"S": 100.0, "K": 100.0, "T": 1.0, "r": 0.05, "sigma": 0.20} | bad
    with pytest.raises(ValueError):
        Option(**params)
