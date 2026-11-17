"""Choose the right pricing engine for a contract, and hand it back as a function.

Everything downstream of a price - implied volatility, Greeks, the scanner's
spread guard - works by repricing the same contract with one input nudged. Until
now each of those called `black_scholes.price` directly, which quietly made the
whole library European-only even though the lattice had handled American
exercise all along.

This module is the seam. A `Pricer` is any callable taking an `Option` and
returning a price, so the callers no longer name an engine; they ask for the one
this contract needs. The default keeps the closed form wherever it is valid, so
nothing that worked before gets slower or less exact.

Why Leisen-Reimer at a hundred-odd steps is the lattice default: it converges at
O(1/n^2) rather than O(1/n) and has none of the Cox-Ross-Rubinstein sawtooth, so
it reaches an accuracy near the closed form at a step count CRR would need
thousands for. That matters because an implied-volatility solve runs the pricer
a dozen times per quote, and a chain holds thousands of quotes.
"""

from __future__ import annotations

from collections.abc import Callable

from optpricing.contracts import Option
from optpricing.models import binomial
from optpricing.models import black_scholes as bs

__all__ = [
    "DEFAULT_METHOD",
    "DEFAULT_STEPS",
    "Pricer",
    "describe",
    "for_option",
    "has_closed_form",
    "lattice",
]

#: Any callable that turns a contract into a price.
Pricer = Callable[[Option], float]

#: Odd, as Leisen-Reimer requires. See the module docstring for why this is enough.
DEFAULT_STEPS = 101

DEFAULT_METHOD = "lr"


def has_closed_form(opt: Option) -> bool:
    """True when Black-Scholes-Merton prices this contract exactly.

    European exercise and no discrete dividend schedule. A continuous yield is
    fine - that is what the Merton extension is for.
    """
    return opt.style == "european" and not opt.dividends


def lattice(n_steps: int = DEFAULT_STEPS, method: str = DEFAULT_METHOD) -> Pricer:
    """A pricer that puts every contract on a binomial lattice."""
    if method not in binomial.METHODS:
        raise ValueError(f"method must be one of {binomial.METHODS}, got {method!r}")
    if method == "lr" and n_steps % 2 == 0:
        raise ValueError(f"Leisen-Reimer requires an odd n_steps, got {n_steps}")

    def priced(opt: Option) -> float:
        return binomial.price(opt, n_steps=n_steps, method=method)

    return priced


def for_option(
    opt: Option, *, n_steps: int = DEFAULT_STEPS, method: str = DEFAULT_METHOD
) -> Pricer:
    """The cheapest engine that prices `opt` correctly.

    Closed form where it is exact, lattice otherwise. Callers that want to force
    one engine can build it themselves and pass it in; this is only the default.
    """
    if has_closed_form(opt):
        return bs.price
    return lattice(n_steps, method)


def describe(pricer: Pricer) -> str:
    """A short name for a pricer, for reporting which engine produced a number.

    The UI has to be able to say whether a price is closed form or numerical,
    because the two carry different error stories.
    """
    if pricer is bs.price:
        return "black-scholes"
    return getattr(pricer, "__qualname__", "pricer").split(".")[0].replace("priced", "lattice")
