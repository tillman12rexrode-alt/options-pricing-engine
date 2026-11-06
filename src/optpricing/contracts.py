"""Contract definitions for vanilla options."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

OptionKind = Literal["call", "put"]
ExerciseStyle = Literal["european", "american"]


@dataclass(frozen=True, slots=True)
class Option:
    """A vanilla option contract together with its market inputs.

    Parameters
    ----------
    S : spot price of the underlying
    K : strike price
    T : time to expiry, in years
    r : continuously compounded risk-free rate
    sigma : annualized volatility of the underlying
    q : continuous dividend yield; 0.0 for a non-dividend-paying underlying
    kind : "call" or "put"
    style : "european" or "american"

    The contract is frozen so that a priced option cannot be mutated
    underneath a study that is sweeping over inputs.
    """

    S: float
    K: float
    T: float
    r: float
    sigma: float
    q: float = 0.0
    kind: OptionKind = "call"
    style: ExerciseStyle = "european"

    def __post_init__(self) -> None:
        if self.S <= 0:
            raise ValueError(f"spot must be positive, got {self.S}")
        if self.K <= 0:
            raise ValueError(f"strike must be positive, got {self.K}")
        if self.T < 0:
            raise ValueError(f"time to expiry cannot be negative, got {self.T}")
        if self.sigma < 0:
            raise ValueError(f"volatility cannot be negative, got {self.sigma}")
        if self.kind not in ("call", "put"):
            raise ValueError(f"kind must be 'call' or 'put', got {self.kind!r}")
        if self.style not in ("european", "american"):
            raise ValueError(f"style must be european or american, got {self.style!r}")

    @property
    def is_call(self) -> bool:
        return self.kind == "call"

    def intrinsic(self, spot: float | None = None) -> float:
        """Payoff from exercising now at `spot`, defaulting to the contract's spot."""
        s = self.S if spot is None else spot
        return max(s - self.K, 0.0) if self.is_call else max(self.K - s, 0.0)
