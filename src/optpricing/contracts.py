"""Contract definitions for vanilla options."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

OptionKind = Literal["call", "put"]
ExerciseStyle = Literal["european", "american"]

#: Cash dividends as (time in years from now, amount per share) pairs. A tuple
#: rather than a list so the frozen contract stays hashable.
DividendSchedule = tuple[tuple[float, float], ...]


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
    dividends : discrete cash dividends as (time in years, amount) pairs, for
        single-name equities. Mutually exclusive with q, because the two are
        alternative descriptions of the same cash flow. An index is modelled
        with q; a stock that pays on known ex-dates is modelled here, since
        early exercise of an American call is a dated event that a yield
        smeared across the year cannot represent.

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
    dividends: DividendSchedule = ()

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

        for entry in self.dividends:
            if len(entry) != 2:
                raise ValueError(f"each dividend must be a (time, amount) pair, got {entry!r}")
            when, amount = entry
            if when < 0:
                raise ValueError(f"dividend time cannot be negative, got {when}")
            if amount < 0:
                raise ValueError(f"dividend amount cannot be negative, got {amount}")

        # A continuous yield and a discrete schedule are two ways of saying the same
        # thing. Accepting both would silently double-count the dividend, which is
        # the kind of error that produces a plausible wrong price rather than a crash.
        if self.dividends and self.q != 0.0:
            raise ValueError(
                "use either a continuous yield q or a discrete dividend schedule, not both"
            )

    @property
    def is_call(self) -> bool:
        return self.kind == "call"

    def pv_dividends(self, before: float | None = None) -> float:
        """Present value, as of now, of the dividends paid up to `before`.

        Defaults to every dividend falling on or before expiry. Dividends dated
        after expiry are irrelevant to the contract and are ignored rather than
        rejected, because a schedule is a property of the stock and is shared
        across contracts with different maturities.
        """
        horizon = self.T if before is None else before
        # Strictly after now, matching the lattice's escrow: a dividend going ex
        # today is already out of the spot price.
        return sum(
            amount * math.exp(-self.r * when)
            for when, amount in self.dividends
            if 0.0 < when <= horizon
        )

    def intrinsic(self, spot: float | None = None) -> float:
        """Payoff from exercising now at `spot`, defaulting to the contract's spot."""
        s = self.S if spot is None else spot
        return max(s - self.K, 0.0) if self.is_call else max(self.K - s, 0.0)
