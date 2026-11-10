"""Two numerical facts about Greeks that are easy to state and easy to get wrong.

First: a smaller finite-difference bump is not always more accurate. Truncation
error falls as the step shrinks while floating-point cancellation grows, so the
total error is U-shaped and has an optimum - near eps^(1/3) for a first
difference and eps^(1/4) for a second.

Second: bumping a Monte Carlo price only works if both repricings share their
random draws. Give them independent draws and the simulation noise, divided by
the bump size, destroys the estimate completely.

Run from the repository root:

    python analysis/greeks_study.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import plotstyle as ps

from optpricing import greeks as gk
from optpricing.contracts import Option
from optpricing.models import black_scholes as bs

ROOT = Path(__file__).resolve().parent.parent
FIGURES = ROOT / "figures"
RESULTS = ROOT / "analysis" / "results"

CONTRACT = Option(S=100.0, K=95.0, T=1.0, r=0.05, sigma=0.30, q=0.02)


def figure_bump_size(opt: Option) -> None:
    """The U-curve: error against step size for a first and a second difference."""
    exact = bs.greeks(opt)
    steps = np.logspace(-12, 1, 80)

    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    for greek, color in (("delta", ps.BLUE), ("gamma", ps.ORANGE)):
        errors = [
            max(abs(gk.finite_difference(opt, greek, h) - exact[greek]), 1e-18) for h in steps
        ]
        ax.loglog(steps, errors, color=color, label=f"{greek} (central difference)")
        ax.axvline(
            gk.default_step(opt, greek),
            color=color,
            linestyle=":",
            linewidth=1.4,
            alpha=0.9,
        )

    ax.set_xlabel("Finite-difference step h (spot units)")
    ax.set_ylabel("Absolute error against the analytic Greek")
    ax.set_title(
        "Smaller steps stop helping: truncation error falls, roundoff rises",
        loc="left",
        pad=12,
    )
    ax.plot(
        [],
        [],
        color=ps.MUTED,
        linestyle=":",
        linewidth=1.4,
        label="step the library picks",
    )
    ax.grid(True, which="both", alpha=0.6)
    ax.legend(frameon=False, loc="lower left", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURES / "greeks_bump_size.png", bbox_inches="tight")
    plt.close(fig)


def figure_common_random_numbers(opt: Option) -> tuple[float, float]:
    """Bumped Monte Carlo delta, with and without shared draws."""
    exact = bs.greeks(opt)["delta"]
    steps = np.logspace(-3, 0.7, 24)
    n_paths = 200_000

    shared = [gk.bumped_delta(opt, h, n_paths, seed=7, common_random=True) for h in steps]
    independent = [
        gk.bumped_delta(opt, h, n_paths, seed=7, common_random=False) for h in steps
    ]

    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    ax.axhline(exact, color=ps.MUTED, linewidth=1.2, alpha=0.9)
    ax.annotate(
        f"analytic delta {exact:.4f}",
        xy=(steps[0], exact),
        xytext=(2, 8),
        textcoords="offset points",
        color=ps.MUTED,
        fontsize=9,
    )
    ax.semilogx(steps, shared, color=ps.BLUE, marker="o", markersize=4, label="Common draws")
    ax.semilogx(
        steps, independent, color=ps.ORANGE, marker="o", markersize=4, label="Independent draws"
    )

    ax.set_xlabel("Bump size h (spot units)")
    ax.set_ylabel("Bumped Monte Carlo delta")
    ax.set_title("Bumping a simulation only works with shared random draws", loc="left", pad=12)
    ax.grid(True, which="both", alpha=0.6)
    ax.legend(frameon=False, loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURES / "greeks_common_random_numbers.png", bbox_inches="tight")
    plt.close(fig)

    small = int(np.argmin(np.abs(steps - 0.01)))
    return shared[small], independent[small]


def main() -> None:
    ps.apply()
    FIGURES.mkdir(exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)

    exact = bs.greeks(CONTRACT)

    print(f"{'greek':<8}{'analytic':>14}{'finite diff':>14}{'rel error':>12}")
    lines = ["greek,analytic,finite_difference,relative_error"]
    for greek in gk.GREEKS:
        estimate = gk.finite_difference(CONTRACT, greek)
        relative = abs(estimate - exact[greek]) / abs(exact[greek])
        print(f"{greek:<8}{exact[greek]:>14.6f}{estimate:>14.6f}{relative:>12.2e}")
        lines.append(f"{greek},{exact[greek]:.10f},{estimate:.10f},{relative:.3e}")
    (RESULTS / "greeks.csv").write_text("\n".join(lines) + "\n")

    figure_bump_size(CONTRACT)
    shared, independent = figure_common_random_numbers(CONTRACT)

    print(f"\npathwise delta        {gk.pathwise_delta(CONTRACT, 400_000, seed=1):.6f}")
    print(f"bumped, shared draws  {shared:.6f}")
    print(f"bumped, independent   {independent:.6f}   <- same paths, no shared draws")
    print(f"analytic delta        {exact['delta']:.6f}")
    print(f"\nFigures written to {FIGURES}")


if __name__ == "__main__":
    main()
