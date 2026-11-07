"""Convergence study: how fast each engine approaches the exact price.

Black-Scholes is exact for European options, so it serves as the reference.
For every approximate engine we measure the error against it across a sweep of
work (lattice steps, or simulated paths) and fit the empirical convergence rate
as the slope of log(error) against log(work).

Run from the repository root:

    python analysis/convergence.py

Writes four figures to figures/ and a summary table to analysis/results/.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from optpricing.contracts import Option
from optpricing.models import binomial
from optpricing.models import black_scholes as bs
from optpricing.models import monte_carlo as mc

ROOT = Path(__file__).resolve().parent.parent
FIGURES = ROOT / "figures"
RESULTS = ROOT / "analysis" / "results"

# --- palette (categorical slots 1-4, light mode) -----------------------------
SERIES = {"crr": "#2a78d6", "jr": "#eb6834", "lr": "#1baf7a", "mc": "#eda100"}
LABELS = {
    "crr": "Cox-Ross-Rubinstein",
    "jr": "Jarrow-Rudd",
    "lr": "Leisen-Reimer",
    "mc": "Monte Carlo",
}
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e3e2de"

THEORY = {"crr": -1.0, "jr": -1.0, "lr": -2.0, "mc": -0.5}


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "text.color": INK,
            "axes.labelcolor": INK,
            "axes.edgecolor": GRID,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "lines.linewidth": 2.0,
            "figure.dpi": 140,
        }
    )


@dataclass(frozen=True)
class Contract:
    label: str
    option: Option


def build_grid() -> list[Contract]:
    """Six contracts: in, at and out of the money, at one month and one year."""
    base = {"S": 100.0, "r": 0.05, "sigma": 0.25, "q": 0.02, "kind": "call"}
    grid = []
    for money, strike in (("ITM", 80.0), ("ATM", 100.0), ("OTM", 120.0)):
        for tenor, years in (("1m", 1 / 12), ("1y", 1.0)):
            grid.append(
                Contract(f"{money} {tenor}", Option(**base, K=strike, T=years))
            )
    return grid


def odd(n: float) -> int:
    """Nearest odd integer at or above n - Leisen-Reimer needs an odd step count."""
    i = round(n)
    return i if i % 2 else i + 1


def lattice_errors(grid: list[Contract], steps: list[int]) -> dict[str, np.ndarray]:
    """RMS absolute error across the grid, per method, at each step count."""
    exact = np.array([bs.price(c.option) for c in grid])
    out = {}
    for method in ("crr", "jr", "lr"):
        errors = []
        for n in steps:
            priced = np.array([binomial.price(c.option, n, method) for c in grid])
            errors.append(float(np.sqrt(np.mean((priced - exact) ** 2))))
        out[method] = np.array(errors)
    return out


def monte_carlo_errors(opt: Option, paths: list[int], n_seeds: int = 30) -> np.ndarray:
    """RMSE over independent seeds, at each path count."""
    exact = bs.price(opt)
    out = []
    for n in paths:
        draws = [mc.price(opt, n, seed=s) for s in range(n_seeds)]
        out.append(float(np.sqrt(np.mean((np.array(draws) - exact) ** 2))))
    return np.array(out)


def fit_slope(work: np.ndarray, error: np.ndarray) -> tuple[float, float]:
    """OLS slope of log(error) on log(work), with R^2."""
    x, y = np.log(np.asarray(work, float)), np.log(error)
    slope, intercept = np.polyfit(x, y, 1)
    predicted = slope * x + intercept
    r2 = 1.0 - np.sum((y - predicted) ** 2) / np.sum((y - y.mean()) ** 2)
    return float(slope), float(r2)


# Aqua and yellow fall below 3:1 against the light surface, so they carry a
# direct label as relief; blue and orange are identified by the legend alone.
NEEDS_RELIEF = ("lr", "mc")


def _label_end(ax, x, y, text, color) -> None:
    """Direct label at the right end of a line - relief for low-contrast hues."""
    ax.annotate(
        text,
        xy=(x[-1], y[-1]),
        xytext=(6, 0),
        textcoords="offset points",
        color=color,
        fontsize=9,
        va="center",
    )


def figure_lattice(steps: list[int], errors: dict[str, np.ndarray], fits: dict) -> None:
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    for method in ("crr", "jr", "lr"):
        slope, _ = fits[method]
        ax.loglog(
            steps,
            errors[method],
            color=SERIES[method],
            marker="o",
            markersize=4,
            label=f"{LABELS[method]}  (slope {slope:+.2f})",
        )
        if method in NEEDS_RELIEF:
            _label_end(ax, steps, errors[method], method.upper(), SERIES[method])

    ax.set_xlabel("Lattice steps (n)")
    ax.set_ylabel("RMS absolute pricing error (currency units)")
    ax.set_title("Lattice convergence against the closed form", loc="left", pad=12)
    ax.grid(True, which="both", alpha=0.6)
    ax.legend(frameon=False, loc="lower left", fontsize=9)
    ax.margins(x=0.08)
    fig.tight_layout()
    fig.savefig(FIGURES / "convergence_lattice.png", bbox_inches="tight")
    plt.close(fig)


def figure_monte_carlo(paths: list[int], error: np.ndarray, fit: tuple) -> None:
    slope, _ = fit
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    ax.loglog(paths, error, color=SERIES["mc"], marker="o", markersize=4)
    _label_end(ax, paths, error, f"slope {slope:+.2f}", SERIES["mc"])

    reference = error[0] * (np.array(paths, float) / paths[0]) ** -0.5
    ax.loglog(paths, reference, color=MUTED, linestyle="--", linewidth=1.2, alpha=0.8)
    ax.annotate(
        "theoretical $n^{-1/2}$",
        xy=(paths[len(paths) // 2], reference[len(paths) // 2]),
        xytext=(0, -18),
        textcoords="offset points",
        color=MUTED,
        fontsize=9,
    )

    ax.set_xlabel("Simulated paths (n)")
    ax.set_ylabel("RMSE over 30 seeds (currency units)")
    ax.set_title("Monte Carlo convergence, ATM one-year call", loc="left", pad=12)
    ax.grid(True, which="both", alpha=0.6)
    ax.margins(x=0.1)
    fig.tight_layout()
    fig.savefig(FIGURES / "convergence_monte_carlo.png", bbox_inches="tight")
    plt.close(fig)


def figure_oscillation(opt: Option) -> None:
    """The sawtooth that a log-log plot hides, and the method that removes it."""
    exact = bs.price(opt)
    crr_steps = list(range(50, 121))
    crr = [binomial.price(opt, n, "crr") - exact for n in crr_steps]
    lr_steps = [n for n in crr_steps if n % 2]
    lr = [binomial.price(opt, n, "lr") - exact for n in lr_steps]

    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    ax.axhline(0.0, color=MUTED, linewidth=1.0, alpha=0.7)
    ax.plot(crr_steps, crr, color=SERIES["crr"], linewidth=1.6, label="Cox-Ross-Rubinstein")
    ax.plot(lr_steps, lr, color=SERIES["lr"], linewidth=2.0, label="Leisen-Reimer")
    _label_end(ax, lr_steps, lr, "LR", SERIES["lr"])

    ax.set_xlabel("Lattice steps (n)")
    ax.set_ylabel("Signed pricing error (currency units)")
    ax.set_title(
        "CRR oscillates as the strike shifts between terminal nodes; LR does not",
        loc="left",
        pad=12,
    )
    ax.grid(True, axis="y", alpha=0.6)
    ax.legend(frameon=False, loc="lower right", fontsize=9)
    ax.margins(x=0.06)
    fig.tight_layout()
    fig.savefig(FIGURES / "binomial_oscillation.png", bbox_inches="tight")
    plt.close(fig)


def _time_once(fn, *args) -> float:
    start = time.perf_counter()
    fn(*args)
    return time.perf_counter() - start


def figure_cost(opt: Option) -> None:
    """Error against wall-clock time - the comparison a practitioner actually makes."""
    exact = bs.price(opt)
    series: dict[str, tuple[list[float], list[float]]] = {}

    for method in ("crr", "jr", "lr"):
        times, errors = [], []
        for n in [odd(x) for x in np.logspace(1.2, 3.6, 9)]:
            best = min(_time_once(binomial.price, opt, n, method) for _ in range(3))
            times.append(best)
            errors.append(abs(binomial.price(opt, n, method) - exact))
        series[method] = (times, errors)

    times, errors = [], []
    for n in [int(x) for x in np.logspace(3, 6, 7)]:
        start = time.perf_counter()
        draws = [mc.price(opt, n, seed=s) for s in range(5)]
        times.append((time.perf_counter() - start) / 5)
        errors.append(float(np.sqrt(np.mean((np.array(draws) - exact) ** 2))))
    series["mc"] = (times, errors)

    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    for method in ("crr", "jr", "lr", "mc"):
        t, e = series[method]
        ax.loglog(t, e, color=SERIES[method], marker="o", markersize=4, label=LABELS[method])
        if method in NEEDS_RELIEF:
            _label_end(ax, t, e, method.upper(), SERIES[method])

    ax.set_xlabel("Wall-clock seconds per price")
    ax.set_ylabel("Absolute pricing error (currency units)")
    ax.set_title("Accuracy bought per second of compute", loc="left", pad=12)
    ax.grid(True, which="both", alpha=0.6)
    ax.legend(frameon=False, loc="lower left", fontsize=9)
    ax.margins(x=0.12)
    fig.tight_layout()
    fig.savefig(FIGURES / "error_vs_time.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    _style()
    FIGURES.mkdir(exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)

    grid = build_grid()
    atm_1y = Option(S=100.0, K=100.0, T=1.0, r=0.05, sigma=0.25, q=0.02)

    steps = [odd(x) for x in np.logspace(1.0, 3.6, 10)]
    paths = [int(x) for x in np.logspace(2, 6, 9)]

    lattice = lattice_errors(grid, steps)
    mc_error = monte_carlo_errors(atm_1y, paths)

    fits = {m: fit_slope(np.array(steps), lattice[m]) for m in ("crr", "jr", "lr")}
    fits["mc"] = fit_slope(np.array(paths), mc_error)

    figure_lattice(steps, lattice, fits)
    figure_monte_carlo(paths, mc_error, fits["mc"])
    figure_oscillation(atm_1y)
    figure_cost(atm_1y)

    lines = ["method,theoretical_slope,fitted_slope,r_squared"]
    print(f"{'method':<22}{'theory':>8}{'fitted':>9}{'R^2':>8}")
    for method in ("crr", "jr", "lr", "mc"):
        slope, r2 = fits[method]
        lines.append(f"{method},{THEORY[method]},{slope:.4f},{r2:.4f}")
        print(f"{LABELS[method]:<22}{THEORY[method]:>8.1f}{slope:>9.3f}{r2:>8.4f}")

    (RESULTS / "convergence_rates.csv").write_text("\n".join(lines) + "\n")
    print(f"\nFigures written to {FIGURES}")
    print(f"Table written to {RESULTS / 'convergence_rates.csv'}")


if __name__ == "__main__":
    main()
