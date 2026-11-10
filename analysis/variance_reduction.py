"""How much variance reduction actually buys, measured per second rather than per path.

Antithetic and control variates both lower the variance of a Monte Carlo price.
The honest question is whether they lower it faster than they cost, so the
comparison here is efficiency - inversely proportional to variance times run
time - and not variance alone. A technique that halves variance and doubles
runtime has gained nothing.

Run from the repository root:

    python analysis/variance_reduction.py
"""

from __future__ import annotations

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import plotstyle as ps

from optpricing.contracts import Option
from optpricing.models import black_scholes as bs
from optpricing.models import monte_carlo as mc

ROOT = Path(__file__).resolve().parent.parent
FIGURES = ROOT / "figures"
RESULTS = ROOT / "analysis" / "results"

VARIANTS = {
    "plain": ({}, "Plain", ps.BLUE),
    "antithetic": ({"antithetic": True}, "Antithetic", ps.ORANGE),
    "control": ({"control_variate": True}, "Control variate", ps.AQUA),
    "both": ({"antithetic": True, "control_variate": True}, "Both", ps.YELLOW),
}

CONTRACT = Option(S=100.0, K=100.0, T=1.0, r=0.05, sigma=0.25, q=0.02)
PATHS = 400_000


def measure(opt: Option, n_paths: int, kwargs: dict, repeats: int = 5) -> tuple[float, float]:
    """Return the standard error and the median run time over `repeats` runs."""
    errors, times = [], []
    for seed in range(repeats):
        start = time.perf_counter()
        result = mc.price_with_error(opt, n_paths, seed=seed, **kwargs)
        times.append(time.perf_counter() - start)
        errors.append(result.std_error)
    return float(np.mean(errors)), float(np.median(times))


def efficiency_table(opt: Option) -> list[dict]:
    rows = []
    baseline_se, baseline_time = measure(opt, PATHS, {})
    baseline_efficiency = 1.0 / (baseline_se**2 * baseline_time)

    for key, (kwargs, label, _) in VARIANTS.items():
        std_error, runtime = measure(opt, PATHS, kwargs)
        efficiency = 1.0 / (std_error**2 * runtime)
        rows.append(
            {
                "variant": key,
                "label": label,
                "std_error": std_error,
                "variance_reduction": (baseline_se / std_error) ** 2,
                "seconds": runtime,
                "efficiency_gain": efficiency / baseline_efficiency,
            }
        )
    return rows


def figure_efficiency(opt: Option) -> None:
    """Standard error against wall-clock time, swept over path counts."""
    fig, ax = plt.subplots(figsize=(7.6, 4.8))

    for kwargs, label, color in VARIANTS.values():
        times, errors = [], []
        for n in (25_000, 50_000, 100_000, 200_000, 400_000, 800_000):
            std_error, runtime = measure(opt, n, kwargs, repeats=3)
            times.append(runtime)
            errors.append(std_error)
        ax.loglog(times, errors, color=color, marker="o", markersize=4, label=label)
        if ps.needs_relief(color):
            ps.label_end(ax, times, errors, label, color)

    ax.set_xlabel("Wall-clock seconds per price")
    ax.set_ylabel("Standard error of the price estimate")
    ax.set_title("Variance reduction, measured per second of compute", loc="left", pad=12)
    ax.grid(True, which="both", alpha=0.6)
    ax.legend(frameon=False, loc="lower left", fontsize=9)
    ax.margins(x=0.16)
    fig.tight_layout()
    fig.savefig(FIGURES / "variance_reduction.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ps.apply()
    FIGURES.mkdir(exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)

    exact = bs.price(CONTRACT)
    rows = efficiency_table(CONTRACT)

    header = f"{'variant':<18}{'std error':>12}{'var. redux':>12}{'seconds':>10}{'efficiency':>12}"
    print(f"ATM one-year call, exact price {exact:.6f}, {PATHS:,} paths\n")
    print(header)
    for row in rows:
        print(
            f"{row['label']:<18}{row['std_error']:>12.6f}"
            f"{row['variance_reduction']:>11.1f}x{row['seconds']:>10.4f}"
            f"{row['efficiency_gain']:>11.1f}x"
        )

    lines = ["variant,std_error,variance_reduction,seconds,efficiency_gain"]
    lines += [
        f"{r['variant']},{r['std_error']:.8f},{r['variance_reduction']:.3f},"
        f"{r['seconds']:.6f},{r['efficiency_gain']:.3f}"
        for r in rows
    ]
    (RESULTS / "variance_reduction.csv").write_text("\n".join(lines) + "\n")

    figure_efficiency(CONTRACT)
    print(f"\nFigure written to {FIGURES / 'variance_reduction.png'}")
    print(f"Table written to {RESULTS / 'variance_reduction.csv'}")


if __name__ == "__main__":
    main()
