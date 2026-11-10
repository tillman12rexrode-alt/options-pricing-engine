"""Shared figure styling for the analysis scripts.

Colours are the first four slots of a categorical palette validated for
colour-vision deficiency on a light surface. Two of them - aqua and yellow -
fall below 3:1 contrast against that surface, so any line drawn in those
carries a direct label as well as a legend entry.
"""

from __future__ import annotations

import matplotlib.pyplot as plt

__all__ = ["AQUA", "BLUE", "MUTED", "ORANGE", "YELLOW", "apply", "label_end", "needs_relief"]

BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
YELLOW = "#eda100"

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e3e2de"

#: Hues below 3:1 contrast on the light surface; these need a direct label.
_LOW_CONTRAST = frozenset({AQUA, YELLOW})


def needs_relief(color: str) -> bool:
    return color in _LOW_CONTRAST


def apply() -> None:
    """Install the shared rcParams. Call once at the top of a script."""
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


def label_end(ax, x, y, text: str, color: str) -> None:
    """Direct label at the right-hand end of a line."""
    ax.annotate(
        text,
        xy=(x[-1], y[-1]),
        xytext=(6, 0),
        textcoords="offset points",
        color=color,
        fontsize=9,
        va="center",
    )
