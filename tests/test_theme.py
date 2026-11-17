"""Guard the app theme's contrast, so a tweaked hex code cannot quietly fail AA.

Colours are easy to change and hard to check by eye, especially on the dark
theme. These assert WCAG 2.x contrast ratios straight from the config file the
deployed app reads, so CI catches a regression before a visitor does.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

CONFIG = Path(__file__).resolve().parent.parent / ".streamlit" / "config.toml"

#: WCAG AA for body text. Accent colours carry text (tab labels, links,
#: slider values), so they are held to the same bar as the text itself.
AA = 4.5


def _luminance(hex_colour: str) -> float:
    channels = [int(hex_colour.lstrip("#")[i : i + 2], 16) / 255 for i in (0, 2, 4)]
    linear = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in channels]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def contrast(a: str, b: str) -> float:
    high, low = sorted((_luminance(a), _luminance(b)), reverse=True)
    return (high + 0.05) / (low + 0.05)


@pytest.fixture(scope="module")
def theme() -> dict:
    return tomllib.loads(CONFIG.read_text())["theme"]


def test_contrast_helper_matches_the_reference_values():
    assert contrast("#000000", "#ffffff") == pytest.approx(21.0)
    assert contrast("#ffffff", "#ffffff") == pytest.approx(1.0)


@pytest.mark.parametrize("mode", ["light", "dark"])
def test_both_modes_are_defined(theme: dict, mode: str):
    assert mode in theme, f"[theme.{mode}] is missing; that mode would fall back to defaults"


@pytest.mark.parametrize("mode", ["light", "dark"])
def test_text_is_comfortably_readable(theme: dict, mode: str):
    palette = theme[mode]
    assert contrast(palette["textColor"], palette["backgroundColor"]) >= 7.0


@pytest.mark.parametrize("mode", ["light", "dark"])
@pytest.mark.parametrize("surface", ["backgroundColor", "secondaryBackgroundColor"])
def test_accent_reads_as_text_on_every_main_surface(theme: dict, mode: str, surface: str):
    palette = theme[mode]
    assert contrast(palette["primaryColor"], palette[surface]) >= AA


@pytest.mark.parametrize("mode", ["light", "dark"])
def test_accent_reads_on_the_sidebar(theme: dict, mode: str):
    palette = theme[mode]
    sidebar = palette["sidebar"]["backgroundColor"]
    assert contrast(palette["primaryColor"], sidebar) >= AA


def test_white_text_is_readable_on_the_light_accent(theme: dict):
    """The headline tile and any primary button put white text on this green."""
    assert contrast("#ffffff", theme["light"]["primaryColor"]) >= AA
