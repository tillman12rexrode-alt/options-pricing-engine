"""Motion and a few finishing touches the Streamlit theme config cannot express.

Colour, type, corner radius and the chart palette live in
.streamlit/config.toml, where Streamlit applies them natively. This module
adds only what config has no key for: transitions, and the metric tiles.

Every animation here follows three rules. It moves only transform and opacity,
so the browser can run it off the main thread. It marks a real change of state:
the page arriving, a tab being opened, a section expanding. And it switches off
entirely for anyone whose system asks for reduced motion.

Colours are translucent tints of the accent rather than fixed hex values, so
the same rules read correctly on the light and the dark theme without knowing
which one is active.

One trap worth knowing: a transformed element becomes the containing block for
any position:fixed descendant, which would trap the data grid's fullscreen view
inside its card. Every keyframe therefore ends at `transform: none`, which does
not create a containing block, and nothing here sets `will-change`.
"""

from __future__ import annotations

import streamlit as st

__all__ = ["apply"]

# The dark-mode accent at low opacity: a visible green tint on both themes.
_TINT = "52, 179, 109"

_CSS = f"""
<style>
@keyframes opt-rise {{
  from {{ opacity: 0; transform: translateY(8px); }}
  to   {{ opacity: 1; transform: none; }}
}}
@keyframes opt-fade {{
  from {{ opacity: 0; }}
  to   {{ opacity: 1; }}
}}

/* Arrival. The page's top-level blocks settle in on first load, in reading
   order. Streamlit keeps these mounted across reruns, so moving a slider does
   not replay it. */
[data-testid="stMainBlockContainer"] > [data-testid="stVerticalBlock"] > * {{
  animation: opt-rise 420ms cubic-bezier(0.2, 0.7, 0.2, 1) both;
}}
[data-testid="stMainBlockContainer"] > [data-testid="stVerticalBlock"] > :nth-child(2) {{ animation-delay: 60ms; }}
[data-testid="stMainBlockContainer"] > [data-testid="stVerticalBlock"] > :nth-child(3) {{ animation-delay: 120ms; }}
[data-testid="stMainBlockContainer"] > [data-testid="stVerticalBlock"] > :nth-child(4) {{ animation-delay: 180ms; }}
[data-testid="stMainBlockContainer"] > [data-testid="stVerticalBlock"] > :nth-child(n+5) {{ animation-delay: 240ms; }}

/* Switching tabs. A hidden panel restarts its animation when shown, so the
   new panel fades in on every switch: a cue that the content changed. */
[data-testid="stTabPanel"] {{
  animation: opt-fade 220ms ease-out both;
}}

/* Expanders: the summary row answers the pointer, the body fades open. */
[data-testid="stExpander"] summary {{
  transition: background-color 160ms ease;
}}
[data-testid="stExpander"] summary:hover {{
  background-color: rgba({_TINT}, 0.08);
}}
[data-testid="stExpanderDetails"] {{
  animation: opt-fade 200ms ease-out both;
}}

/* Tabs: colour eases on hover, and a press gives a small tactile dip. */
[data-baseweb="tab"] {{
  transition: color 160ms ease, transform 120ms ease;
}}
[data-baseweb="tab"]:active {{
  transform: scale(0.97);
}}

/* Metric tiles. The three headline numbers are the summary of the whole
   analysis, which is the hierarchy a card should communicate. Static on
   purpose: they are not clickable, so they do not lift on hover. */
[data-testid="stMetric"] {{
  background: rgba({_TINT}, 0.07);
  border: 1px solid rgba({_TINT}, 0.22);
  border-radius: 0.75rem;
  padding: 0.85rem 1.1rem 0.9rem;
}}
[data-testid="stMetricValue"] {{
  font-variant-numeric: tabular-nums;
  letter-spacing: -0.01em;
}}
[data-testid="stMetricLabel"] p {{
  font-weight: 500;
  opacity: 0.75;
}}

/* The one deliberate block of colour: the underlying's level, the number every
   other figure on the page is measured against. One strong tile among quiet
   ones is found at a glance; three would cancel out. White on this green is
   5.4:1, so it passes AA on either theme. */
.st-key-headline [data-testid="stColumn"]:first-child [data-testid="stMetric"] {{
  background: #157a46;
  border-color: #157a46;
}}
.st-key-headline [data-testid="stColumn"]:first-child [data-testid="stMetric"] * {{
  color: #ffffff;
}}
.st-key-headline [data-testid="stColumn"]:first-child [data-testid="stMetricLabel"] p {{
  opacity: 0.92;
}}
/* A delta badge (a day's price change) on that tile. Its usual red or green
   tint would vanish into the green, so it becomes a translucent white pill;
   the arrow and the sign still say which way the price moved. */
.st-key-headline [data-testid="stColumn"]:first-child [data-testid="stMetricDelta"] {{
  background: rgba(255, 255, 255, 0.18);
}}
.st-key-headline [data-testid="stColumn"]:first-child [data-testid="stMetricDelta"] svg {{
  fill: #ffffff;
}}

@media (prefers-reduced-motion: reduce) {{
  *, *::before, *::after {{
    animation: none !important;
    transition: none !important;
  }}
}}
</style>
"""


def apply() -> None:
    """Inject the motion layer. Call once, straight after st.set_page_config."""
    st.html(_CSS)
