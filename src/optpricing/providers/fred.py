"""The risk-free curve from FRED, for options where put-call parity cannot supply it.

For index options the library implies its rate from the chain itself, by
regressing the call-put spread on strike. That works because European parity
is an equality. American equity options break it, so the rate has to come from
outside, and the natural source is the Treasury constant-maturity curve the
Federal Reserve publishes daily through FRED.

Three conventions, each a deliberate simplification:

- Constant-maturity yields are quoted bond-equivalent, compounding twice a
  year. They are converted to continuous compounding, r = 2 ln(1 + y/2), because
  that is what every pricer in this library expects.
- They are par yields, not zero rates. Out to two or three years, where nearly
  every listed option expires, the two differ by a few basis points, so the curve
  is used as though it were a zero curve rather than bootstrapped.
- Between tenors the rate is interpolated linearly; outside them it is held flat.

The API key is read from an explicit argument or the FRED_API_KEY environment
variable, never from a file this module chooses. Streamlit's st.secrets belongs
to the app; the library does not import Streamlit.
"""

from __future__ import annotations

import json
import math
import os
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date

import numpy as np

__all__ = [
    "SERIES",
    "FREDError",
    "MissingKeyError",
    "YieldCurve",
    "fetch_curve",
    "flat_curve",
    "latest_value",
    "resolve_api_key",
]

URL = "https://api.stlouisfed.org/fred/series/observations"

#: Treasury constant-maturity series and their maturities in years.
SERIES: dict[str, float] = {
    "DGS1MO": 1 / 12,
    "DGS3MO": 0.25,
    "DGS6MO": 0.5,
    "DGS1": 1.0,
    "DGS2": 2.0,
    "DGS3": 3.0,
    "DGS5": 5.0,
    "DGS7": 7.0,
    "DGS10": 10.0,
    "DGS20": 20.0,
    "DGS30": 30.0,
}

#: Enough recent rows to step over holidays, which FRED marks with ".".
_LOOKBACK = 10


class FREDError(RuntimeError):
    """A request to FRED failed. The message never contains the API key."""


class MissingKeyError(FREDError):
    """No API key was supplied and none is set in the environment."""


def resolve_api_key(explicit: str | None = None) -> str:
    """The key to use: an explicit one if given, else FRED_API_KEY from the environment."""
    key = (explicit or os.environ.get("FRED_API_KEY") or "").strip()
    if not key:
        raise MissingKeyError(
            "No FRED API key. Set FRED_API_KEY in the environment, or put it in "
            ".streamlit/secrets.toml for the app. Keys are free at fred.stlouisfed.org."
        )
    return key


@dataclass(frozen=True)
class YieldCurve:
    """Continuously compounded risk-free rates by maturity, as of one day."""

    asof: date
    tenors: tuple[float, ...]
    rates: tuple[float, ...]
    source: str
    #: The date each series was last observed, so staleness is visible per tenor.
    observed: dict[str, date] = field(default_factory=dict, compare=False)

    def __post_init__(self) -> None:
        if len(self.tenors) != len(self.rates) or len(self.tenors) < 1:
            raise ValueError("a curve needs matching, non-empty tenors and rates")
        if any(b <= a for a, b in zip(self.tenors, self.tenors[1:], strict=False)):
            raise ValueError("tenors must be strictly increasing")
        if not all(math.isfinite(r) for r in self.rates):
            raise ValueError("every rate must be finite")

    def rate(self, years: float) -> float:
        """Continuously compounded rate to `years`, interpolated, flat beyond the ends."""
        if years < 0:
            raise ValueError(f"maturity cannot be negative, got {years}")
        return float(np.interp(years, self.tenors, self.rates))

    def discount(self, years: float) -> float:
        return math.exp(-self.rate(years) * years)


def flat_curve(rate: float, asof: date, source: str = "flat assumption") -> YieldCurve:
    """A single rate at every maturity: the documented fallback when FRED is unavailable."""
    return YieldCurve(asof=asof, tenors=(0.0,), rates=(rate,), source=source)


def latest_value(payload: dict) -> tuple[date, float] | None:
    """The most recent observation with a value, as (date, percent), or None.

    FRED reports days with no data, such as market holidays, as the string ".".
    """
    for row in payload.get("observations", []):
        raw = row.get("value", ".")
        if raw in (".", "", None):
            continue
        try:
            return date.fromisoformat(row["date"]), float(raw)
        except (KeyError, ValueError):
            continue
    return None


def _to_continuous(percent: float) -> float:
    """Bond-equivalent yield in percent to a continuously compounded decimal rate."""
    return 2.0 * math.log1p(percent / 100.0 / 2.0)


def _urllib_get_json(url: str, timeout: float) -> dict:
    request = urllib.request.Request(url, headers={"User-Agent": "optpricing-research/0.1"})
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 - fixed https host
        return json.load(response)


def fetch_curve(
    api_key: str | None = None,
    *,
    series: dict[str, float] | None = None,
    get_json: Callable[[str, float], dict] | None = None,
    timeout: float = 10.0,
) -> YieldCurve:
    """Download the latest Treasury curve from FRED.

    `get_json` exists so tests can supply recorded responses instead of touching
    the network. Errors are re-raised without the request URL, because the URL
    carries the API key and exception text tends to end up in logs and in the UI.
    """
    key = resolve_api_key(api_key)
    fetch = get_json or _urllib_get_json
    wanted = SERIES if series is None else series

    points: list[tuple[float, float]] = []
    observed: dict[str, date] = {}
    for series_id, maturity in sorted(wanted.items(), key=lambda kv: kv[1]):
        query = urllib.parse.urlencode({
            "series_id": series_id,
            "api_key": key,
            "file_type": "json",
            "sort_order": "desc",
            "limit": _LOOKBACK,
        })
        try:
            payload = fetch(f"{URL}?{query}", timeout)
        except urllib.error.HTTPError as error:
            raise FREDError(f"FRED returned HTTP {error.code} for {series_id}") from None
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            reason = getattr(error, "reason", error)
            raise FREDError(f"could not reach FRED for {series_id}: {reason}") from None

        latest = latest_value(payload)
        if latest is None:
            continue
        when, percent = latest
        observed[series_id] = when
        points.append((maturity, _to_continuous(percent)))

    if not points:
        raise FREDError("FRED returned no usable observations for any Treasury series")

    tenors, rates = zip(*points, strict=True)
    # The curve is only as current as its stalest tenor.
    return YieldCurve(
        asof=min(observed.values()),
        tenors=tuple(tenors),
        rates=tuple(rates),
        source="FRED Treasury constant maturity",
        observed=observed,
    )
