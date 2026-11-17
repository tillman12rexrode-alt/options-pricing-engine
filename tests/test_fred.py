"""The Treasury curve: parsing, conversion, interpolation, and keeping the key secret.

Every test runs offline against synthetic FRED-shaped responses. The yields
below are illustrative round numbers chosen for the test, not market data.
"""

from __future__ import annotations

import math
import urllib.error
from datetime import date

import pytest

from optpricing.providers import fred

#: Illustrative yields in percent, keyed by series. Not real observations.
SAMPLE = {
    "DGS1MO": 4.00, "DGS3MO": 4.10, "DGS6MO": 4.20, "DGS1": 4.25, "DGS2": 4.30,
    "DGS3": 4.35, "DGS5": 4.40, "DGS7": 4.50, "DGS10": 4.60, "DGS20": 4.80, "DGS30": 4.90,
}
SECRET = "0123456789abcdef0123456789abcdef"


def _payload(value: float | str, when: str = "2026-11-13", missing_first: bool = False) -> dict:
    rows = [{"date": when, "value": str(value)}]
    if missing_first:
        rows.insert(0, {"date": "2026-11-14", "value": "."})
    return {"observations": rows}


def _fake_fred(requested: list[str] | None = None, **overrides):
    """A stand-in for the network: answers each series from SAMPLE."""

    def get_json(url: str, timeout: float) -> dict:
        series = url.split("series_id=")[1].split("&")[0]
        if requested is not None:
            requested.append(url)
        if series in overrides:
            return overrides[series]
        return _payload(SAMPLE[series])

    return get_json


def test_conversion_to_continuous_compounding():
    """4% bond-equivalent is 2 ln(1.02), a little under 4% continuous."""
    curve = fred.fetch_curve(SECRET, get_json=_fake_fred())
    assert curve.rate(2.0) == pytest.approx(2 * math.log(1 + 0.043 / 2))
    assert curve.rate(2.0) < 0.043


def test_interpolates_linearly_between_tenors():
    curve = fred.fetch_curve(SECRET, get_json=_fake_fred())
    midpoint = 0.5 * (curve.rate(1.0) + curve.rate(2.0))
    assert curve.rate(1.5) == pytest.approx(midpoint)


def test_holds_flat_beyond_both_ends():
    curve = fred.fetch_curve(SECRET, get_json=_fake_fred())
    assert curve.rate(0.0) == pytest.approx(curve.rate(1 / 12))
    assert curve.rate(50.0) == pytest.approx(curve.rate(30.0))


def test_discount_factor_matches_the_rate():
    curve = fred.fetch_curve(SECRET, get_json=_fake_fred())
    assert curve.discount(0.5) == pytest.approx(math.exp(-curve.rate(0.5) * 0.5))


def test_skips_holidays_marked_with_a_dot():
    """FRED reports a closed market as "." rather than leaving the row out."""
    curve = fred.fetch_curve(
        SECRET, get_json=_fake_fred(DGS2=_payload(4.30, missing_first=True))
    )
    assert curve.observed["DGS2"] == date(2026, 11, 13)


def test_a_series_with_no_data_is_left_out_not_zeroed():
    curve = fred.fetch_curve(SECRET, get_json=_fake_fred(DGS20={"observations": []}))
    assert 20.0 not in curve.tenors
    assert len(curve.tenors) == len(SAMPLE) - 1


def test_curve_date_is_the_stalest_tenor():
    """A curve is only as current as its oldest point."""
    curve = fred.fetch_curve(
        SECRET, get_json=_fake_fred(DGS30=_payload(4.9, when="2026-11-10"))
    )
    assert curve.asof == date(2026, 11, 10)


def test_no_data_at_all_is_an_error():
    empty = {series: {"observations": []} for series in SAMPLE}
    with pytest.raises(fred.FREDError, match="no usable observations"):
        fred.fetch_curve(SECRET, get_json=_fake_fred(**empty))


def test_missing_key_explains_how_to_get_one(monkeypatch):
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    with pytest.raises(fred.MissingKeyError, match="fred.stlouisfed.org"):
        fred.resolve_api_key()


def test_key_is_read_from_the_environment(monkeypatch):
    monkeypatch.setenv("FRED_API_KEY", f"  {SECRET}  ")
    assert fred.resolve_api_key() == SECRET


def test_an_explicit_key_wins_over_the_environment(monkeypatch):
    monkeypatch.setenv("FRED_API_KEY", "from-environment")
    assert fred.resolve_api_key("explicit") == "explicit"


def test_the_key_is_sent_to_fred():
    requested: list[str] = []
    fred.fetch_curve(SECRET, get_json=_fake_fred(requested))
    assert all(f"api_key={SECRET}" in url for url in requested)


@pytest.mark.parametrize(
    "failure",
    [
        urllib.error.HTTPError("https://example.invalid", 400, "Bad Request", None, None),
        urllib.error.URLError("connection refused"),
        TimeoutError("timed out"),
    ],
)
def test_errors_never_reveal_the_key(failure: Exception):
    """The request URL carries the key, and error text ends up in logs and the UI."""

    def failing(url: str, timeout: float) -> dict:
        raise failure

    with pytest.raises(fred.FREDError) as caught:
        fred.fetch_curve(SECRET, get_json=failing)
    assert SECRET not in str(caught.value)
    assert caught.value.__cause__ is None


def test_flat_curve_is_flat():
    curve = fred.flat_curve(0.04, date(2026, 11, 16))
    assert curve.rate(0.1) == curve.rate(10.0) == 0.04
    assert curve.source == "flat assumption"


@pytest.mark.parametrize(
    "tenors, rates",
    [((1.0, 0.5), (0.04, 0.04)), ((0.5,), (float("nan"),)), ((0.5, 1.0), (0.04,))],
)
def test_malformed_curves_are_rejected(tenors: tuple, rates: tuple):
    with pytest.raises(ValueError):
        fred.YieldCurve(date(2026, 11, 16), tenors, rates, "test")


def test_negative_maturity_is_rejected():
    with pytest.raises(ValueError, match="negative"):
        fred.flat_curve(0.04, date(2026, 11, 16)).rate(-0.1)


def test_a_past_date_asks_fred_for_that_day_not_today():
    """A saved snapshot is priced with the rates of the day it was quoted."""
    requested: list[str] = []
    fred.fetch_curve(SECRET, asof=date(2026, 11, 16), get_json=_fake_fred(requested))
    assert all("observation_end=2026-11-16" in url for url in requested)


def test_without_a_date_fred_returns_the_latest():
    requested: list[str] = []
    fred.fetch_curve(SECRET, get_json=_fake_fred(requested))
    assert not any("observation_end" in url for url in requested)
