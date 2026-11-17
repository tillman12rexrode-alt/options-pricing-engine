"""The app, driven headless through Streamlit's own test harness.

These run the real pages against the snapshots committed in data/. They catch
what unit tests cannot: a page that raises, a widget that forgets what the
visitor picked, a choice that leaks from one ticker into the next.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = str(Path(__file__).resolve().parent.parent / "app" / "streamlit_app.py")


@pytest.fixture
def app(monkeypatch) -> AppTest:
    monkeypatch.delenv("OPTPRICING_LIVE_DATA", raising=False)
    return AppTest.from_file(APP, default_timeout=120).run()


def _ticker(at: AppTest):
    return next(box for box in at.sidebar.selectbox if box.label == "Ticker")


def test_the_stock_page_opens_without_errors(app):
    assert not app.exception
    assert app.title[0].value == "AAPL"
    assert app.metric[0].label == "Share price"


def test_switching_tickers_works_in_both_directions(app):
    """Regression: the dropdown once discarded a pick when going back."""
    for pick, price in (("RCAT", "$3.53"), ("AAPL", "$352.78"), ("RCAT", "$3.53")):
        _ticker(app).set_value(pick).run()
        assert app.title[0].value == pick
        assert app.metric[0].value == price


def test_an_expiry_never_carries_over_to_another_ticker(app):
    """AAPL's chosen expiry must not survive a switch; RCAT gets its own default."""
    app.selectbox(key="expiry").set_value("2027-01-15").run()
    assert app.selectbox(key="expiry").value == "2027-01-15"
    _ticker(app).set_value("RCAT").run()
    assert app.selectbox(key="expiry").value == "2026-12-18"  # RCAT's nearest to 30 days


def test_the_public_app_offers_no_live_data(app):
    """Bundled snapshots only, unless live data is switched on locally."""
    assert not any(radio.label == "Data" for radio in app.sidebar.radio)


def test_live_data_appears_when_switched_on(monkeypatch):
    monkeypatch.setenv("OPTPRICING_LIVE_DATA", "1")
    at = AppTest.from_file(APP, default_timeout=120).run()
    data = next(radio for radio in at.sidebar.radio if radio.label == "Data")
    assert any("Live" in option for option in data.options)


def test_the_index_page_still_runs(app):
    """The original scanner, now its own page, must load and analyse as before."""
    app.switch_page("views/index.py").run()
    assert not app.exception
    assert app.title[0].value == "Surface Scanner"
    assert app.metric[2].label == "Quotes solved"
    assert app.metric[2].value == "2,260"


def test_a_cheap_stock_still_shows_a_useful_number_of_strikes(app):
    """At $3.53 with $1 strikes, a 25% window alone would leave two rows."""
    _ticker(app).set_value("RCAT").run()
    chain = next(frame for frame in app.dataframe if "Strike" in frame.value.columns)
    assert len(chain.value) >= 8
