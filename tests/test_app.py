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


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    """No test reaches FRED, even on a machine that has a key configured."""
    monkeypatch.setenv("OPTPRICING_OFFLINE", "1")


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


def test_the_model_tabs_are_there(app):
    assert [tab.label for tab in app.tabs] == ["Chain", "Model vs market", "Greeks", "Smile"]


def test_model_vs_market_reports_its_checks(app):
    labels = {metric.label: metric.value for metric in app.metric}
    assert labels["Our IV vs Cboe's"].endswith("vol pts")
    assert float(labels["Our IV vs Cboe's"].split()[0]) < 0.3
    assert "of" in labels["Contracts with an implied vol"]
    assert "Quotes off the smile by more than their spread" in labels


def test_without_fred_the_fallback_is_said_out_loud(app):
    assert any("no FRED key" in warning.value for warning in app.warning)
    rate = next(c.value for c in app.caption if c.value.startswith("Rate:"))
    assert "carry this expiry's own options imply" in rate
    assert "4.50% assumption" in rate


def test_a_thin_chain_says_it_is_not_priced(app):
    _ticker(app).set_value("RCAT").run()
    assert not app.exception
    labels = {metric.label: metric.value for metric in app.metric}
    assert labels["Priced off the smile"] == "No"
    assert any("expensive to borrow" in c.value for c in app.caption)


def _research(app: AppTest) -> AppTest:
    app.switch_page("views/research.py").run()
    assert not app.exception
    return app


def test_the_research_page_opens_on_the_same_ticker(app):
    _ticker(app).set_value("TSLA").run()
    research = _research(app)
    assert research.title[0].value == "TSLA"
    assert "Tesla" in research.caption[0].value


def test_research_shows_filings_and_options(app):
    research = _research(app)
    labels = {metric.label: metric.value for metric in research.metric}
    assert labels["Market cap"].endswith("T")
    assert labels["P/E, last 12 months"].endswith("x")
    assert labels["Revenue"].startswith("$")
    headings = [s.value for s in research.subheader]
    assert headings == ["Last twelve months", "Annual results", "Valuation",
                        "What the options price in", "Filings"]
    assert any("sec.gov/Archives" in m.value for m in research.markdown)


def test_research_says_loss_instead_of_a_negative_pe(app):
    research = _research(app)
    _ticker(research).set_value("RCAT").run()
    assert not research.exception
    labels = {metric.label: metric.value for metric in research.metric}
    assert labels["P/E, last 12 months"] == "Loss"


def test_switching_ticker_on_research_carries_back_to_options(app):
    research = _research(app)
    _ticker(research).set_value("NVDA").run()
    research.switch_page("views/stocks.py").run()
    assert research.title[0].value == "NVDA"
