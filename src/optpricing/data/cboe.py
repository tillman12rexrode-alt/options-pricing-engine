"""Fetch option chains from CBOE's delayed quote feed.

Only cash-settled European index options are offered. Equity options are
American, which breaks put-call parity as an equality - and that equality is
where this library gets its forwards, rates and dividend yields. Inverting
Black-Scholes on an American quote also leaves an early-exercise bias that
cannot be separated from the signal a scanner is looking for.

Quotes are delayed roughly fifteen minutes. That is irrelevant for studying
the shape of a surface and disqualifying for trading on it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, date, datetime
from zoneinfo import ZoneInfo

import pandas as pd

__all__ = ["INDEX_SYMBOLS", "ChainSnapshot", "fetch_chain", "select_expiries"]

URL = "https://cdn.cboe.com/api/global/delayed_quotes/options/_{symbol}.json"

#: European, cash-settled index options. Nothing else is supported on purpose.
INDEX_SYMBOLS = {
    "SPX": "S&P 500",
    "XSP": "Mini S&P 500",
    "NDX": "Nasdaq-100",
    "RUT": "Russell 2000",
    "DJX": "Dow Jones Industrial Average",
}

EXCHANGE = ZoneInfo("America/New_York")

#: OCC option symbol: root, YYMMDD expiry, C or P, strike in thousandths.
SYMBOL = re.compile(r"^(?P<root>[A-Z]+)(?P<expiry>\d{6})(?P<kind>[CP])(?P<strike>\d{8})$")

#: Horizons we would like represented, so a surface has a term structure.
TARGET_DAYS = (7, 30, 60, 90, 180, 365)

_FIELDS = {
    "bid": "bid",
    "ask": "ask",
    "bid_size": "bid_size",
    "ask_size": "ask_size",
    "volume": "volume",
    "open_interest": "open_interest",
    "iv": "cboe_iv",
    "delta": "cboe_delta",
    "theo": "cboe_theo",
    "last_trade_price": "last_trade_price",
}


@dataclass(frozen=True)
class ChainSnapshot:
    """Every quote for one underlying at one moment, plus the index level."""

    symbol: str
    quotes: pd.DataFrame
    spot: float
    asof: datetime

    @property
    def trading_date(self) -> date:
        """The New York session this snapshot belongs to."""
        return self.asof.astimezone(EXCHANGE).date()

    @property
    def expiries(self) -> list[str]:
        return sorted(self.quotes["expiry"].unique())

    def to_frame(self) -> pd.DataFrame:
        """Flat table in the shape analysis/fetch_chain.py writes to disk."""
        out = self.quotes.copy()
        out["underlying"] = self.symbol
        out["spot"] = self.spot
        out["asof"] = self.asof.isoformat(sep=" ", timespec="seconds").replace("+00:00", "")
        return out


def _index_level(meta: dict) -> float | None:
    for key in ("current_price", "close", "last", "prev_day_close", "open"):
        try:
            level = float(meta.get(key))
        except (TypeError, ValueError):
            continue
        if level > 0:
            return level
    return None


def parse_payload(payload: dict, symbol: str) -> ChainSnapshot:
    """Decode CBOE's JSON into a snapshot, one row per quote."""
    data = payload.get("data", {})
    options = data.get("options") or []
    if not options:
        raise ValueError(f"CBOE returned no options for {symbol}")

    rows = []
    for quote in options:
        match = SYMBOL.match(quote.get("option", ""))
        if match is None:
            continue
        row = {
            "root": match["root"],
            "expiry": datetime.strptime(match["expiry"], "%y%m%d").date().isoformat(),
            "kind": "call" if match["kind"] == "C" else "put",
            "strike": int(match["strike"]) / 1000.0,
        }
        row.update({out: quote.get(src) for src, out in _FIELDS.items()})
        rows.append(row)

    if not rows:
        raise ValueError(f"No parseable option symbols for {symbol}")

    stamp = payload.get("timestamp")
    asof = (
        datetime.fromisoformat(str(stamp)).replace(tzinfo=UTC)
        if stamp
        else datetime.now(UTC)
    )
    spot = _index_level({k: v for k, v in data.items() if k != "options"})
    if spot is None:
        raise ValueError(f"Could not read an index level for {symbol}")

    return ChainSnapshot(symbol, pd.DataFrame(rows), spot, asof)


def fetch_chain(symbol: str = "SPX", *, timeout: float = 30.0) -> ChainSnapshot:
    """Download and decode the current delayed chain for an index."""
    symbol = symbol.upper()
    if symbol not in INDEX_SYMBOLS:
        raise ValueError(
            f"{symbol!r} is not a supported index. "
            f"Choose from {', '.join(sorted(INDEX_SYMBOLS))}. "
            "Equity options are American and this library's method does not apply to them."
        )

    import requests

    response = requests.get(
        URL.format(symbol=symbol),
        timeout=timeout,
        allow_redirects=True,
        headers={"User-Agent": "optpricing-research/0.1"},
    )
    response.raise_for_status()
    return parse_payload(response.json(), symbol)


def select_expiries(
    available: list[str], asof: date, targets: tuple[int, ...] = TARGET_DAYS
) -> list[str]:
    """Pick the expiry nearest each target horizon, so the surface spans maturities.

    Index options expire daily, so taking the first few expirations gives a
    fortnight of data and no term structure whatsoever.
    """
    dated = sorted((date.fromisoformat(e), e) for e in available if date.fromisoformat(e) > asof)
    if not dated:
        return []

    chosen: list[str] = []
    for target in targets:
        nearest = min(dated, key=lambda item: abs((item[0] - asof).days - target))
        if nearest[1] not in chosen:
            chosen.append(nearest[1])
    return chosen
