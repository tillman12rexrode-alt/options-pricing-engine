"""Option chains from Cboe's free delayed-quote feed, for indices and stocks.

One endpoint serves both. Index chains live at `_SPX.json`, with a leading
underscore; stock chains at `AAPL.json`, without one. The payload schema is
the same, and each one carries the underlying's own quote alongside the chain.

Indices and stocks differ in three ways this module has to respect.

Which underlying price the options were quoted against. During the regular
session that is the last trade. After the close it is not: the payload's
current_price moves with extended-hours trading, while the option quotes
stopped at the close. On 16 Nov 2026 AAPL's options implied a forward of
$352.72 against a $352.78 close and a $353.61 after-hours price; RCAT's implied
$3.54 against a $3.53 close and a $2.89 after-hours price. Measuring RCAT's
chain against the after-hours price would have marked every contract as
mispriced by a fifth. So for stocks, outside the session, the spot is the close.

What has expired. Stock chains still list contracts that expired that day.
They have no time left to price, so they are removed and counted.

Missing values. Cboe reports an implied volatility it could not compute as 0.0.
For stocks that is converted to missing, so it cannot be read as a real quote.

The index path is deliberately left exactly as it was: same spot rule, same
rows, same columns. Its results are published and reproduce bit for bit.

Quotes are delayed fifteen minutes. That is irrelevant for studying the shape
of a surface and disqualifying for trading on it.
"""

from __future__ import annotations

import gzip
import json
import math
import re
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, time
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

__all__ = [
    "DELAY_MINUTES",
    "EXCHANGE",
    "INDEX_SYMBOLS",
    "SOURCE",
    "ChainSnapshot",
    "InvalidTickerError",
    "UnknownSymbolError",
    "fetch_chain",
    "fetch_payload",
    "in_regular_session",
    "is_index",
    "list_snapshots",
    "load_snapshot",
    "normalize",
    "parse_payload",
    "payload_symbol",
    "save_payload",
    "select_expiries",
]

URL = "https://cdn.cboe.com/api/global/delayed_quotes/options/{path}.json"

SOURCE = "Cboe delayed quotes"
DELAY_MINUTES = 15

#: European, cash-settled index options. Fetched with a leading underscore.
INDEX_SYMBOLS = {
    "SPX": "S&P 500",
    "XSP": "Mini S&P 500",
    "NDX": "Nasdaq-100",
    "RUT": "Russell 2000",
    "DJX": "Dow Jones Industrial Average",
}

EXCHANGE = ZoneInfo("America/New_York")
SESSION_OPEN = time(9, 30)
SESSION_CLOSE = time(16, 0)

#: OCC option symbol: root, YYMMDD expiry, C or P, strike in thousandths.
SYMBOL = re.compile(r"^(?P<root>[A-Z]+)(?P<expiry>\d{6})(?P<kind>[CP])(?P<strike>\d{8})$")

#: Stock option symbols, whose root may carry a digit after a corporate action
#: (AAPL1). Kept separate from SYMBOL so index parsing is untouched.
EQUITY_SYMBOL = re.compile(
    r"^(?P<root>[A-Z]+\d?)(?P<expiry>\d{6})(?P<kind>[CP])(?P<strike>\d{8})$"
)

#: A plain US listing. Share classes written with a dot (BRK.B) are not supported.
TICKER = re.compile(r"^[A-Z]{1,5}$")

#: Every standard US equity option delivers 100 shares.
STANDARD_MULTIPLIER = 100

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

#: Extra per-contract fields kept for stocks, which the index path never had.
_EQUITY_FIELDS = {
    "gamma": "cboe_gamma",
    "vega": "cboe_vega",
    "theta": "cboe_theta",
    "rho": "cboe_rho",
    "last_trade_time": "last_trade_time",
}

#: The underlying's own quote, kept for display. iv30 arrives in percent.
_UNDERLYING_FIELDS = (
    "current_price", "close", "prev_day_close", "price_change", "price_change_percent",
    "bid", "ask", "open", "high", "low", "volume", "iv30", "last_trade_time",
)


class InvalidTickerError(ValueError):
    """The text cannot be a US ticker at all."""


class UnknownSymbolError(LookupError, ValueError):
    """Cboe has no listed options for this symbol.

    Also a ValueError, which is what the index-only version raised, so any
    caller written against that keeps working.
    """


def normalize(symbol: str) -> str:
    """Upper-cased, stripped and checked. Raises before any network call."""
    cleaned = (symbol or "").strip().upper().lstrip("$^")
    if cleaned in INDEX_SYMBOLS or TICKER.match(cleaned):
        return cleaned
    raise InvalidTickerError(
        f"{symbol!r} is not a US ticker. Use letters only, for example AAPL or SPX."
    )


def is_index(symbol: str) -> bool:
    return symbol.upper() in INDEX_SYMBOLS


def in_regular_session(moment: datetime) -> bool:
    """True between 9:30 and 16:00 New York time on a weekday.

    Exchange holidays are not modelled. On a holiday the feed is idle, so the
    last trade and the close coincide and either reading gives the same spot.
    """
    local = moment.astimezone(EXCHANGE)
    return local.weekday() < 5 and SESSION_OPEN <= local.time() < SESSION_CLOSE


@dataclass(frozen=True)
class ChainSnapshot:
    """Every quote for one underlying at one moment, plus where it came from."""

    symbol: str
    quotes: pd.DataFrame
    spot: float
    asof: datetime
    kind: str = "index"
    #: Which price `spot` is: the index level, a last trade, or the close.
    spot_basis: str = "index level"
    underlying: dict = field(default_factory=dict, compare=False)
    #: Rows removed while parsing, by reason, so the UI can say why.
    dropped: dict = field(default_factory=dict, compare=False)
    source: str = SOURCE
    delay_minutes: int = DELAY_MINUTES

    @property
    def trading_date(self) -> date:
        """The New York session this snapshot belongs to."""
        return self.asof.astimezone(EXCHANGE).date()

    @property
    def expiries(self) -> list[str]:
        return sorted(self.quotes["expiry"].unique())

    def age_minutes(self, now: datetime | None = None) -> float:
        now = datetime.now(UTC) if now is None else now
        return (now - self.asof).total_seconds() / 60.0

    def is_stale(self, now: datetime | None = None) -> bool:
        """Older than the feed's own delay would explain, while the market is open.

        Outside the session nothing newer exists, so a snapshot from the close
        is as current as the data can be and is not called stale.
        """
        now = datetime.now(UTC) if now is None else now
        if not in_regular_session(now):
            return False
        return self.age_minutes(now) > self.delay_minutes + 10

    def to_frame(self) -> pd.DataFrame:
        """Flat table in the shape analysis/fetch_chain.py writes to disk."""
        out = self.quotes.copy()
        out["underlying"] = self.symbol
        out["spot"] = self.spot
        out["asof"] = self.asof.isoformat(sep=" ", timespec="seconds").replace("+00:00", "")
        if self.kind == "equity":
            out["spot_basis"] = self.spot_basis
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


def _positive(meta: dict, key: str) -> float | None:
    try:
        value = float(meta.get(key))
    except (TypeError, ValueError):
        return None
    return value if value > 0 and math.isfinite(value) else None


def _equity_spot(meta: dict, asof: datetime) -> tuple[float, str] | None:
    """The price the option quotes were struck against, and what it is.

    See the module docstring for why the close, not current_price, is right
    once the regular session has ended.
    """
    if in_regular_session(asof):
        order = (("current_price", "last trade"), ("close", "close"))
    else:
        order = (
            ("close", "regular-session close"),
            ("prev_day_close", "previous close"),
            ("current_price", "last trade"),
        )
    for key, label in order:
        value = _positive(meta, key)
        if value is not None:
            return value, label
    return None


def _timestamp(payload: dict) -> datetime:
    stamp = payload.get("timestamp")
    return (
        datetime.fromisoformat(str(stamp)).replace(tzinfo=UTC)
        if stamp
        else datetime.now(UTC)
    )


def _rows(options: list, fields: dict, pattern: re.Pattern = SYMBOL) -> list[dict]:
    rows = []
    for quote in options:
        match = pattern.match(quote.get("option", ""))
        if match is None:
            continue
        row = {
            "root": match["root"],
            "expiry": datetime.strptime(match["expiry"], "%y%m%d").date().isoformat(),
            "kind": "call" if match["kind"] == "C" else "put",
            "strike": int(match["strike"]) / 1000.0,
        }
        row.update({out: quote.get(src) for src, out in fields.items()})
        rows.append(row)
    return rows


def _parse_index(payload: dict, symbol: str) -> ChainSnapshot:
    data = payload.get("data", {})
    rows = _rows(data.get("options") or [], _FIELDS)
    if not rows:
        raise ValueError(f"No parseable option symbols for {symbol}")

    spot = _index_level({k: v for k, v in data.items() if k != "options"})
    if spot is None:
        raise ValueError(f"Could not read an index level for {symbol}")

    return ChainSnapshot(symbol, pd.DataFrame(rows), spot, _timestamp(payload))


def _parse_equity(payload: dict, symbol: str) -> ChainSnapshot:
    data = payload.get("data", {})
    asof = _timestamp(payload)
    meta = {k: v for k, v in data.items() if k != "options"}

    priced = _equity_spot(meta, asof)
    if priced is None:
        raise ValueError(f"Could not read a share price for {symbol}")
    spot, basis = priced

    options = data.get("options") or []
    frame = pd.DataFrame(_rows(options, {**_FIELDS, **_EQUITY_FIELDS}, EQUITY_SYMBOL))
    if frame.empty:
        raise UnknownSymbolError(f"Cboe lists no options for {symbol}")

    dropped: dict[str, int] = {}
    if len(frame) < len(options):
        dropped["unrecognised symbol"] = len(options) - len(frame)

    # Adjusted contracts after a split or merger trade under a modified root
    # (AAPL1) and deliver something other than 100 plain shares.
    standard = frame["root"] == symbol
    if (~standard).any():
        dropped["non-standard deliverable"] = int((~standard).sum())
    frame = frame[standard]

    session = asof.astimezone(EXCHANGE).date()
    live = pd.to_datetime(frame["expiry"]).dt.date > session
    if (~live).any():
        dropped["expired or expiring today"] = int((~live).sum())
    frame = frame[live].copy()

    if frame.empty:
        raise UnknownSymbolError(f"Every listed option for {symbol} has expired")

    frame["cboe_iv"] = pd.to_numeric(frame["cboe_iv"], errors="coerce").where(
        lambda iv: iv > 0
    )
    frame["multiplier"] = STANDARD_MULTIPLIER

    underlying = {key: meta.get(key) for key in _UNDERLYING_FIELDS if key in meta}
    if underlying.get("iv30") is not None:
        underlying["iv30"] = float(underlying["iv30"]) / 100.0

    return ChainSnapshot(
        symbol=symbol,
        quotes=frame.reset_index(drop=True),
        spot=spot,
        asof=asof,
        kind="equity",
        spot_basis=basis,
        underlying=underlying,
        dropped=dropped,
    )


def parse_payload(payload: dict, symbol: str) -> ChainSnapshot:
    """Decode Cboe's JSON into a snapshot, one row per quote."""
    symbol = symbol.upper()
    if not (payload.get("data") or {}).get("options"):
        raise UnknownSymbolError(f"Cboe returned no options for {symbol}")
    if is_index(symbol):
        return _parse_index(payload, symbol)
    return _parse_equity(payload, symbol)


def fetch_payload(symbol: str, *, timeout: float = 30.0) -> dict:
    """Cboe's raw JSON for an index or a stock, exactly as served."""
    symbol = normalize(symbol)
    path = f"_{symbol}" if is_index(symbol) else symbol

    import requests

    response = requests.get(
        URL.format(path=path),
        timeout=timeout,
        allow_redirects=True,
        headers={"User-Agent": "optpricing-research/0.1"},
    )
    if response.status_code in (403, 404):
        raise UnknownSymbolError(f"Cboe has no listed options for {symbol}")
    response.raise_for_status()
    return response.json()


def fetch_chain(symbol: str = "SPX", *, timeout: float = 30.0) -> ChainSnapshot:
    """Download and decode the current delayed chain for an index or a stock."""
    symbol = normalize(symbol)
    return parse_payload(fetch_payload(symbol, timeout=timeout), symbol)


def payload_symbol(payload: dict) -> str:
    """The underlying a raw payload belongs to. Index payloads say "_SPX"."""
    raw = str(payload.get("symbol") or (payload.get("data") or {}).get("symbol") or "")
    return raw.lstrip("_").upper()


def save_payload(payload: dict, directory: Path) -> Path:
    """Store a raw payload, compressed, named by symbol and trading session.

    Stocks are kept raw rather than flattened to CSV so that every parsing rule
    - which price is the spot, what counts as expired - is applied afresh when
    the snapshot is read. Improve a rule and every stored snapshot benefits.
    """
    symbol = payload_symbol(payload)
    if not symbol:
        raise ValueError("payload names no symbol")
    session = _timestamp(payload).astimezone(EXCHANGE).date()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{symbol.lower()}_{session.isoformat()}.json.gz"
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle, separators=(",", ":"))
    return path


def load_snapshot(path: Path) -> ChainSnapshot:
    """Read a stored raw payload back into a snapshot."""
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    return parse_payload(payload, payload_symbol(payload))


def list_snapshots(directory: Path) -> dict[str, list[Path]]:
    """Stored snapshots by symbol, oldest first."""
    found: dict[str, list[Path]] = {}
    for path in sorted(directory.glob("*_????-??-??.json.gz")):
        found.setdefault(path.name.split("_")[0].upper(), []).append(path)
    return found


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
