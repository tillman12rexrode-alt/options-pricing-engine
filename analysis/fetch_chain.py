"""Fetch a dated SPX option-chain snapshot from CBOE and write it to data/.

This is the one step that needs live market data, so it has to run on a machine
with direct internet access - not inside a sandbox. Everything downstream reads
the committed snapshot, so the figures regenerate without a network.

SPX rather than SPY: SPX options are European and cash-settled on the index,
which is exactly the contract Black-Scholes was written for. SPY options are
American on a dividend-paying ETF, so implied volatilities taken from them
carry an early-exercise error that cannot be separated out. Yahoo publishes no
book for the index, so the quotes come from the exchange's own delayed feed.

Expiries are sampled across the curve rather than taken in order: SPX expires
daily, so the first eight expirations span about ten calendar days and contain
no term structure at all.

Run from the repository root:

    python analysis/fetch_chain.py             # write a snapshot
    python analysis/fetch_chain.py --probe     # report only, write nothing
"""

from __future__ import annotations

import argparse
import re
import sys
from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

CBOE_URL = "https://cdn.cboe.com/api/global/delayed_quotes/options/_{symbol}.json"

#: OCC option symbol: root, YYMMDD expiry, C or P, strike in thousandths.
SYMBOL = re.compile(r"^(?P<root>[A-Z]+)(?P<expiry>\d{6})(?P<kind>[CP])(?P<strike>\d{8})$")

#: Days to expiry we would like represented, so the surface has a term structure.
TARGET_DAYS = (7, 30, 60, 90, 180, 365)

KEEP = {
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


def download(symbol: str) -> dict:
    import requests

    response = requests.get(
        CBOE_URL.format(symbol=symbol),
        timeout=30,
        allow_redirects=True,
        headers={"User-Agent": "optpricing-research/0.1"},
    )
    response.raise_for_status()
    return response.json()


def parse(payload: dict) -> tuple[pd.DataFrame, dict]:
    """Flatten the option list, decoding each OCC symbol into its fields."""
    data = payload.get("data", {})
    options = data.get("options") or []
    if not options:
        raise SystemExit("CBOE returned no options; the symbol or endpoint may have changed.")

    rows = []
    unparsed = 0
    for quote in options:
        match = SYMBOL.match(quote.get("option", ""))
        if match is None:
            unparsed += 1
            continue

        row = {
            "root": match["root"],
            "expiry": datetime.strptime(match["expiry"], "%y%m%d").date().isoformat(),
            "kind": "call" if match["kind"] == "C" else "put",
            "strike": int(match["strike"]) / 1000.0,
        }
        row.update({out: quote.get(src) for src, out in KEEP.items()})
        rows.append(row)

    meta = {k: v for k, v in data.items() if k != "options"}
    meta["timestamp"] = payload.get("timestamp")
    meta["unparsed_symbols"] = unparsed
    return pd.DataFrame(rows), meta


def underlying_level(meta: dict) -> float | None:
    """CBOE labels the index level differently across endpoints, so try each."""
    for key in ("current_price", "close", "last", "prev_day_close", "open"):
        value = meta.get(key)
        try:
            level = float(value)
        except (TypeError, ValueError):
            continue
        if level > 0:
            return level
    return None


def select_expiries(available: list[str], asof: date, targets=TARGET_DAYS) -> list[str]:
    """Pick the expiry closest to each target horizon, skipping duplicates."""
    dated = sorted((date.fromisoformat(e), e) for e in available if date.fromisoformat(e) > asof)
    if not dated:
        return []

    chosen: list[str] = []
    for target in targets:
        nearest = min(dated, key=lambda item: abs((item[0] - asof).days - target))
        if nearest[1] not in chosen:
            chosen.append(nearest[1])
    return chosen


def report(chain: pd.DataFrame, meta: dict, spot: float | None, selected: list[str]) -> None:
    print(f"timestamp         {meta.get('timestamp')}")
    print(f"index level       {spot}")
    print(f"quotes returned   {len(chain)}  (unparsed symbols: {meta['unparsed_symbols']})")
    print(f"expiries offered  {chain['expiry'].nunique()}")
    print(f"roots             {sorted(chain['root'].unique())}\n")

    print("metadata keys from CBOE:")
    for key, value in meta.items():
        shown = value if not isinstance(value, (list, dict)) else f"<{type(value).__name__}>"
        print(f"  {key:<22}{shown}")

    zero_bid = int((chain["bid"].fillna(0) <= 0).sum())
    print(f"\nquotes with a zero or missing bid: {zero_bid} of {len(chain)}")

    print("\nselected expiries:")
    asof = date.fromisoformat(str(meta.get("timestamp", ""))[:10])
    for expiry in selected:
        days = (date.fromisoformat(expiry) - asof).days
        rows = chain[chain["expiry"] == expiry]
        print(f"  {expiry}  {days:>4}d  {len(rows):>5} quotes  roots={sorted(rows['root'].unique())}")

    print("\nfirst five rows of the nearest selected expiry:")
    columns = ["expiry", "kind", "strike", "bid", "ask", "volume", "open_interest", "cboe_iv"]
    print(chain[chain["expiry"] == selected[0]][columns].head(5).to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default="SPX")
    parser.add_argument("--probe", action="store_true", help="report only, write nothing")
    parser.add_argument("--all-expiries", action="store_true", help="keep every expiry")
    args = parser.parse_args()

    try:
        import requests  # noqa: F401
    except ImportError:
        raise SystemExit('requests is not installed. Run:  pip install -e ".[data]"') from None

    chain, meta = parse(download(args.symbol))
    spot = underlying_level(meta)

    stamp = str(meta.get("timestamp") or datetime.now(UTC).isoformat())
    asof = date.fromisoformat(stamp[:10])
    selected = (
        sorted(chain["expiry"].unique())
        if args.all_expiries
        else select_expiries(sorted(chain["expiry"].unique()), asof)
    )

    report(chain, meta, spot, selected)

    if args.probe:
        print("\n--probe: nothing written.")
        return

    snapshot = chain[chain["expiry"].isin(selected)].copy()
    snapshot["underlying"] = args.symbol
    snapshot["spot"] = spot
    snapshot["asof"] = stamp

    DATA.mkdir(exist_ok=True)
    path = DATA / f"{args.symbol.lower()}_chain_{asof:%Y-%m-%d}.csv"
    snapshot.to_csv(path, index=False)
    print(f"\nSnapshot written to {path}  ({len(snapshot)} rows, {len(selected)} expiries)")
    print("Commit this file - every figure downstream is built from it.")


if __name__ == "__main__":
    sys.exit(main())
