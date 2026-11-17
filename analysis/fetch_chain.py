"""Write a dated option-chain snapshot: an index to data/, a stock to data/equity/.

This is the one step that needs live market data, so it has to run on a machine
with direct internet access - not inside a sandbox. Everything downstream reads
the committed snapshot, so every figure regenerates without a network.

The fetching and decoding live in optpricing.data.cboe; this is only the
command-line wrapper around them.

    python analysis/fetch_chain.py                 # SPX, expiries across the curve
    python analysis/fetch_chain.py --symbol NDX
    python analysis/fetch_chain.py --symbol AAPL   # a stock: raw payload, every expiry
    python analysis/fetch_chain.py --symbol AAPL NVDA TSLA KTOS RCAT
    python analysis/fetch_chain.py --probe         # report only, write nothing
    python analysis/fetch_chain.py --all-expiries  # keep every expiry offered

Index snapshots are flattened to CSV, which is what the published analysis
reads. Stock snapshots are stored as Cboe's raw response, compressed, so every
parsing rule is applied afresh whenever they are read.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from optpricing.providers import cboe
from optpricing.providers.cboe import (
    INDEX_SYMBOLS,
    ChainSnapshot,
    fetch_chain,
    select_expiries,
)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
EQUITY = DATA / "equity"


def report(snapshot: ChainSnapshot, selected: list[str]) -> None:
    quotes = snapshot.quotes
    name = INDEX_SYMBOLS.get(snapshot.symbol, "stock")
    print(f"symbol            {snapshot.symbol} ({name})")
    print(f"timestamp         {snapshot.asof:%Y-%m-%d %H:%M:%S %Z}")
    print(f"trading session   {snapshot.trading_date}")
    print(f"underlying        {snapshot.spot:,.2f} ({snapshot.spot_basis})")
    print(f"quotes returned   {len(quotes):,}")
    print(f"expiries offered  {quotes['expiry'].nunique()}")
    print(f"roots             {sorted(quotes['root'].unique())}")

    zero_bid = int((quotes["bid"].fillna(0) <= 0).sum())
    print(f"zero or missing bid: {zero_bid:,} of {len(quotes):,}\n")

    print("selected expiries:" if snapshot.kind == "index" else "expiries (all kept):")
    for expiry in selected:
        rows = quotes[quotes["expiry"] == expiry]
        days = (type(snapshot.trading_date).fromisoformat(expiry) - snapshot.trading_date).days
        print(f"  {expiry}  {days:>4}d  {len(rows):>5} quotes  roots={sorted(rows['root'].unique())}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", nargs="+", default=["SPX"], help="index or stock tickers")
    parser.add_argument("--probe", action="store_true", help="report only, write nothing")
    parser.add_argument("--all-expiries", action="store_true", help="keep every expiry")
    args = parser.parse_args()

    try:
        import requests  # noqa: F401
    except ImportError:
        raise SystemExit('requests is not installed. Run:  pip install -e ".[data]"') from None

    for symbol in args.symbol:
        try:
            symbol = cboe.normalize(symbol)
        except cboe.InvalidTickerError as error:
            print(f"skipping: {error}")
            continue
        if cboe.is_index(symbol):
            fetch_index(symbol, args)
        else:
            fetch_stock(symbol, args)
        print()


def fetch_index(symbol: str, args: argparse.Namespace) -> None:
    snapshot = fetch_chain(symbol)
    selected = (
        snapshot.expiries
        if args.all_expiries
        else select_expiries(snapshot.expiries, snapshot.trading_date)
    )

    report(snapshot, selected)

    if args.probe:
        print("\n--probe: nothing written.")
        return

    frame = snapshot.to_frame()
    frame = frame[frame["expiry"].isin(selected)]

    DATA.mkdir(exist_ok=True)
    path = DATA / f"{symbol.lower()}_chain_{snapshot.trading_date:%Y-%m-%d}.csv"
    frame.to_csv(path, index=False)
    print(f"\nSnapshot written to {path}  ({len(frame):,} rows, {len(selected)} expiries)")
    print("Commit this file - every figure downstream is built from it.")


def fetch_stock(symbol: str, args: argparse.Namespace) -> None:
    try:
        payload = cboe.fetch_payload(symbol)
        snapshot = cboe.parse_payload(payload, symbol)
    except cboe.UnknownSymbolError as error:
        print(f"skipping: {error}")
        return

    report(snapshot, snapshot.expiries)
    for reason, count in snapshot.dropped.items():
        print(f"  dropped {count:>5}  {reason}")

    if args.probe:
        print("\n--probe: nothing written.")
        return

    path = cboe.save_payload(payload, EQUITY)
    print(f"\nSnapshot written to {path.relative_to(ROOT)}  ({path.stat().st_size // 1024} KB)")

if __name__ == "__main__":
    sys.exit(main())
