"""Write a dated index option-chain snapshot to data/.

This is the one step that needs live market data, so it has to run on a machine
with direct internet access - not inside a sandbox. Everything downstream reads
the committed snapshot, so every figure regenerates without a network.

The fetching and decoding live in optpricing.data.cboe; this is only the
command-line wrapper around them.

    python analysis/fetch_chain.py                 # SPX, expiries across the curve
    python analysis/fetch_chain.py --symbol NDX
    python analysis/fetch_chain.py --probe         # report only, write nothing
    python analysis/fetch_chain.py --all-expiries  # keep every expiry offered
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from optpricing.data.cboe import INDEX_SYMBOLS, ChainSnapshot, fetch_chain, select_expiries

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"


def report(snapshot: ChainSnapshot, selected: list[str]) -> None:
    quotes = snapshot.quotes
    print(f"symbol            {snapshot.symbol} ({INDEX_SYMBOLS[snapshot.symbol]})")
    print(f"timestamp         {snapshot.asof:%Y-%m-%d %H:%M:%S %Z}")
    print(f"trading session   {snapshot.trading_date}")
    print(f"index level       {snapshot.spot:,.2f}")
    print(f"quotes returned   {len(quotes):,}")
    print(f"expiries offered  {quotes['expiry'].nunique()}")
    print(f"roots             {sorted(quotes['root'].unique())}")

    zero_bid = int((quotes["bid"].fillna(0) <= 0).sum())
    print(f"zero or missing bid: {zero_bid:,} of {len(quotes):,}\n")

    print("selected expiries:")
    for expiry in selected:
        rows = quotes[quotes["expiry"] == expiry]
        days = (type(snapshot.trading_date).fromisoformat(expiry) - snapshot.trading_date).days
        print(f"  {expiry}  {days:>4}d  {len(rows):>5} quotes  roots={sorted(rows['root'].unique())}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default="SPX", choices=sorted(INDEX_SYMBOLS))
    parser.add_argument("--probe", action="store_true", help="report only, write nothing")
    parser.add_argument("--all-expiries", action="store_true", help="keep every expiry")
    args = parser.parse_args()

    try:
        import requests  # noqa: F401
    except ImportError:
        raise SystemExit('requests is not installed. Run:  pip install -e ".[data]"') from None

    snapshot = fetch_chain(args.symbol)
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
    path = DATA / f"{args.symbol.lower()}_chain_{snapshot.trading_date:%Y-%m-%d}.csv"
    frame.to_csv(path, index=False)
    print(f"\nSnapshot written to {path}  ({len(frame):,} rows, {len(selected)} expiries)")
    print("Commit this file - every figure downstream is built from it.")


if __name__ == "__main__":
    sys.exit(main())
