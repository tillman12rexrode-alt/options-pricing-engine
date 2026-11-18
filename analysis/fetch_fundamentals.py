"""Write a dated SEC EDGAR snapshot of each company to data/fundamentals/.

Like fetch_chain.py, this is the one step that needs the network, so it runs on
your own machine. The app and the tests read only the committed snapshots.

The SEC asks every request to name who is making it, with a contact email
(https://www.sec.gov/os/accessing-edgar-data). Supply yours either way:

    export SEC_USER_AGENT="Your Name you@example.com"
    python analysis/fetch_fundamentals.py --symbol AAPL NVDA TSLA KTOS RCAT

or let the script ask for it. It is sent to the SEC only, and is never written
into the snapshot.

    python analysis/fetch_fundamentals.py --symbol AAPL --probe   # report, write nothing
"""

from __future__ import annotations

import argparse
import getpass
import os
import sys
from datetime import date
from pathlib import Path

from optpricing import fundamentals
from optpricing.providers import sec

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "fundamentals"


def report(f: fundamentals.Fundamentals) -> None:
    print(f"company           {f.name} (CIK {f.cik})")
    print(f"industry          {f.industry}")
    print(f"fiscal year ends  {f.fiscal_year_end}")
    print(f"latest period     {f.latest_end}")
    years = ", ".join(fundamentals.fiscal_label(y) for y in f.years)
    print(f"annual history    {years or 'none'}")
    revenue = f.ttm.get("revenue")
    print(f"revenue, 12 mo    {'n/a' if revenue is None else f'${revenue:,.0f}'}")
    if f.missing:
        print(f"not reported      {', '.join(f.missing)}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--symbol", nargs="+", required=True, help="stock tickers")
    parser.add_argument("--probe", action="store_true", help="report only, write nothing")
    args = parser.parse_args()

    agent = os.environ.get("SEC_USER_AGENT") or getpass.getpass(
        "Your name and email for the SEC (not shown, not saved): "
    )
    try:
        agent = sec.resolve_user_agent(agent)
    except sec.MissingUserAgentError as error:
        print(error)
        return 1

    tickers = None
    failures = 0
    for symbol in args.symbol:
        symbol = symbol.strip().upper()
        try:
            if tickers is None:
                tickers = sec.fetch_tickers(agent)
            cik = sec.lookup_cik(symbol, tickers)
            snapshot = sec.fetch_snapshot(symbol, agent, cik=cik, today=date.today())
        except (sec.SECError, sec.UnknownCompanyError) as error:
            print(f"{symbol}: skipped. {error}\n")
            failures += 1
            continue
        report(fundamentals.analyze(snapshot))
        if args.probe:
            print("--probe: nothing written.\n")
            continue
        path = sec.save_snapshot(snapshot, OUT)
        print(f"written           {path.relative_to(ROOT)}  ({path.stat().st_size // 1024} KB)\n")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
