"""Company filings from SEC EDGAR: who a company is, and what it reported.

EDGAR's JSON APIs are free, need no key, and carry the numbers exactly as each
company tagged them in its 10-K and 10-Q filings. The SEC asks two things of
anyone calling them (https://www.sec.gov/os/accessing-edgar-data):

- Say who you are. Every request carries a User-Agent naming the requester and
  a contact email. It is read from the SEC_USER_AGENT environment variable or
  passed in, never hard-coded, because it is the user's own email address.
- Stay under ten requests a second. This module waits between requests.

Two endpoints are used, per company, keyed by its ten-digit CIK:

- submissions: name, industry (SIC), exchange, fiscal year-end, and recent
  filings. Saved trimmed to the profile and the 10-K/10-Q filings.
- companyfacts: every XBRL fact the company has ever filed, typically 2-4 MB.
  Saved trimmed to the concepts `fundamentals` reads, which brings a snapshot
  down to tens of kilobytes.

Snapshots are stored as gzipped JSON in data/fundamentals/, one per company
per fetch day, the same way option chains are stored in data/equity/.
"""

from __future__ import annotations

import gzip
import json
import os
import re
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

__all__ = [
    "FILING_FORMS",
    "FundamentalsSnapshot",
    "MissingUserAgentError",
    "SECError",
    "UnknownCompanyError",
    "build_snapshot",
    "fetch_snapshot",
    "fetch_tickers",
    "filing_url",
    "list_snapshots",
    "load_snapshot",
    "lookup_cik",
    "resolve_user_agent",
    "save_snapshot",
    "trim_facts",
    "trim_submissions",
]

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{document}"
SOURCE = "SEC EDGAR"

#: The SEC's published ceiling is ten a second; stay well under it.
MIN_INTERVAL = 0.15

#: Periodic reports. Amendments are kept for the filings list but a 10-K/A
#: often restates only Part III (executive pay), so `fundamentals` takes its
#: numbers from the original forms first.
FILING_FORMS = ("10-K", "10-Q", "10-KT", "10-QT", "10-K/A", "10-Q/A")

#: Profile fields kept from the submissions endpoint.
_PROFILE = (
    "cik", "name", "tickers", "exchanges", "sic", "sicDescription", "fiscalYearEnd",
    "category", "entityType", "stateOfIncorporation", "formerNames",
)

#: How many periodic filings to keep in a snapshot: three years of 10-Qs and 10-Ks.
_KEEP_FILINGS = 16

_EMAIL = re.compile(r"[^@\s]+@[^@\s]+\.[^@\s]+")


class SECError(RuntimeError):
    """A request to EDGAR failed."""


class MissingUserAgentError(SECError):
    """No contact User-Agent was supplied, and the SEC requires one."""


class UnknownCompanyError(LookupError, ValueError):
    """The ticker is not in the SEC's list of companies that file with it."""


def resolve_user_agent(explicit: str | None = None) -> str:
    """The User-Agent to send: a name and an email, as the SEC asks."""
    agent = (explicit or os.environ.get("SEC_USER_AGENT") or "").strip()
    if not agent:
        raise MissingUserAgentError(
            "The SEC asks every request to say who is making it. Set SEC_USER_AGENT to "
            "your name and email, e.g. 'Jane Doe jane@example.com'."
        )
    if not _EMAIL.search(agent):
        raise MissingUserAgentError("SEC_USER_AGENT needs a contact email address in it.")
    return agent


# ------------------------------------------------------------------- the snapshot


@dataclass(frozen=True)
class FundamentalsSnapshot:
    """One company's profile, filings list and reported facts, as of one fetch."""

    symbol: str
    cik: int
    fetched: date
    profile: dict
    filings: list[dict]
    facts: dict
    source: str = SOURCE
    path: Path | None = field(default=None, compare=False)

    @property
    def name(self) -> str:
        return self.profile.get("name") or self.symbol

    def latest_filing(self, *forms: str) -> dict | None:
        """The most recent filing of any of `forms`, by filing date."""
        wanted = forms or ("10-K", "10-Q")
        matches = [f for f in self.filings if f["form"] in wanted]
        return max(matches, key=lambda f: f["filed"]) if matches else None


def filing_url(cik: int, filing: dict) -> str:
    """Link to a filing's main document on sec.gov."""
    return ARCHIVE_URL.format(
        cik=int(cik),
        accession=filing["accession"].replace("-", ""),
        document=filing["document"],
    )


def trim_submissions(payload: dict) -> tuple[dict, list[dict]]:
    """The profile fields, and the recent periodic filings, newest first."""
    profile = {key: payload.get(key) for key in _PROFILE if key in payload}
    recent = payload.get("filings", {}).get("recent", {})
    rows = zip(
        recent.get("form", []),
        recent.get("filingDate", []),
        recent.get("reportDate", []),
        recent.get("accessionNumber", []),
        recent.get("primaryDocument", []),
        strict=False,
    )
    filings = [
        {"form": form, "filed": filed, "period": period, "accession": acc, "document": doc}
        for form, filed, period, acc, doc in rows
        if form in FILING_FORMS
    ]
    filings.sort(key=lambda f: f["filed"], reverse=True)
    return profile, filings[:_KEEP_FILINGS]


def trim_facts(payload: dict, concepts: Iterable[tuple[str, str]]) -> dict:
    """Only the (taxonomy, concept) pairs asked for, from periodic reports.

    Each companyfacts fact appears once per filing that reported it, including
    as a prior-period comparative. Those repeats are kept: `fundamentals` uses
    the filing date to prefer the latest version of each number.
    """
    facts = payload.get("facts", {})
    kept: dict[str, dict] = {}
    for taxonomy, concept in concepts:
        source = facts.get(taxonomy, {}).get(concept)
        if not source:
            continue
        units = {}
        for unit, rows in source.get("units", {}).items():
            periodic = [
                {k: r[k] for k in ("start", "end", "val", "fy", "fp", "form", "filed", "accn") if k in r}
                for r in rows
                if r.get("form") in FILING_FORMS
            ]
            if periodic:
                units[unit] = periodic
        if units:
            kept.setdefault(taxonomy, {})[concept] = {"label": source.get("label"), "units": units}
    return kept


def build_snapshot(
    symbol: str,
    submissions: dict,
    companyfacts: dict,
    fetched: date,
    concepts: Iterable[tuple[str, str]] | None = None,
) -> FundamentalsSnapshot:
    """A snapshot from the two raw EDGAR payloads."""
    if concepts is None:
        from optpricing.fundamentals import CONCEPTS as concepts  # noqa: N811
    profile, filings = trim_submissions(submissions)
    cik = int(profile.get("cik") or companyfacts.get("cik"))
    return FundamentalsSnapshot(
        symbol=symbol.upper(),
        cik=cik,
        fetched=fetched,
        profile=profile,
        filings=filings,
        facts=trim_facts(companyfacts, concepts),
    )


# ---------------------------------------------------------------------- storage


def _name(symbol: str, fetched: date) -> str:
    return f"{symbol.lower()}_{fetched.isoformat()}.json.gz"


def save_snapshot(snapshot: FundamentalsSnapshot, directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / _name(snapshot.symbol, snapshot.fetched)
    body = {
        "symbol": snapshot.symbol,
        "cik": snapshot.cik,
        "fetched": snapshot.fetched.isoformat(),
        "source": snapshot.source,
        "profile": snapshot.profile,
        "filings": snapshot.filings,
        "facts": snapshot.facts,
    }
    # mtime=0 keeps the file byte-identical when the content has not changed.
    path.write_bytes(gzip.compress(json.dumps(body, sort_keys=True).encode(), mtime=0))
    return path


def load_snapshot(path: Path) -> FundamentalsSnapshot:
    body = json.loads(gzip.decompress(Path(path).read_bytes()))
    return FundamentalsSnapshot(
        symbol=body["symbol"],
        cik=int(body["cik"]),
        fetched=date.fromisoformat(body["fetched"]),
        profile=body["profile"],
        filings=body["filings"],
        facts=body["facts"],
        source=body.get("source", SOURCE),
        path=Path(path),
    )


def list_snapshots(directory: Path) -> dict[str, list[Path]]:
    """Saved snapshots by ticker, newest last."""
    found: dict[str, list[Path]] = {}
    for path in sorted(Path(directory).glob("*_????-??-??.json.gz")):
        found.setdefault(path.name.split("_")[0].upper(), []).append(path)
    return found


# ------------------------------------------------------------------------ fetch


def _urllib_get_json(url: str, user_agent: str, timeout: float) -> dict:
    request = urllib.request.Request(
        url, headers={"User-Agent": user_agent, "Accept-Encoding": "gzip"}
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 - fixed https hosts
        raw = response.read()
        if response.headers.get("Content-Encoding") == "gzip":
            raw = gzip.decompress(raw)
    return json.loads(raw)


class _Client:
    """Sends the User-Agent, spaces requests out, and turns failures into SECError."""

    def __init__(self, user_agent: str, get_json, timeout: float, sleep=time.sleep):
        self.user_agent = user_agent
        self.get_json = get_json or _urllib_get_json
        self.timeout = timeout
        self.sleep = sleep
        self._last = 0.0

    def __call__(self, url: str) -> dict:
        wait = MIN_INTERVAL - (time.monotonic() - self._last)
        if wait > 0:
            self.sleep(wait)
        try:
            return self.get_json(url, self.user_agent, self.timeout)
        except urllib.error.HTTPError as error:
            hint = " The SEC refuses requests without a contact User-Agent." if error.code == 403 else ""
            raise SECError(f"EDGAR returned HTTP {error.code} for {url}.{hint}") from None
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            raise SECError(f"could not reach EDGAR: {getattr(error, 'reason', error)}") from None
        finally:
            self._last = time.monotonic()


def lookup_cik(symbol: str, tickers: dict) -> int:
    """The CIK for a ticker, from the SEC's company_tickers.json."""
    wanted = symbol.strip().upper()
    for row in tickers.values():
        if str(row.get("ticker", "")).upper() == wanted:
            return int(row["cik_str"])
    raise UnknownCompanyError(
        f"{wanted} is not in the SEC's list of companies. Funds, ETFs and indices "
        "do not file 10-Ks, so they have no fundamentals here."
    )


def fetch_tickers(
    user_agent: str | None = None,
    *,
    get_json: Callable[[str, str, float], dict] | None = None,
    timeout: float = 30.0,
) -> dict:
    """The SEC's ticker-to-CIK list. Fetch once and pass it to `lookup_cik` per ticker."""
    return _Client(resolve_user_agent(user_agent), get_json, timeout)(TICKERS_URL)


def fetch_snapshot(
    symbol: str,
    user_agent: str | None = None,
    *,
    cik: int | None = None,
    today: date | None = None,
    get_json: Callable[[str, str, float], dict] | None = None,
    timeout: float = 30.0,
) -> FundamentalsSnapshot:
    """Download one company's profile and facts: two or three requests."""
    client = _Client(resolve_user_agent(user_agent), get_json, timeout)
    if cik is None:
        cik = lookup_cik(symbol, client(TICKERS_URL))
    padded = f"{int(cik):010d}"
    submissions = client(SUBMISSIONS_URL.format(cik=padded))
    facts = client(FACTS_URL.format(cik=padded))
    return build_snapshot(symbol, submissions, facts, today or date.today())
