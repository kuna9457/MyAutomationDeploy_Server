"""
kite_symbols.py
Maps this codebase's instruments onto Zerodha Kite's own tradingsymbols.

Why this is needed
------------------
Equity maps for free: Kite's tradingsymbol for NSE cash is the plain symbol
(RELIANCE), which is exactly `Instrument.symbol`. MCX FUTURES do not — Kite
names them with an exchange-assigned expiry suffix:

    our Instrument.symbol        CRUDEOILM
    Kite tradingsymbol           CRUDEOILM26AUGFUT

Because of that mismatch, commodity orders to Zerodha used to be refused
outright rather than guessed at. Guessing the suffix format would risk sending
a malformed REAL order, which is worse than not trading. This module removes
the guess: it reads Kite's own published instrument list and matches exactly.

How the match is made
---------------------
Kite's MCX dump carries `name` (the contract root, e.g. "CRUDEOILM") and
`expiry` (YYYY-MM-DD). Our Instrument carries the same two facts — `symbol`
and, since the MCX refresh tool started writing it, `expiry`. So the join is
an exact (root, expiry) lookup, not a fuzzy prefix match: there is exactly one
CRUDEOILM contract expiring 2026-08-19, and that is the one we hold a
position in.

A symbol that does NOT resolve returns None, and the broker refuses the order
with a clear message. Silently falling back to the front month would trade a
DIFFERENT contract from the one the strategy priced, which is the one outcome
worth avoiding entirely.

Quantity units
--------------
Kite reports lot_size=1 for every MCX future, i.e. Kite counts commodity
quantity in LOTS (quantity=1 => one contract) — the same convention Upstox
uses and the same one the engine sizes in. No conversion is needed anywhere.

The dump is a public CSV (no authentication), so this works before a client
has connected and cannot fail because someone's token expired.
"""
from __future__ import annotations

import csv
import io
import threading
import time
import urllib.request
from typing import Optional

#: Kite's published instrument list for one exchange. Public — no auth header.
DUMP_URL = "https://api.kite.trade/instruments/{exchange}"

#: Contracts roll, so the map cannot be cached forever; they roll on a known
#: date though, so re-fetching more than a few times a day is pointless. Six
#: hours re-reads it a couple of times per session without being chatty.
CACHE_TTL_SECONDS = 6 * 60 * 60

_lock = threading.Lock()
#: exchange -> (fetched_at, {(name, expiry): row})
_cache: dict[str, tuple[float, dict[tuple[str, str], dict]]] = {}


def _fetch(exchange: str) -> dict[tuple[str, str], dict]:
    """Download and index one exchange's FUTURES contracts by (root, expiry).

    Never raises: a network failure yields an empty map, the caller resolves
    nothing, and the broker refuses the order — the same safe outcome as an
    unknown symbol."""
    try:
        url = DUMP_URL.format(exchange=exchange)
        raw = urllib.request.urlopen(url, timeout=60).read().decode("utf-8", "replace")
    except Exception as exc:
        print(f"[kite_symbols] could not fetch the {exchange} instrument list "
              f"({exc}); Kite {exchange} orders will be refused until it "
              f"succeeds.")
        return {}
    out: dict[tuple[str, str], dict] = {}
    try:
        for row in csv.DictReader(io.StringIO(raw)):
            if row.get("instrument_type") != "FUT":
                continue
            key = (str(row.get("name", "")), str(row.get("expiry", ""))[:10])
            if key[0] and key[1]:
                out[key] = row
    except Exception as exc:
        print(f"[kite_symbols] {exchange} instrument list was unreadable ({exc}).")
        return {}
    print(f"[kite_symbols] loaded {len(out)} {exchange} futures contracts.")
    return out


def _contracts(exchange: str) -> dict[tuple[str, str], dict]:
    now = time.time()
    with _lock:
        cached = _cache.get(exchange)
        if cached is not None and now - cached[0] < CACHE_TTL_SECONDS:
            return cached[1]
    fresh = _fetch(exchange)
    # Only cache a SUCCESSFUL fetch, so a transient network failure is retried
    # on the next order rather than poisoning the map for six hours.
    if fresh:
        with _lock:
            _cache[exchange] = (now, fresh)
        return fresh
    with _lock:
        cached = _cache.get(exchange)
    return cached[1] if cached else {}


def futures_tradingsymbol(root: str, expiry: str,
                          exchange: str = "MCX") -> Optional[str]:
    """Kite's tradingsymbol for one futures contract, or None if unknown.

    `root` is the contract root as this codebase names it (CRUDEOILM) and
    `expiry` is YYYY-MM-DD. Both must match — an instrument whose expiry we
    don't know cannot be resolved safely, because picking a contract for the
    caller is exactly the guess this module exists to avoid.
    """
    if not root or not expiry:
        return None
    row = _contracts(exchange).get((str(root), str(expiry)[:10]))
    return str(row["tradingsymbol"]) if row else None


def resolve(instrument, exchange: str = "MCX") -> Optional[str]:
    """Kite tradingsymbol for one of our Instruments, or None."""
    return futures_tradingsymbol(
        getattr(instrument, "symbol", ""),
        getattr(instrument, "expiry", ""),
        exchange)


def describe_failure(instrument, exchange: str = "MCX") -> str:
    """Why a resolve() failed, in terms the operator can act on."""
    expiry = getattr(instrument, "expiry", "")
    symbol = getattr(instrument, "symbol", "?")
    if not expiry:
        return (f"{symbol}: no contract expiry on file, so its Kite "
                f"tradingsymbol can't be resolved. Run "
                f"tools/refresh_mcx.py to write expiries.")
    if not _contracts(exchange):
        return (f"{symbol}: Kite's {exchange} instrument list is unavailable "
                f"right now, so the order was refused rather than guessed.")
    return (f"{symbol}: no {exchange} contract found expiring {expiry}. It has "
            f"probably rolled — run tools/refresh_mcx.py to move to the "
            f"current contract.")
