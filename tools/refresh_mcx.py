"""
tools/refresh_mcx.py
Regenerate ../mcx_instruments.py from the live Upstox MCX instrument master.

Unlike NSE equity keys (ISIN-based, stable), MCX FUTURES instrument_keys EXPIRE:
each contract has an expiry, and once it rolls off you must trade the next one.
Run this whenever a contract expires (or monthly, to be safe):

    python tools/refresh_mcx.py

It downloads the Upstox MCX master, and for each root symbol below picks the
NEAREST still-active future (front month), writing real instrument_keys with the
exchange's own lot_size / tick_size / qty_multiplier. Any root with no active
future is reported and skipped, so a rolled-off contract can never break config
import — config falls back to its inline list when this module is absent.

Both FULL-size and MINI/MICRO contracts are refreshed. The mini contracts track
the same underlying as their full sibling but require a fraction of the margin
(e.g. CRUDEOILM ≈ ₹22k/lot vs CRUDEOIL ≈ ₹2.5L/lot), which is why they exist.
The REAL margin is always fetched live from the broker at trade time
(broker_api.fetch_upstox_margin); config.MCX_MARGIN_PER_LOT is only an offline
fallback and is NOT written by this tool.
"""
from __future__ import annotations

import datetime as dt
import gzip
import json
import os
import sys
import urllib.request

# Root asset_symbols to include, in display order. Full-size first, then the
# mini/micro variants that trade the same underlying at a fraction of the size.
MCX_ROOTS = [
    "GOLD", "CRUDEOIL", "NATURALGAS", "SILVER",       # full-size
    "GOLDM", "CRUDEOILM", "NATGASMINI", "SILVERM", "SILVERMIC",  # mini / micro
]

# Nominal reference prices (quoted-unit price) to seed the SIMULATED feed ONLY.
# Real backtest and live trading use actual market data, so these need not be
# exact. A mini shares its full sibling's quote unit, hence the same seed.
REF_PRICE = {
    "GOLD": 142419.0, "GOLDM": 142419.0,
    "CRUDEOIL": 7580.0, "CRUDEOILM": 7580.0,
    "NATURALGAS": 279.7, "NATGASMINI": 279.7,
    "SILVER": 223320.0, "SILVERM": 223320.0, "SILVERMIC": 223320.0,
}

# Skip contracts expiring within this many days and take the NEXT one instead.
#
# Picking the bare nearest-active future is a trap on a roll date: run this on
# the morning a contract expires and it re-selects that same contract, which is
# dead by the next session — the exact failure this tool exists to prevent.
# Rolling a few days early costs a little liquidity and buys a working feed.
# Override with:  python tools/refresh_mcx.py --min-days 3
MIN_DAYS_TO_EXPIRY = 5

URL = "https://assets.upstox.com/market-quote/instruments/exchange/MCX.json.gz"
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "mcx_instruments.py")


def main() -> int:
    min_days = MIN_DAYS_TO_EXPIRY
    if "--min-days" in sys.argv:
        try:
            min_days = int(sys.argv[sys.argv.index("--min-days") + 1])
        except (IndexError, ValueError):
            print("--min-days needs a whole number of days", file=sys.stderr)
            return 2

    print(f"Downloading {URL} ...", file=sys.stderr)
    rows = json.loads(gzip.decompress(urllib.request.urlopen(URL, timeout=60).read()))
    now = dt.datetime.now()
    # The cutoff, not "now": a contract must still be alive in `min_days` time,
    # otherwise we would happily write a key that dies before the next run.
    cutoff_ms = (now + dt.timedelta(days=min_days)).timestamp() * 1000

    # All futures still active BEYOND the cutoff, grouped by root, nearest first.
    futs: dict[str, list[dict]] = {}
    skipped: list[str] = []
    for r in rows:
        if r.get("instrument_type") != "FUT":
            continue
        exp = r.get("expiry", 0)
        if exp <= cutoff_ms:
            # Report the ones we deliberately rolled past, so the operator can
            # see WHY a root moved to a later month than they expected.
            if exp > now.timestamp() * 1000 and r.get("asset_symbol") in MCX_ROOTS:
                skipped.append(
                    f"{r['asset_symbol']} "
                    f"{dt.datetime.fromtimestamp(exp / 1000):%d-%b}")
            continue
        futs.setdefault(r.get("asset_symbol"), []).append(r)
    for lst in futs.values():
        lst.sort(key=lambda r: r["expiry"])

    if skipped:
        print(f"rolled past (expiring within {min_days}d): {', '.join(sorted(set(skipped)))}",
              file=sys.stderr)

    found, missing = [], []
    for root in MCX_ROOTS:
        cand = futs.get(root)
        if not cand:
            missing.append(root)
            continue
        r = cand[0]  # nearest contract that survives the cutoff
        exp = dt.datetime.fromtimestamp(r["expiry"] / 1000).strftime("%d-%b-%Y")
        found.append({
            "symbol": root,
            "instrument_key": r["instrument_key"],
            "lot_size": int(r["lot_size"]),
            # The Upstox master expresses tick_size scaled ×100 (paise); config
            # stores rupees, matching how the full contracts were entered.
            "tick_size": float(r["tick_size"]) / 100.0,
            "ref_price": REF_PRICE.get(root, 100.0),
            "multiplier": int(float(r.get("qty_multiplier", 1) or 1)),
            "unit": r.get("price_quote_unit", ""),
            "expiry": exp,
            "expiry_iso": dt.datetime.fromtimestamp(
                r["expiry"] / 1000).strftime("%Y-%m-%d"),
        })

    print(f"matched {len(found)} / {len(MCX_ROOTS)}  missing={missing}",
          file=sys.stderr)
    if not found:
        print("no active futures matched — leaving mcx_instruments.py untouched",
              file=sys.stderr)
        return 1

    body = "\n".join(
        f'    Instrument("{x["symbol"]}", Segment.MCX, "{x["instrument_key"]}", '
        f'{x["lot_size"]}, {x["tick_size"]}, {x["ref_price"]}, {x["multiplier"]}, '
        f'expiry="{x["expiry_iso"]}"),'
        f'   # {x["unit"]}, expires {x["expiry"]}'
        for x in found)
    header = (
        '"""\n'
        "mcx_instruments.py\n"
        "The MCX commodity futures universe (full-size + mini/micro), with REAL\n"
        "front-month Upstox instrument_keys pulled from the Upstox MCX instrument\n"
        "master.\n\n"
        "Generated by tools/refresh_mcx.py — do not hand-edit. MCX futures keys\n"
        "EXPIRE, so regenerate this each expiry (or monthly) by running\n"
        "`python tools/refresh_mcx.py`. config prefers this module when present and\n"
        "falls back to its inline list otherwise.\n\n"
        "`reference_price` seeds the SIMULATED feed only; real backtest and live\n"
        "trading use actual market data, so it need not be exact.\n"
        '"""\n'
        "from config import Instrument, Segment\n\n"
        "MCX_INSTRUMENTS = [\n")
    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write(header + body + "\n]\n")
    print(f"wrote {len(found)} instruments to {OUT}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
