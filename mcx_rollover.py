"""
mcx_rollover.py
Rolls MCX futures to the next contract AUTOMATICALLY, before they expire.

MCX instrument_keys die at expiry: Upstox answers "Invalid Instrument key" and
the symbol silently stops producing data. Until now the only cure was someone
remembering to run `tools/refresh_mcx.py` and redeploy. This module does the
same selection on the server, unattended:

  * WHEN — a contract is "due" once it is within ROLL_DAYS (default 5) of
    expiry. Rolling a few days early costs a little liquidity and guarantees a
    working key; rolling on expiry day would re-pick a contract that is dead by
    the next session (the trap tools/refresh_mcx.py documents).
  * WHAT — the nearest future in the Upstox MCX master that is still alive
    beyond ROLL_DAYS, with the exchange's own lot / tick / multiplier.
  * WHERE IT RUNS — at server start, every CHECK_INTERVAL (6h) in a background
    thread, and on every bot start (TradingEngine.__init__ and the fleet
    broadcast's resolve_run), so a Start always trades the current contract.

SAFETY. Every lookup in this codebase goes symbol -> config.INSTRUMENTS_BY_SYMBOL
-> instrument_key, and trade documents do not store the key. So swapping a
symbol's key while something still holds the OLD contract would send that
position's exit order to the NEW contract — opening a fresh position instead
of closing the old one. A symbol is therefore rolled only when:
  1. no OPEN trade on it exists (paper or live, any account), and
  2. no running bot / broadcast / node runner is subscribed to the old key.
Otherwise the roll is DEFERRED, the reason is logged to the affected bots, and
it is retried on the next check. Stop the bot while flat and the next Start
picks up the new contract.

PERSISTENCE. Rolled contracts are written to data/mcx_rolls.json (gitignored),
NOT to mcx_instruments.py — the server must never dirty a tracked file, or the
next `git pull` on deploy fails. config overlays that file at import, keeping
whichever of the two has the LATER expiry, so a freshly committed
mcx_instruments.py still wins once it is newer.

Disable with MCX_AUTO_ROLL=false (the tests do). Change the window with
MCX_ROLL_DAYS.
"""
from __future__ import annotations

import gzip
import json
import os
import threading
import time
import urllib.request
from dataclasses import replace
from datetime import date, datetime, timedelta
from typing import Optional

import config
from config import Instrument, Segment

MASTER_URL = "https://assets.upstox.com/market-quote/instruments/exchange/MCX.json.gz"
ROLL_DAYS = int(os.getenv("MCX_ROLL_DAYS", "5") or 5)
CHECK_INTERVAL_SECONDS = 6 * 3600
#: Minimum gap between master downloads while something is due but blocked,
#: so repeated bot starts don't each re-download several MB.
REFETCH_SECONDS = 30 * 60
ROLLS_FILE = config.MCX_ROLLS_FILE

_lock = threading.RLock()
_master_cache: tuple[float, list[dict]] = (0.0, [])
_last_report: dict = {}
_thread: Optional[threading.Thread] = None


def enabled() -> bool:
    return os.getenv("MCX_AUTO_ROLL", "true").strip().lower() not in (
        "0", "false", "no", "off")


def _log(msg: str) -> None:
    print(f"[mcx_rollover] {msg}", flush=True)


# --------------------------------------------------------------------------- #
#  Selection
# --------------------------------------------------------------------------- #
def due(within_days: int = ROLL_DAYS) -> list[tuple[Instrument, int]]:
    """MCX contracts within `within_days` of expiry (negative = already dead)."""
    return config.expiring_soon(list(config.MCX_INSTRUMENTS), within_days)


def _expiry_date(ms: float) -> date:
    return datetime.fromtimestamp(ms / 1000, tz=config.IST).date()


def _fetch_master(force: bool = False) -> list[dict]:
    global _master_cache
    fetched_at, rows = _master_cache
    if rows and not force and time.time() - fetched_at < REFETCH_SECONDS:
        return rows
    with urllib.request.urlopen(MASTER_URL, timeout=30) as resp:
        rows = json.loads(gzip.decompress(resp.read()))
    _master_cache = (time.time(), rows)
    return rows


def pick_next(current: Instrument, rows: list[dict], today: date,
              within_days: int = ROLL_DAYS) -> Optional[Instrument]:
    """The nearest future for `current.symbol` still alive beyond the roll
    window, as an Instrument carrying the exchange's own contract spec — or
    None if the master has nothing usable."""
    cutoff = today + timedelta(days=within_days)
    cands = [r for r in rows
             if r.get("instrument_type") == "FUT"
             and r.get("asset_symbol") == current.symbol
             and r.get("expiry")
             and _expiry_date(r["expiry"]) > cutoff]
    if not cands:
        return None
    r = min(cands, key=lambda r: r["expiry"])
    if r["instrument_key"] == current.instrument_key:
        return None
    return replace(
        current,
        instrument_key=r["instrument_key"],
        lot_size=int(r.get("lot_size") or current.lot_size),
        # The master's tick_size is in paise; config stores rupees (same
        # conversion tools/refresh_mcx.py makes).
        tick_size=(float(r["tick_size"]) / 100.0 if r.get("tick_size")
                   else current.tick_size),
        contract_multiplier=int(float(r.get("qty_multiplier", 0) or 0)
                                or current.contract_multiplier),
        expiry=_expiry_date(r["expiry"]).isoformat(),
    )


# --------------------------------------------------------------------------- #
#  Safety checks
# --------------------------------------------------------------------------- #
def _open_symbols() -> Optional[set[str]]:
    """Tickers with an OPEN trade in either environment, any account. None if
    the store can't be read — the caller then defers everything."""
    try:
        from db_manager import DBManager
        db = DBManager()
        out: set[str] = set()
        for env in (config.Environment.PAPER, config.Environment.LIVE):
            for t in db.get_open_trades(env):
                if t.get("ticker"):
                    out.add(str(t["ticker"]))
        return out
    except Exception as exc:  # pragma: no cover - defensive
        _log(f"could not read open trades ({exc}); deferring all rolls")
        return None


def _runners_using(key: str) -> list:
    """Every live runner (local bot, broadcast, node) subscribed to `key`."""
    import strategy_runner
    with strategy_runner._registry_lock:
        runners = list(strategy_runner._runners.values())
    return [r for r in runners
            if any(i.instrument_key == key for i in getattr(r, "instruments", []))]


def _tell(runners: list, msg: str) -> None:
    for r in runners:
        for acc in list(getattr(r, "_accounts", [])):
            state = getattr(acc, "state", None)
            if state is not None and hasattr(state, "push_log"):
                try:
                    state.push_log(msg)
                except Exception:
                    pass


# --------------------------------------------------------------------------- #
#  Apply + persist
# --------------------------------------------------------------------------- #
def _swap(new: Instrument) -> None:
    for lst in (config.MCX_INSTRUMENTS, config.ALL_INSTRUMENTS):
        for idx, inst in enumerate(lst):
            if inst.symbol == new.symbol and inst.segment == Segment.MCX:
                lst[idx] = new
    config.INSTRUMENTS_BY_SYMBOL[new.symbol] = new


def _persist(new: Instrument) -> None:
    try:
        with open(ROLLS_FILE, encoding="utf-8") as fh:
            saved = json.load(fh)
    except Exception:
        saved = {}
    saved[new.symbol] = {
        "instrument_key": new.instrument_key,
        "lot_size": new.lot_size,
        "tick_size": new.tick_size,
        "contract_multiplier": new.contract_multiplier,
        "expiry": new.expiry,
        "rolled_at": config.now_ist().isoformat(timespec="seconds"),
    }
    os.makedirs(os.path.dirname(ROLLS_FILE), exist_ok=True)
    tmp = ROLLS_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(saved, fh, indent=2)
    os.replace(tmp, ROLLS_FILE)


def roll_if_due(force_fetch: bool = False) -> dict:
    """Roll every due MCX contract that is safe to roll. Returns a report:
    {"rolled": [...], "deferred": [...], "unavailable": [...]} of messages.
    Cheap when nothing is due (no network)."""
    report = {"rolled": [], "deferred": [], "unavailable": []}
    if not enabled():
        return report
    with _lock:
        pending = due()
        if not pending:
            return report
        try:
            rows = _fetch_master(force=force_fetch)
        except Exception as exc:
            report["unavailable"].append(f"Upstox MCX master unreachable: {exc}")
            _log(report["unavailable"][-1])
            _last_report.update(report, at=config.now_ist().isoformat(timespec="seconds"))
            return report

        today = config.now_ist().date()
        open_syms = _open_symbols()
        for old, days_left in pending:
            new = pick_next(old, rows, today)
            if new is None:
                report["unavailable"].append(
                    f"{old.symbol}: no later contract in the Upstox master")
                continue
            when = (f"expired {old.expiry}" if days_left < 0
                    else f"expires {old.expiry}")
            if open_syms is None or old.symbol in open_syms:
                msg = (f"⏸ {old.symbol} roll to {new.expiry} deferred: an OPEN "
                       f"position is on the current contract ({when}). It rolls "
                       f"automatically once that position is closed.")
                report["deferred"].append(msg)
                _tell(_runners_using(old.instrument_key), msg)
                continue
            users = _runners_using(old.instrument_key)
            if users:
                msg = (f"⏸ {old.symbol} contract {when}; next contract "
                       f"({new.expiry}) is ready. Stop and Start the bot while "
                       f"flat to switch — a running bot is not switched "
                       f"underneath itself.")
                report["deferred"].append(msg)
                _tell(users, msg)
                continue
            _swap(new)
            try:
                _persist(new)
            except Exception as exc:
                _log(f"rolled {old.symbol} in memory but could not save "
                     f"{ROLLS_FILE}: {exc}")
            report["rolled"].append(
                f"🔄 {old.symbol}: {old.instrument_key} ({old.expiry}) -> "
                f"{new.instrument_key} ({new.expiry})")
        for line in report["rolled"] + report["deferred"] + report["unavailable"]:
            _log(line)
        _last_report.clear()
        _last_report.update(report, at=config.now_ist().isoformat(timespec="seconds"))
    return report


def current(instruments: list[Instrument]) -> list[Instrument]:
    """Roll anything due, then return `instruments` re-resolved by symbol, so a
    bot being started trades the contract that is current NOW rather than the
    one its caller looked up. Never raises — a failed roll keeps the input."""
    try:
        roll_if_due()
    except Exception as exc:  # pragma: no cover - defensive
        _log(f"roll check failed: {exc}")
    return [config.INSTRUMENTS_BY_SYMBOL.get(i.symbol, i)
            if i.segment == Segment.MCX else i for i in instruments]


def last_report() -> dict:
    return dict(_last_report)


def start_background() -> None:
    """Check at startup and then every CHECK_INTERVAL_SECONDS. Idempotent."""
    global _thread
    if not enabled() or (_thread is not None and _thread.is_alive()):
        return

    def _loop():
        while True:
            try:
                roll_if_due()
            except Exception as exc:  # pragma: no cover - defensive
                _log(f"roll check failed: {exc}")
            time.sleep(CHECK_INTERVAL_SECONDS)

    _thread = threading.Thread(target=_loop, daemon=True, name="mcx-rollover")
    _thread.start()
