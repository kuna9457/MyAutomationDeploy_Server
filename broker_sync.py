"""
broker_sync.py
The Live trade log, rebuilt from the broker's own fills.

WHY. Every Live trade the bot books is its own record of what it asked for: an
order call returns an id, not a fill, so prices, quantities and exits could all
drift from what Kite actually executed (a GTT that fired, a partial fill, a
manual square-off, an auto square-off at 15:20). The day-wise P&L then disagreed
with the broker's. Here the broker is the source of truth instead.

WHAT. For one account and one day: read Kite's fills (`trades()`) and
positions, rebuild each round trip (flat -> position -> flat) per
tradingsymbol+product, and store those as CLOSED trades with
`source="broker"` in `live_trades`. A sync record (`broker_syncs`) then tells
`DBManager.get_trades` to hide the bot's OWN closed rows for that account/day,
so every reader — trade log, day-wise P&L, analytics, Excel, the admin's client
stats, fleet P&L, commission — reports the broker's numbers without any of them
changing.

WHAT IT NEVER TOUCHES.
  * The bot's own rows are not modified or deleted, only hidden at read time
    (`get_trades(broker_overlay=False)` still returns them), so a bad sync can
    be undone by deleting its record.
  * OPEN rows are never hidden: the engine manages positions from them.
  * A symbol whose fills do not start AND end flat today (a position carried
    from yesterday, or one still open) is UNRESOLVED: no broker row is written
    for it and the bot's rows for it stay visible, so it is never counted twice.
  * Paper. This is Live-only, and Kite-only (the only broker with a day
    tradebook wired up).

LIMIT. Kite's API returns TODAY's fills only, so a day can be synced only on
that day (the scheduler runs through the session and once after each close).
Days before this existed keep the bot's own ledger.
"""
from __future__ import annotations

import os
import threading
import time
import uuid
from datetime import datetime, timedelta
from typing import Any, Optional

import config
from config import Environment, Segment

SOURCE = "broker"
BROKER = "Zerodha"

#: A bot row belongs to a broker round trip when it was entered this close to
#: (or inside) the trip's fill window. Generous on the early side because the
#: bot stamps its row after the order call returns.
_MATCH_BEFORE = timedelta(minutes=3)
_MATCH_AFTER = timedelta(minutes=1)

_NAMESPACE = uuid.UUID("6f1c1a52-7c55-4b0e-9a54-3c1c0b6f1e20")
_IST_OFFSET = timedelta(hours=5, minutes=30)

_user_locks: dict[str, threading.Lock] = {}
_locks_guard = threading.Lock()


def _lock_for(username: str) -> threading.Lock:
    with _locks_guard:
        return _user_locks.setdefault(username, threading.Lock())


# --------------------------------------------------------------------------- #
#  Fills -> round trips (pure; no I/O)
# --------------------------------------------------------------------------- #
def _parse_ts(value: Any) -> Optional[datetime]:
    """Kite timestamps are IST wall-clock, either datetimes (kiteconnect parses
    them) or 'YYYY-MM-DD HH:MM:SS' strings. Returned naive, in IST."""
    if isinstance(value, datetime):
        return value.replace(tzinfo=None)
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("T", " ")[:19])
    except ValueError:
        return None


def _utc_iso(ist: datetime) -> str:
    """Same shape as DBManager.new_trade's timestamp (naive UTC isoformat), so a
    broker row lands on the same trading-day key as a bot row."""
    return (ist - _IST_OFFSET).isoformat()


def normalize_fills(raw: list[dict], day: str) -> list[dict]:
    """Kite trade rows for `day` (IST date), in a stable time order. Rows with
    no usable price/quantity/side/time are dropped rather than guessed."""
    out = []
    for t in raw or []:
        ts = _parse_ts(t.get("fill_timestamp") or t.get("exchange_timestamp")
                       or t.get("order_timestamp"))
        side = str(t.get("transaction_type", "")).upper()
        try:
            price = float(t.get("average_price") or 0)
            qty = int(float(t.get("quantity") or 0))
        except (TypeError, ValueError):
            continue
        if ts is None or ts.date().isoformat() != day or side not in ("BUY", "SELL") \
                or price <= 0 or qty <= 0:
            continue
        out.append({
            "symbol": str(t.get("tradingsymbol", "")),
            "exchange": str(t.get("exchange", "")),
            "product": str(t.get("product", "")),
            "side": side, "qty": qty, "price": price, "ts": ts,
            "order_id": str(t.get("order_id", "")),
            "trade_id": str(t.get("trade_id", "")),
        })
    out.sort(key=lambda f: (f["ts"], f["trade_id"]))
    return out


def _vwap(legs: list[tuple[float, int]]) -> float:
    qty = sum(q for _, q in legs)
    return sum(p * q for p, q in legs) / qty if qty else 0.0


def rebuild_round_trips(fills: list[dict], opening: dict[tuple, int],
                        multipliers: dict[str, float]
                        ) -> tuple[list[dict], set[tuple]]:
    """Round trips from normalised fills: (trips, unresolved keys).

    A key is (symbol, exchange, product) — Kite nets positions per product, so
    an MIS and a CNC position in one stock are separate. A trip runs from flat
    back to flat; a fill that crosses zero closes the trip and opens the next
    with the remainder. A key that did not open flat (`opening`, from Kite's
    overnight quantity) or is not flat at the end has every trip dropped and is
    returned as unresolved — its P&L cannot be attributed from today's fills
    alone, and a half-attributed symbol would double count against the bot's
    own rows for it."""
    by_key: dict[tuple, list[dict]] = {}
    for f in fills:
        by_key.setdefault((f["symbol"], f["exchange"], f["product"]), []).append(f)

    trips: list[dict] = []
    unresolved: set[tuple] = set()
    for key, rows in by_key.items():
        if opening.get(key, 0):
            unresolved.add(key)
            continue
        mult = float(multipliers.get(key[0]) or 1.0)
        pos = 0
        cur: Optional[dict] = None
        done: list[dict] = []
        for f in rows:
            remaining = f["qty"] if f["side"] == "BUY" else -f["qty"]
            while remaining:
                if pos == 0:
                    cur = {"side": "BUY" if remaining > 0 else "SELL",
                           "entries": [], "exits": [], "start": f["ts"],
                           "end": f["ts"], "entry_orders": [],
                           "exit_orders": [], "fills": 0}
                assert cur is not None
                if pos == 0 or (pos > 0) == (remaining > 0):
                    cur["entries"].append((f["price"], abs(remaining)))
                    _add_order(cur["entry_orders"], f["order_id"])
                    pos += remaining
                    remaining = 0
                else:
                    close = min(abs(remaining), abs(pos))
                    step = close if remaining > 0 else -close
                    cur["exits"].append((f["price"], close))
                    _add_order(cur["exit_orders"], f["order_id"])
                    pos += step
                    remaining -= step
                cur["fills"] += 1
                cur["end"] = f["ts"]
                if pos == 0:
                    done.append(cur)
                    cur = None
        if pos != 0:
            unresolved.add(key)
            continue
        for c in done:
            qty = sum(q for _, q in c["entries"])
            entry, exit_ = _vwap(c["entries"]), _vwap(c["exits"])
            direction = 1 if c["side"] == "BUY" else -1
            trips.append({
                "key": key, "side": c["side"], "quantity": qty,
                "entry_price": entry, "exit_price": exit_,
                "multiplier": mult,
                "realized_pnl": (exit_ - entry) * qty * direction * mult,
                "start": c["start"], "end": c["end"],
                "entry_order_ids": c["entry_orders"],
                "exit_order_ids": c["exit_orders"], "fills": c["fills"],
            })
    trips.sort(key=lambda t: t["start"])
    return trips, unresolved


def _add_order(ids: list[str], oid: str) -> None:
    if oid and oid not in ids:
        ids.append(oid)


def position_maps(positions: dict) -> tuple[dict[tuple, int], dict[str, float],
                                            float]:
    """(opening quantity per key, multiplier per symbol, Kite's day realised
    total) from a Kite positions() response."""
    opening: dict[tuple, int] = {}
    mults: dict[str, float] = {}
    for p in (positions or {}).get("net", []) or []:
        key = (str(p.get("tradingsymbol", "")), str(p.get("exchange", "")),
               str(p.get("product", "")))
        try:
            opening[key] = int(float(p.get("overnight_quantity") or 0))
        except (TypeError, ValueError):
            opening[key] = 0
    realised = 0.0
    for bucket in ("day", "net"):
        for p in (positions or {}).get(bucket, []) or []:
            try:
                m = float(p.get("multiplier") or 0)
            except (TypeError, ValueError):
                m = 0.0
            if m > 0:
                mults.setdefault(str(p.get("tradingsymbol", "")), m)
    for p in (positions or {}).get("day", []) or []:
        try:
            realised += float(p.get("realised") or 0)
        except (TypeError, ValueError):
            pass
    return opening, mults, round(realised, 2)


# --------------------------------------------------------------------------- #
#  Round trips -> trade documents
# --------------------------------------------------------------------------- #
_mcx_reverse: dict[str, str] = {}


def bot_ticker(symbol: str, exchange: str) -> str:
    """The bot's own symbol for a Kite tradingsymbol. Equity is the same name;
    an MCX contract (CRUDEOILM26AUGFUT) maps back to its root instrument so it
    lines up with the bot's rows. Unknown contracts keep Kite's name."""
    if exchange != "MCX":
        return symbol
    if symbol not in _mcx_reverse:
        try:
            import kite_symbols
            for inst in config.ALL_INSTRUMENTS:
                if inst.segment == Segment.MCX:
                    ts = kite_symbols.resolve(inst)
                    if ts:
                        _mcx_reverse[ts] = inst.symbol
        except Exception:
            pass
    return _mcx_reverse.get(symbol, symbol)


def _row_time(row: dict, field: str = "timestamp") -> Optional[datetime]:
    """A stored row's naive-UTC timestamp, as IST."""
    try:
        return datetime.fromisoformat(str(row.get(field) or "")[:26]) + _IST_OFFSET
    except ValueError:
        return None


def _matching_bot_rows(trip: dict, ticker: str, bot_rows: list[dict]) -> list[dict]:
    lo, hi = trip["start"] - _MATCH_BEFORE, trip["end"] + _MATCH_AFTER
    hits = []
    for r in bot_rows:
        t = _row_time(r)
        if (r.get("ticker") == ticker and r.get("side") == trip["side"]
                and t is not None and lo <= t <= hi):
            hits.append(r)
    hits.sort(key=lambda r: str(r.get("timestamp") or ""))
    return hits


def _num(value: Any) -> float:
    try:
        f = float(value)
        return f if f == f else 0.0
    except (TypeError, ValueError):
        return 0.0


def build_docs(username: str, day: str, trips: list[dict],
               bot_rows: list[dict], synced_at: str) -> list[dict]:
    """One `live_trades` document per round trip. Numbers come from the broker;
    only labels (mode, strategy, entry reason, the stop/target the bot set, exit
    reason) are borrowed from the bot's own row for the same trade, when there
    is one. No bot row = a trade placed outside the bot (`origin="manual"`)."""
    docs = []
    for n, trip in enumerate(trips):
        symbol, exchange, product = trip["key"]
        ticker = bot_ticker(symbol, exchange)
        hits = _matching_bot_rows(trip, ticker, bot_rows)
        first = hits[0] if hits else {}
        last_exit = max(hits, key=lambda r: str(r.get("exit_timestamp") or ""),
                        default={})
        segment = first.get("segment") or (
            Segment.MCX.value if exchange == "MCX" else Segment.EQUITY.value)
        mode = first.get("mode") or (config.Mode.SWING.value if product == "CNC"
                                     else config.Mode.INTRADAY.value)
        mult = trip["multiplier"]
        docs.append({
            "trade_id": str(uuid.uuid5(
                _NAMESPACE, f"{username}|{day}|{symbol}|{exchange}|{product}|{n}")),
            "timestamp": _utc_iso(trip["start"]),
            "user_id": username,
            "mode": mode,
            "strategy": first.get("strategy") or "",
            "entry_reason": first.get("entry_reason") or "",
            "environment": Environment.LIVE.value,
            "broker": BROKER,
            "segment": segment,
            "category": config.category_for_segment(segment),
            "ticker": ticker,
            "side": trip["side"],
            "entry_price": round(trip["entry_price"], 4),
            "stop_loss": round(_num(first.get("stop_loss")), 4),
            "target": round(_num(first.get("target")), 4),
            "quantity": int(trip["quantity"]),
            "contract_multiplier": int(mult) if float(mult).is_integer() else mult,
            "risk_amount": round(_num(first.get("risk_amount")), 2),
            "broker_gtt_id": "",
            "status": "CLOSED",
            "exit_price": round(trip["exit_price"], 4),
            "realized_pnl": round(trip["realized_pnl"], 2),
            "exit_timestamp": _utc_iso(trip["end"]),
            "exit_reason": last_exit.get("exit_reason") or (
                "BROKER" if hits else "MANUAL"),
            "source": SOURCE,
            "origin": "bot" if hits else "manual",
            "broker_day": day,
            "broker_symbol": symbol,
            "product": product,
            "fills": trip["fills"],
            "entry_order_ids": trip["entry_order_ids"],
            "exit_order_ids": trip["exit_order_ids"],
            "synced_at": synced_at,
        })
    return docs


# --------------------------------------------------------------------------- #
#  One account
# --------------------------------------------------------------------------- #
def today() -> str:
    """The day a sync covers: today's IST date, Kite's tradebook day."""
    return config.now_ist().date().isoformat()


def kite_for(username: str):
    """(connected ZerodhaBroker, "") for this account's own Kite session, or
    (None, why not). A client's token comes from their stored, encrypted
    credentials; the admin falls back to the .env Kite app it trades with."""
    import broker_api
    import user_manager
    user = user_manager.get_user(username)
    if user is None:
        return None, "No such account."
    key, _secret = user_manager.get_broker_credentials(user.get("user_id", ""),
                                                       BROKER)
    token = user_manager.get_broker_token(user.get("user_id", ""), BROKER)
    if key and token:
        b = broker_api.ZerodhaBroker(access_token=token, api_key=key)
    elif user.get("role") == "admin" and config.has_zerodha():
        b = broker_api.ZerodhaBroker()
    else:
        return None, "No Kite (Zerodha) account is connected."
    if not b.connect():
        return None, "The Kite session has expired — reconnect Kite, then sync."
    return b, ""


def sync_user(username: str, db=None, broker=None,
              now: Optional[datetime] = None) -> dict:
    """Rebuild `username`'s Live trades for today from their Kite account.
    Idempotent: a re-sync replaces that day's broker rows. Never raises."""
    from db_manager import DBManager
    now = now or config.now_ist()
    day = now.date().isoformat()
    report: dict[str, Any] = {"ok": False, "username": username, "day": day}
    with _lock_for(username):
        try:
            if broker is None:
                broker, why = kite_for(username)
                if broker is None:
                    report["reason"] = why
                    return report
            raw = broker.day_fills()
            positions = broker.day_positions()
            if raw is None or positions is None:
                report["reason"] = "Kite did not return today's trades — try again."
                return report
            db = db or DBManager()
            fills = normalize_fills(raw, day)
            opening, mults, kite_realised = position_maps(positions)
            bot_rows = [
                r for r in db.get_trades(Environment.LIVE, user_id=username,
                                         broker_overlay=False)
                .to_dict("records")
                if r.get("source") != SOURCE and r.get("broker") == BROKER
                and str(r.get("timestamp") or "")[:10] == day
            ]
            if not fills:
                if any(r.get("status") == "CLOSED" for r in bot_rows):
                    # Hiding the bot's trades on an empty tradebook is how a
                    # token for the wrong account would wipe a day. Refuse.
                    report["reason"] = ("Kite reports no fills today but the bot "
                                        "booked trades — nothing changed. Check "
                                        "the Kite account is the one the bot "
                                        "trades.")
                    report["broker_realised"] = kite_realised
                    return report
                report.update(ok=True, trades=0, unresolved=[],
                              broker_realised=kite_realised, ledger_total=0.0,
                              gap=0.0, matches=abs(kite_realised) < 1.0,
                              reason="No fills in Kite today.")
                return report

            trips, unresolved = rebuild_round_trips(fills, opening, mults)
            synced_at = datetime.utcnow().isoformat()
            docs = build_docs(username, day, trips, bot_rows, synced_at)
            unresolved_tickers = sorted({bot_ticker(k[0], k[1]) for k in unresolved})
            synced_total = round(sum(d["realized_pnl"] for d in docs), 2)
            kept_bot = round(sum(
                _num(r.get("realized_pnl")) for r in bot_rows
                if r.get("status") == "CLOSED"
                and r.get("ticker") in unresolved_tickers), 2)
            ledger_total = round(synced_total + kept_bot, 2)
            gap = round(ledger_total - kite_realised, 2)
            record = {
                "user_id": username, "day": day, "broker": BROKER,
                "synced_at": synced_at, "trades": len(docs),
                "manual": sum(1 for d in docs if d["origin"] == "manual"),
                "unresolved": unresolved_tickers,
                "synced_total": synced_total, "ledger_total": ledger_total,
                "broker_realised": kite_realised, "gap": gap,
            }
            db.replace_broker_day(username, day, BROKER, docs, record)
            report.update(ok=True, **{k: v for k, v in record.items()
                                      if k not in ("user_id",)},
                          matches=abs(gap) < 1.0)
            return report
        except Exception as exc:  # never let a sync break a caller
            report["reason"] = f"Sync failed: {exc}"[:240]
            return report


# --------------------------------------------------------------------------- #
#  Every account this server can read
# --------------------------------------------------------------------------- #
def eligible_usernames() -> list[str]:
    """Accounts this process should sync. A client NODE syncs only the client
    it is assigned; the hub skips clients that have their own server (that
    server holds their decryptable credentials). Everything else: every active
    account — kite_for() quietly skips those with no Kite connected."""
    import user_manager
    try:
        import hub_link
        if hub_link.is_worker():
            return list(hub_link.link().assigned)
    except Exception:
        pass
    try:
        import fleet_registry
        has_node = fleet_registry.node_for_client
    except Exception:
        has_node = None
    names = []
    for u in user_manager.list_users():
        name = u.get("username")
        if not name or u.get("status") != "active":
            continue
        if u.get("role") == "client" and has_node is not None:
            try:
                if has_node(name) is not None:
                    continue
            except Exception:
                pass
        names.append(name)
    return names


def sync_all(db=None) -> list[dict]:
    from db_manager import DBManager
    db = db or DBManager()
    return [sync_user(name, db=db) for name in eligible_usernames()]


# --------------------------------------------------------------------------- #
#  Scheduler
# --------------------------------------------------------------------------- #
_thread: Optional[threading.Thread] = None
_last_run: dict[str, Any] = {}


def enabled() -> bool:
    return os.getenv("BROKER_SYNC_ENABLED", "true").strip().lower() not in (
        "0", "false", "no", "off")


def interval_minutes() -> int:
    try:
        return max(5, int(os.getenv("BROKER_SYNC_MINUTES", "15")))
    except ValueError:
        return 15


def in_window(now: datetime) -> bool:
    """Weekdays, from just after the equity open to just before midnight —
    which covers the MCX evening session and the final post-close sync, while
    Kite still holds the day's tradebook."""
    if now.weekday() >= 5:
        return False
    minutes = now.hour * 60 + now.minute
    return 9 * 60 + 20 <= minutes <= 23 * 60 + 55


def last_run() -> dict:
    return dict(_last_run)


def start_background() -> None:
    """Sync every eligible account every BROKER_SYNC_MINUTES inside the trading
    window. Idempotent; BROKER_SYNC_ENABLED=false turns it off (tests do)."""
    global _thread
    if not enabled() or (_thread is not None and _thread.is_alive()):
        return

    def _loop():
        while True:
            now = config.now_ist()
            if in_window(now):
                try:
                    results = sync_all()
                    _last_run.update(at=now.isoformat(timespec="seconds"),
                                     accounts=len(results),
                                     ok=sum(1 for r in results if r.get("ok")))
                except Exception as exc:  # pragma: no cover - defensive
                    print(f"[broker_sync] run failed: {exc}")
            time.sleep(interval_minutes() * 60)

    _thread = threading.Thread(target=_loop, daemon=True, name="broker-sync")
    _thread.start()
