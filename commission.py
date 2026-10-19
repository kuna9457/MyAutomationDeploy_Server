"""
commission.py
WelthWest's weekly profit share: each client is charged an admin-chosen
percent of the profit the bot booked for them in a Monday-Sunday week, payable
the following Monday by UPI.

    week's realized profit  x  client's percent  =  one invoice
    invoice  ->  UPI link (rendered as a QR by the UI)  ->  client marks it paid

READ-ONLY toward trading. This module reads the stored trade ledger and writes
only its own records; it imports no engine, strategy, broker or market-data
code, so nothing here can start, stop, size or alter a trade.

Rules, all of them deliberate:

  * PROFIT is the realized PnL of CLOSED trades in the charged environment
    (Live unless the admin switches it to Paper for a dry run), grouped by the
    same trading-day key the Daily PnL table uses (`DBManager._trade_day`), so
    a week's invoice is exactly the sum of the seven daily rows the client
    already sees. It is the BOT'S OWN LEDGER — gross of brokerage and taxes.
  * A LOSING or flat week owes nothing ("nil"). Losses are NOT carried forward
    against later weeks.
  * The PERCENT is frozen onto an invoice when it is created. Changing a
    client's rate first freezes every finished week at the old rate, so a new
    rate never reprices the past.
  * Billing starts at `commission_from` (a Monday) — set when a rate is first
    given — so switching commission on never back-bills a client's history.
  * An UNPAID invoice is re-derived from the ledger on every read (a swing
    trade that closes late still counts); a PAID one never changes again.
  * Nobody's money moves here. "Paid" is the client's own statement, "verified"
    is the admin's; the bank statement stays the source of truth.

Storage: rates live on the user record, settings in config_store, invoices in
their own Mongo collection (one document each, so a client marking one paid on
their server and the hub generating the next cannot overwrite each other),
with the usual local-JSON fallback.
"""
from __future__ import annotations

import json
import os
import re
import threading
from datetime import date, datetime, timedelta, timezone
from typing import Any, Optional
from urllib.parse import quote

import pandas as pd

import config
import config_store
import mongo_client
import user_manager
from config import Environment

COLLECTION = "commission_invoices"
_SETTINGS_KEY = "commission_settings"
_LOCAL_PATH = os.path.join(config.LOCAL_DB_DIR, f"{COLLECTION}.json")

#: Where commission is paid unless the admin changes it in the panel
#: (COMMISSION_UPI_ID in .env overrides this default; the panel overrides both).
DEFAULT_UPI_ID = "9458603249@ybl"
DEFAULT_PAYEE_NAME = "WelthWest"

DUE, NIL, PAID, VERIFIED = "due", "nil", "paid", "verified"
#: Statuses still tied to the ledger — re-derived on read. The rest are frozen.
_OPEN = (DUE, NIL)

MAX_PCT = 100.0
_VPA = re.compile(r"^[A-Za-z0-9.\-_]{2,256}@[A-Za-z][A-Za-z0-9]{1,63}$")

_lock = threading.Lock()
_trades_db = None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _today() -> date:
    return config.now_ist().date()


def week_start(day: date) -> date:
    """The Monday of `day`'s week."""
    return day - timedelta(days=day.weekday())


def _parse_day(value: Any) -> Optional[date]:
    try:
        return date.fromisoformat(str(value or "")[:10])
    except ValueError:
        return None


# --------------------------------------------------------------------------- #
#  Settings — where the money goes, and which book is charged
# --------------------------------------------------------------------------- #
def settings() -> dict:
    saved = config_store.load(_SETTINGS_KEY) or {}
    env = saved.get("environment")
    return {
        "upi_id": (saved.get("upi_id")
                   or os.getenv("COMMISSION_UPI_ID", "").strip()
                   or DEFAULT_UPI_ID),
        "payee_name": (saved.get("payee_name")
                       or os.getenv("COMMISSION_PAYEE_NAME", "").strip()
                       or DEFAULT_PAYEE_NAME),
        "environment": (env if env in (Environment.LIVE.value,
                                       Environment.PAPER.value)
                        else Environment.LIVE.value),
    }


def update_settings(upi_id: str, payee_name: str, environment: str) -> dict:
    upi_id = (upi_id or "").strip()
    if not _VPA.match(upi_id):
        raise ValueError("That doesn't look like a UPI ID (expected name@bank).")
    payee_name = (payee_name or "").strip()[:60] or DEFAULT_PAYEE_NAME
    if environment not in (Environment.LIVE.value, Environment.PAPER.value):
        raise ValueError("environment must be 'Live' or 'Paper'.")
    config_store.save(_SETTINGS_KEY, {"upi_id": upi_id, "payee_name": payee_name,
                                      "environment": environment})
    return settings()


def upi_uri(upi_id: str, payee_name: str, amount: float, note: str) -> str:
    """The standard UPI deep link. Every UPI app reads it from a QR, and on a
    phone it opens the app directly with the amount already filled in."""
    return (f"upi://pay?pa={quote(upi_id, safe='@.-_')}"
            f"&pn={quote(payee_name)}&am={amount:.2f}&cu=INR"
            f"&tn={quote(note[:50])}")


# --------------------------------------------------------------------------- #
#  Invoice storage
# --------------------------------------------------------------------------- #
def _col() -> Optional[Any]:
    db = mongo_client.get_db()
    return None if db is None else db[COLLECTION]


def backend() -> str:
    return "MongoDB" if _col() is not None else "Local JSON"


def _read_local() -> dict:
    try:
        with open(_LOCAL_PATH, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _write_local(data: dict) -> None:
    os.makedirs(os.path.dirname(_LOCAL_PATH), exist_ok=True)
    with open(_LOCAL_PATH, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)


def _invoice_id(username: str, week: str) -> str:
    return f"{username}:{week}"


def _load(username: Optional[str] = None) -> list[dict]:
    """Stored invoices, newest week first. `None` = every client's."""
    col = _col()
    if col is not None:
        query = {"username": username} if username else {}
        rows = list(col.find(query, {"_id": 0}))
    else:
        with _lock:
            rows = [r for r in _read_local().values()
                    if username is None or r.get("username") == username]
    return sorted(rows, key=lambda r: r.get("week_start", ""), reverse=True)


def _create(doc: dict) -> None:
    """Insert unless this client already has an invoice for the week — two
    servers generating the same week at once must not produce two."""
    col = _col()
    if col is not None:
        col.update_one({"_id": doc["invoice_id"]}, {"$setOnInsert": doc},
                       upsert=True)
        return
    with _lock:
        data = _read_local()
        if doc["invoice_id"] not in data:
            data[doc["invoice_id"]] = doc
            _write_local(data)


def _patch(username: str, week: str, patch: dict,
           only_if: tuple[str, ...]) -> Optional[dict]:
    """Update one invoice, but only while its status is one of `only_if` — the
    guard is part of the write, so a stale read can never reopen a paid
    invoice or mark one paid twice. Returns the updated record, or None."""
    key = _invoice_id(username, week)
    col = _col()
    if col is not None:
        res = col.update_one({"_id": key, "status": {"$in": list(only_if)}},
                             {"$set": patch})
        if not res.matched_count:
            return None
        return col.find_one({"_id": key}, {"_id": 0})
    with _lock:
        data = _read_local()
        row = data.get(key)
        if row is None or row.get("status") not in only_if:
            return None
        row.update(patch)
        _write_local(data)
        return dict(row)


# --------------------------------------------------------------------------- #
#  The ledger: realized profit per week
# --------------------------------------------------------------------------- #
def _trades_manager():
    global _trades_db
    if _trades_db is None:
        from db_manager import DBManager
        _trades_db = DBManager()
    return _trades_db


def _read_trades(environment: str) -> pd.DataFrame:
    """Every trade in one environment (all users) — one read, sliced per
    client by the callers."""
    return _trades_manager().get_trades(Environment(environment))


def _ledger_is_shared() -> bool:
    """Whether trades and invoices are on the SAME kind of store.

    DBManager drops to a local file the moment a Mongo call fails, and stays
    there. A server in that state sees none of the real trades, so letting it
    re-derive invoices would rewrite a genuine amount due to zero. When the two
    stores disagree, invoices are served as stored and nothing is recomputed.
    """
    return _trades_manager().backend == backend()


class _Ledger:
    """Per-request cache: one trade read per environment, however many clients
    and invoices are then priced from it."""

    def __init__(self) -> None:
        self._weeks: dict[str, dict[str, dict[str, dict]]] = {}

    def weeks(self, environment: str, username: str) -> dict[str, dict]:
        if environment not in self._weeks:
            self._weeks[environment] = profit_by_user_week(
                _read_trades(environment))
        return self._weeks[environment].get(username, {})


def profit_by_user_week(df: pd.DataFrame) -> dict[str, dict[str, dict]]:
    """{username: {week_start: {"profit", "trades"}}} from a trade frame.

    Only CLOSED trades count — an open position has booked nothing. The week
    is that of the trade's trading day, the same key Daily PnL groups by.
    Trades stored before `user_id` existed belong to "admin", as everywhere
    else in the app.
    """
    out: dict[str, dict[str, dict]] = {}
    if df is None or df.empty or "status" not in df.columns:
        return out
    closed = df[df["status"] == "CLOSED"]
    owners = (closed["user_id"] if "user_id" in closed.columns
              else pd.Series("admin", index=closed.index))
    for owner, ts, pnl in zip(owners, closed["timestamp"],
                              closed["realized_pnl"]):
        day = _parse_day(ts)
        if day is None:
            continue
        try:
            value = float(pnl)
        except (TypeError, ValueError):
            continue
        if value != value:                      # NaN
            continue
        user = owner if isinstance(owner, str) and owner else "admin"
        row = out.setdefault(user, {}).setdefault(
            week_start(day).isoformat(), {"profit": 0.0, "trades": 0})
        row["profit"] += value
        row["trades"] += 1
    for weeks in out.values():
        for row in weeks.values():
            row["profit"] = round(row["profit"], 2)
    return out


def commission_for(profit: float, pct: float) -> float:
    """A percent of the week's PROFIT. A loss owes nothing."""
    return round(max(float(profit), 0.0) * float(pct) / 100.0, 2)


# --------------------------------------------------------------------------- #
#  Rates
# --------------------------------------------------------------------------- #
def _rate(user: dict) -> tuple[float, str]:
    """(percent, billing-start Monday) for a user record. (0, "") = not charged."""
    try:
        pct = float(user.get("commission_pct") or 0.0)
    except (TypeError, ValueError):
        pct = 0.0
    start = _parse_day(user.get("commission_from"))
    return max(pct, 0.0), (week_start(start).isoformat() if start else "")


def _client(username: str) -> dict:
    user = user_manager.get_user(username)
    if user is None or user.get("role") != "client":
        raise KeyError(username)
    return user


def set_rate(username: str, pct: float, effective_from: str = "") -> dict:
    """Set a client's percent. 0 switches commission off for weeks not yet
    invoiced.

    `effective_from` (any date; its week's Monday is used) is the first week
    billed. Left blank it is the CURRENT week when commission is being switched
    on, and unchanged when only the percent is being adjusted.
    """
    user = _client(username)
    pct = float(pct)
    if not 0.0 <= pct <= MAX_PCT:
        raise ValueError(f"Commission must be between 0 and {MAX_PCT:g} percent.")
    old_pct, old_from = _rate(user)
    # Freeze every finished week at the rate it was earned under BEFORE the
    # rate changes, so the new one cannot reprice the past.
    sync(username)
    if effective_from:
        start = _parse_day(effective_from)
        if start is None:
            raise ValueError("effective_from must be a date (YYYY-MM-DD).")
        new_from = week_start(start).isoformat()
    elif pct > 0 and (old_pct <= 0 or not old_from):
        new_from = week_start(_today()).isoformat()
    else:
        new_from = old_from
    user_manager.set_commission(user["user_id"], pct, new_from)
    return {"username": username, "commission_pct": pct,
            "commission_from": new_from}


# --------------------------------------------------------------------------- #
#  Invoices
# --------------------------------------------------------------------------- #
def sync(username: str, ledger: Optional[_Ledger] = None) -> list[dict]:
    """Bring one client's invoices up to date and return them, newest first.

    Creates an invoice for every FINISHED week since billing started that has
    none, and re-derives the amount on any still-unpaid one. Idempotent, and a
    no-op for a client with no rate and no open invoices.
    """
    existing = {i["week_start"]: i for i in _load(username)}
    if not _ledger_is_shared():
        return list(existing.values())

    user = user_manager.get_user(username) or {}
    pct, start = _rate(user)
    ledger = ledger or _Ledger()
    last_finished = week_start(_today()) - timedelta(days=7)

    if pct > 0 and start:
        env = settings()["environment"]
        week = date.fromisoformat(start)
        while week <= last_finished:
            key = week.isoformat()
            if key not in existing:
                profit = ledger.weeks(env, username).get(key, {})
                amount = commission_for(profit.get("profit", 0.0), pct)
                _create({
                    "invoice_id": _invoice_id(username, key),
                    "username": username,
                    "week_start": key,
                    "week_end": (week + timedelta(days=6)).isoformat(),
                    "environment": env,
                    "profit": profit.get("profit", 0.0),
                    "trades": profit.get("trades", 0),
                    "pct": pct,
                    "amount": amount,
                    "status": DUE if amount > 0 else NIL,
                    "created_at": _now(),
                    "paid_at": "", "paid_ref": "", "verified_at": "",
                })
            week += timedelta(days=7)

    for inv in _load(username):
        if inv.get("status") not in _OPEN:
            continue
        profit = ledger.weeks(inv["environment"], username).get(
            inv["week_start"], {})
        amount = commission_for(profit.get("profit", 0.0), inv["pct"])
        fresh = {"profit": profit.get("profit", 0.0),
                 "trades": profit.get("trades", 0),
                 "amount": amount, "status": DUE if amount > 0 else NIL}
        if any(inv.get(k) != v for k, v in fresh.items()):
            _patch(username, inv["week_start"], fresh, only_if=_OPEN)
    return _load(username)


def mark_paid(username: str, week: str, reference: str = "") -> dict:
    """The CLIENT says they have paid. Only a due invoice can be marked."""
    ref = re.sub(r"[^A-Za-z0-9 \-_/]", "", reference or "").strip()[:40]
    row = _patch(username, week, {"status": PAID, "paid_at": _now(),
                                  "paid_ref": ref}, only_if=(DUE,))
    if row is None:
        raise ValueError("That invoice isn't awaiting payment.")
    return row


def admin_set_status(username: str, week: str, status: str) -> dict:
    """The ADMIN's side: `verified` = money seen in the statement (also covers
    a payment the client never marked); `due` = reopen one marked by mistake."""
    if status == VERIFIED:
        row = _patch(username, week, {"status": VERIFIED, "verified_at": _now()},
                     only_if=(DUE, PAID))
    elif status == DUE:
        row = _patch(username, week, {"status": DUE, "paid_at": "",
                                      "paid_ref": "", "verified_at": ""},
                     only_if=(PAID, VERIFIED))
    else:
        raise ValueError("status must be 'verified' or 'due'.")
    if row is None:
        raise ValueError("That invoice can't be moved to that status.")
    return row


def delete_client_invoices(username: str) -> int:
    """Remove every invoice of one client — used only when the account itself
    is deleted. Returns how many were removed."""
    if not username:
        return 0
    col = _col()
    if col is not None:
        return int(col.delete_many({"username": username}).deleted_count)
    with _lock:
        data = _read_local()
        keep = {k: v for k, v in data.items() if v.get("username") != username}
        _write_local(keep)
        return len(data) - len(keep)


# --------------------------------------------------------------------------- #
#  Views
# --------------------------------------------------------------------------- #
def _with_link(inv: dict, cfg: dict) -> dict:
    out = dict(inv)
    out["upi_uri"] = (upi_uri(cfg["upi_id"], cfg["payee_name"], inv["amount"],
                              f"{cfg['payee_name']} {inv['username']} "
                              f"wk {inv['week_start']}")
                      if inv.get("status") == DUE else "")
    return out


def _statement(username: str, user: dict, cfg: dict, ledger: _Ledger) -> dict:
    pct, start = _rate(user)
    invoices = [_with_link(i, cfg) for i in sync(username, ledger)]
    this_week = week_start(_today())
    billable = bool(pct > 0 and start and this_week.isoformat() >= start)
    running = (ledger.weeks(cfg["environment"], username).get(
        this_week.isoformat(), {}) if _ledger_is_shared() else {})
    profit = running.get("profit", 0.0)
    return {
        "username": username,
        "commission_pct": pct,
        "commission_from": start,
        "current_week": {
            "week_start": this_week.isoformat(),
            "week_end": (this_week + timedelta(days=6)).isoformat(),
            "payable_on": (this_week + timedelta(days=7)).isoformat(),
            "profit": profit,
            "trades": running.get("trades", 0),
            "accrued": commission_for(profit, pct) if billable else 0.0,
        },
        "total_due": round(sum(i["amount"] for i in invoices
                               if i["status"] == DUE), 2),
        "total_paid": round(sum(i["amount"] for i in invoices
                                if i["status"] in (PAID, VERIFIED)), 2),
        "total_unverified": round(sum(i["amount"] for i in invoices
                                      if i["status"] == PAID), 2),
        "invoices": invoices,
    }


def client_view(username: str) -> dict:
    """Everything one client's own Commission tab shows."""
    cfg = settings()
    out = _statement(username, user_manager.get_user(username) or {}, cfg,
                     _Ledger())
    out.update(upi_id=cfg["upi_id"], payee_name=cfg["payee_name"],
               environment=cfg["environment"])
    return out


def admin_overview() -> dict:
    """Every client's statement, from one trade read per environment."""
    cfg = settings()
    ledger = _Ledger()
    clients = []
    for user in user_manager.list_users(role="client"):
        row = _statement(user["username"], user, cfg, ledger)
        row["display_name"] = user.get("display_name") or user["username"]
        row["account_status"] = user.get("status", "active")
        clients.append(row)
    payments = sorted(
        ({**inv, "display_name": c["display_name"]}
         for c in clients for inv in c["invoices"]
         if inv["status"] in (PAID, VERIFIED)),
        key=lambda i: i.get("paid_at") or i.get("verified_at") or "",
        reverse=True)
    return {
        "settings": cfg,
        "backend": backend(),
        #: Every payment across clients, newest first — the admin's history.
        "payments": payments,
        "ledger_shared": _ledger_is_shared(),
        "totals": {k: round(sum(c[k] for c in clients), 2)
                   for k in ("total_due", "total_paid", "total_unverified")},
        "clients": clients,
    }
