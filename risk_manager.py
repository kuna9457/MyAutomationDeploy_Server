"""
risk_manager.py
LIVE-only, user-editable risk guardrails on top of real broker money.

Deliberately separate from StrategyParams (config.py): StrategyParams is a
strategy's own risk/RR contract — fixed per mode/strategy, identical in Paper
and Live, and untouched by this module (Immutable Rule #1 keeps holding
regardless of anything here). LiveRiskLimits is an EXTRA, independent layer
that only ever *tightens* what a strategy would otherwise do, applies in Live
only, and is meant to change daily as the user's real-money risk appetite
changes for the day — hence persisted to disk (survives a restart) and read
fresh every tick by engine.py rather than being baked in at start() (so
editing it from the sidebar takes effect on a RUNNING bot immediately, no
restart required).

Storage is keyed per user_id (frontend_migration_plan.md §3, Phase 2): each
account gets its own limits file under live_risk_limits/, so one client's
daily-loss kill switch never affects another's. `user_id="admin"` (the
default everywhere) transparently migrates the original single-file
live_risk_limits.json the first time it's read, so the admin account's
existing settings survive this change untouched.
"""
from __future__ import annotations

import json
import os
import threading
from dataclasses import asdict, dataclass

import config
import config_store

_LEGACY_PATH = os.path.join(config.LOCAL_DB_DIR, "live_risk_limits.json")
_DIR = os.path.join(config.LOCAL_DB_DIR, "live_risk_limits")
#: All users' limits in ONE durable document, {user_id: {...fields}}. Storage
#: moved here (config_store: MongoDB when reachable, local JSON otherwise)
#: because the per-user files below live on an ephemeral filesystem — on a
#: container host a redeploy silently reset everyone's guardrails back to
#: "no extra restriction". Both older layouts are still read on the way in
#: (see `load`), so no existing account loses its settings.
_KEY = "live_risk_limits_by_user"
_lock = threading.Lock()


@dataclass
class LiveRiskLimits:
    # 0 / falsy everywhere below means "no extra restriction" — the strategy's
    # own StrategyParams caps (risk_per_trade, max_capital_per_trade_pct, ...)
    # still apply either way; these only add tighter ceilings on top.

    # ₹ ceiling on the TOTAL capital the live bot may deploy today, across all
    # open positions combined. 0 = fall back to the sidebar's "Total Capital"
    # figure as-is. This is what makes "whatever capital I choose, use only
    # that — not the whole broker account" strict: engine._available_capital
    # sizes every trade against min(total_capital, capital_allocated), never
    # against the account's real (possibly much larger) balance.
    capital_allocated: float = 0.0
    # ₹ kill-switch: once today's REALIZED loss reaches this, the bot stops
    # opening new trades for the rest of the day (existing open positions are
    # still managed to their SL/TP/time-exit as normal — this blocks new risk,
    # it does not panic-close what's already on).
    max_daily_loss_cash: float = 0.0
    # Same kill-switch, expressed as a % of capital_allocated (or total_capital
    # if capital_allocated is 0). Whichever of the two (cash/pct) is set AND
    # tighter wins; 0 disables that leg.
    max_daily_loss_pct: float = 0.0
    # Hard cap on how many NEW trades may be opened today. 0 = unlimited.
    max_trades_per_day: int = 0
    # Hard cap on the quantity (shares or lots) of any SINGLE order. 0 =
    # unlimited (the strategy's own risk/capital sizing still bounds it).
    max_qty_per_trade: int = 0
    # Real MIS leverage to size EQUITY Intraday/Scalper trades against, LIVE
    # only. Swing is never affected (overnight = delivery = 1x, a hard
    # real-world constraint, not a preference). 1.0 = no leverage (the
    # historical default). Set this to what your broker's MIS product actually
    # offers (commonly ~5x) — the bot does not verify this number against the
    # broker; it verifies the RESULTING trade against the broker's real
    # available funds instead (see broker.available_funds() in engine.py).
    intraday_leverage: float = 1.0


def _path(user_id: str) -> str:
    safe = "".join(c for c in user_id if c.isalnum() or c in "-_") or "admin"
    return os.path.join(_DIR, f"{safe}.json")


def _load_json(path: str) -> LiveRiskLimits:
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    return LiveRiskLimits(**{k: v for k, v in data.items()
                             if k in LiveRiskLimits.__dataclass_fields__})


def _coerce(data: dict) -> LiveRiskLimits:
    return LiveRiskLimits(**{k: v for k, v in (data or {}).items()
                             if k in LiveRiskLimits.__dataclass_fields__})


def load(user_id: str = "admin") -> LiveRiskLimits:
    """This user's limits, newest storage layout first.

    Three layouts are read, in order, so nobody's settings are lost as
    storage has moved: the durable combined document, then the per-user file
    it replaced, then the single legacy file that predated per-user storage.
    The first hit wins and is returned as-is; it gets written into the durable
    document on the next set_limits().
    """
    try:
        by_user = config_store.load(_KEY)
        if isinstance(by_user, dict) and user_id in by_user:
            return _coerce(by_user[user_id])
    except Exception:
        pass
    try:
        return _load_json(_path(user_id))
    except Exception:
        pass
    # One-time migration: the admin account's limits used to live at the
    # single legacy path, before storage became per-user.
    if user_id == "admin" and os.path.exists(_LEGACY_PATH):
        try:
            return _load_json(_LEGACY_PATH)
        except Exception:
            pass
    return LiveRiskLimits()


_limits_cache: dict[str, LiveRiskLimits] = {}


def get_limits(user_id: str = "admin") -> LiveRiskLimits:
    """A snapshot copy — callers must never mutate the shared instance
    in-place, since that would race with set_limits()."""
    with _lock:
        if user_id not in _limits_cache:
            _limits_cache[user_id] = load(user_id)
        return LiveRiskLimits(**asdict(_limits_cache[user_id]))


def set_limits(user_id: str = "admin", **kwargs) -> LiveRiskLimits:
    """Update one or more fields and persist immediately to disk. Unknown
    kwargs are ignored rather than raising, so a stale UI field never crashes
    the save. Takes effect on a RUNNING live bot on its very next tick."""
    with _lock:
        current = asdict(_limits_cache.get(user_id) or load(user_id))
        current.update({k: v for k, v in kwargs.items()
                        if k in LiveRiskLimits.__dataclass_fields__})
        updated = LiveRiskLimits(**current)
        _limits_cache[user_id] = updated
        try:
            by_user = config_store.load(_KEY)
            if not isinstance(by_user, dict):
                by_user = {}
            by_user[user_id] = asdict(updated)
            config_store.save(_KEY, by_user)
        except Exception as exc:
            print(f"[risk_manager] could not persist limits for {user_id}: {exc}")
        return LiveRiskLimits(**asdict(updated))


def daily_loss_limit(limits: LiveRiskLimits, base_capital: float) -> float:
    """Resolve the effective ₹ daily-loss kill-switch from the cash/pct pair —
    the STRICTER (smaller) of whichever legs are actually set. Returns 0.0
    (disabled) if neither is set."""
    candidates = []
    if limits.max_daily_loss_cash and limits.max_daily_loss_cash > 0:
        candidates.append(limits.max_daily_loss_cash)
    if limits.max_daily_loss_pct and limits.max_daily_loss_pct > 0:
        candidates.append(base_capital * limits.max_daily_loss_pct / 100.0)
    return min(candidates) if candidates else 0.0
