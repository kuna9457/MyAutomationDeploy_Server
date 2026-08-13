"""
ai_auditor/audit_pack.py
Turn the trade store into the compact JSON an LLM is allowed to see.

THE LOAD-BEARING RULE: raw trades never leave this module. A year of Intraday on
a handful of symbols is thousands of rows — past any sensible context window,
expensive per run, and (worst of all) a model asked to sum 8,000 numbers will
get it wrong and then reason confidently from the wrong total. So every figure
the model receives is computed HERE, in Python, and the prompt forbids the model
from deriving any number that is not present.

Construction is by ALLOW-LIST, never by dumping documents: a field reaches the
pack only because it is named below. That is what keeps tokens, ids and other
accounts' figures out of a payload bound for a third-party API — see
`assert_no_secrets`, which is run before any network call.
"""
from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Any, Optional

import pandas as pd

import admin_config
import backtester
import config
import risk_manager
import strategy
import strategy_groups
import symbol_config
from config import Environment, Mode

#: Per-slice caps. These are what keep the pack bounded as the trade log grows —
#: without them a year of the strategy board would produce a pack no model could
#: read and no wallet would enjoy.
TOP_SETUPS = 10
TOP_SYMBOLS = 15

#: Below this a slice is reported but explicitly marked as too small to conclude
#: from. Mirrors backtester.MIN_BUCKET_TRADES' intent at the slice level.
MIN_SLICE_TRADES = 30

#: Refuse to send a pack larger than this. A pack this big means something is
#: wrong with the caps above, and silently sending it would be expensive.
MAX_PACK_BYTES = 400_000


# --------------------------------------------------------------------------- #
#  Redaction
# --------------------------------------------------------------------------- #
#: Substrings that must never appear in a serialised pack. Checked literally
#: against the real values, so this catches a leak however it got there —
#: including one introduced by a future edit that forgets the allow-list rule.
def _secret_values() -> list[str]:
    out = []
    for name in ("UPSTOX_SANDBOX_TOKEN", "UPSTOX_LIVE_ACCESS_TOKEN",
                 "UPSTOX_LIVE_API_KEY", "UPSTOX_LIVE_SECRET",
                 "DHAN_CLIENT_ID", "DHAN_ACCESS_TOKEN",
                 "ZERODHA_API_KEY", "ZERODHA_API_SECRET", "ZERODHA_ACCESS_TOKEN",
                 "KOTAK_NEO_ACCESS_TOKEN", "MONGO_URI",
                 "OPENROUTER_API_KEY", "GEMINI_API_KEY"):
        val = getattr(config, name, "") or ""
        # Short values would false-positive against ordinary text.
        if isinstance(val, str) and len(val) >= 8:
            out.append(val)
    return out


_FORBIDDEN_KEYS = re.compile(
    r"(token|secret|api_key|apikey|password|mongo_uri|user_id|username|"
    r"broker_gtt_id)", re.I)


def assert_no_secrets(pack: dict) -> None:
    """Raise if the pack contains a credential or an account identifier.

    Called immediately before every provider request. Belt and braces: the pack
    is already built from an allow-list, but this is the check that survives
    someone later adding a field without thinking about where it goes.
    """
    blob = json.dumps(pack, default=str)
    for secret in _secret_values():
        if secret in blob:
            raise ValueError(
                "Refusing to send the audit pack: it contains a credential. "
                "This is a bug in audit_pack.py — report it rather than "
                "working around it.")

    def walk(node: Any, path: str = "") -> None:
        if isinstance(node, dict):
            for k, v in node.items():
                if _FORBIDDEN_KEYS.search(str(k)):
                    raise ValueError(
                        f"Refusing to send the audit pack: field {path}.{k} "
                        f"looks like a credential or an account identifier.")
                walk(v, f"{path}.{k}")
        elif isinstance(node, list):
            for i, v in enumerate(node):
                walk(v, f"{path}[{i}]")

    walk(pack)


# --------------------------------------------------------------------------- #
#  Trade-store adapter
# --------------------------------------------------------------------------- #
def _closed_trades(db, env: Environment, start: str = "", end: str = "",
                   ) -> tuple[pd.DataFrame, int]:
    """(closed trades in window, count of open trades excluded).

    Only CLOSED trades can be audited — an open position has no realised PnL.
    The open count is carried into the pack so it is honest about what it left
    out rather than quietly dropping rows.
    """
    df = db.get_trades(env)
    if df.empty:
        return pd.DataFrame(), 0
    open_n = int((df["status"] != "CLOSED").sum()) if "status" in df else 0
    closed = df[df["status"] == "CLOSED"].copy() if "status" in df else pd.DataFrame()
    if closed.empty:
        return closed, open_n

    closed["_entry_dt"] = pd.to_datetime(closed["timestamp"], errors="coerce",
                                         utc=True)
    closed = closed[closed["_entry_dt"].notna()]
    # Stored UTC; the bot trades IST and every day/hour cut below must be read
    # in the market's own clock or "09:00" means nothing.
    closed["_entry_dt"] = closed["_entry_dt"].dt.tz_convert("Asia/Kolkata")
    if start:
        closed = closed[closed["_entry_dt"] >= pd.Timestamp(start, tz="Asia/Kolkata")]
    if end:
        closed = closed[closed["_entry_dt"]
                        <= pd.Timestamp(end, tz="Asia/Kolkata") + pd.Timedelta(days=1)]
    if closed.empty:
        return closed, open_n

    closed["_pnl"] = pd.to_numeric(closed["realized_pnl"], errors="coerce").fillna(0.0)
    closed["_risk"] = pd.to_numeric(closed.get("risk_amount"), errors="coerce")
    exit_dt = pd.to_datetime(closed.get("exit_timestamp"), errors="coerce", utc=True)
    closed["_hold_min"] = (exit_dt - closed["_entry_dt"]).dt.total_seconds() / 60.0
    return closed, open_n


def _to_analytics_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Rename the live trade columns onto the shape backtester.trade_analytics
    already consumes.

    Reusing that function rather than writing a second aggregator is deliberate:
    it is already unit-tested, already applies the 8-trade bucket floor, and a
    second implementation would eventually disagree with the Backtest tab's
    charts about the same trades.
    """
    return pd.DataFrame({
        "entry_time": df["_entry_dt"].dt.tz_localize(None),
        "pnl": df["_pnl"],
        "side": df.get("side", "BUY"),
        "win": df["_pnl"] > 0,
        "entry_reason": df.get("entry_reason", ""),
    })


# --------------------------------------------------------------------------- #
#  Aggregation
# --------------------------------------------------------------------------- #
def _round(v: Any, n: int = 2) -> Any:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if pd.isna(f) else round(f, n)


def _core_stats(df: pd.DataFrame) -> dict:
    """The figures every level of the pack reports, computed one way only."""
    pnl = df["_pnl"]
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    gross_win = float(wins.sum())
    gross_loss = float(-losses.sum())
    risk = df["_risk"].dropna()
    return {
        "trades": int(len(df)),
        "pnl": _round(pnl.sum()),
        "win_rate": _round(100.0 * len(wins) / len(df)) if len(df) else 0.0,
        "avg_pnl": _round(pnl.mean()),
        "avg_win": _round(wins.mean()) if len(wins) else 0.0,
        "avg_loss": _round(losses.mean()) if len(losses) else 0.0,
        "best": _round(pnl.max()),
        "worst": _round(pnl.min()),
        # Profit factor: None (not 0, not inf) when there is nothing to divide,
        # so the model can tell "no losses yet" from "lost everything".
        "profit_factor": _round(gross_win / gross_loss, 3) if gross_loss > 0 else None,
        # Expectancy in R — average PnL per rupee risked. The one figure that
        # compares a commodity lot against an equity position honestly.
        "expectancy_r": (_round(float(pnl.mean()) / float(risk.mean()), 3)
                         if len(risk) and float(risk.mean()) > 0 else None),
        "avg_hold_minutes": _round(df["_hold_min"].dropna().mean(), 1),
    }


def _exit_mix(df: pd.DataFrame) -> dict:
    if "exit_reason" not in df.columns:
        return {}
    # Normalise "TIME-EXIT (7m)" -> "TIME-EXIT" so the mix has three buckets
    # rather than one per hold length.
    reasons = (df["exit_reason"].astype(str).str.split("(").str[0].str.strip()
               .replace("", "UNKNOWN"))
    return {str(k): int(v) for k, v in reasons.value_counts().items()}


def _by_symbol(df: pd.DataFrame) -> list[dict]:
    rows = []
    for sym, g in df.groupby("ticker", sort=False):
        s = _core_stats(g)
        rows.append({"symbol": str(sym), "trades": s["trades"], "pnl": s["pnl"],
                     "win_rate": s["win_rate"], "expectancy_r": s["expectancy_r"]})
    rows.sort(key=lambda r: (r["pnl"] is None, -(r["pnl"] or 0)))
    if len(rows) <= TOP_SYMBOLS * 2:
        return rows
    # Best and worst only: the middle of a long list is where tokens go to die,
    # and the actionable names are at the ends.
    return rows[:TOP_SYMBOLS] + rows[-TOP_SYMBOLS:]


def _slice_rows(df: pd.DataFrame) -> list[dict]:
    """One row per (strategy, mode, category) that actually has trades."""
    out = []
    keys = [k for k in ("strategy", "mode", "category") if k in df.columns]
    if not keys:
        return out
    for key, g in df.groupby(keys, sort=False):
        key = key if isinstance(key, tuple) else (key,)
        named = dict(zip(keys, [str(k) for k in key]))
        stats = _core_stats(g)
        a = backtester.trade_analytics(_to_analytics_frame(g),
                                       top_setups=TOP_SETUPS)
        out.append({
            **named,
            **stats,
            "too_small_to_conclude": stats["trades"] < MIN_SLICE_TRADES,
            "exit_mix": _exit_mix(g),
            "by_side": a["by_side"],
            "by_weekday": a["by_weekday"],
            "by_hour": a["by_hour"],
            "by_setup": a["by_setup"],
            "by_symbol": _by_symbol(g),
        })
    out.sort(key=lambda r: (r["pnl"] is None, -(r["pnl"] or 0)))
    return out


# --------------------------------------------------------------------------- #
#  Current configuration — so advice targets real levers
# --------------------------------------------------------------------------- #
#: The ONLY settings the auditor may propose changing. Handed to the model so a
#: recommendation names something that actually exists instead of inventing a
#: knob nobody can turn.
AVAILABLE_LEVERS = [
    "strategy_board.enabled", "strategy_board.symbols",
    "per_mode.risk_reward", "per_mode.min_score",
    "per_mode.square_off_time", "per_mode.square_off_enabled",
    "per_symbol.trade_days", "per_symbol.trade_hours",
    "per_symbol.risk_reward", "per_symbol.square_off_at_end",
    "per_symbol.trail_enabled", "per_symbol.trail_atr_mult",
    "risk_limits.capital_allocated", "risk_limits.max_daily_loss_cash",
    "risk_limits.max_daily_loss_pct", "risk_limits.max_trades_per_day",
    "risk_limits.max_qty_per_trade", "risk_limits.intraday_leverage",
]


def _current_config(user_id: str) -> dict:
    """What the bot is set to right now. No credentials — see the allow-list."""
    from dataclasses import asdict

    board = {}
    per_mode = {}
    per_symbol = {}
    for m in Mode:
        try:
            groups = strategy_groups.get_all(m.value)
            if groups:
                board[m.value] = [
                    {"strategy": g.strategy_key, "symbols": g.symbols,
                     "risk_reward": g.risk_reward, "min_score": g.min_score,
                     "enabled": g.enabled}
                    for g in groups]
            mc = admin_config.get_mode_config(m.value)
            per_mode[m.value] = {
                "strategy_key": mc.strategy_key, "symbols": mc.symbols,
                "risk_reward": mc.risk_reward, "min_score": mc.min_score,
                "square_off_time": mc.square_off_time,
                "square_off_enabled": mc.square_off_enabled}
            syms = {s: asdict(c) for s, c in symbol_config.get_all(m.value).items()}
            if syms:
                per_symbol[m.value] = syms
        except Exception:
            # Config is context, not the point of the audit. A store that fails
            # to read must not cost you the whole report.
            continue

    limits = asdict(risk_manager.get_limits(user_id))
    defaults = {}
    for key, sd in strategy._REGISTRY.items():
        defaults[key] = {
            m.value: {
                "risk_reward": p.risk_reward, "risk_per_trade": p.risk_per_trade,
                "cs_min_score": p.cs_min_score, "atr_sl_mult": p.atr_sl_mult,
                "atr_period": p.atr_period, "allow_short": p.allow_short,
                "max_hold_minutes": p.max_hold_minutes,
                "entry_skip_minutes": p.entry_skip_minutes,
                "use_limit_entry": p.use_limit_entry,
                "timeframe": p.timeframe,
            } for m, p in sd.params_by_mode.items()}
    return {"strategy_board": board, "per_mode": per_mode,
            "per_symbol": per_symbol, "risk_limits": limits,
            "strategy_defaults": defaults}


# --------------------------------------------------------------------------- #
#  The pack
# --------------------------------------------------------------------------- #
def build_pack(db, environment: Environment, start: str = "", end: str = "",
               user_id: str = "admin") -> dict:
    """The complete document the model will see. Never raises on empty data —
    an empty book is a legitimate audit result ("nothing to judge yet")."""
    closed, open_n = _closed_trades(db, environment, start, end)

    caveats = [
        "Fills are recorded at the REQUESTED price, not the broker's average "
        "fill price. Real entries and exits will differ.",
        "No slippage or brokerage is modelled in any figure in this pack. A "
        "configuration that trades more often is therefore flattered relative "
        "to one that trades less; an edge thinner than real costs is not an edge.",
        "Only CLOSED trades are included. Open positions carry no realised PnL.",
    ]
    if environment == Environment.PAPER:
        caveats.append(
            "This is PAPER data: fills are simulated and always succeed. It "
            "shows whether the LOGIC works, not whether the edge survives a "
            "real order book.")

    meta = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "environment": environment.value,
        "window": {"from": start or "all", "to": end or "all"},
        "closed_trades": int(len(closed)),
        "open_trades_excluded": open_n,
        "data_caveats": caveats,
    }

    if closed.empty:
        return {"meta": meta, "overall": {}, "by_slice": [],
                "current_config": _current_config(user_id),
                "available_levers": AVAILABLE_LEVERS}

    overall = _core_stats(closed)
    overall["exit_mix"] = _exit_mix(closed)
    days = closed["_entry_dt"].dt.date
    overall["trading_days"] = int(days.nunique())
    # Cumulative realised PnL drawdown, in the order trades actually closed.
    ordered = closed.sort_values("_entry_dt")
    curve = ordered["_pnl"].cumsum()
    peak = curve.cummax()
    overall["max_drawdown_cash"] = _round((curve - peak).min())

    a = backtester.trade_analytics(_to_analytics_frame(closed), top_setups=TOP_SETUPS)
    overall["by_weekday"] = a["by_weekday"]
    overall["by_hour"] = a["by_hour"]
    overall["by_side"] = a["by_side"]

    return {
        "meta": meta,
        "overall": overall,
        "by_slice": _slice_rows(closed),
        "current_config": _current_config(user_id),
        "available_levers": AVAILABLE_LEVERS,
    }


def pack_size_bytes(pack: dict) -> int:
    return len(json.dumps(pack, default=str).encode("utf-8"))
