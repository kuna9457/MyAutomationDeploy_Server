"""
fleet_reporting.py
How ALL the clients are doing, in one view — the roll-up the admin's per-client
stats panel cannot give.

Where the numbers come from
---------------------------
The shared database. Every client's engine tags each trade with its own
`user_id` and writes it to `paper_trades` / `live_trades` (DBManager.new_trade);
when hub and nodes share one MongoDB, a client's trades are here the moment
they are written, wherever the client's bot actually runs. This module only
READS — it never asks a node for a number, so it cannot disagree with the
per-client drill-down (`/admin/clients/{u}/stats`), which reads the same rows.

One read, sliced per client. The obvious version calls DBManager's per-user
helpers once per client, and each of those re-reads the WHOLE collection; with
a few dozen clients that is a few dozen full scans per refresh.

Consistency: every figure is built from DBManager._perf_row, the same function
behind the existing daily / strategy breakdowns, so a client's total here can
never differ from the one on their own stats panel.

"Day" follows the existing convention — the UTC date of the trade's ENTRY
timestamp — so a day here is the same day the rest of the app means.
"""
from __future__ import annotations

from typing import Callable, Optional

import pandas as pd

from db_manager import DBManager

#: Days of history returned. The daily table is a glance, not an archive — a
#: client's full history is one click away on their own panel.
MAX_DAYS = 60


def _owner(df: pd.DataFrame) -> pd.Series:
    """Whose trade each row is. Rows written before multi-tenancy carry no
    user_id and belong to "admin" — the same rule DBManager.get_trades uses."""
    if "user_id" in df.columns:
        return df["user_id"].fillna("admin")
    return pd.Series("admin", index=df.index)


def _closed_pnl(g: pd.DataFrame) -> float:
    closed = g[g["status"] == "CLOSED"]
    if closed.empty:
        return 0.0
    return float(closed["realized_pnl"].astype(float).sum())


def client_pnl_report(
    trades: Optional[pd.DataFrame],
    clients: list[dict],
    live_of: Callable[[str], Optional[dict]],
    today: str,
) -> dict:
    """Totals, one row per client, and a per-day series across all of them.

    `trades` is the WHOLE environment's trades (admin's included — they are
    sliced out below, because this is about clients). `live_of(username)` says
    which server a client is on and whether their bot is running, or None.
    """
    have = trades is not None and not trades.empty
    if have:
        trades = trades.assign(_owner=_owner(trades))
        trades["_day"] = trades["timestamp"].map(DBManager._trade_day)

    rows = []
    for c in clients:
        u = c["username"]
        g = trades[trades["_owner"] == u] if have else None
        if g is not None and not g.empty:
            perf = DBManager._perf_row(g)
            today_pnl = _closed_pnl(g[g["_day"] == today])
        else:
            perf = {"Trades": 0, "Closed": 0, "Open": 0, "Wins": 0,
                    "Win Rate %": 0.0, "Realized PnL (₹)": 0.0}
            today_pnl = 0.0
        live = live_of(u) or {}
        rows.append({
            "username": u,
            "display_name": c.get("display_name", u),
            "status": c.get("status", "active"),
            "node": live.get("node_name"),
            "running": bool(live.get("running")),
            "environment": live.get("environment"),
            "total_pnl": round(perf["Realized PnL (₹)"], 2),
            "today_pnl": round(today_pnl, 2),
            "trades": perf["Trades"],
            "closed": perf["Closed"],
            "open": perf["Open"],
            "wins": perf["Wins"],
            "win_rate": perf["Win Rate %"],
        })

    closed = sum(r["closed"] for r in rows)
    wins = sum(r["wins"] for r in rows)
    totals = {
        "clients": len(rows),
        "running": sum(1 for r in rows if r["running"]),
        "total_pnl": round(sum(r["total_pnl"] for r in rows), 2),
        "today_pnl": round(sum(r["today_pnl"] for r in rows), 2),
        "trades": sum(r["trades"] for r in rows),
        "closed": closed,
        "open": sum(r["open"] for r in rows),
        "win_rate": round(100 * wins / closed, 2) if closed else 0.0,
    }

    daily: list[dict] = []
    names = {c["username"] for c in clients}
    if have and names:
        mine = trades[trades["_owner"].isin(names)]
        for day, g in mine.groupby("_day"):
            if not day:
                continue
            perf = DBManager._perf_row(g)
            daily.append({"date": day, "pnl": round(perf["Realized PnL (₹)"], 2),
                          "trades": perf["Trades"], "closed": perf["Closed"],
                          "wins": perf["Wins"],
                          "clients": int(g["_owner"].nunique())})
        daily.sort(key=lambda r: r["date"], reverse=True)
        daily = daily[:MAX_DAYS]

    rows.sort(key=lambda r: r["total_pnl"], reverse=True)
    return {"totals": totals, "clients": rows, "daily": daily}
