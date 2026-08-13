"""
api/routers/chart.py
Trade replay: the candles a trade was taken on, with the trade drawn over them.

READ-ONLY. It reads the trade store and fetches history through the same
backtester.fetch_history the Backtest tab uses, so a chart shows the same
candles a backtest would have seen. Nothing here can place or change anything.

ON TIMESTAMPS — the one subtle part. Trades are stored in UTC; candles come back
on a tz-naive IST index. lightweight-charts renders UNIX seconds in UTC with no
timezone support, so both are converted to IST wall-clock and then labelled as
if that wall-clock were UTC. The chart therefore shows 09:15 for the 09:15 bar
instead of 03:45, and — because candles and trades get the identical treatment —
a 09:20 entry lands on the 09:15 bar it actually belongs to.
"""
from __future__ import annotations

from typing import Optional

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

import backtester
import config
from api.auth import require_admin
from config import Environment, Mode
from db_manager import DBManager

router = APIRouter(prefix="/chart", tags=["chart"],
                   dependencies=[Depends(require_admin)])

_db = DBManager()

#: The timeframe each mode trades. A chart must be drawn on the bars the
#: strategy actually decided on — showing a 1-minute scalp on daily candles
#: would hide the entire trade inside one bar.
_INTERVAL = {Mode.SWING.value: "1d", Mode.INTRADAY.value: "15m",
             Mode.SCALPER.value: "1m"}

#: Bars of context to pad around the first and last trade so an entry is not
#: pinned to the edge of the chart.
_PAD_BARS = 40


class ChartQuery(BaseModel):
    environment: str = "Live"
    #: "" = every strategy.
    strategy: str = ""
    start: str = ""
    end: str = ""
    symbol: str = ""


def _env(name: str) -> Environment:
    try:
        return Environment(name)
    except ValueError:
        raise HTTPException(400, "environment must be 'Paper' or 'Live'.")


def _ist(series: pd.Series) -> pd.Series:
    """UTC-stored timestamps -> tz-naive IST wall-clock."""
    dt = pd.to_datetime(series, errors="coerce", utc=True)
    return dt.dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)


def _epoch(ts: pd.Timestamp) -> int:
    """A tz-naive IST wall-clock -> the UNIX second lightweight-charts must be
    given for that wall-clock to APPEAR on the axis. See the module docstring."""
    return int(ts.tz_localize("UTC").timestamp())


def _intraday_today(inst, mode: str, since: pd.Timestamp) -> pd.DataFrame:
    """Candles including TODAY, via the same helper the live REST feed uses.

    Never raises: today's bars are a bonus on top of the historical fetch, and a
    missing/expired token must degrade to "no chart for today" rather than an
    error on a page that is otherwise fine.
    """
    token = config.UPSTOX_LIVE_ACCESS_TOKEN or config.UPSTOX_SANDBOX_TOKEN
    if not token or not inst.instrument_key:
        return pd.DataFrame()
    try:
        import upstox_client
        import data_feed

        cfg = upstox_client.Configuration()
        cfg.access_token = token
        api = upstox_client.HistoryApi(upstox_client.ApiClient(cfg))
        days = max(2, (pd.Timestamp.now().normalize() - since.normalize()).days + 2)
        return data_feed._fetch_rest_candles(api, inst, Mode(mode), days)
    except Exception as exc:
        print(f"[chart] intra-day candles unavailable for {inst.symbol}: {exc}")
        return pd.DataFrame()


def _closed(environment: Environment, strategy: str, start: str,
            end: str) -> pd.DataFrame:
    df = _db.get_trades(environment)
    if df.empty or "status" not in df.columns:
        return pd.DataFrame()
    df = df[df["status"] == "CLOSED"].copy()
    if df.empty:
        return df
    df["_entry"] = _ist(df["timestamp"])
    df = df[df["_entry"].notna()]
    if strategy:
        df = df[df.get("strategy", "") == strategy]
    if start:
        df = df[df["_entry"] >= pd.Timestamp(start)]
    if end:
        df = df[df["_entry"] <= pd.Timestamp(end) + pd.Timedelta(days=1)]
    return df


@router.post("/symbols")
def symbols(q: ChartQuery):
    """Every symbol traded in the window, newest activity first.

    This is the picker: choose a strategy and a date range, and see what the bot
    actually touched — rather than guessing a symbol and finding an empty chart.
    """
    df = _closed(_env(q.environment), q.strategy, q.start, q.end)
    if df.empty:
        return {"symbols": [], "strategies": []}

    out = []
    for sym, g in df.groupby("ticker", sort=False):
        pnl = pd.to_numeric(g["realized_pnl"], errors="coerce").fillna(0.0)
        wins = int((pnl > 0).sum())
        modes = g["mode"].dropna().tolist() if "mode" in g else []
        out.append({
            "symbol": str(sym),
            "trades": int(len(g)),
            "pnl": round(float(pnl.sum()), 2),
            "wins": wins,
            "win_rate": round(100.0 * wins / len(g), 1) if len(g) else 0.0,
            # A symbol can be traded on more than one mode; the chart uses the
            # most common one, and the UI shows which so the choice is visible.
            "mode": max(set(modes), key=modes.count) if modes else "",
            "category": str(g["category"].iloc[0]) if "category" in g else "",
            "first_trade": g["_entry"].min().strftime("%Y-%m-%d"),
            "last_trade": g["_entry"].max().strftime("%Y-%m-%d"),
        })
    out.sort(key=lambda r: r["last_trade"], reverse=True)

    # Strategies present in this environment, so the picker only offers ones
    # that actually have trades to show.
    all_df = _closed(_env(q.environment), "", q.start, q.end)
    strategies = sorted(set(all_df.get("strategy", pd.Series(dtype=str))
                            .dropna().astype(str)) - {""})
    return {"symbols": out, "strategies": strategies}


@router.post("/candles")
def candles(q: ChartQuery):
    """Candles for one symbol over the window, with its trades attached."""
    if not q.symbol:
        raise HTTPException(400, "symbol is required.")
    inst = config.INSTRUMENTS_BY_SYMBOL.get(q.symbol)
    if inst is None:
        raise HTTPException(400, f"Unknown instrument: {q.symbol}")

    df = _closed(_env(q.environment), q.strategy, q.start, q.end)
    df = df[df["ticker"] == q.symbol] if not df.empty else df
    if df.empty:
        raise HTTPException(404, f"No closed trades for {q.symbol} in that window.")

    modes = df["mode"].dropna().tolist() if "mode" in df else []
    mode = max(set(modes), key=modes.count) if modes else Mode.INTRADAY.value
    interval = _INTERVAL.get(mode, "15m")

    # Pad the fetch window so the first and last trades have context either
    # side. Padding is in DAYS because fetch_history takes dates; the bar counts
    # differ per timeframe, hence the per-interval widths.
    per_day = {"1d": 1, "15m": 25, "1m": 375}.get(interval, 25)
    pad_days = max(2, int(_PAD_BARS / per_day) + 1)
    first = (df["_entry"].min() - pd.Timedelta(days=pad_days)).strftime("%Y-%m-%d")
    last = (df["_entry"].max() + pd.Timedelta(days=pad_days)).strftime("%Y-%m-%d")

    token = config.UPSTOX_LIVE_ACCESS_TOKEN or config.UPSTOX_SANDBOX_TOKEN
    try:
        candles_df, source = backtester.fetch_history(
            q.symbol, first, last, interval, inst.instrument_key, token)
    except Exception as exc:
        raise HTTPException(502, f"Could not fetch history: {exc}")
    if candles_df is None or candles_df.empty:
        candles_df = pd.DataFrame()

    # Upstox's HISTORICAL endpoint stops at yesterday, so a trade taken TODAY —
    # the most likely thing anyone wants to look at — would fall past the last
    # candle and be undrawable. The live feed already solves this by appending
    # the intra-day endpoint, so reuse that rather than reimplement it.
    if df["_entry"].max().date() >= pd.Timestamp.now(tz="Asia/Kolkata").date():
        today_df = _intraday_today(inst, mode, df["_entry"].min())
        if not today_df.empty:
            candles_df = (today_df if candles_df.empty else
                          pd.concat([candles_df, today_df]))
            # Intra-day wins on overlap: it is the fresher read of the same bar.
            candles_df = candles_df[~candles_df.index.duplicated(keep="last")]
            candles_df = candles_df.sort_index()
            if source not in ("upstox", "yfinance"):
                source = "upstox"

    if candles_df.empty:
        raise HTTPException(404, "No candles returned for that window.")

    bars = [{
        "time": _epoch(ts),
        "open": round(float(r["open"]), 2), "high": round(float(r["high"]), 2),
        "low": round(float(r["low"]), 2), "close": round(float(r["close"]), 2),
        "volume": int(r["volume"]) if pd.notna(r.get("volume")) else 0,
    } for ts, r in candles_df.iterrows()]

    # Attached to the frame rather than looked up per row: a label lookup inside
    # the loop is both slower and easy to get wrong once the frame is sorted.
    df = df.copy()
    df["_exit"] = (_ist(df["exit_timestamp"]) if "exit_timestamp" in df
                   else pd.NaT)

    trades = []
    for i, (_idx, t) in enumerate(df.sort_values("_entry").iterrows()):
        pnl = float(pd.to_numeric(t.get("realized_pnl"), errors="coerce") or 0.0)
        x = t["_exit"]
        trades.append({
            "id": str(t.get("trade_id", i)),
            "side": str(t.get("side", "BUY")),
            "entry_time": _epoch(t["_entry"]),
            "entry_price": float(t.get("entry_price") or 0),
            "exit_time": _epoch(x) if pd.notna(x) else None,
            "exit_price": (float(t["exit_price"])
                           if pd.notna(t.get("exit_price")) else None),
            "stop_loss": (float(t["stop_loss"])
                          if pd.notna(t.get("stop_loss")) else None),
            "target": float(t["target"]) if pd.notna(t.get("target")) else None,
            "quantity": int(t.get("quantity") or 0),
            "pnl": round(pnl, 2),
            "win": pnl > 0,
            "strategy": str(t.get("strategy", "")),
            "mode": str(t.get("mode", "")),
            "entry_reason": str(t.get("entry_reason", "") or ""),
            "exit_reason": str(t.get("exit_reason", "") or ""),
        })

    return {
        "symbol": q.symbol, "interval": interval, "mode": mode,
        "source": source, "candles": bars, "trades": trades,
        # Surfaced so a chart drawn on a random walk is never mistaken for one
        # drawn on the market — fetch_history falls back to synthetic data.
        "is_real_data": source in ("upstox", "yfinance", "binance"),
    }
