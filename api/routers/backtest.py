"""
api/routers/backtest.py
Wraps backtester.run_backtest exactly as app.py's "Run Backtest" button does
— same inputs, same BacktestResult, only the rendering differs (JSON instead
of st.dataframe/plotly). Bulk/save-report parity (app.py's "Bulk Backtest"
and "Saved Backtest Analyses" sections) is deferred to a follow-up pass —
see frontend_migration_plan.md.
"""
from __future__ import annotations

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException

import backtester
import config
from api.auth import require_admin
from api.schemas import BacktestRequest, BulkBacktestRequest, RRSweepRequest

#: Ceiling on one bulk request. Each symbol is a full simulation; the executor
#: runs 5 at a time, so 40 is a few minutes rather than an unbounded wait.
BULK_MAX_TICKERS = 40
from config import Mode

router = APIRouter(prefix="/backtest", tags=["backtest"], dependencies=[Depends(require_admin)])


def _jsonable_trades(trades: pd.DataFrame) -> list[dict]:
    if trades.empty:
        return []
    t = trades.copy()
    for col in ("entry_time", "exit_time"):
        if col in t.columns:
            t[col] = t[col].astype(str)
    return t.to_dict("records")


@router.post("/run")
def run_backtest(req: BacktestRequest):
    if req.ticker not in config.INSTRUMENTS_BY_SYMBOL:
        raise HTTPException(400, f"Unknown ticker: {req.ticker}")
    try:
        mode = Mode(req.mode)
    except ValueError:
        raise HTTPException(400, "Invalid mode.")
    inst = config.INSTRUMENTS_BY_SYMBOL[req.ticker]
    try:
        filters = backtester.parse_filters(req.trade_days, req.trade_hours,
                                           req.side)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    result = backtester.run_backtest(
        req.ticker, req.start, req.end, req.initial_capital, mode,
        lot_size=inst.lot_size, strategy_key=req.strategy_key,
        risk_reward=req.risk_reward, min_score=req.min_score,
        filters=filters, patterns=req.patterns,
    )
    equity = result.equity_curve
    return {
        "metrics": result.metrics,
        "equity_curve": [{"t": str(ts), "equity": float(v)}
                         for ts, v in equity.items()] if not equity.empty else [],
        "trades": _jsonable_trades(result.trades),
        # Derived from the very same trade rows returned above, so the charts
        # and the table can never disagree.
        "analytics": backtester.trade_analytics(result.trades),
        "filters": filters.describe() if filters else "",
    }


@router.post("/bulk")
def bulk_backtest(req: BulkBacktestRequest):
    """Run one strategy across many symbols and rank them.

    Ranked by return, and every row carries its trade count so a chart-topping
    symbol with four trades is visibly not a finding. Analytics are computed on
    the POOLED trade log across all symbols — the day/hour/setup edges worth
    acting on are the ones that survive a whole bucket, not one lucky name.
    """
    if not req.tickers:
        raise HTTPException(400, "Select at least one symbol.")
    unknown = [t for t in req.tickers if t not in config.INSTRUMENTS_BY_SYMBOL]
    if unknown:
        raise HTTPException(400, f"Unknown ticker(s): {', '.join(unknown)}")
    if len(req.tickers) > BULK_MAX_TICKERS:
        raise HTTPException(
            400, f"{len(req.tickers)} symbols requested; the limit is "
                 f"{BULK_MAX_TICKERS} per run.")
    try:
        mode = Mode(req.mode)
    except ValueError:
        raise HTTPException(400, "Invalid mode.")
    try:
        filters = backtester.parse_filters(req.trade_days, req.trade_hours,
                                           req.side)
    except ValueError as exc:
        raise HTTPException(400, str(exc))

    results = backtester.run_bulk_backtest(
        req.tickers, req.start, req.end, req.initial_capital, mode,
        strategy_key=req.strategy_key, risk_reward=req.risk_reward,
        min_score=req.min_score, filters=filters, patterns=req.patterns,
    )
    summary = backtester.bulk_summary_frame(results)

    pooled = [r.trades for r in results.values()
              if r.trades is not None and not r.trades.empty]
    all_trades = pd.concat(pooled, ignore_index=True) if pooled else pd.DataFrame()

    return {
        "ranking": summary.to_dict("records") if not summary.empty else [],
        "analytics": backtester.trade_analytics(all_trades),
        "filters": filters.describe() if filters else "",
        "tickers": len(req.tickers),
    }


@router.post("/rr-sweep")
def rr_sweep(req: RRSweepRequest):
    """The same backtest at every RR in a ladder — one row each.

    Answers "which risk:reward actually suits this strategy on this symbol"
    in one request instead of re-running the form by hand. Only RR moves;
    every other input is held constant.
    """
    if req.ticker not in config.INSTRUMENTS_BY_SYMBOL:
        raise HTTPException(400, f"Unknown ticker: {req.ticker}")
    try:
        mode = Mode(req.mode)
    except ValueError:
        raise HTTPException(400, "Invalid mode.")
    inst = config.INSTRUMENTS_BY_SYMBOL[req.ticker]
    try:
        rows = backtester.run_rr_sweep(
            req.ticker, req.start, req.end, req.initial_capital, mode,
            rr_start=req.rr_start, rr_step=req.rr_step, rr_end=req.rr_end,
            lot_size=inst.lot_size, strategy_key=req.strategy_key,
            min_score=req.min_score, patterns=req.patterns,
        )
    except ValueError as exc:
        # Bad ladder (start<=0, end<start, step too small, too many runs) —
        # the message is written to be shown to the user as-is.
        raise HTTPException(400, str(exc))
    # `best` is by RETURN, and named rather than implied, so the table can
    # highlight it without the UI re-deriving a different notion of "best".
    ok = [r for r in rows if not r["error"]]
    best = max(ok, key=lambda r: r["return_pct"])["risk_reward"] if ok else None
    return {"rows": rows, "best_by_return": best}
