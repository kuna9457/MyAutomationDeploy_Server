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
from api.schemas import BacktestRequest
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
    result = backtester.run_backtest(
        req.ticker, req.start, req.end, req.initial_capital, mode,
        lot_size=inst.lot_size, strategy_key=req.strategy_key,
        risk_reward=req.risk_reward,
    )
    equity = result.equity_curve
    return {
        "metrics": result.metrics,
        "equity_curve": [{"t": str(ts), "equity": float(v)}
                         for ts, v in equity.items()] if not equity.empty else [],
        "trades": _jsonable_trades(result.trades),
    }
