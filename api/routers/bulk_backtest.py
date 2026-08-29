"""
api/routers/bulk_backtest.py
The multi-axis optimizer surface. Admin-only, read-only — it changes nothing.
A funnel reads history, runs simulations, and returns a ranked table.

COMPLETELY SEPARATE from the existing advanced_backtest router.
"""
from __future__ import annotations

import os

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, field_validator

import config
import strategy
from api.auth import require_admin
from bulk_backtest import jobs
from bulk_backtest.funnel import FunnelSpec
from config import Mode, Segment

router = APIRouter(prefix="/bulk-backtest", tags=["bulk-backtest"],
                   dependencies=[Depends(require_admin)])

try:
    MAX_SYMBOLS = max(1, int(os.getenv("BULK_BACKTEST_MAX_SYMBOLS", "150")))
except ValueError:
    MAX_SYMBOLS = 150


class FunnelRequest(BaseModel):
    """Everything the funnel needs to run."""
    symbols: list[str]
    strategy_keys: list[str] = ["candlestick_engine"]
    start: str
    end: str
    capital: float = 100_000.0
    mode: str = "Intraday"
    rr_ladder: list[float] = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    folds: int = 4
    min_trades: int = 30
    turnover_cap_pct: float = 0.0
    #: Round 0b — the unfiltered control run at every rung of the RR ladder.
    #: On by default; an admin sweeping a very large symbol list can switch it
    #: off to save one simulation per (pair × rung).
    baseline_rr_sweep: bool = True

    @field_validator("rr_ladder")
    @classmethod
    def rr_ladder_valid(cls, v):
        if not v or len(v) < 2:
            raise ValueError("RR ladder needs at least 2 values.")
        if len(v) > 20:
            raise ValueError("RR ladder is capped at 20 rungs.")
        if any(r <= 0 for r in v):
            raise ValueError("All RR values must be > 0.")
        return sorted(set(v))


@router.post("/start")
def start(req: FunnelRequest):
    symbols = [s for s in req.symbols if s in config.INSTRUMENTS_BY_SYMBOL]
    if not symbols:
        raise HTTPException(400, "Select at least one known symbol.")
    if len(symbols) > MAX_SYMBOLS:
        raise HTTPException(
            400, f"{len(symbols)} symbols requested; the limit is {MAX_SYMBOLS}.")
    try:
        mode = Mode(req.mode)
    except ValueError:
        raise HTTPException(400, "Invalid mode.")

    # Validate strategy keys against THIS mode. A key that exists but does not
    # support the mode must be rejected, not passed through: resolve_strategy()
    # silently falls back to the mode default, so the funnel would rank a
    # strategy the admin never asked for under the name they did ask for.
    if not req.strategy_keys:
        raise HTTPException(400, "Select at least one strategy.")
    valid_keys = {s.key for s in strategy.strategies_for_mode(mode)}
    bad = [k for k in req.strategy_keys if k not in valid_keys]
    if bad:
        raise HTTPException(
            400, f"Strategy key(s) {bad} are not available on {mode.value}.")

    if not 2 <= req.folds <= 10:
        raise HTTPException(400, "Folds must be between 2 and 10.")

    spec = FunnelSpec(
        symbols=symbols,
        strategy_keys=req.strategy_keys,
        start=req.start,
        end=req.end,
        capital=req.capital,
        mode=req.mode,
        rr_ladder=req.rr_ladder,
        folds=req.folds,
        min_trades=req.min_trades,
        turnover_cap_pct=req.turnover_cap_pct,
        baseline_rr_sweep=req.baseline_rr_sweep,
    )
    try:
        return {"job_id": jobs.start(spec)}
    except RuntimeError as exc:
        raise HTTPException(409, str(exc))


@router.get("/jobs")
def recent():
    return jobs.recent()


@router.get("/jobs/{job_id}")
def job(job_id: str):
    out = jobs.get(job_id)
    if out is None:
        raise HTTPException(404, "No such bulk backtest.")
    return out


@router.post("/jobs/{job_id}/cancel")
def cancel(job_id: str):
    if not jobs.cancel(job_id):
        raise HTTPException(404, "That bulk backtest is not running.")
    return {"ok": True}


@router.get("/limits")
def limits():
    """Form metadata for the UI."""
    # The whole catalogue, each tagged with the modes it supports — the form
    # picks a mode independently, so the UI needs to know which strategies that
    # choice leaves selectable.
    strategy_list = [
        {"key": s.key, "name": s.name,
         "modes": [m.value for m in s.modes]}
        for s in strategy.all_strategies()
    ]
    margins = {
        i.symbol: config.mcx_margin_per_lot(i.symbol)
        for i in config.ALL_INSTRUMENTS
        if i.segment == Segment.MCX
    }
    return {
        "max_symbols": MAX_SYMBOLS,
        "strategies": strategy_list,
        "mcx_margin_per_lot": margins,
        "default_rr_ladder": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
    }
