"""
api/routers/advanced_backtest.py
The combination-search surface. Admin-only, and it changes nothing: a search
reads history and returns a ranking. Applying a result stays a human action.
"""
from __future__ import annotations

import os

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

import config
from advanced_backtest import jobs
from advanced_backtest.search import (DEFAULT_VERIFY_TOP, MAX_WORKERS,
                                      SearchSpec)
from api.auth import require_admin
from config import Mode

router = APIRouter(prefix="/advanced-backtest", tags=["advanced-backtest"],
                   dependencies=[Depends(require_admin)])

#: Ceiling on symbols per search. Each one is a full simulation in the screen
#: pass (~1.8 s, five at a time) plus a one-off download if it is not cached
#: yet, so the cost is linear and modest: 114 symbols is roughly 40 s of
#: simulation, plus about a minute of fetching the FIRST time only — the
#: superset cache means later runs re-use it whatever the date range.
#:
#: The default comfortably covers the whole equity universe (114 instruments).
#: Raise it via the environment if you add more; nothing here breaks at a
#: higher number, it simply takes proportionally longer.
try:
    MAX_SYMBOLS = max(1, int(os.getenv("ADV_BACKTEST_MAX_SYMBOLS", "150")))
except ValueError:
    MAX_SYMBOLS = 150


class SearchRequest(BaseModel):
    """Everything held FIXED, plus the symbols to search over.

    RR, signal score, session hours and direction are inputs rather than search
    axes by decision — the question is "given this configuration, which symbol
    and which pattern", not "which configuration".
    """
    symbols: list[str]
    start: str
    end: str
    capital: float = 100_000.0
    mode: str = "Intraday"
    strategy_key: str = "candlestick_engine"
    #: 0 = the strategy's own RR.
    risk_reward: float = 0.0
    #: 0 = the strategy's own threshold.
    min_score: float = 0.0
    #: Fraction of the window used to CHOOSE; the rest scores what was chosen.
    split: float = 0.7
    verify_top: int = DEFAULT_VERIFY_TOP


@router.post("/start")
def start(req: SearchRequest):
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
    if not 0.3 <= req.split <= 0.9:
        raise HTTPException(
            400, "Split must be between 0.3 and 0.9 — outside that one half is "
                 "too small to conclude anything from.")

    spec = SearchSpec(
        symbols=symbols, start=req.start, end=req.end, capital=req.capital,
        mode=mode, strategy_key=req.strategy_key,
        risk_reward=req.risk_reward, min_score=req.min_score,
        split=req.split, verify_top=max(5, min(req.verify_top, 50)))
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
        raise HTTPException(404, "No such search.")
    return out


@router.post("/jobs/{job_id}/cancel")
def cancel(job_id: str):
    if not jobs.cancel(job_id):
        raise HTTPException(404, "That search is not running.")
    return {"ok": True}


@router.get("/limits")
def limits():
    """What the UI needs to describe the form honestly — rather than repeating
    a number that then drifts from the server's."""
    return {"max_symbols": MAX_SYMBOLS, "workers": MAX_WORKERS}
