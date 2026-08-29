"""
HTTP adapter for the isolated `crudeoil_pipeline` package.

    WHY THIS FILE LIVES HERE AND NOT IN THE PACKAGE
    crudeoil_pipeline/CLAUDE.md RULE #1 forbids the package from importing any
    WelthWest module, and tests/test_isolation.py enforces it. The dependency
    therefore has to point ONE WAY: this adapter (a WelthWest module) imports
    the package; the package never imports back. That keeps the pipeline
    independently testable and portable while still being reachable from the
    existing frontend.

    This module is ADDITIVE. It adds a router; it changes no existing route,
    no existing strategy and no existing engine behaviour.
"""
from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from api.auth import require_admin

# The package sits beside `backend/`, so make the repo root importable. This is
# a sys.path entry, not an import of repo code — the isolation rule is intact.
_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Admin-gated, matching every other research router (advanced_backtest, etc).
# A backtest endpoint is compute-heavy and reveals strategy internals, so it
# must not be reachable unauthenticated.
router = APIRouter(prefix="/crudeoil", tags=["crudeoil-pipeline"],
                   dependencies=[Depends(require_admin)])


class BacktestRequest(BaseModel):
    symbol: str = "CRUDEOILM"
    start: Optional[str] = None
    end: Optional[str] = None
    initial_capital: float = 1_000_000.0
    # 5-minute is the design timeframe (crudeoil_pipeline/config/timeframe.py).
    # Bar-count rules auto-scale, so 15 still works if only 15m data exists.
    timeframe_minutes: int = 5
    # Exit rules — the parameters the in-sample sweep showed matter most.
    atr_stop_mult: Optional[float] = None
    target_r: Optional[float] = None
    trail_atr_mult: Optional[float] = None
    breakeven_at_r: Optional[float] = None
    risk_pct_per_trade: Optional[float] = None
    # Which strategies to run.
    us_open_momentum: bool = True
    opening_range_breakout: bool = True
    eia_event: bool = True
    vwap_meanrev: bool = False
    #: Run walk-forward instead of a single in-sample pass. THIS IS THE ONLY
    #: MODE WHOSE NUMBERS MEAN ANYTHING (CLAUDE.md §8) — a plain backtest over
    #: the whole span is in-sample by construction and is labelled as such.
    walk_forward: bool = False
    train_days: int = 30
    test_days: int = 10
    monte_carlo: bool = True


def _settings(req: BacktestRequest):
    from crudeoil_pipeline.config.schema import Settings
    s = Settings.load()
    risk = s.risk.model_copy(update={
        k: v for k, v in {
            "atr_stop_mult": req.atr_stop_mult,
            "momentum_target_r": req.target_r,
            "trail_atr_mult": req.trail_atr_mult,
            "breakeven_at_r": req.breakeven_at_r,
        }.items() if v is not None})
    sizing = s.sizing.model_copy(update=(
        {"risk_pct_per_trade": req.risk_pct_per_trade}
        if req.risk_pct_per_trade is not None else {}))
    strategies = s.strategies.model_copy(update={
        "us_open_momentum": req.us_open_momentum,
        "opening_range_breakout": req.opening_range_breakout,
        "eia_event": req.eia_event,
        "vwap_meanrev": req.vwap_meanrev,
    })
    return s.model_copy(update={
        "symbol": req.symbol.upper(),
        "initial_capital": req.initial_capital,
        "timeframe_minutes": req.timeframe_minutes,
        "risk": risk, "sizing": sizing, "strategies": strategies,
    })


@router.get("/info")
def info() -> dict:
    """Static description of the pipeline — powers the frontend panel."""
    try:
        from crudeoil_pipeline.config.contracts import (COST_ROUNDTRIP_POINTS,
                                                        CONTRACTS)
        from crudeoil_pipeline.backtest.costs import roundtrip_points
        hurdles = {sym: round(roundtrip_points(5500.0, 1, sym), 2)
                   for sym in CONTRACTS}
        return {
            "available": True,
            "symbols": sorted(CONTRACTS),
            "strategies": [
                {"key": "us_open_momentum", "name": "US-Open Momentum",
                 "primary": True,
                 "summary": "Waits ~15 min after the NYMEX open, then takes the "
                            "first impulse reclaim/loss of the US-anchored VWAP."},
                {"key": "opening_range_breakout", "name": "Opening-Range Breakout",
                 "primary": True,
                 "summary": "Break of the 30-min US opening range with volume "
                            "confirmation and an ATR range-sanity filter."},
                {"key": "eia_event", "name": "EIA Event Overlay", "primary": False,
                 "summary": "Wednesday only, after the EIA print. Trades the "
                            "confirmed reaction at half size."},
                {"key": "vwap_meanrev", "name": "VWAP Mean-Reversion",
                 "primary": False,
                 "summary": "Fades a stretch from VWAP. RANGE regime only — it "
                            "is disabled in a trend by design. Off by default."},
            ],
            "cost_hurdle_points": hurdles,
            "cost_hurdle_expected": COST_ROUNDTRIP_POINTS,
            "notes": [
                "Every metric is NET OF COSTS (CLAUDE.md RULE #3).",
                "Session times are DST-aware: MCX closes 23:30 IST on US DST, "
                "23:55 IST on US standard time.",
                "A plain backtest over the whole span is IN-SAMPLE. Use "
                "walk-forward for a number that means anything.",
            ],
        }
    except Exception as exc:                            # noqa: BLE001
        return {"available": False, "error": str(exc)}


@router.get("/data-status")
def data_status(symbol: str = "CRUDEOILM") -> dict:
    """What bars the pipeline can actually see. Surfaced in the UI because
    'no trades' and 'no data' look identical in a results panel."""
    try:
        from crudeoil_pipeline.data.adapters.parquet_source import (
            ParquetBarSource, default_data_dir)
        src = ParquetBarSource()
        files = src.available(symbol)
        if not files:
            return {"ok": False, "data_dir": str(default_data_dir()),
                    "files": [], "message":
                        f"No parquet bars for {symbol}. Drop {symbol}.parquet "
                        f"into {default_data_dir()}."}
        from crudeoil_pipeline.data.adapters.parquet_source import _infer_minutes
        # timeframe_minutes=0 => no resampling, so this reports the data as it
        # actually is. Asking for a fixed width here would report the count
        # AFTER aggregation and hide the native resolution, which is exactly
        # the thing the UI needs to show (a 5m request against 15m data errors).
        df = src.load(symbol, timeframe_minutes=0)
        native = _infer_minutes(df.index)
        return {"ok": True, "data_dir": str(default_data_dir()),
                "files": [os.path.basename(f) for f in files],
                "bars": int(len(df)), "native_timeframe_minutes": native,
                "start": str(df.index.min()), "end": str(df.index.max())}
    except Exception as exc:                            # noqa: BLE001
        return {"ok": False, "error": str(exc)}


@router.post("/backtest")
def run(req: BacktestRequest) -> dict:
    """Backtest (in-sample) or walk-forward (out-of-sample) the pipeline."""
    try:
        from crudeoil_pipeline import pipeline
        from crudeoil_pipeline.validation.montecarlo import reshuffle
        from crudeoil_pipeline.validation.walkforward import walk_forward
    except ImportError as exc:
        raise HTTPException(500, f"crudeoil_pipeline not importable: {exc}")

    s = _settings(req)
    try:
        feats, report = pipeline.load_features(s, start=req.start, end=req.end)
    except FileNotFoundError as exc:
        raise HTTPException(400, str(exc))
    except Exception as exc:                            # noqa: BLE001
        raise HTTPException(500, f"feature build failed: {exc}")

    try:
        if req.walk_forward:
            wf = walk_forward(feats, s,
                              lambda f, st: pipeline.generate_signals(f, st),
                              train_days=req.train_days, test_days=req.test_days)
            trades, metrics = wf.oos_trades, wf.oos_metrics
            folds = wf.fold_table().to_dict("records")
            stable = wf.stable
            equity = wf.oos_equity
        else:
            from crudeoil_pipeline.backtest.engine import run_backtest
            sigs = pipeline.generate_signals(feats, s)
            res = run_backtest(feats, sigs, s, label="in-sample")
            trades, metrics = res.trades, res.metrics
            folds, stable = [], None
            equity = res.equity
            metrics["signals_generated"] = res.signals
            metrics["rejected"] = res.rejected
    except Exception as exc:                            # noqa: BLE001
        raise HTTPException(500, f"backtest failed: {exc}\n"
                                 f"{traceback.format_exc(limit=3)}")

    mc = None
    if req.monte_carlo and trades is not None and not trades.empty:
        try:
            mc = reshuffle(trades, s.initial_capital, runs=1000,
                           seed=s.seed).as_dict()
        except Exception:                               # noqa: BLE001
            mc = None

    eq_points = []
    if equity is not None and len(equity):
        step = max(len(equity) // 500, 1)               # cap payload size
        eq_points = [{"t": str(t), "equity": float(v)}
                     for t, v in equity.iloc[::step].items()]

    gate8 = _gate8(metrics, stable, mc) if req.walk_forward else None

    return {
        "metrics": metrics,
        "is_out_of_sample": bool(req.walk_forward),
        "folds": folds,
        "stable_across_folds": stable,
        "monte_carlo": mc,
        "gate8": gate8,
        "equity_curve": eq_points,
        "trades": (trades.assign(
            entry_ts=trades["entry_ts"].astype(str),
            exit_ts=trades["exit_ts"].astype(str)).to_dict("records")
            if trades is not None and not trades.empty else []),
        "data_quality": report.summary(),
    }


def _gate8(metrics: dict, stable: Optional[bool], mc: Optional[dict]) -> dict:
    """CLAUDE.md GATE 8 — the explicit GO/NO-GO for live.

    Reported as a structured verdict rather than prose so the UI cannot
    accidentally present a failing strategy as a passing one.
    """
    pf = metrics.get("profit_factor")
    checks = {
        "net_sharpe_ge_1.0": None,   # bar Sharpe is undefined across OOS folds
        "profit_factor_ge_1.3": (pf is not None and pf >= 1.3),
        "stable_across_windows": bool(stable),
        "mc_p95_dd_survivable": (mc is not None
                                 and mc.get("p95_max_dd_pct", -100) > -25.0),
    }
    decided = [v for v in checks.values() if v is not None]
    return {"checks": checks,
            "passed": bool(decided) and all(decided),
            "verdict": ("GO" if decided and all(decided) else "NO-GO"),
            "note": "Sharpe is not evaluated across disjoint OOS windows; "
                    "profit factor and cross-window stability decide the gate."}
