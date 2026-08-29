"""
Paper/live bridge for the isolated `crudeoil_pipeline` package.

Registers the pipeline as an ordinary strategy in the existing registry, so it
can be selected in the sidebar and traded by the existing engine in Paper or
Live exactly like any other — while all of its DECISION logic stays inside the
greenfield package.

    THE DEPENDENCY POINTS ONE WAY.
    This module (WelthWest) imports crudeoil_pipeline. The package never
    imports back — crudeoil_pipeline/CLAUDE.md RULE #1, enforced by
    tests/test_isolation.py. So the pipeline can be developed, tested and
    replaced without touching the trading platform, and the platform gains a
    strategy without inheriting the pipeline's dependencies.

    WHAT THIS BRIDGE DOES NOT DO
    It does not re-implement the pipeline's risk or sizing. The engine's own
    MCX path (fixed lots from the sidebar, real broker margin) sizes the trade,
    and the engine's own square-off closes it. The pipeline supplies the
    DECISION — side, entry reference, stop and target — which is exactly the
    boundary strategy/base.py draws.

    GATE 9 WARNING
    crudeoil_pipeline/CLAUDE.md gates live trading behind GATE 8, and as of the
    last walk-forward the pipeline FAILS it (out-of-sample profit factor 0.64
    against a 1.3 requirement, and not stable across windows). This bridge
    exists so the strategy can be PAPER traded and inspected. Do not enable it
    on a live account until a walk-forward passes.
"""
from __future__ import annotations

import sys
from datetime import time as dtime
from pathlib import Path
from typing import Optional

import pandas as pd

from config import CRUDEOIL_PIPELINE_PARAMS, Mode, Segment, StrategyParams
from strategy import Signal, StrategyDef, register

# The package sits beside `backend/`. A sys.path entry, not an import of repo
# code — the isolation rule is about imports, and it stays intact.
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_UNAVAILABLE = ""
try:
    from crudeoil_pipeline import pipeline as _pipe
    from crudeoil_pipeline.config.schema import Settings as _Settings
    from crudeoil_pipeline.data.align import to_utc as _to_utc
except Exception as exc:                                # noqa: BLE001
    _pipe = None
    _UNAVAILABLE = f"{type(exc).__name__}: {exc}"


def _bar_minutes(df: pd.DataFrame, default: int = 15) -> int:
    """The MODAL bar width of the frame the engine handed us.

    Inferred rather than assumed, because the platform's feed width is keyed
    off Mode (data_feed.TF_MINUTES: Scalper=1, Intraday=15) and could change.
    The mode is used rather than the mean because every overnight and weekend
    gap drags a mean upward.

    This matters more than it looks: the pipeline expresses its look-backs as
    DURATIONS and converts them to bars against this number
    (crudeoil_pipeline/config/timeframe.py). Get it wrong and every rule
    silently spans the wrong amount of market time.
    """
    if not isinstance(df.index, pd.DatetimeIndex) or len(df) < 3:
        return default
    d = pd.Series(df.index).diff().dropna()
    if d.empty:
        return default
    mins = (d.dt.total_seconds() / 60).round().astype(int)
    mins = mins[mins > 0]
    return int(mins.mode().iloc[0]) if not mins.empty else default


def _settings_for(params: StrategyParams, symbol: str, bar_minutes: int):
    """Map the platform's StrategyParams onto the pipeline's own Settings.

    Only the handful of knobs the sidebar actually exposes are mapped; the rest
    come from crudeoil_pipeline/config/settings.yaml, which stays the single
    source of truth for the pipeline (§6, config-driven).

    `atr_period` is deliberately NOT taken from the platform params: the
    pipeline derives it from the timeframe so it always spans ~70 minutes.
    Passing the platform's raw 14 would mean 3.5 hours on 15m bars.
    """
    s = _Settings.load()
    return s.model_copy(update={
        "symbol": symbol,
        "timeframe_minutes": bar_minutes,
        "risk": s.risk.model_copy(update={
            "atr_stop_mult": params.atr_sl_mult,
            "momentum_target_r": params.risk_reward,
            "min_reward_risk": min(params.risk_reward, s.risk.min_reward_risk),
        }),
    })


def crudeoil_pipeline_signal(df: pd.DataFrame, params: StrategyParams,
                             session_open: Optional[dtime] = None
                             ) -> Optional[Signal]:
    """Adapt the pipeline's Signal to the platform's Signal.

    Returns a signal ONLY when the pipeline fires on the most recent bar —
    `pipeline.latest_signal` enforces that — so a historical signal deeper in
    the window can never be replayed as if it were new.

    The pipeline decides side/entry/stop/target; the engine sizes and executes.
    """
    if _pipe is None or df is None or len(df) < 60:
        return None
    if not isinstance(df.index, pd.DatetimeIndex):
        return None

    symbol = str(getattr(params, "_bridge_symbol", "") or "CRUDEOILM")
    try:
        bars = df[["open", "high", "low", "close", "volume"]].copy()
        bars = _to_utc(bars, assume="Asia/Kolkata")
        st = _settings_for(params, symbol, _bar_minutes(bars))
        sig = _pipe.latest_signal(st, bars)
        if sig is None:
            return None

        from crudeoil_pipeline.config.calendar import MarketCalendar
        from crudeoil_pipeline.features.build import atr as _atr
        from crudeoil_pipeline.risk.manager import RiskManager

        # The pipeline's OWN ATR, over its timeframe-scaled period — not the
        # platform's `df["atr"]`, which uses the platform's atr_period and would
        # size the stop against a different span of market time than the signal
        # was generated with.
        prof = _pipe.profile_for(st)
        atr_series = _atr(bars, prof.atr_period)
        feats_atr = float(atr_series.iloc[-1])
        if not (feats_atr > 0):
            return None
        rm = RiskManager(st.risk, st.costs, MarketCalendar(), symbol)
        br = rm.build(sig, feats_atr)
        if not br.ok:
            # A rejected bracket is a real decision (usually "the target does
            # not clear round-trip friction"), not an error. Staying flat is
            # the correct outcome.
            return None
        return Signal(sig.side, br.entry, br.stop, br.target,
                      f"[pipeline] {sig.reason}")
    except Exception:                                   # noqa: BLE001
        # A bridge failure must never take the trading loop down with it. The
        # engine treats None as "no signal", which is the safe outcome.
        return None


register(StrategyDef(
    key="crudeoil_pipeline", name="Crudeoil Pipeline (US-session)",
    params_by_mode={Mode.INTRADAY: CRUDEOIL_PIPELINE_PARAMS},
    fn=crudeoil_pipeline_signal,
    segments=(Segment.MCX,),        # see crudeoil_engine.py for why
    summary=("MCX crude only. Isolated crudeoil_pipeline package: US-open "
             "momentum + opening-range breakout + EIA overlay, DST-aware, "
             "cost-gated (a target that cannot clear round-trip friction is "
             "refused). PAPER ONLY — it currently fails its own GATE 8 "
             "(out-of-sample PF 0.64 vs 1.3 required)."),
))


def availability() -> dict:
    """Surfaced by the API so the UI can explain a missing pipeline."""
    return {"available": _pipe is not None, "error": _UNAVAILABLE}
