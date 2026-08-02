"""
strategies/confidence_engine.py

plan.md PHASE 5 — the pattern-prediction / confidence engine, wired up as a
tradable strategy.

This is the phase plan.md calls the one "that would make your script stand out":
instead of a binary "pattern detected", it produces a CONTINUOUSLY UPDATING
confidence score from weighted evidence, and trades only when that score clears a
bar. plan.md specifies the exact weighting, and this file implements it verbatim:

    Trend alignment ........ 20%
    Market structure ....... 20%
    Pattern geometry ....... 25%
    Candle confirmation .... 10%
    Volume confirmation .... 10%
    Support/Resistance ..... 10%
    Volatility conditions .. 5%
                             ----
                             100%

Each factor is scored on a signed [-1, +1] scale (positive = bullish evidence,
negative = bearish), multiplied by its weight, and summed. The weighted sum is
itself in [-1, +1]; its magnitude × 100 is the confidence percentage, and its
sign is the predicted direction. Because it recomputes every bar, the score is
the live "what is forming" read plan.md describes, not a one-shot label.

    SELF-CONTAINED BY DESIGN (the brief)
    ------------------------------------
    plan.md envisions Phase 5 CONSUMING Phases 2-4. The brief for now is
    "independent from each other", so this file does NOT import the Phase 1-4
    modules; it computes its own lightweight version of each factor from the
    enriched OHLCV frame. When the phases are unified, these factor functions are
    the natural seams to swap for the real engines (detect_patterns for geometry,
    the market-structure engine for structure, the context engine for S/R).
"""
from __future__ import annotations

from datetime import time as dtime
from typing import Optional

import numpy as np
import pandas as pd

from config import Mode, StrategyParams
from strategy import (Signal, StrategyDef, _atr_in_normal_range,
                      _past_entry_window, register)

# --------------------------------------------------------------------------- #
#  plan.md's confidence weights (must sum to 1.0).
# --------------------------------------------------------------------------- #
W_TREND = 0.20
W_STRUCTURE = 0.20
W_GEOMETRY = 0.25
W_CANDLE = 0.10
W_VOLUME = 0.10
W_SR = 0.10
W_VOLATILITY = 0.05

# The confidence bar (percent). Below this the tape is too undecided to act — the
# engine still reports the number, it just won't trade on it.
CONF_MIN = 62.0

PIVOT_K = 3
SR_NEAR_ATR = 1.0
ATR_MED_WINDOW = 20
ATR_EXPANSION = 1.10
SL_BUFFER_ATR = 0.3
MIN_SL_ATR = 0.5
MAX_SL_ATR = 4.0


def _clamp(x: float) -> float:
    return max(-1.0, min(1.0, x))


def _pivots(df: pd.DataFrame, k: int) -> list[tuple[int, float, int]]:
    """Local fractal pivots (see market_structure_engine for the rationale)."""
    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()
    n = len(highs)
    out: list[tuple[int, float, int]] = []
    for i in range(k, n - k):
        wh = highs[i - k:i + k + 1]
        wl = lows[i - k:i + k + 1]
        if highs[i] == wh.max() and wh.argmax() == k:
            out.append((i, float(highs[i]), 1))
        elif lows[i] == wl.min() and wl.argmin() == k:
            out.append((i, float(lows[i]), -1))
    return out


# --------------------------------------------------------------------------- #
#  The seven factor functions. Each returns signed evidence in [-1, +1].
# --------------------------------------------------------------------------- #
def _f_trend(df: pd.DataFrame) -> float:
    last = df.iloc[-1]
    close = float(last["close"])
    ema_t = float(last.get("ema_trend", np.nan))
    ema_f = float(last.get("ema_fast", np.nan))
    if not (np.isfinite(ema_t) and np.isfinite(ema_f)):
        return 0.0
    score = 0.5 if close > ema_t else -0.5
    if len(df) > 5:
        slope = float(df["ema_fast"].iloc[-1] - df["ema_fast"].iloc[-5])
        score += 0.5 if slope > 0 else (-0.5 if slope < 0 else 0.0)
    return _clamp(score)


def _f_structure(df: pd.DataFrame) -> float:
    piv = _pivots(df, PIVOT_K)
    highs = [p[1] for p in piv if p[2] == 1]
    lows = [p[1] for p in piv if p[2] == -1]
    if len(highs) < 2 or len(lows) < 2:
        return 0.0
    hi = 1 if highs[-1] > highs[-2] else (-1 if highs[-1] < highs[-2] else 0)
    lo = 1 if lows[-1] > lows[-2] else (-1 if lows[-1] < lows[-2] else 0)
    return _clamp((hi + lo) / 2.0)          # both up = +1 (HH+HL), both down = -1


def _f_geometry(df: pd.DataFrame) -> float:
    """Pattern-geometry proxy: the last bar's signed body conviction. A full-bodied
    bull bar is +1, a full-bodied bear bar -1, a doji ~0 — a compact stand-in for
    Phase 1's detect_patterns until the phases are unified."""
    last = df.iloc[-1]
    rng = float(last["high"] - last["low"])
    if rng <= 0:
        return 0.0
    return _clamp((float(last["close"]) - float(last["open"])) / rng)


def _f_candle(df: pd.DataFrame) -> float:
    """Candle confirmation: where the close sits within the bar's range. Closing
    on the highs is bullish confirmation, on the lows bearish."""
    last = df.iloc[-1]
    rng = float(last["high"] - last["low"])
    if rng <= 0:
        return 0.0
    mid = (float(last["high"]) + float(last["low"])) / 2.0
    return _clamp((float(last["close"]) - mid) / (rng / 2.0))


def _f_volume(df: pd.DataFrame) -> float:
    last = df.iloc[-1]
    vol = float(last.get("volume", 0.0) or 0.0)
    vsma = float(last.get("vol_sma", np.nan))
    if not (np.isfinite(vsma) and vsma > 0):
        return 0.0
    mag = _clamp(vol / vsma - 1.0)                     # 0 at average, 1 at ≥2×
    direction = 1.0 if last["close"] > last["open"] else (
        -1.0 if last["close"] < last["open"] else 0.0)
    return _clamp(abs(mag) * direction)


def _f_sr(df: pd.DataFrame, atr_val: float) -> float:
    """Support below → bullish, resistance above → bearish, scaled by closeness."""
    last = df.iloc[-1]
    close = float(last["close"])
    piv = _pivots(df, PIVOT_K)
    near = SR_NEAR_ATR * atr_val
    if near <= 0:
        return 0.0
    sup = max((p[1] for p in piv if p[2] == -1 and p[1] <= close), default=None)
    res = min((p[1] for p in piv if p[2] == 1 and p[1] >= close), default=None)
    score = 0.0
    if sup is not None and (close - sup) <= near:
        score += 1.0 - (close - sup) / near
    if res is not None and (res - close) <= near:
        score -= 1.0 - (res - close) / near
    return _clamp(score)


def _f_volatility(df: pd.DataFrame, atr_val: float) -> float:
    """Volatility conditions: expansion amplifies the bar's direction; a dead
    tape contributes nothing (its 5% weight simply drops out)."""
    a = df["atr"].dropna()
    if len(a) < ATR_MED_WINDOW:
        return 0.0
    med = float(a.tail(ATR_MED_WINDOW).median())
    if med <= 0 or atr_val < ATR_EXPANSION * med:
        return 0.0
    last = df.iloc[-1]
    return 1.0 if last["close"] > last["open"] else (
        -1.0 if last["close"] < last["open"] else 0.0)


def confidence(df: pd.DataFrame, atr_val: float) -> tuple[float, dict[str, float]]:
    """The weighted confidence in [-1, +1] and the per-factor breakdown.

    Public so Phase 6 (visualization / the confidence dashboard) can render the
    same numbers the trade decision uses.
    """
    factors = {
        "trend": (_f_trend(df), W_TREND),
        "structure": (_f_structure(df), W_STRUCTURE),
        "geometry": (_f_geometry(df), W_GEOMETRY),
        "candle": (_f_candle(df), W_CANDLE),
        "volume": (_f_volume(df), W_VOLUME),
        "sr": (_f_sr(df, atr_val), W_SR),
        "volatility": (_f_volatility(df, atr_val), W_VOLATILITY),
    }
    net = sum(val * w for val, w in factors.values())
    breakdown = {k: round(val, 3) for k, (val, _w) in factors.items()}
    return _clamp(net), breakdown


# --------------------------------------------------------------------------- #
#  The strategy
# --------------------------------------------------------------------------- #
def confidence_signal(df: pd.DataFrame, params: StrategyParams,
                      session_open: Optional[dtime] = None) -> Optional[Signal]:
    """
    Trade the weighted confidence score.

      Direction : sign of the weighted evidence (net > 0 long, net < 0 short).
      Trigger   : |net| × 100 ≥ CONF_MIN — enough weighted evidence has stacked
                  on one side. Recomputed every bar, so it fires as conviction
                  crosses the bar, not on a stale label.
      Stop      : structural (recent swing on the trade's side) when available,
                  else an ATR band; clamped in ATR. Target follows the mode's RR.
    """
    need = max(params.atr_period, params.ema_trend // 4, ATR_MED_WINDOW,
               PIVOT_K * 2 + 2) + 6
    if len(df) < max(need, 12):
        return None
    last = df.iloc[-1]
    if not np.isfinite(last.get("atr", np.nan)):
        return None
    if not _past_entry_window(df, params, session_open):
        return None
    if params.use_atr_gate and not _atr_in_normal_range(df, params):
        return None

    atr_val = float(last["atr"])
    entry = float(last["close"])
    if atr_val <= 0 or entry <= 0:
        return None

    net, breakdown = confidence(df, atr_val)
    conf_pct = abs(net) * 100.0
    if conf_pct < CONF_MIN:
        return None

    top = ", ".join(f"{k} {v:+.2f}" for k, v in
                    sorted(breakdown.items(), key=lambda kv: -abs(kv[1]))[:3])
    buf = SL_BUFFER_ATR * atr_val
    piv = _pivots(df, PIVOT_K)

    # -- Long --------------------------------------------------------------- #
    if net > 0:
        sl_lows = [p[1] for p in piv if p[2] == -1 and p[1] < entry]
        anchor = max(sl_lows) if sl_lows else entry - MAX_SL_ATR * atr_val
        stop = anchor - buf
        stop = min(stop, entry - MIN_SL_ATR * atr_val)
        stop = max(stop, entry - MAX_SL_ATR * atr_val)
        if stop < entry:
            target = entry + params.risk_reward * (entry - stop)
            return Signal("BUY", entry, stop, target,
                          f"Confidence {conf_pct:.0f}% [{top}]")

    # -- Short -------------------------------------------------------------- #
    if net < 0 and params.allow_short:
        sh_highs = [p[1] for p in piv if p[2] == 1 and p[1] > entry]
        anchor = min(sh_highs) if sh_highs else entry + MAX_SL_ATR * atr_val
        stop = anchor + buf
        stop = max(stop, entry + MIN_SL_ATR * atr_val)
        stop = min(stop, entry + MAX_SL_ATR * atr_val)
        if stop > entry:
            target = entry - params.risk_reward * (stop - entry)
            if target > 0:
                return Signal("SELL", entry, stop, target,
                              f"Confidence {conf_pct:.0f}% [{top}]")

    return None


# --------------------------------------------------------------------------- #
#  Params (one per mode) + registration. Risk/RR obey Immutable Rule #1.
# --------------------------------------------------------------------------- #
_SCALPER = StrategyParams(
    mode=Mode.SCALPER, timeframe="1m", risk_per_trade=0.01, risk_reward=1.0,
    atr_period=7, risk_per_trade_cash=2000.0, allow_short=True,
    entry_skip_minutes=15, max_hold_minutes=7, use_limit_entry=True,
    use_atr_gate=True, max_leverage=15.0,
)
_INTRADAY = StrategyParams(
    mode=Mode.INTRADAY, timeframe="15m", risk_per_trade=0.01, risk_reward=2.0,
    atr_period=14, allow_short=True, max_leverage=15.0,
    max_capital_per_trade_pct=0.20,
)
_SWING = StrategyParams(
    mode=Mode.SWING, timeframe="1d", risk_per_trade=0.03, risk_reward=3.0,
    atr_period=14, allow_short=False, max_leverage=1.0,
)

register(StrategyDef(
    key="confidence_engine",
    name="Confidence Engine (Phase 5)",
    params_by_mode={Mode.SCALPER: _SCALPER, Mode.INTRADAY: _INTRADAY,
                    Mode.SWING: _SWING},
    fn=confidence_signal,
    summary="plan.md Phase 5: a weighted-evidence confidence score (trend 20%, "
            "structure 20%, geometry 25%, candle 10%, volume 10%, S/R 10%, "
            "volatility 5%), updated every bar; trades when it clears the bar. "
            "Structural/ATR stop; RR per mode.",
))
