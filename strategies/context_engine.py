"""
strategies/context_engine.py

plan.md PHASE 4 — the context engine (Modules 8, 9, 10), wired up as a tradable
strategy.

plan.md is blunt about why this phase exists: "Without context such as volume or
volatility, many visually correct patterns fail." Phases 1-3 answer WHAT the tape
is doing; this phase answers WHETHER the surroundings support acting on it. It
scores four independent context dimensions and only trades when enough of them
agree:

  Module 8  Trend          — price vs. 200 EMA, 20 EMA slope, and their stacking
  Module 9  Support/Resist.— proximity to a real pivot level + VWAP side
  Module 10 Volatility     — ATR expanding out of compression (a move is starting)
  Module 10 Volume         — this bar's volume beating its 20-period average

    SELF-CONTAINED BY DESIGN
    ------------------------
    Independent of the other phase files (the brief). It reads only indicator
    columns supplied by strategy.enrich (ema_trend, ema_fast, vwap, vol_sma, atr)
    plus a tiny local pivot scan for S/R levels.

    WHY IT'S A CONFLUENCE SCORE, NOT A GATE-CHAIN
    ---------------------------------------------
    A hard AND of four conditions almost never fires and is brittle to one noisy
    input. Instead each dimension contributes evidence toward a side; the trade
    triggers when one side's confluence clears CONFLUENCE_MIN and the other side
    is quiet. That mirrors plan.md's weighted-evidence philosophy while staying
    honest that this is CONTEXT, not a pattern — the trigger is a fresh momentum
    push (a close back above/below the fast EMA), context is the filter.
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
#  Tuning — weights (unitless) and geometry (ATR / bars). No price constants.
# --------------------------------------------------------------------------- #
W_TREND = 2.0           # weight of the trend dimension
W_SR = 1.5              # weight of the support/resistance dimension
W_VOLATILITY = 1.0      # weight of the volatility-expansion dimension
W_VOLUME = 1.5          # weight of the volume-confirmation dimension
CONFLUENCE_MIN = 3.5    # summed weight one side needs before a trade is allowed

PIVOT_K = 3             # pivot strength for locating S/R levels
SR_NEAR_ATR = 0.8       # price is "at" a level within this × ATR
PULLBACK_BARS = 5       # bars scanned for a pullback to the fast EMA
ATR_EXPANSION = 1.15    # ATR now vs. its recent median to count as "expanding"
ATR_MED_WINDOW = 20     # window for the ATR median reference
SL_BUFFER_ATR = 0.3     # stop beyond the anchoring level
MIN_SL_ATR = 0.5
MAX_SL_ATR = 4.0


def _nearest_levels(df: pd.DataFrame, k: int, price: float
                    ) -> tuple[Optional[float], Optional[float]]:
    """(nearest support below price, nearest resistance above price) from recent
    fractal pivots — plan.md Module 9's static S/R, found the simplest honest way.
    """
    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()
    n = len(highs)
    sup: Optional[float] = None
    res: Optional[float] = None
    for i in range(k, n - k):
        if lows[i] == lows[i - k:i + k + 1].min() and lows[i - k:i + k + 1].argmin() == k:
            lv = float(lows[i])
            if lv <= price and (sup is None or lv > sup):
                sup = lv
        if highs[i] == highs[i - k:i + k + 1].max() and highs[i - k:i + k + 1].argmax() == k:
            lv = float(highs[i])
            if lv >= price and (res is None or lv < res):
                res = lv
    return sup, res


def _context_scores(df: pd.DataFrame, params: StrategyParams, atr_val: float
                    ) -> tuple[float, float, float, float, str]:
    """Return (bull_score, bear_score, support, resistance, note).

    Each dimension pushes weight onto exactly one side (or neither). This is the
    heart of the phase — every branch maps directly to a plan.md module.
    """
    last = df.iloc[-1]
    close = float(last["close"])
    bull = bear = 0.0
    notes: list[str] = []

    # -- Module 8: trend ---------------------------------------------------- #
    ema_t = float(last.get("ema_trend", np.nan))
    ema_f = float(last.get("ema_fast", np.nan))
    fast_slope = 0.0
    if len(df) > 5 and "ema_fast" in df.columns:
        fast_slope = float(df["ema_fast"].iloc[-1] - df["ema_fast"].iloc[-5])
    if np.isfinite(ema_t) and np.isfinite(ema_f):
        if close > ema_t and ema_f > ema_t and fast_slope > 0:
            bull += W_TREND; notes.append("uptrend")
        elif close < ema_t and ema_f < ema_t and fast_slope < 0:
            bear += W_TREND; notes.append("downtrend")

    # -- Module 10: volume confirmation ------------------------------------- #
    vol = float(last.get("volume", 0.0) or 0.0)
    vsma = float(last.get("vol_sma", np.nan))
    volume_ok = np.isfinite(vsma) and vsma > 0 and vol > vsma
    if volume_ok:
        # Volume confirms whoever is currently in control of this bar.
        if last["close"] > last["open"]:
            bull += W_VOLUME; notes.append("vol+")
        elif last["close"] < last["open"]:
            bear += W_VOLUME; notes.append("vol+")

    # -- Module 10: volatility expansion ------------------------------------ #
    a = df["atr"].dropna()
    if len(a) >= ATR_MED_WINDOW:
        med = float(a.tail(ATR_MED_WINDOW).median())
        if med > 0 and atr_val >= ATR_EXPANSION * med:
            # Expansion favours the side of the bar's push.
            if last["close"] > last["open"]:
                bull += W_VOLATILITY
            elif last["close"] < last["open"]:
                bear += W_VOLATILITY
            notes.append("ATR↑")

    # -- Module 9: support / resistance proximity + VWAP -------------------- #
    sup, res = _nearest_levels(df, PIVOT_K, close)
    near = SR_NEAR_ATR * atr_val
    vwap_v = float(last.get("vwap", np.nan))
    if sup is not None and abs(close - sup) <= near and last["close"] > last["open"]:
        bull += W_SR; notes.append("at support")
    if res is not None and abs(close - res) <= near and last["close"] < last["open"]:
        bear += W_SR; notes.append("at resistance")
    if np.isfinite(vwap_v):
        if close > vwap_v:
            bull += 0.5
        elif close < vwap_v:
            bear += 0.5

    return bull, bear, (sup if sup is not None else np.nan), \
        (res if res is not None else np.nan), ", ".join(dict.fromkeys(notes))


# --------------------------------------------------------------------------- #
#  The strategy
# --------------------------------------------------------------------------- #
def context_signal(df: pd.DataFrame, params: StrategyParams,
                   session_open: Optional[dtime] = None) -> Optional[Signal]:
    """
    Enter when context confluence backs a pullback-and-resume push.

      Trigger : price PULLED BACK to the 20 EMA within the last PULLBACK_BARS and
                this bar CLOSES back above it (long) / below it (short) with a
                bar in that direction. A strict same-bar EMA cross almost never
                fires in a real trend (price rides above the fast EMA), so the
                trigger is a pullback resumption, like the scalper's VWAP dip.
      Filter  : trend + S/R + volatility + volume confluence for that side must
                clear CONFLUENCE_MIN, and the opposite side must be weaker.
      Stop    : just beyond the anchoring level (nearest support/resistance, else
                an ATR band), clamped in ATR. Target follows the mode's RR.
    """
    need = max(params.atr_period, params.ema_fast, ATR_MED_WINDOW,
               PIVOT_K * 2 + 2, PULLBACK_BARS) + 4
    if len(df) < need:
        return None
    last = df.iloc[-1]
    if not (np.isfinite(last.get("atr", np.nan))
            and np.isfinite(last.get("ema_fast", np.nan))):
        return None
    if not _past_entry_window(df, params, session_open):
        return None
    if params.use_atr_gate and not _atr_in_normal_range(df, params):
        return None

    atr_val = float(last["atr"])
    entry = float(last["close"])
    if atr_val <= 0 or entry <= 0:
        return None

    ema_f = float(last["ema_fast"])
    # The pullback must have happened BEFORE the trigger bar, so exclude it.
    pull = df.iloc[-(PULLBACK_BARS + 1):-1]
    dipped_to_ema = bool((pull["low"] <= pull["ema_fast"]).any())
    rose_to_ema = bool((pull["high"] >= pull["ema_fast"]).any())
    push_up = (dipped_to_ema and last["close"] > ema_f
               and last["close"] > last["open"])
    push_dn = (rose_to_ema and last["close"] < ema_f
               and last["close"] < last["open"])
    if not (push_up or push_dn):
        return None

    bull, bear, sup, res, note = _context_scores(df, params, atr_val)
    buf = SL_BUFFER_ATR * atr_val

    # -- Long --------------------------------------------------------------- #
    if push_up and bull >= CONFLUENCE_MIN and bull > bear:
        anchor = sup if np.isfinite(sup) else entry - MAX_SL_ATR * atr_val
        stop = anchor - buf
        stop = min(stop, entry - MIN_SL_ATR * atr_val)
        stop = max(stop, entry - MAX_SL_ATR * atr_val)
        if stop < entry:
            target = entry + params.risk_reward * (entry - stop)
            return Signal("BUY", entry, stop, target,
                          f"Context confluence {bull:.1f} [{note}]")

    # -- Short -------------------------------------------------------------- #
    if params.allow_short and push_dn and bear >= CONFLUENCE_MIN and bear > bull:
        anchor = res if np.isfinite(res) else entry + MAX_SL_ATR * atr_val
        stop = anchor + buf
        stop = max(stop, entry + MIN_SL_ATR * atr_val)
        stop = min(stop, entry + MAX_SL_ATR * atr_val)
        if stop > entry:
            target = entry - params.risk_reward * (stop - entry)
            if target > 0:
                return Signal("SELL", entry, stop, target,
                              f"Context confluence {bear:.1f} [{note}]")

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
    key="context_engine",
    name="Context Confluence (Phase 4)",
    params_by_mode={Mode.SCALPER: _SCALPER, Mode.INTRADAY: _INTRADAY,
                    Mode.SWING: _SWING},
    fn=context_signal,
    summary="plan.md Phase 4: scores trend, support/resistance, volatility "
            "expansion and volume, and trades a fresh 20-EMA push only when the "
            "context confluence agrees. Stop anchored to S/R; RR per mode.",
))
