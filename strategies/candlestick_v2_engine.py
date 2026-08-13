"""
strategies/candlestick_v2_engine.py

"Candlestick Engine (Phase 1 New)" — items 1-3 of the Phase-1 improvement plan,
as a SEPARATE strategy so the original can keep running untouched.

    WHY A NEW FILE RATHER THAN AN EDIT
    ----------------------------------
    candlestick_engine.py is live. Changing its scoring or its entry gate would
    silently change every open run, every saved preset pointing at it, and every
    backtest number already recorded against it — with no way to compare old
    against new. This module imports its pattern DETECTION (a pure function of
    an OHLCV frame, no state) and rebuilds only the decision layer on top, so the
    two strategies see byte-identical patterns and differ exactly and only by the
    three changes below. That is what makes an A/B meaningful.

    WHAT CHANGED, AND WHY
    ---------------------
    1. THE EVIDENCE BAR NOW MATCHES ITS INTENT (plan item 1).
       config.CANDLE_INTRADAY_PARAMS carries the comment "Raised from 3.0 ... to
       6.0 on request — now needs roughly two agreeing patterns" while the code
       says 3.0. At 3.0 ONE high-strength pattern (weight 3.0) opens an intraday
       trade — the low-conviction bar the comment says was turned off. This
       strategy runs intraday at 6.0, so an entry really does need two agreeing
       patterns. Scalper (4.0) and Swing (3.0) are left where they are: the
       drift was intraday-only.

    2. REGIME-CONDITIONAL DIRECTION, NOT A DIRECTIONAL PREFERENCE (plan item 2).
       Trade WITH the trend at the normal bar; AGAINST it only at a raised bar.
       Regime is the close versus the trend EMA (params.ema_trend, already
       supplied by strategy.enrich). A counter-trend entry must clear
       cs_min_score x COUNTER_TREND_MULT.

       This is deliberately NOT "prefer shorts". A permanent short tilt is a bet
       that the market stays weak forever; this instead raises the bar on
       whichever side is fighting the trend, which fixes the long/short
       asymmetry in both directions and keeps working when the regime flips.

    3. VOLUME CONFIRMS THE PATTERN (plan item 3).
       The original scores an engulfing on dead volume identically to one on a
       volume spike. Here each pattern's weight is scaled by the completing
       bar's volume against its own 20-bar average, clamped so volume can
       support or temper evidence but never manufacture or erase it.

    NOT CHANGED (deliberately, so the A/B stays clean): pattern detection, the
    stop (beyond the pattern extreme, clamped in ATR), RR, risk per trade, the
    entry-window and ATR gates. Items 4-6 of the plan (confirmation candle,
    per-pattern expectancy, the real market-structure filter) are NOT here.
"""
from __future__ import annotations

from dataclasses import replace
from datetime import time as dtime
from typing import Optional

import numpy as np
import pandas as pd

from config import (CANDLE_INTRADAY_PARAMS, CANDLE_SCALPER_PARAMS,
                    CANDLE_SWING_PARAMS, Mode, StrategyParams)
from strategy import (Signal, StrategyDef, _atr_in_normal_range,
                      _past_entry_window, register)
# Detection is imported, never copied: one definition of a Bullish Engulfing for
# both strategies, so a fix to the geometry benefits both and the A/B can never
# be contaminated by the two drifting apart.
from strategies.candlestick_engine import PatternHit, detect_patterns

# --------------------------------------------------------------------------- #
#  Tuning — module-level on purpose. Putting these in config.StrategyParams
#  would edit a dataclass every other strategy shares; keeping them here means
#  this experiment cannot reach anything else.
# --------------------------------------------------------------------------- #

#: Evidence multiplier for an entry that fights the trend EMA. 1.5 means a
#: counter-trend intraday trade needs 9.0 against the with-trend 6.0 — roughly
#: three agreeing patterns instead of two.
COUNTER_TREND_MULT = 1.5

#: How hard volume moves a pattern's weight. weight x (1 + SENS x (ratio - 1)),
#: where ratio is the completing bar's volume over its 20-bar average. At 0.5,
#: a 2x volume spike adds 50% to the evidence.
VOL_SENSITIVITY = 0.5

#: Hard bounds on that multiplier. Volume TEMPERS evidence, it never creates or
#: destroys it: without the floor a dead-volume bar could zero out a genuine
#: three-candle reversal, and without the ceiling one volume spike could push a
#: single weak pattern over a bar meant to need two.
VOL_MULT_MIN, VOL_MULT_MAX = 0.6, 1.5


def volume_multiplier(df: pd.DataFrame) -> float:
    """Evidence multiplier from the COMPLETING bar's volume vs its own average.

    Returns exactly 1.0 (no effect) whenever volume is unusable — missing
    column, zero/NaN average, or a feed that reports no volume at all. Several
    Indian feeds report 0 volume on index instruments, and a strategy that
    silently stopped trading them would be a far worse bug than one that simply
    declines to use volume as evidence.
    """
    if "volume" not in df.columns or "vol_sma" not in df.columns:
        return 1.0
    try:
        vol = float(df["volume"].iloc[-1])
        avg = float(df["vol_sma"].iloc[-1])
    except (TypeError, ValueError, IndexError):
        return 1.0
    if not np.isfinite(vol) or not np.isfinite(avg) or avg <= 0 or vol <= 0:
        return 1.0
    mult = 1.0 + VOL_SENSITIVITY * ((vol / avg) - 1.0)
    return max(VOL_MULT_MIN, min(VOL_MULT_MAX, mult))


def score_patterns_volume(hits: list[PatternHit],
                          vol_mult: float) -> tuple[float, float]:
    """(bullish, bearish) evidence, volume-scaled.

    Same shape as candlestick_engine.score_patterns — direction-0 patterns (the
    Dojis) still contribute to neither side, because hesitation is information
    about indecision, not about direction.
    """
    bull = sum(h.weight for h in hits if h.direction > 0) * vol_mult
    bear = sum(h.weight for h in hits if h.direction < 0) * vol_mult
    return bull, bear


def trend_regime(df: pd.DataFrame) -> int:
    """+1 above the trend EMA, -1 below, 0 when it cannot be judged.

    0 is the honest answer while the EMA is still warming up, and it makes the
    gate below fall back to the plain evidence bar rather than blocking every
    trade or waving every trade through.
    """
    if "ema_trend" not in df.columns:
        return 0
    try:
        ema = float(df["ema_trend"].iloc[-1])
        close = float(df["close"].iloc[-1])
    except (TypeError, ValueError, IndexError):
        return 0
    if not np.isfinite(ema) or ema <= 0:
        return 0
    if close > ema:
        return 1
    if close < ema:
        return -1
    return 0


def required_score(params: StrategyParams, regime: int, want: int) -> float:
    """The evidence this side must clear right now.

    `want` is +1 for a long, -1 for a short. With the trend (or in an
    unjudgeable regime) that is the plain bar; against it, the raised one.
    """
    if regime == 0 or regime == want:
        return params.cs_min_score
    return params.cs_min_score * COUNTER_TREND_MULT


def candlestick_v2_signal(df: pd.DataFrame, params: StrategyParams,
                          session_open: Optional[dtime] = None
                          ) -> Optional[Signal]:
    """Phase 1 New. Identical to candlestick_signal except for the three
    changes in the module docstring; the stop and target maths below are a
    deliberate mirror of the original so an A/B compares decisions, not
    arithmetic."""
    # IDENTICAL to candlestick_signal's guard, and it must stay that way.
    #
    # The regime gate reads params.ema_trend (200), so it is tempting to demand
    # ~200 bars of window here. That would be wrong twice over. The EMA is a
    # PRE-COMPUTED COLUMN: strategy.enrich() runs once over the whole frame
    # (backtester.py:373 says so explicitly, and run_strategy re-enriches the
    # full lookback live), so df["ema_trend"].iloc[-1] is the true 200-EMA
    # regardless of how many rows this particular window carries.
    #
    # And demanding more silently kills the strategy: the backtester hands
    # Intraday a window of only `macd_slow + 2 + 6` = 34 bars, so a `need` of
    # 56 made this return None on EVERY bar and produce a blank backtest.
    # Matching v1's guard also keeps the A/B honest — both strategies get the
    # same tradable bars. A warming-up EMA is handled where it belongs, in
    # trend_regime(), which returns 0 (neutral) and falls back to the plain
    # evidence bar rather than blocking the trade.
    need = max(params.atr_period, params.cs_trend_lookback) + 6
    if len(df) < need:
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

    hits = detect_patterns(df, params)
    if not hits:
        return None

    vol_mult = volume_multiplier(df)                       # change 3
    bull, bear = score_patterns_volume(hits, vol_mult)

    # Both sides firing at once is real ambiguity, not a 50/50 bet. Sit it out.
    if bull > 0 and bear > 0:
        return None

    regime = trend_regime(df)                              # change 2
    need_long = required_score(params, regime, 1)
    need_short = required_score(params, regime, -1)

    if bull >= need_long:
        side, want, evidence, bar = "BUY", 1, bull, need_long
    elif bear >= need_short and params.allow_short:
        side, want, evidence, bar = "SELL", -1, bear, need_short
    else:
        return None

    chosen = [h for h in hits if h.direction == want]
    win = df.iloc[-max(h.n for h in chosen):]
    named = ", ".join(dict.fromkeys(h.name for h in chosen))
    buf = params.cs_sl_buffer_atr * atr_val

    if side == "BUY":
        stop = float(win["low"].min()) - buf
        stop = min(stop, entry - params.cs_min_sl_atr * atr_val)
        stop = max(stop, entry - params.cs_max_sl_atr * atr_val)
        if stop >= entry:
            return None
        target = entry + params.risk_reward * (entry - stop)
    else:
        stop = float(win["high"].max()) + buf
        stop = max(stop, entry + params.cs_min_sl_atr * atr_val)
        stop = min(stop, entry + params.cs_max_sl_atr * atr_val)
        if stop <= entry:
            return None
        target = entry - params.risk_reward * (stop - entry)
        if target <= 0:
            return None

    # The reason names the regime and the bar that was actually applied, so the
    # trade log shows WHY this one cleared — without it a counter-trend entry is
    # indistinguishable from a with-trend one after the fact.
    regime_txt = {1: "up", -1: "down", 0: "neutral"}[regime]
    against = regime != 0 and regime != want
    return Signal(
        side, entry, stop, target,
        f"{named} (evidence {evidence:.1f}/{bar:.1f}, "
        f"{regime_txt}-trend{' COUNTER' if against else ''}, "
        f"vol x{vol_mult:.2f})")


# --------------------------------------------------------------------------- #
#  Registration. `replace` inherits every risk/RR/gate value from the original
#  params, so this strategy can never drift on anything except the one field it
#  deliberately changes — and Immutable Rule #1's caps come along untouched.
# --------------------------------------------------------------------------- #
V2_SCALPER_PARAMS = CANDLE_SCALPER_PARAMS
V2_INTRADAY_PARAMS = replace(CANDLE_INTRADAY_PARAMS, cs_min_score=6.0)
V2_SWING_PARAMS = CANDLE_SWING_PARAMS

register(StrategyDef(
    key="candlestick_engine_v2",
    name="Candlestick Engine (Phase 1 New)",
    params_by_mode={
        Mode.SCALPER: V2_SCALPER_PARAMS,
        Mode.INTRADAY: V2_INTRADAY_PARAMS,
        Mode.SWING: V2_SWING_PARAMS,
    },
    fn=candlestick_v2_signal,
    summary="Phase 1 + improvements 1-3: intraday evidence bar restored to 6.0 "
            "(two agreeing patterns), regime gate (with-trend at the normal "
            "bar, counter-trend only at 1.5x), and volume-scaled pattern "
            "weights. Same patterns and same stop as Phase 1 — run both to "
            "compare.",
    uses_min_score=True,
))
