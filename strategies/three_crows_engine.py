"""
strategies/three_crows_engine.py

"Identical Three Crows (Short)" — newstrategy.md, as a SEPARATE strategy.

    WHY A NEW FILE AND NOT AN EDIT
    ------------------------------
    Every rule below already exists somewhere in this bot: the pattern is in
    candlestick_engine.py, the two exit rules are exit_manager.py's `fixed` and
    the 2b/2c family, the costs are cost_model.py. What did NOT exist is the
    COMBINATION — one pattern, traded short-only, with no scoring, no trend
    filter and no volatility gate in front of it. Bending the candlestick
    engine into that shape would have changed a live strategy, every preset
    pointing at it and every backtest number already recorded against it.
    So this module reaches nothing shared: it imports Signal/register/two
    guards from strategy.py, reads StrategyParams, and writes its own geometry.
    Deleting this file removes the strategy and nothing else.

    THE SETUP (newstrategy.md section 4)
    ------------------------------------
    Three consecutive bearish candles, each opening within OPEN_TOLERANCE of
    the previous close (no bounce was even attempted), each closing below the
    last. Enter SHORT at the third candle's close. This is "Identical Three
    Crows" — the strict cousin of Three Black Crows, which only wants the opens
    inside the previous BODY.

    Short-only by construction. A bearish continuation pattern has no long
    reading, so `allow_short` is not a preference here, it is the strategy.

    THE STOP
    --------
    entry + atr_sl_mult x ATR(atr_period), i.e. the standard volatility stop
    this bot uses everywhere, with the script's 1.5 x ATR(14) as the default.
    ATR IS COMPUTED HERE, as a simple mean of the last N true ranges, because
    that is what newstrategy.md measures and it produces a materially different
    stop from the Wilder ATR in strategy.enrich() (Wilder's is an EMA with
    alpha = 1/N, so it reacts slower and, after a volatility spike, sits
    wider). Nothing else in the bot reads this number: the frame's own `atr`
    column still feeds the ATR gate and the chandelier trail, and this strategy
    switches both of those off — see THE EXITS.

    THE EXITS — both of the script's rules, and neither is new code
    --------------------------------------------------------------
    newstrategy.md offers two, and they are ALREADY exit_manager.py styles.
    Pick between them with the mode's Exit style control (ModeConfig.exit_style
    live, `exit_style` per backtest run):

      "strategy" (the default, = the script's Rule 1 "Trail SL TP")
          book 50% at TP1 = 1R, move the runner's stop to entry - 10 bps
          (the script's `entry * (1 - 0.001)`), promote that stop to TP1 once
          price reaches the MIDPOINT of TP1..TP2, and hold for TP2 = 2R.
          That is Approach 2c, configured below on the params themselves
          rather than through config.EXIT_STYLES["partial_ladder"] — the
          shipped ladder books 30% at a 20 bps buffer on a 1.5xATR trigger,
          which is a different (and deliberately parked) experiment. Setting
          the fields here leaves that style untouched for everyone else.

      "fixed" (= the script's Rule 2 "Fixed RR")
          all-in / all-out at risk_reward x R. Choose the ratio with the
          mode's RR control (config.RR_CHOICES: 1, 1.5, 2, 2.5, 3).

    A ONE-UNIT POSITION cannot be halved. exit_manager leaves it whole, moves
    the stop to break-even + buffer at TP1 and runs it to TP2 — the same
    `is_single_lot` branch the script writes by hand.

    LEAVE THE RR CONTROL ALONE WHILE THE MANAGED EXIT IS SELECTED. TP1 is
    `risk_reward x R`, and both the live engine (engine._rr_for) and the
    backtester will honour a per-mode or per-symbol RR override — so setting
    the Intraday RR to 1:2 moves TP1 out to 2R while `runner_rr_mult` stays at
    2R, collapsing TP1 onto TP2. Nothing breaks (the partial books at 2R and
    the runner is promoted and squared off there, i.e. it degrades to a plain
    all-out at 2R), but the ladder the style exists for is gone. The RR
    control is for the "fixed" style; here, leave the override unset — 0 means
    "inherit the strategy's own", which is the 1:1 the script specifies.

    WHAT IS DELIBERATELY *NOT* HERE
    -------------------------------
    * No pattern SCORE. One pattern is the whole signal, so `uses_min_score`
      is False and cs_min_score is never read. The admin panel will say so.
    * No trend filter and no ATR gate (`use_atr_gate=False`). The script has
      neither, and adding one would be testing a different strategy.
    * No stop-distance band (`min_stop_pct`/`max_stop_pct` left at 0). The
      script does not clamp, and on Candlestick Phase 1 the band was measured
      to move results by percentage points — it belongs in a sweep, not in the
      first honest reading of this setup.
    * No SWING mode. This strategy only sells, and an overnight NSE cash short
      is a delivery you cannot make (the same reason CANDLE_SWING_PARAMS is
      long-only). Registering it there would offer a mode that can only ever
      return zero trades.
    * Gap-open fills. newstrategy.md checks whether the bar OPENED through the
      stop and fills at the open; exit_manager.step_bar fills at the stop
      level. That is a property of the shared simulator, not of this strategy,
      so it is left alone here — the backtest is optimistic by the size of the
      overnight gap on any position that survives to the next session.
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
#  Tuning — module-level on purpose. These are the two numbers newstrategy.md
#  hardcodes. Putting them in config.StrategyParams would edit a frozen
#  dataclass every other strategy shares, so this experiment cannot reach
#  anything outside this file.
# --------------------------------------------------------------------------- #

#: How close the next candle must open to the previous close before the pair
#: counts as "identical" — a FRACTION OF PRICE, which is the script's unit
#: (`(open2 - close1).abs() / close1 <= 0.002`).
#:
#: Note this differs from candlestick_engine's own Identical Three Crows,
#: which measures the same gap against the CANDLE RANGE. Neither is wrong;
#: they are different definitions, and this one is the one being tested.
#: 0.2% of price is roughly a third of a typical 15-minute Nifty-100 candle,
#: so it is the looser of the two on quiet tape and the tighter on violent
#: tape — which is the behaviour the script's author chose.
OPEN_TOLERANCE = 0.002

#: Break-even buffer for the runner after the partial, in basis points of
#: entry. 10 bps is the script's `entry * (1 - 0.001)` exactly.
#:
#: IT IS THINNER THAN A ROUND TRIP. config.BREAKEVEN_BUFFER_BPS is 20 for a
#: reason — measured against cost_model, a half-size intraday equity runner's
#: own round trip costs ~19 bps. At 10 the runner is stopped out fractionally
#: NEGATIVE after costs rather than free. Kept at the script's number so the
#: first run measures the script; raise it to 20.0 to measure the runner the
#: rest of this bot would give you.
BREAKEVEN_BUFFER_BPS = 10.0


# --------------------------------------------------------------------------- #
#  ATR — simple mean of true range, per newstrategy.md.
# --------------------------------------------------------------------------- #
def sma_atr(df: pd.DataFrame, period: int) -> float:
    """Mean of the last `period` true ranges. Returns NaN if there is not
    enough history for a full window.

    Deliberately NOT strategy.atr(), which is Wilder's EMA. Every true range in
    this average is computed against a real previous close (the script's first
    row silently falls back to high-low), so the number is the script's after
    its own warm-up and better-defined before it.
    """
    if len(df) < period + 1:
        return float("nan")
    tail = df.iloc[-(period + 1):]
    high, low, close = tail["high"], tail["low"], tail["close"]
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(),
                    (low - prev_close).abs()], axis=1).max(axis=1)
    return float(tr.iloc[1:].mean())


# --------------------------------------------------------------------------- #
#  The pattern.
# --------------------------------------------------------------------------- #
def is_identical_three_crows(c1, c2, c3,
                             tolerance: float = OPEN_TOLERANCE) -> bool:
    """Three bearish candles, each opening at the last one's close and closing
    lower. `c1` is the OLDEST.

    Every condition is the script's, in its order:
      1. all three close below their own open,
      2. c2 opens within `tolerance` of c1's close,
      3. c3 opens within `tolerance` of c2's close,
      4. closes step strictly down.
    """
    o1, cl1 = float(c1["open"]), float(c1["close"])
    o2, cl2 = float(c2["open"]), float(c2["close"])
    o3, cl3 = float(c3["open"]), float(c3["close"])
    if cl1 <= 0 or cl2 <= 0:
        return False                       # the tolerance is a % of these
    if not (cl1 < o1 and cl2 < o2 and cl3 < o3):
        return False
    if abs(o2 - cl1) / cl1 > tolerance:
        return False
    if abs(o3 - cl2) / cl2 > tolerance:
        return False
    return cl3 < cl2 < cl1


# --------------------------------------------------------------------------- #
#  Signal.
# --------------------------------------------------------------------------- #
def three_crows_signal(df: pd.DataFrame, params: StrategyParams,
                       session_open: Optional[dtime] = None
                       ) -> Optional[Signal]:
    """One short signal on the bar that completes the pattern, or None."""
    # +3 for the pattern itself, on top of the ATR window.
    if len(df) < params.atr_period + 4:
        return None
    # SHORT-ONLY, and it must be checked rather than assumed: an admin can
    # reach `allow_short` through the params, and a False there has to mean
    # "do not sell", not "sell anyway".
    if not params.allow_short:
        return None
    if not _past_entry_window(df, params, session_open):
        return None
    # Off by default for this strategy (the script has no gate); honoured
    # anyway so switching it on in a sweep does what it says.
    if params.use_atr_gate and not _atr_in_normal_range(df, params):
        return None

    c1, c2, c3 = df.iloc[-3], df.iloc[-2], df.iloc[-1]
    if not is_identical_three_crows(c1, c2, c3, OPEN_TOLERANCE):
        return None

    atr_val = sma_atr(df, params.atr_period)
    if not np.isfinite(atr_val) or atr_val <= 0:
        return None

    entry = float(c3["close"])
    sl_distance = atr_val * params.atr_sl_mult
    if entry <= 0 or sl_distance <= 0:
        return None
    stop = entry + sl_distance
    # TP1 for the managed exit, or the whole target under the "fixed" style.
    # Read off params.risk_reward like every other strategy here, so the RR
    # control (and any per-symbol override) still means what it means
    # everywhere else. Left at 1.0 below, which is the script's TP1.
    target = entry - params.risk_reward * sl_distance
    if target <= 0:
        return None

    return Signal(
        "SELL", entry, stop, target,
        f"Identical Three Crows (opens within {OPEN_TOLERANCE:.2%}, "
        f"SL {params.atr_sl_mult:g}xATR{params.atr_period} = {sl_distance:.2f})")


# --------------------------------------------------------------------------- #
#  Registration.
#
#  The exit block below IS the script's "Trail SL TP" rule. It is set on the
#  params rather than selected through config.EXIT_STYLES so that:
#    * the default run of this strategy is the script, with nothing to
#      configure, and
#    * config's own `partial_ladder` — 30% at 20 bps on a 1.5xATR trigger,
#      measured and deliberately parked — stays exactly as it is for every
#      other strategy that selects it.
#  Choosing "fixed" for the mode still overrides all of it, which is how the
#  script's second exit rule is reached.
# --------------------------------------------------------------------------- #
_EXIT_RULE_TRAIL_SL_TP = dict(
    partial_exit_fraction=0.5,          # book half at TP1
    breakeven_buffer_ticks=0.0,
    breakeven_buffer_bps=BREAKEVEN_BUFFER_BPS,   # runner stop = entry -10 bps
    runner_rr_mult=2.0,                 # TP2 = 2R
    lock_first_target=True,             # ...and the runner's stop may reach TP1
    lock_trigger_frac=0.5,              # promoted at the TP1..TP2 midpoint
    lock_trigger_atr_mult=0.0,          # the script's trigger is the midpoint,
                                        # not a volatility distance
    trail_remainder=False,              # no chandelier anywhere in the script
    trail_from_entry=False,
)

_COMMON = dict(
    risk_per_trade=0.01,        # 1% — inside the 2% intraday ceiling (Rule #1)
    risk_reward=1.0,            # TP1 = 1R, per the script
    max_leverage=15.0,          # the segment's own cap still binds (equity 1x)
    max_capital_per_trade_pct=0.0,
    atr_period=14,
    atr_sl_mult=1.5,
    allow_short=True,           # the strategy only sells
    use_atr_gate=False,         # the script has no volatility gate
    #: No new entries after 15:00 IST (15:09 flat-out minus 9), which is the
    #: script's `if current_time.hour >= 15: continue`. Derived from the
    #: square-off, so shortening the day moves it too.
    entry_cutoff_before_close=9,
    **_EXIT_RULE_TRAIL_SL_TP,
)

THREE_CROWS_INTRADAY_PARAMS = StrategyParams(
    mode=Mode.INTRADAY, timeframe="15m", **_COMMON)
THREE_CROWS_SCALPER_PARAMS = StrategyParams(
    mode=Mode.SCALPER, timeframe="1m", **_COMMON)

register(StrategyDef(
    key="identical_three_crows",
    name="Identical Three Crows (Short)",
    params_by_mode={
        Mode.INTRADAY: THREE_CROWS_INTRADAY_PARAMS,
        Mode.SCALPER: THREE_CROWS_SCALPER_PARAMS,
    },
    fn=three_crows_signal,
    summary="Short-only. Three bearish candles, each opening within 0.2% of "
            "the last close and closing lower. Stop 1.5xATR(14) above entry "
            "(simple-mean ATR). Default exit books 50% at 1R, floors the "
            "runner at entry -10bps, promotes it to 1R at the 1R..2R midpoint "
            "and targets 2R; choose the 'fixed' exit style for plain all-in / "
            "all-out at the mode's RR. No score, no trend filter, no ATR gate.",
    uses_min_score=False,
))
