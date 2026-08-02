"""
strategies/market_structure_engine.py

plan.md PHASE 2 — the swing & market-structure engine (Module 4), wired up as a
tradable strategy.

Covers plan.md Module 4: swing highs/lows, Higher-High / Higher-Low /
Lower-High / Lower-Low labelling, trend vs. range, Break of Structure (BOS) and
Change of Character (CHOCH / MSS). plan.md itself says why this matters:
"Without market structure, candlestick signals generate many false positives."
Phase 1 recognises shapes; THIS phase reads the skeleton the shapes hang on.

    SELF-CONTAINED BY DESIGN
    ------------------------
    The brief is "each phase in its own file, independent from each other for
    now", so this module imports nothing from the other phase files. Its pivot
    detector (_pivots) is deliberately re-implemented here rather than shared —
    Phase 3 keeps its own copy too. When the phases are later unified, these
    become one helper; until then independence is the requirement.

    WHY THIS WORKS ON EVERY TIMEFRAME (architectural rule 4)
    --------------------------------------------------------
    Structure is defined purely by the RELATIVE ordering of swing pivots — "this
    high is above the last high" — which carries no rupee or percentage constant.
    The only absolute yardstick is the stop, which is anchored to a real swing
    level and clamped in ATR units. So the same logic runs on 1m, 15m and daily;
    only risk, RR and the pivot strength differ, and those come from params.

    WHAT THIS PHASE IS
    ------------------
    A structure-break entry:
      * find confirmed swing pivots (fractals),
      * label the trend from the last two highs and last two lows,
      * enter when price BREAKS the most recent opposing swing level —
        a BOS in the trend direction (continuation) or a CHOCH against a
        prior trend/range (early reversal),
      * stop beyond the protected swing on the other side.
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
#  Tuning — pivot geometry (bars) and stop placement (ATR). No price constants.
# --------------------------------------------------------------------------- #
PIVOT_K = 2             # a swing pivot must be the extreme of ±K neighbouring bars
MAX_PIVOTS = 12         # only the most recent pivots describe the live structure
BREAK_LOOKBACK = 60     # the level being broken must be no older than this (bars)
SL_BUFFER_ATR = 0.25    # stop sits this far beyond the protected swing
MIN_SL_ATR = 0.5        # ...but never closer to entry than this
MAX_SL_ATR = 4.0        # ...and never further (else position_size shrinks to ~0)


# --------------------------------------------------------------------------- #
#  Swing pivots (fractals)
# --------------------------------------------------------------------------- #
def _pivots(df: pd.DataFrame, k: int) -> list[tuple[int, float, int]]:
    """Confirmed fractal pivots as (bar_index, price, kind) with kind +1 = swing
    high, -1 = swing low.

    A bar is a swing high if its high is the strict maximum of the 2k+1 window
    centred on it, and symmetrically for a swing low. Requiring k bars on the
    RIGHT is what makes the pivot 'confirmed' — it cannot be known until k bars
    later, so the newest pivot is always ≥ k bars old and never repaints. That is
    exactly the property that keeps the entry honest: we break levels the market
    has already finished printing, not provisional ones.
    """
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


def _last_two(pivots: list[tuple[int, float, int]], kind: int
              ) -> Optional[tuple[float, float]]:
    """(previous, latest) price of the last two pivots of one kind, or None."""
    same = [p for p in pivots if p[2] == kind]
    if len(same) < 2:
        return None
    return same[-2][1], same[-1][1]


def _trend(pivots: list[tuple[int, float, int]]) -> int:
    """+1 uptrend (HH & HL), -1 downtrend (LH & LL), 0 range/undecided.

    plan.md Module 4 in one function: a market is trending up only when it is
    making both higher highs AND higher lows; anything else is a range, where a
    break is far more likely to be a CHOCH (reversal) than a BOS (continuation).
    """
    hh = _last_two(pivots, 1)
    ll = _last_two(pivots, -1)
    if hh is None or ll is None:
        return 0
    highs_up, lows_up = hh[1] > hh[0], ll[1] > ll[0]
    highs_dn, lows_dn = hh[1] < hh[0], ll[1] < ll[0]
    if highs_up and lows_up:
        return 1
    if highs_dn and lows_dn:
        return -1
    return 0


def _recent(pivots: list[tuple[int, float, int]], kind: int, n_bars: int,
            max_age: int) -> Optional[tuple[int, float]]:
    """The newest pivot of `kind` that is not older than max_age bars: (idx, px)."""
    for idx, px, knd in reversed(pivots):
        if knd == kind:
            return (idx, px) if (n_bars - 1 - idx) <= max_age else None
    return None


# --------------------------------------------------------------------------- #
#  The strategy
# --------------------------------------------------------------------------- #
def market_structure_signal(df: pd.DataFrame, params: StrategyParams,
                            session_open: Optional[dtime] = None
                            ) -> Optional[Signal]:
    """
    Enter on a confirmed break of market structure.

      Long : price CLOSES above the most recent confirmed swing HIGH on this bar
             (and did not on the previous bar — a fresh break). If the prior
             trend was down/range it is a CHOCH (reversal); if up, a BOS
             (continuation). Stop below the most recent swing LOW.
      Short: mirror — close below the most recent swing LOW, stop above the last
             swing HIGH. (Long-only in Swing: delivery can't be shorted.)

    The stop is STRUCTURAL — the swing the break is supposed to protect — clamped
    into [MIN_SL_ATR, MAX_SL_ATR] × ATR. Target follows the mode's RR.
    """
    need = max(params.atr_period, PIVOT_K * 2 + 2) + BREAK_LOOKBACK // 2
    if len(df) < max(need, 2 * PIVOT_K + 4):
        return None
    last = df.iloc[-1]
    prev = df.iloc[-2]
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

    pivots = _pivots(df, PIVOT_K)
    if len(pivots) < 4:
        return None
    pivots = pivots[-MAX_PIVOTS:]
    n = len(df)
    trend = _trend(pivots)
    buf = SL_BUFFER_ATR * atr_val

    # -- Long: break of the most recent swing high -------------------------- #
    sh = _recent(pivots, 1, n, BREAK_LOOKBACK)
    sl = _recent(pivots, -1, n, BREAK_LOOKBACK)
    if sh is not None and sl is not None:
        level = sh[1]
        fresh_break = prev["close"] <= level and last["close"] > level
        if fresh_break and last["close"] > last["open"]:
            kind = "BOS" if trend > 0 else "CHOCH"
            stop = sl[1] - buf                      # beyond the protected swing low
            stop = min(stop, entry - MIN_SL_ATR * atr_val)
            stop = max(stop, entry - MAX_SL_ATR * atr_val)
            if stop < entry:
                target = entry + params.risk_reward * (entry - stop)
                return Signal("BUY", entry, stop, target,
                              f"Bullish {kind}: broke swing high {level:.2f} "
                              f"(trend {trend:+d})")

    # -- Short: break of the most recent swing low -------------------------- #
    if params.allow_short and sh is not None and sl is not None:
        level = sl[1]
        fresh_break = prev["close"] >= level and last["close"] < level
        if fresh_break and last["close"] < last["open"]:
            kind = "BOS" if trend < 0 else "CHOCH"
            stop = sh[1] + buf                      # beyond the protected swing high
            stop = max(stop, entry + MIN_SL_ATR * atr_val)
            stop = min(stop, entry + MAX_SL_ATR * atr_val)
            if stop > entry:
                target = entry - params.risk_reward * (stop - entry)
                if target > 0:
                    return Signal("SELL", entry, stop, target,
                                  f"Bearish {kind}: broke swing low {level:.2f} "
                                  f"(trend {trend:+d})")

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
    atr_period=14, allow_short=False, max_leverage=1.0,   # delivery = long-only
)

register(StrategyDef(
    key="market_structure_engine",
    name="Market Structure (Phase 2)",
    params_by_mode={Mode.SCALPER: _SCALPER, Mode.INTRADAY: _INTRADAY,
                    Mode.SWING: _SWING},
    fn=market_structure_signal,
    summary="plan.md Phase 2: HH/HL/LH/LL swing structure, trades a Break of "
            "Structure (BOS) or Change of Character (CHOCH). Structural stop "
            "beyond the protected swing; RR follows the mode. Runs on 1m/15m/1d.",
))
