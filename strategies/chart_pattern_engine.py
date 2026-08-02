"""
strategies/chart_pattern_engine.py

plan.md PHASE 3 — the chart-pattern engine (Module 5), wired up as a tradable
strategy.

Covers plan.md Module 5's classic figures at their true, pivot-based scale:
  Reversal    — Double Top, Double Bottom, Head & Shoulders, Inverse H&S
  Continuation/reversal by breakout — Ascending, Descending and Symmetrical
                Triangles
plan.md notes these "require identifying pivots, trendlines, and breakout
confirmation rather than a fixed number of candles" — so this engine works off
swing pivots and only signals on a confirmed neckline / trendline BREAK, never on
a half-formed shape.

    SELF-CONTAINED BY DESIGN
    ------------------------
    Per the brief ("each phase independent for now") this file shares no code
    with Phase 2; it carries its own _pivots. When the phases are unified this
    and Phase 2's copy collapse into one helper.

    WHY THIS WORKS ON EVERY TIMEFRAME (architectural rule 4)
    --------------------------------------------------------
    Every tolerance is a RATIO of ATR or of the pattern's own height, so "the two
    tops are equal" and "the shoulders match" mean the same on a ₹800 stock and a
    ₹1.4L gold contract. The stop is anchored to the pattern's invalidation level
    and clamped in ATR. The strategy therefore registers on all three modes.
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
#  Tuning — all ratios / bar counts. No absolute price constants.
# --------------------------------------------------------------------------- #
PIVOT_K = 3             # chart figures are built from stronger pivots than structure
MAX_PIVOTS = 14         # window of recent pivots a pattern may span
EQUAL_ATR = 0.6         # two levels are "equal" within this × ATR (tops, shoulders)
FLAT_SLOPE_ATR = 0.15   # a trendline is "flat" if |slope| ≤ this × ATR per bar
SL_BUFFER_ATR = 0.3     # stop sits this far beyond the invalidation level
MIN_SL_ATR = 0.5        # ...never closer to entry than this
MAX_SL_ATR = 4.5        # ...never further than this


def _pivots(df: pd.DataFrame, k: int) -> list[tuple[int, float, int]]:
    """Confirmed fractal pivots as (bar_index, price, +1 high / -1 low).
    See market_structure_engine._pivots for the confirmation rationale."""
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


def _slope(points: list[tuple[int, float]]) -> float:
    """Least-squares slope (price per bar) through (index, price) points."""
    if len(points) < 2:
        return 0.0
    xs = np.array([p[0] for p in points], dtype=float)
    ys = np.array([p[1] for p in points], dtype=float)
    xs -= xs.mean()
    denom = float((xs * xs).sum())
    if denom == 0:
        return 0.0
    return float((xs * (ys - ys.mean())).sum() / denom)


# --------------------------------------------------------------------------- #
#  Pattern detectors. Each returns (side, invalidation_price, name) or None.
#  `side` is "BUY"/"SELL"; invalidation_price is where the pattern is proven
#  wrong (used to place the structural stop). All fire ONLY on the bar that
#  confirms the break, using prev/last so the same setup can't re-trigger.
# --------------------------------------------------------------------------- #
def _double(pivots, prev, last, atr) -> Optional[tuple[str, float, str]]:
    """Double Bottom (long) / Double Top (short): two equal extremes either side
    of an opposite pivot (the neckline); break of the neckline confirms."""
    eq = EQUAL_ATR * atr
    lows = [p for p in pivots if p[2] == -1]
    highs = [p for p in pivots if p[2] == 1]

    # Double Bottom: low1, (high between), low2 ~ equal; neckline = that high.
    if len(lows) >= 2 and highs:
        l1, l2 = lows[-2], lows[-1]
        mids = [h for h in highs if l1[0] < h[0] < l2[0]]
        if mids and abs(l1[1] - l2[1]) <= eq:
            neck = max(mids, key=lambda h: h[0])[1]
            if prev["close"] <= neck < last["close"]:
                return "BUY", min(l1[1], l2[1]), "Double Bottom"

    # Double Top: high1, (low between), high2 ~ equal; neckline = that low.
    if len(highs) >= 2 and lows:
        h1, h2 = highs[-2], highs[-1]
        mids = [l for l in lows if h1[0] < l[0] < h2[0]]
        if mids and abs(h1[1] - h2[1]) <= eq:
            neck = min(mids, key=lambda l: l[0])[1]
            if prev["close"] >= neck > last["close"]:
                return "SELL", max(h1[1], h2[1]), "Double Top"
    return None


def _head_shoulders(pivots, prev, last, atr) -> Optional[tuple[str, float, str]]:
    """Head & Shoulders (short) / Inverse H&S (long): three extremes with the
    middle the most extreme and the outer two ~equal; neckline joins the two
    opposite pivots between them; break of the neckline confirms."""
    eq = EQUAL_ATR * atr

    highs = [p for p in pivots if p[2] == 1]
    lows = [p for p in pivots if p[2] == -1]

    # Head & Shoulders top → SELL. LS < HEAD > RS, LS ≈ RS, HEAD highest.
    if len(highs) >= 3:
        ls, head, rs = highs[-3], highs[-2], highs[-1]
        if (head[1] > ls[1] and head[1] > rs[1] and abs(ls[1] - rs[1]) <= eq):
            necks = [l[1] for l in lows if ls[0] < l[0] < rs[0]]
            if necks:
                neck = float(np.mean(necks))
                if prev["close"] >= neck > last["close"]:
                    return "SELL", head[1], "Head & Shoulders"

    # Inverse H&S bottom → BUY. LS > HEAD < RS, LS ≈ RS, HEAD lowest.
    if len(lows) >= 3:
        ls, head, rs = lows[-3], lows[-2], lows[-1]
        if (head[1] < ls[1] and head[1] < rs[1] and abs(ls[1] - rs[1]) <= eq):
            necks = [h[1] for h in highs if ls[0] < h[0] < rs[0]]
            if necks:
                neck = float(np.mean(necks))
                if prev["close"] <= neck < last["close"]:
                    return "BUY", head[1], "Inverse Head & Shoulders"
    return None


def _triangle(pivots, prev, last, atr) -> Optional[tuple[str, float, str]]:
    """Ascending / Descending / Symmetrical triangle, by the slopes of the recent
    swing-high line (resistance) and swing-low line (support). A close through the
    relevant boundary confirms the breakout.

      Ascending  : flat highs + rising lows  → break UP  (long)
      Descending : flat lows  + falling highs → break DOWN (short)
      Symmetrical: converging (falling highs + rising lows) → break either way
    """
    flat = FLAT_SLOPE_ATR * atr
    highs = [(p[0], p[1]) for p in pivots if p[2] == 1][-3:]
    lows = [(p[0], p[1]) for p in pivots if p[2] == -1][-3:]
    if len(highs) < 2 or len(lows) < 2:
        return None

    hi_slope, lo_slope = _slope(highs), _slope(lows)
    res = max(h[1] for h in highs)     # resistance = the flat/converging top
    sup = min(l[1] for l in lows)      # support = the flat/converging bottom
    if res - sup <= 0:
        return None

    # Ascending: highs flat, lows rising → bullish break of resistance.
    if abs(hi_slope) <= flat and lo_slope > flat:
        if prev["close"] <= res < last["close"]:
            return "BUY", sup, "Ascending Triangle"
    # Descending: lows flat, highs falling → bearish break of support.
    if abs(lo_slope) <= flat and hi_slope < -flat:
        if prev["close"] >= sup > last["close"]:
            return "SELL", res, "Descending Triangle"
    # Symmetrical: converging → trade whichever side breaks first.
    if hi_slope < -flat and lo_slope > flat:
        if prev["close"] <= res < last["close"]:
            return "BUY", sup, "Symmetrical Triangle"
        if prev["close"] >= sup > last["close"]:
            return "SELL", res, "Symmetrical Triangle"
    return None


_DETECTORS = (_double, _head_shoulders, _triangle)


# --------------------------------------------------------------------------- #
#  The strategy
# --------------------------------------------------------------------------- #
def chart_pattern_signal(df: pd.DataFrame, params: StrategyParams,
                         session_open: Optional[dtime] = None
                         ) -> Optional[Signal]:
    """
    Trade the confirmed breakout of a classic chart figure.

    Runs each detector; the first that confirms on THIS bar wins. Entry is the
    breakout close; the stop sits beyond the pattern's invalidation level (the
    double's extreme, the H&S head, the triangle's far boundary), clamped in ATR;
    target follows the mode's RR. Bearish figures are skipped when the mode is
    long-only (Swing delivery).
    """
    need = max(params.atr_period, PIVOT_K * 2 + 2) + 10
    if len(df) < max(need, 2 * PIVOT_K + 6):
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
    if len(pivots) < 3:
        return None
    pivots = pivots[-MAX_PIVOTS:]

    for detect in _DETECTORS:
        hit = detect(pivots, prev, last, atr_val)
        if hit is None:
            continue
        side, invalidation, name = hit
        if side == "SELL" and not params.allow_short:
            continue
        buf = SL_BUFFER_ATR * atr_val
        if side == "BUY":
            stop = invalidation - buf
            stop = min(stop, entry - MIN_SL_ATR * atr_val)
            stop = max(stop, entry - MAX_SL_ATR * atr_val)
            if stop >= entry:
                continue
            target = entry + params.risk_reward * (entry - stop)
        else:
            stop = invalidation + buf
            stop = max(stop, entry + MIN_SL_ATR * atr_val)
            stop = min(stop, entry + MAX_SL_ATR * atr_val)
            if stop <= entry:
                continue
            target = entry - params.risk_reward * (stop - entry)
            if target <= 0:
                continue
        return Signal(side, entry, stop, target, f"{name} breakout")

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
    key="chart_pattern_engine",
    name="Chart Patterns (Phase 3)",
    params_by_mode={Mode.SCALPER: _SCALPER, Mode.INTRADAY: _INTRADAY,
                    Mode.SWING: _SWING},
    fn=chart_pattern_signal,
    summary="plan.md Phase 3: Double Top/Bottom, Head & Shoulders (+inverse) and "
            "ascending/descending/symmetrical triangles, traded on a confirmed "
            "neckline/trendline breakout. Stop beyond the pattern; RR per mode.",
))
