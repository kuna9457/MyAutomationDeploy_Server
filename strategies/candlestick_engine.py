"""
strategies/candlestick_engine.py

plan.md PHASE 1 — the candlestick recognition engine (single, double and triple
candle patterns), wired up as a tradable strategy.

Covers Modules 1-3 of plan.md: every named pattern is implemented. Detection is a
pure function of an OHLCV frame — no broker, no database, no state (Immutable
Rule #3) — so Phases 2-5 (market structure, chart patterns, context, confidence
scoring) can consume detect_patterns() directly rather than reimplementing it.

    WHY THIS WORKS ON EVERY TIMEFRAME (architectural rule 4)
    --------------------------------------------------------
    Not one threshold in this file is a rupee amount or a percentage of price.
    Pattern GEOMETRY is expressed as ratios of the candle's own range ("the body
    is under 10% of the range"), which is scale-free: identical on a ₹800 SBIN
    1-minute bar and a ₹1.4L GOLD daily bar. The two places that need an absolute
    yardstick — "is this candle big enough to mean anything" and "how far away is
    the stop" — use ATR, which self-calibrates per instrument and per timeframe.
    The strategy therefore registers against all three modes; only risk, RR and
    the evidence threshold differ, and those come from config.

    WHAT THIS PHASE HONESTLY IS
    ---------------------------
    plan.md says it plainly: "Without market structure, candlestick signals
    generate many false positives." That is Module 4 / Phase 2, and it is not
    built yet. What IS here is the mitigation the pattern definitions themselves
    demand: a prior-trend check (Module 2: "previous trend [must] be evaluated"),
    without which a Hammer cannot even be distinguished from a Hanging Man. Treat
    Phase 1 as the recognition layer it is billed as, and paper-trade it until
    Phase 2 supplies the structural filter.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import time as dtime
from typing import Optional

import numpy as np
import pandas as pd

from config import (CANDLE_INTRADAY_PARAMS, CANDLE_SCALPER_PARAMS,
                    CANDLE_SWING_PARAMS, Mode, StrategyParams)
from strategy import (Signal, StrategyDef, _atr_in_normal_range,
                      _past_entry_window, register)

# --------------------------------------------------------------------------- #
#  Geometry thresholds — all RATIOS of the candle's own range (see module docs).
# --------------------------------------------------------------------------- #
DOJI_BODY = 0.10        # body <= 10% of range -> open and close are "the same"
SMALL_BODY = 0.30       # a body this small has no conviction behind it
MARUBOZU_BODY = 0.90    # the body IS the candle: no rejection at either end
LONG_WICK = 2.0         # wick >= 2x its own body -> a genuine rejection tail
TINY_WICK = 0.15        # wick <= 15% of range -> effectively no wick
NEAR = 0.10             # two prices count as "equal" within 10% of range

# The only ATR-scaled geometry rule. A candle with no meaningful range is noise,
# not a pattern: a one-tick "doji" on a dead 1-minute bar is arithmetically a
# perfect doji and means absolutely nothing. "Meaningful" is instrument- and
# timeframe-specific, which is exactly what ATR measures.
MIN_RANGE_ATR = 0.25

# How far price must travel for the preceding move to count as a trend, in ATR.
TREND_ATR = 1.0

# Evidence weights. plan.md: three-candle patterns are "among the strongest
# because they incorporate changing momentum over several bars" — so span is
# weighted alongside the per-pattern strength that plan.md's Module 1 table
# assigns (Hammer High, Hanging Man Medium, Spinning Top Weak, ...).
STRENGTH_WEIGHT = {"high": 3.0, "medium": 2.0, "weak": 1.0}
SPAN_WEIGHT = {1: 1.0, 2: 1.25, 3: 1.5, 5: 1.75}


# --------------------------------------------------------------------------- #
#  Candle primitives
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class _C:
    """One candle, with the derived measurements every pattern is phrased in."""
    o: float
    h: float
    l: float
    c: float
    v: float

    @classmethod
    def of(cls, row) -> "_C":
        return cls(float(row["open"]), float(row["high"]), float(row["low"]),
                   float(row["close"]), float(row.get("volume", 0.0) or 0.0))

    @property
    def body(self) -> float: return abs(self.c - self.o)
    @property
    def rng(self) -> float: return self.h - self.l
    @property
    def upper(self) -> float: return self.h - max(self.o, self.c)
    @property
    def lower(self) -> float: return min(self.o, self.c) - self.l
    @property
    def bull(self) -> bool: return self.c > self.o
    @property
    def bear(self) -> bool: return self.c < self.o
    @property
    def body_top(self) -> float: return max(self.o, self.c)
    @property
    def body_bot(self) -> float: return min(self.o, self.c)
    @property
    def body_mid(self) -> float: return (self.o + self.c) / 2.0

    @property
    def is_doji(self) -> bool:
        return self.rng > 0 and self.body <= DOJI_BODY * self.rng

    @property
    def small(self) -> bool:
        return self.rng > 0 and self.body <= SMALL_BODY * self.rng


@dataclass(frozen=True)
class PatternHit:
    """One recognised pattern.

    `direction` is 0 for the indecision patterns (Doji, Spinning Top). plan.md
    marks those "Both" — which in trading terms means neither, so they carry no
    directional weight. They are still reported: Phase 5's confidence engine wants
    to know the tape hesitated, even though Phase 1 won't trade on it.
    """
    name: str
    direction: int      # +1 bullish, -1 bearish, 0 indecision
    strength: str       # "high" | "medium" | "weak"
    n: int              # how many candles the pattern spans

    @property
    def weight(self) -> float:
        return STRENGTH_WEIGHT[self.strength] * SPAN_WEIGHT[self.n]


def _prior_trend(df: pd.DataFrame, atr_val: float, lookback: int,
                 skip: int) -> int:
    """Direction of the move that PRECEDES the pattern: +1 up, -1 down, 0 neither.

    Candlestick patterns are not self-describing. A Hammer and a Hanging Man are
    the identical shape; a Shooting Star and an Inverted Hammer are the identical
    shape. What separates each pair is only what came before, which is why
    plan.md's Module 2 requires "previous trend to be evaluated". Without this,
    half the library would be coin-flips.

    `skip` excludes the pattern's own candles — otherwise a big bullish engulfing
    would itself create the "uptrend" it is supposed to be reversing.

    Measured in ATR units so one threshold holds across instruments and
    timeframes. This is a deliberately minimal stand-in for plan.md Module 4
    (swing structure: HH/HL/LH/LL, BOS, CHOCH), which is Phase 2's job.
    """
    end = len(df) - skip
    start = end - lookback
    if start < 0 or atr_val <= 0:
        return 0
    seg = df["close"].iloc[start:end]
    if len(seg) < 2:
        return 0
    move = (float(seg.iloc[-1]) - float(seg.iloc[0])) / atr_val
    if move >= TREND_ATR:
        return 1
    if move <= -TREND_ATR:
        return -1
    return 0


# --------------------------------------------------------------------------- #
#  MODULE 1 — single candle patterns
# --------------------------------------------------------------------------- #
def _singles(c: _C, trend: int) -> list[PatternHit]:
    hits: list[PatternHit] = []
    if c.rng <= 0:
        return hits
    body_r = c.body / c.rng
    up_r = c.upper / c.rng
    lo_r = c.lower / c.rng

    # -- Doji family: open == close, the market could not decide -------------- #
    # Returns early: the doji family is mutually exclusive with everything below,
    # since a body this small cannot also be a marubozu or carry a hammer's body.
    if body_r <= DOJI_BODY:
        if lo_r >= 0.6 and up_r <= TINY_WICK:
            hits.append(PatternHit("Dragonfly Doji", 1, "high", 1))
        elif up_r >= 0.6 and lo_r <= TINY_WICK:
            hits.append(PatternHit("Gravestone Doji", -1, "high", 1))
        elif up_r >= 0.3 and lo_r >= 0.3:
            hits.append(PatternHit("Long-legged Doji", 0, "medium", 1))
        else:
            hits.append(PatternHit("Doji", 0, "medium", 1))
        return hits

    # -- Marubozu: all body, no wick — one side never gave an inch ----------- #
    if body_r >= MARUBOZU_BODY:
        hits.append(PatternHit("Marubozu Bullish" if c.bull else "Marubozu Bearish",
                               1 if c.bull else -1, "high", 1))
        return hits

    # -- Long lower wick: price was pushed down and rejected ----------------- #
    # Hammer and Hanging Man are the SAME shape. Only the prior trend names them.
    if c.lower >= LONG_WICK * c.body and up_r <= TINY_WICK and body_r <= SMALL_BODY:
        if trend < 0:
            hits.append(PatternHit("Hammer", 1, "high", 1))
        elif trend > 0:
            hits.append(PatternHit("Hanging Man", -1, "medium", 1))
        # trend == 0: genuinely neither. Naming one would be a guess, so we don't.

    # -- Long upper wick: price was pushed up and rejected ------------------- #
    if c.upper >= LONG_WICK * c.body and lo_r <= TINY_WICK and body_r <= SMALL_BODY:
        if trend < 0:
            hits.append(PatternHit("Inverted Hammer", 1, "medium", 1))
        elif trend > 0:
            hits.append(PatternHit("Shooting Star", -1, "high", 1))

    # -- Spinning Top: small body, rejection at BOTH ends = a pause ---------- #
    if (not hits and body_r <= SMALL_BODY
            and c.upper >= c.body and c.lower >= c.body):
        hits.append(PatternHit("Spinning Top", 0, "weak", 1))

    return hits


# --------------------------------------------------------------------------- #
#  MODULE 2 — two candle patterns  (a = older, b = newer)
# --------------------------------------------------------------------------- #
def _doubles(a: _C, b: _C, trend: int) -> list[PatternHit]:
    hits: list[PatternHit] = []
    if a.rng <= 0 or b.rng <= 0:
        return hits
    near = NEAR * max(a.rng, b.rng)

    # -- Engulfing: b's body completely swallows a's ------------------------- #
    if trend < 0 and a.bear and b.bull and b.c >= a.o and b.o <= a.c and b.body > a.body:
        hits.append(PatternHit("Bullish Engulfing", 1, "high", 2))
    if trend > 0 and a.bull and b.bear and b.c <= a.o and b.o >= a.c and b.body > a.body:
        hits.append(PatternHit("Bearish Engulfing", -1, "high", 2))

    # -- Harami: b's body sits INSIDE a's — the move suddenly stops ---------- #
    inside = (b.body_top <= a.body_top and b.body_bot >= a.body_bot
              and a.body > b.body and not a.is_doji)
    if inside and trend < 0 and a.bear:
        hits.append(PatternHit("Bullish Harami Cross" if b.is_doji else "Bullish Harami",
                               1, "high" if b.is_doji else "medium", 2))
    if inside and trend > 0 and a.bull:
        hits.append(PatternHit("Bearish Harami Cross" if b.is_doji else "Bearish Harami",
                               -1, "high" if b.is_doji else "medium", 2))

    # -- Tweezers: two candles rejected at the SAME level -------------------- #
    if trend < 0 and a.bear and b.bull and abs(a.l - b.l) <= near:
        hits.append(PatternHit("Tweezer Bottom", 1, "medium", 2))
    if trend > 0 and a.bull and b.bear and abs(a.h - b.h) <= near:
        hits.append(PatternHit("Tweezer Top", -1, "medium", 2))

    # -- Piercing / Dark Cloud: b stabs deep into a's body but not through --- #
    if trend < 0 and a.bear and b.bull and b.o < a.c and a.body_mid < b.c < a.o:
        hits.append(PatternHit("Piercing Pattern", 1, "high", 2))
    if trend > 0 and a.bull and b.bear and b.o > a.c and a.o < b.c < a.body_mid:
        hits.append(PatternHit("Dark Cloud Cover", -1, "high", 2))

    # -- Matching Low / High: two closes at an identical level = a floor/ceiling #
    if trend < 0 and a.bear and b.bear and abs(a.c - b.c) <= near:
        hits.append(PatternHit("Matching Low", 1, "medium", 2))
    if trend > 0 and a.bull and b.bull and abs(a.c - b.c) <= near:
        hits.append(PatternHit("Matching High", -1, "medium", 2))

    # -- Kicker: a violent gap the other way, with no overlap at all --------- #
    # Trend-agnostic on purpose: a kicker IS the reversal. Demanding that a trend
    # preceded it would throw away the strongest signal in this module.
    if a.bear and b.bull and b.o >= a.o and b.l > a.c:
        hits.append(PatternHit("Bullish Kicker", 1, "high", 2))
    if a.bull and b.bear and b.o <= a.o and b.h < a.c:
        hits.append(PatternHit("Bearish Kicker", -1, "high", 2))

    # -- Failed bounces inside a downtrend: bearish CONTINUATION ------------- #
    # All three are the same story at different depths — a rally that could not
    # reach back into the previous body. Ordered shallowest-first; they are
    # mutually exclusive by where b closes.
    if trend < 0 and a.bear and b.bull and b.o < a.l:
        if abs(b.c - a.l) <= near:
            hits.append(PatternHit("On Neck", -1, "weak", 2))
        elif a.l < b.c <= a.c + near:
            hits.append(PatternHit("In Neck", -1, "weak", 2))
        elif a.c < b.c < a.body_mid:
            hits.append(PatternHit("Thrusting Pattern", -1, "weak", 2))

    return hits


# --------------------------------------------------------------------------- #
#  MODULE 3 — three candle patterns  (a, b, c oldest -> newest)
# --------------------------------------------------------------------------- #
def _triples(a: _C, b: _C, c: _C, trend: int) -> list[PatternHit]:
    hits: list[PatternHit] = []
    if min(a.rng, b.rng, c.rng) <= 0:
        return hits
    near = NEAR * max(a.rng, b.rng, c.rng)

    # -- Morning / Evening Star: drive, hesitation, decisive reversal -------- #
    if (trend < 0 and a.bear and b.small and c.bull
            and b.body_top < a.body_bot        # b's body gapped below a's
            and b.body_top < c.body_bot        # ...and c leaves it behind
            and c.c > a.body_mid):             # c takes back half of a
        hits.append(PatternHit("Morning Star", 1, "high", 3))
    if (trend > 0 and a.bull and b.small and c.bear
            and b.body_bot > a.body_top
            and b.body_bot > c.body_top
            and c.c < a.body_mid):
        hits.append(PatternHit("Evening Star", -1, "high", 3))

    # -- Three rising / falling candles, and the ways they can go wrong ------ #
    stepping_up = (a.bull and b.bull and c.bull and b.c > a.c and c.c > b.c
                   and min(a.body, b.body, c.body) > 0
                   and a.body_bot < b.o < a.body_top
                   and b.body_bot < c.o < b.body_top)
    if stepping_up:
        closes_strong = b.upper <= 0.3 * b.rng and c.upper <= 0.3 * c.rng
        shrinking = c.body < b.body < a.body
        if closes_strong and not shrinking:
            hits.append(PatternHit("Three White Soldiers", 1, "high", 3))
        elif shrinking and c.upper > c.body:
            # Same three green candles, but each one is struggling harder than
            # the last: buyers are being absorbed, not winning.
            hits.append(PatternHit("Advance Block", -1, "medium", 3))
    # Deliberation: the third candle simply gives up, opening at the second's
    # close and going nowhere.
    if (a.bull and b.bull and c.bull and b.c > a.c and b.body > 0
            and c.small and c.body < 0.4 * b.body and abs(c.o - b.c) <= near):
        hits.append(PatternHit("Deliberation", -1, "medium", 3))

    stepping_down = (a.bear and b.bear and c.bear and b.c < a.c and c.c < b.c
                     and min(a.body, b.body, c.body) > 0)
    if (stepping_down and a.body_bot < b.o < a.body_top
            and b.body_bot < c.o < b.body_top
            and b.lower <= 0.3 * b.rng and c.lower <= 0.3 * c.rng):
        hits.append(PatternHit("Three Black Crows", -1, "high", 3))
    # Identical Three Crows: each opens exactly where the last closed — no bounce
    # was even attempted, which is what makes it the harsher version.
    if stepping_down and abs(b.o - a.c) <= near and abs(c.o - b.c) <= near:
        hits.append(PatternHit("Identical Three Crows", -1, "high", 3))

    # -- Three Inside / Outside: a 2-candle signal CONFIRMED by the third ---- #
    # Reuses _doubles() rather than restating engulfing/harami geometry: one
    # definition, so the pair can never drift apart.
    prior = {h.name for h in _doubles(a, b, trend)}
    if prior & {"Bullish Harami", "Bullish Harami Cross"} and c.bull and c.c > b.c and c.c > a.o:
        hits.append(PatternHit("Three Inside Up", 1, "high", 3))
    if prior & {"Bearish Harami", "Bearish Harami Cross"} and c.bear and c.c < b.c and c.c < a.o:
        hits.append(PatternHit("Three Inside Down", -1, "high", 3))
    if "Bullish Engulfing" in prior and c.bull and c.c > b.c:
        hits.append(PatternHit("Three Outside Up", 1, "high", 3))
    if "Bearish Engulfing" in prior and c.bear and c.c < b.c:
        hits.append(PatternHit("Three Outside Down", -1, "high", 3))

    # -- Abandoned Baby: a doji marooned by gaps on BOTH sides -------------- #
    # Rare by construction, and rarer still on Indian intraday tape, which gaps
    # only at the open. Kept exact rather than relaxed: a "nearly abandoned" baby
    # is a Morning Star, and that is already detected above.
    if trend < 0 and a.bear and b.is_doji and c.bull and b.h < a.l and b.h < c.l:
        hits.append(PatternHit("Abandoned Baby", 1, "high", 3))
    if trend > 0 and a.bull and b.is_doji and c.bear and b.l > a.h and b.l > c.h:
        hits.append(PatternHit("Abandoned Baby", -1, "high", 3))

    # -- Tri-Star: three dojis running — the trend has stopped breathing ----- #
    if a.is_doji and b.is_doji and c.is_doji and trend != 0:
        hits.append(PatternHit("Tri-Star", -trend, "high", 3))

    # -- Upside Gap Two Crows: the gap up gets sold, but the floor holds ----- #
    if (trend > 0 and a.bull and b.bear and c.bear
            and b.body_bot > a.c                 # b gapped above a
            and c.o > b.o and c.c < b.c          # c engulfs b
            and c.c > a.c):                      # yet still holds above a's close
        hits.append(PatternHit("Upside Gap Two Crows", -1, "medium", 3))

    # -- Stick Sandwich: two closes at the same level with a rally between --- #
    if a.bear and b.bull and c.bear and b.c > a.c and abs(c.c - a.c) <= near:
        hits.append(PatternHit("Stick Sandwich", 1, "medium", 3))

    # -- Tasuki Gap: a gap the pullback fails to fill = CONTINUATION --------- #
    if (a.bull and b.bull and b.body_bot > a.body_top
            and c.bear and b.body_bot < c.o < b.body_top
            and a.body_top < c.c < b.body_bot):  # closes inside the gap, not past
        hits.append(PatternHit("Upside Tasuki Gap", 1, "medium", 3))
    if (a.bear and b.bear and b.body_top < a.body_bot
            and c.bull and b.body_bot < c.o < b.body_top
            and b.body_top < c.c < a.body_bot):
        hits.append(PatternHit("Downside Tasuki Gap", -1, "medium", 3))

    return hits


# --------------------------------------------------------------------------- #
#  MODULE 3 (continued) — the five-candle formations
# --------------------------------------------------------------------------- #
def _fives(cs: list[_C], trend: int) -> list[PatternHit]:
    """plan.md files Rising/Falling Three Methods, Mat Hold and Breakaway under
    "Three Candle Patterns", but all four are classically FIVE-candle formations
    — the names count the middle candles, not the whole pattern. Implemented at
    their true length: a three-candle "Rising Three Methods" is just a small bull
    flag, and would fire constantly on something that isn't the pattern.
    """
    hits: list[PatternHit] = []
    if len(cs) < 5 or min(x.rng for x in cs) <= 0:
        return hits
    a, b, c, d, e = cs[-5:]
    mids = (b, c, d)

    # -- Rising / Falling Three Methods: a shallow pause inside a big candle - #
    contained = all(a.l <= x.l and x.h <= a.h for x in mids)
    smaller = all(x.body < a.body for x in mids)
    if (a.bull and e.bull and contained and smaller
            and d.c < b.c            # the pause genuinely drifts against the move
            and e.c > a.h):          # and the move then resumes to a new high
        hits.append(PatternHit("Rising Three Methods", 1, "high", 5))
    if (a.bear and e.bear and contained and smaller
            and d.c > b.c
            and e.c < a.l):
        hits.append(PatternHit("Falling Three Methods", -1, "high", 5))

    # -- Mat Hold: same idea, but the pause gaps up and never gives back the pole #
    if (a.bull and e.bull and b.body_bot > a.body_top
            and all(x.small for x in mids)
            and all(x.l > a.body_mid for x in mids)
            and e.c > max(b.h, c.h, d.h)):
        hits.append(PatternHit("Mat Hold", 1, "high", 5))

    # -- Breakaway: the trend gaps, exhausts itself, then snaps back to the gap #
    if (trend < 0 and a.bear and b.bear and b.body_top < a.body_bot
            and c.c < b.c and d.c < c.c
            and e.bull and b.body_top < e.c < a.body_bot):
        hits.append(PatternHit("Breakaway", 1, "high", 5))
    if (trend > 0 and a.bull and b.bull and b.body_bot > a.body_top
            and c.c > b.c and d.c > c.c
            and e.bear and a.body_top < e.c < b.body_bot):
        hits.append(PatternHit("Breakaway", -1, "high", 5))

    return hits


# --------------------------------------------------------------------------- #
#  Detection + scoring  (the reusable public surface for Phases 2-5)
# --------------------------------------------------------------------------- #
def detect_patterns(df: pd.DataFrame, params: StrategyParams) -> list[PatternHit]:
    """Every pattern that COMPLETES on the last bar of `df`.

    Only the last bar is considered. A pattern that completed three bars ago is
    history, not a trade — and re-detecting it every bar is exactly how a bot ends
    up entering the same setup over and over.

    `df` must already carry an "atr" column (strategy.enrich supplies it).
    """
    if len(df) < 2 or "atr" not in df.columns:
        return []
    atr_val = float(df["atr"].iloc[-1])
    if not np.isfinite(atr_val) or atr_val <= 0:
        return []

    n = min(len(df), 5)
    cs = [_C.of(df.iloc[i]) for i in range(-n, 0)]
    last = cs[-1]
    if last.rng < MIN_RANGE_ATR * atr_val:
        return []

    lb = params.cs_trend_lookback
    hits = _singles(last, _prior_trend(df, atr_val, lb, 1))
    if n >= 2:
        hits += _doubles(cs[-2], last, _prior_trend(df, atr_val, lb, 2))
    if n >= 3:
        hits += _triples(cs[-3], cs[-2], last, _prior_trend(df, atr_val, lb, 3))
    if n >= 5:
        hits += _fives(cs, _prior_trend(df, atr_val, lb, 5))
    return hits


def score_patterns(hits: list[PatternHit]) -> tuple[float, float]:
    """(bullish_evidence, bearish_evidence) as weighted sums.

    Indecision patterns contribute to neither, by design — a Doji is information
    about hesitation, not about direction, and adding it to a side would be
    inventing a signal plan.md never claimed.

    NOTE: this is a placeholder for plan.md's Confidence Engine, which weights
    trend alignment, market structure, geometry, volume and S/R context. Those
    inputs are Phases 2-4; until they exist there is nothing honest to weight, so
    scoring stays a simple sum of pattern evidence.
    """
    bull = sum(h.weight for h in hits if h.direction > 0)
    bear = sum(h.weight for h in hits if h.direction < 0)
    return bull, bear


# --------------------------------------------------------------------------- #
#  The strategy
# --------------------------------------------------------------------------- #
def candlestick_signal(df: pd.DataFrame, params: StrategyParams,
                       session_open: Optional[dtime] = None) -> Optional[Signal]:
    """
    Trade a completed candlestick pattern, sized off the pattern's own invalidation
    level. Runs unchanged on 1m / 15m / daily — see the module docstring.

      Entry : the close of the bar that completes the pattern, once the weighted
              evidence for one side clears params.cs_min_score and the other side
              is silent.
      Stop  : beyond the pattern's extreme (the level the pattern claims will
              hold), clamped into [cs_min_sl_atr, cs_max_sl_atr] x ATR.
      Target: the mode's RR applied to that stop distance — 1:1 Scalper,
              1:2 Intraday, 1:3 Swing (Immutable Rule #1).
    """
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
    bull, bear = score_patterns(hits)

    # Both sides firing at once is real ambiguity, not a 50/50 bet. Sit it out.
    if bull > 0 and bear > 0:
        return None

    if bull >= params.cs_min_score:
        side, want = "BUY", 1
    elif bear >= params.cs_min_score and params.allow_short:
        side, want = "SELL", -1
    else:
        return None

    # The stop goes beyond the pattern's own extreme — the level the pattern says
    # must hold. Break it and the pattern was simply wrong, so there is nothing
    # left to be right about. That is what makes this a candlestick stop rather
    # than a generic "entry -/+ k x ATR".
    chosen = [h for h in hits if h.direction == want]
    win = df.iloc[-max(h.n for h in chosen):]
    named = ", ".join(dict.fromkeys(h.name for h in chosen))
    buf = params.cs_sl_buffer_atr * atr_val

    if side == "BUY":
        stop = float(win["low"].min()) - buf
        # Clamp both ways. Too close and noise takes it out for free; too far and
        # position_size() shrinks the trade to nothing (or to zero shares). Both
        # bounds are in ATR, so they travel across timeframes with everything else.
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

    return Signal(side, entry, stop, target,
                  f"{named} (evidence {max(bull, bear):.1f})")


register(StrategyDef(
    key="candlestick_engine",
    name="Candlestick Engine (Phase 1)",
    # The same logic on every timeframe the bot supports; only risk, RR and the
    # evidence bar change. See config for why Swing is long-only.
    params_by_mode={
        Mode.SCALPER: CANDLE_SCALPER_PARAMS,
        Mode.INTRADAY: CANDLE_INTRADAY_PARAMS,
        Mode.SWING: CANDLE_SWING_PARAMS,
    },
    fn=candlestick_signal,
    summary="plan.md Phase 1: 40+ single/double/triple candle patterns, scored by "
            "strength and confirmed against the prior trend. Stop sits beyond the "
            "pattern; RR follows the mode. Runs on 1m, 15m and daily.",
))
