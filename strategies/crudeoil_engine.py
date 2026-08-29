"""
strategies/crudeoil_engine.py
CRUDEOIL — US-session Opening Range Breakout.  See Commodity.md.

WHY THIS EXISTS
---------------
MCX crude is a follower. It drifts through the Indian morning on thin volume and
then moves when the US session opens and NYMEX starts trading it for real. Every
other strategy in this system anchors its VWAP to the local session open, which
for a commodity that takes its direction from New York measures the wrong thing.

So this strategy ignores the 09:00 IST MCX open completely. It anchors both its
VWAP and its opening range to 09:00 America/New_York, waits for the first 30
minutes of US trade to define a range, and trades the break of that range in the
direction the anchored VWAP already agrees with.

    THE ANCHOR IS A TIMEZONE, NOT A CLOCK TIME
    09:00 New York is 18:30 IST under EDT and 19:30 IST under EST. Hardcoding
    "18:30 IST" — as the original draft did — silently builds the opening range
    an hour early for roughly five months a year, on pre-open tape, and every
    signal that follows is derived from the wrong two candles. The anchor is
    therefore resolved per trading day from the tz database.

WHAT THIS FILE IS AND IS NOT
----------------------------
It is PURE SIGNAL LOGIC: candles in, an optional Signal out. It imports no
broker, no database and no engine (Immutable Rule #6), and it knows nothing
about lots, margin or money. Position sizing for commodities is the admin's
fixed lot count, applied by engine._mcx_fixed_size on the same path every other
MCX trade takes — this file never sees it.

The scale-out behaviour people associate with this strategy (book half at the
target, move the rest to break-even, trail it) lives in the engine, driven by
the params set in config.CRUDEOIL_PARAMS. It is deliberately not here: how a
position is *managed* is execution, not strategy, and a backtest that replays
this function must see the same entry decision either way.

    ONE LOT NEVER SCALES OUT. A single-lot position is indivisible, so it runs
    to the FULL target and closes there — no partial, no break-even shuffle, no
    trail. Scaling out of a 1-lot position would mean closing all of it at the
    first target, which is strictly worse than simply letting it reach the
    target it was sized for. Only a position of 2+ lots takes the runner path.
    That rule is enforced in engine._maybe_scale_out(); it is restated here
    because it is the thing most likely to be "helpfully" broken later.
"""
from __future__ import annotations

from datetime import date, datetime, time as dtime, timedelta
from typing import Optional
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from config import CRUDEOIL_PARAMS, Mode, Segment, StrategyParams
from strategy import Signal, StrategyDef, _atr_in_normal_range, register

_IST = ZoneInfo("Asia/Kolkata")


# --------------------------------------------------------------------------- #
#  Session anchor
# --------------------------------------------------------------------------- #
def _anchor_ist(day: date, params: StrategyParams) -> Optional[dtime]:
    """The IST wall-clock time at which the US session opens on `day`.

    Resolved through the tz database rather than stored, so the EDT/EST shift
    (18:30 IST vs 19:30 IST) is handled by definition instead of by remembering
    to change a constant twice a year. Returns None when the strategy has no
    anchor configured, which is what makes every OTHER strategy's params inert
    here.
    """
    if not params.orb_anchor_tz or not params.orb_anchor_hhmm:
        return None
    try:
        hh, mm = (int(x) for x in params.orb_anchor_hhmm.split(":", 1))
        src = datetime(day.year, day.month, day.day, hh, mm,
                       tzinfo=ZoneInfo(params.orb_anchor_tz))
    except Exception:                                    # noqa: BLE001
        return None
    return src.astimezone(_IST).replace(tzinfo=None).time()


def _session_slice(df: pd.DataFrame, anchor: dtime) -> pd.DataFrame:
    """Bars from this trading day's anchor up to and including the last bar.

    Sliced on the LAST bar's own date, so a backtest walking a window of history
    anchors to the day it is currently evaluating rather than to today's.
    """
    last_ts = df.index[-1]
    start = datetime.combine(last_ts.date(), anchor)
    if last_ts < start:                    # anchor hasn't arrived yet today
        return df.iloc[0:0]
    return df.loc[start:]


def _anchored_vwap(session: pd.DataFrame) -> float:
    """VWAP of the US session only.

    Deliberately NOT df["vwap"] from enrich(), which resets on the calendar day.
    MCX runs 09:00-23:30 inside ONE calendar day, so the day-anchored VWAP has
    already absorbed nine hours of thin Indian-session tape by the time the US
    opens — exactly the prints this strategy exists to ignore.
    """
    vol = session["volume"].astype(float)
    typical = (session["high"] + session["low"] + session["close"]) / 3.0
    total = float(vol.sum())
    if total <= 0:
        return float("nan")
    return float((typical * vol).sum() / total)


def _opening_range(session: pd.DataFrame, anchor: dtime,
                   orb_minutes: int) -> Optional[tuple[float, float, int]]:
    """(high, low, bar_count) of the opening range, or None until it is CLOSED.

    Returning None while the window is still forming is what stops the strategy
    trading a breakout of a range that does not exist yet. The range is only
    honoured once at least one bar has printed AFTER it, which on 15-minute
    candles means the two range bars are genuinely complete.
    """
    if session.empty:
        return None
    day = session.index[-1].date()
    start = datetime.combine(day, anchor)
    end = start + timedelta(minutes=orb_minutes)
    window = session.loc[start:end - timedelta(microseconds=1)]
    if window.empty:
        return None
    if session.index[-1] < end:            # still inside the window
        return None
    return (float(window["high"].max()), float(window["low"].min()), len(window))


# --------------------------------------------------------------------------- #
#  Entry
# --------------------------------------------------------------------------- #
def crudeoil_signal(df: pd.DataFrame, params: StrategyParams,
                    session_open: Optional[dtime] = None) -> Optional[Signal]:
    """
    Long entry (all on one CLOSED bar, after the opening range has locked):

        1. Context  — close held above the US-anchored VWAP on >= 60% of the
                      last 10 bars. The break has to agree with where US
                      participants have actually been trading.
        2. Trigger  — this bar CLOSES above the opening range high. A close,
                      not a touch: intrabar pokes through the range are the
                      single most common way an ORB strategy bleeds.
        3. Momentum — this bar's high exceeds the previous bar's high, so the
                      break is being carried rather than faded.
        4. Gate     — ATR sits inside its own normal band, and the opening
                      range is a sane multiple of ATR.

    Short entry is the exact mirror (below VWAP, close below the range low,
    low below the previous low).

    Stop is a pure volatility stop, entry -/+ atr_sl_mult x ATR. NO structural
    leg, unlike the scalper's hybrid stop: the opening range low IS the
    structure here, it sits just under a long entry by construction, and a stop
    placed there would be the first thing a normal retest takes out.

    `session_open` is accepted for registry-signature compatibility and
    deliberately unused — this strategy's session starts in New York, not at the
    MCX open.
    """
    anchor = _anchor_ist(df.index[-1].date(), params) if _usable(df, params) else None
    if anchor is None:
        return None

    session = _session_slice(df, anchor)
    if session.empty:
        return None
    rng = _opening_range(session, anchor, params.orb_minutes)
    if rng is None:
        return None
    orb_high, orb_low, _bars = rng

    last, prev = df.iloc[-1], df.iloc[-2]
    atr_val = float(last["atr"]) if np.isfinite(last.get("atr", np.nan)) else 0.0
    entry = float(last["close"])
    if atr_val <= 0 or entry <= 0:
        return None
    if params.use_atr_gate and not _atr_in_normal_range(df, params):
        return None

    # The opening range must be worth trading. A collapsed range breaks on
    # noise; a blown-out one means the move already happened inside the window
    # and the ATR stop would sit a very long way from a very late entry.
    orb_range = orb_high - orb_low
    if orb_range <= 0:
        return None
    if params.orb_range_min_atr and orb_range < params.orb_range_min_atr * atr_val:
        return None
    if params.orb_range_max_atr and orb_range > params.orb_range_max_atr * atr_val:
        return None

    # No new entries in the last stretch before square-off — a position opened
    # at 23:10 gets flattened at 23:15 on the clock, having proved nothing.
    if not _before_entry_cutoff(df, params):
        return None

    vwap_val = _anchored_vwap(session)
    if not np.isfinite(vwap_val):
        return None

    ctx = _context_fractions(df, session, anchor, params)
    if ctx is None:
        return None
    frac_above, frac_below = ctx
    # Multi-timeframe veto. 0 (no clear higher-timeframe trend) blocks BOTH
    # sides — an opening-range break with the 60m chopping is the setup that
    # fails most often.
    bias = htf_bias(df, params)

    # -- Long ---------------------------------------------------------------- #
    if (bias > 0
            and frac_above >= params.context_min_frac
            and entry > vwap_val                      # break agrees with VWAP
            and entry > orb_high                      # CLOSES above the range
            and float(prev["close"]) <= orb_high      # ...for the first time
            and float(last["high"]) > float(prev["high"])):
        stop = entry - params.atr_sl_mult * atr_val
        if stop >= entry:
            return None
        target = entry + params.risk_reward * (entry - stop)
        return Signal("BUY", entry, stop, target,
                      f"US-ORB break {orb_high:.2f} + VWAP {vwap_val:.2f} "
                      f"+ {params.htf_minutes}m bias up (ATR {atr_val:.2f})")

    # -- Short --------------------------------------------------------------- #
    if (params.allow_short
            and bias < 0
            and frac_below >= params.context_min_frac
            and entry < vwap_val
            and entry < orb_low
            and float(prev["close"]) >= orb_low
            and float(last["low"]) < float(prev["low"])):
        stop = entry + params.atr_sl_mult * atr_val
        if stop <= entry:
            return None
        target = entry - params.risk_reward * (stop - entry)
        if target <= 0:
            return None
        return Signal("SELL", entry, stop, target,
                      f"US-ORB breakdown {orb_low:.2f} + VWAP {vwap_val:.2f} "
                      f"+ {params.htf_minutes}m bias down (ATR {atr_val:.2f})")

    return None


def _usable(df: pd.DataFrame, params: StrategyParams) -> bool:
    """Enough history, a real time index, and an opening range configured."""
    if not params.orb_minutes:
        return False
    if not isinstance(df.index, pd.DatetimeIndex) or len(df) < params.atr_period + 3:
        return False
    return True


def _context_fractions(df: pd.DataFrame, session: pd.DataFrame, anchor: dtime,
                       params: StrategyParams
                       ) -> Optional[tuple[float, float]]:
    """How much of the recent window closed each side of the ANCHORED VWAP.

    Computed bar-by-bar against the VWAP as it stood at that bar (a running
    anchored VWAP), not against today's final value — judging an earlier bar by
    a VWAP that had not been printed yet would be look-ahead, and would make the
    context filter agree with itself far too often.
    """
    n = int(params.context_bars)
    if n <= 0 or len(session) < 2:
        return None
    vol = session["volume"].astype(float)
    typical = (session["high"] + session["low"] + session["close"]) / 3.0
    cum_vol = vol.cumsum()
    running = (typical * vol).cumsum() / cum_vol.replace(0, np.nan)
    tail_close = session["close"].tail(n)
    tail_vwap = running.tail(n)
    ok = tail_vwap.notna()
    if not bool(ok.any()):
        return None
    above = float((tail_close[ok] > tail_vwap[ok]).mean())
    below = float((tail_close[ok] < tail_vwap[ok]).mean())
    return above, below


def htf_bias(df: pd.DataFrame, params: StrategyParams) -> int:
    """MULTI-TIMEFRAME bias: +1 bullish, -1 bearish, 0 no agreement.

    The higher timeframe is RESAMPLED from the base candles rather than pulled
    from a second feed, so live and backtest derive it identically and there is
    no second stream to fall out of sync.

    Two invariants make this safe:

    1. ONLY COMPLETED HTF BARS COUNT. The bar the base candle is currently
       inside is dropped. Including it would let a 60-minute bias flip
       mid-formation and, in a backtest, would read parts of the future — the
       classic multi-timeframe look-ahead bug, and the reason MTF filters so
       often look brilliant in test and useless live.
    2. NO PARTIAL BUCKETS AT THE EDGES. `label`/`closed` are left-aligned so a
       resampled bar is stamped at its own OPEN, matching how the base feed
       stamps candles.

    Bias is the direction of the last `htf_bars` closes: all rising = +1, all
    falling = -1, mixed = 0. Deliberately blunt — this is a veto on trading
    against the larger trend, not a signal in its own right.
    """
    if not params.htf_minutes or not isinstance(df.index, pd.DatetimeIndex):
        return 0
    n = max(int(params.htf_bars), 2)
    htf = (df.resample(f"{int(params.htf_minutes)}min",
                       label="left", closed="left")
             .agg({"open": "first", "high": "max", "low": "min",
                   "close": "last", "volume": "sum"})
             .dropna())
    # Drop the bucket the current base bar still sits inside — it is not a
    # closed bar yet, and treating it as one is look-ahead.
    bucket = df.index[-1].floor(f"{int(params.htf_minutes)}min")
    htf = htf[htf.index < bucket]
    if len(htf) < n:
        return 0
    closes = htf["close"].tail(n).to_numpy(dtype=float)
    diffs = np.diff(closes)
    if np.all(diffs > 0):
        return 1
    if np.all(diffs < 0):
        return -1
    return 0


def _before_entry_cutoff(df: pd.DataFrame, params: StrategyParams) -> bool:
    """True while new entries are still allowed.

    The cutoff is expressed relative to the MCX square-off (23:15 IST) rather
    than as its own wall-clock setting, so moving the square-off moves this with
    it instead of leaving a stale constant behind.
    """
    if not params.entry_cutoff_before_close:
        return True
    from config import DEFAULT_SQUARE_OFF, Segment          # local: avoid cycle
    close_t = DEFAULT_SQUARE_OFF.get(Segment.MCX)
    if close_t is None:
        return True
    bar_t = df.index[-1].time()
    cutoff = (datetime.combine(date.today(), close_t)
              - timedelta(minutes=params.entry_cutoff_before_close)).time()
    return bar_t < cutoff


register(StrategyDef(
    key="crudeoil_us_orb", name="Crudeoil Strategy",
    params_by_mode={Mode.INTRADAY: CRUDEOIL_PARAMS}, fn=crudeoil_signal,
    # MCX ONLY. On an equity the US-session anchor (18:30/19:30 IST) falls
    # outside the 09:15-15:30 session, so no opening range ever forms and the
    # strategy silently returns nothing. Declaring the segment lets the UI say
    # so instead of showing an empty result that looks like "no setups".
    segments=(Segment.MCX,),
    summary="MCX crude only. Anchors VWAP and a 30-min opening range to the US "
            "session open (18:30 IST in summer / 19:30 in winter), then trades "
            "the range break in the VWAP's direction. 1:1.5 RR, long+short. "
            "1 lot runs to the full target; 2+ lots book half there, move the "
            "rest to break-even and trail it.",
))
