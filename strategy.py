"""
strategy.py
Pure strategy logic — indicators + entry signals + position sizing.

This module NEVER imports a broker or a database. It takes a candle DataFrame
in, and returns signals/sizes out. That decoupling is Immutable Rule #3, and it
is what lets the same strategy run over Upstox, Dhan, Kotak or the simulator, and
over NSE equity or MCX commodities, without change.

Strategies live in a REGISTRY (bottom of this file), keyed by a stable id. A Mode
picks the timeframe; the registry picks the logic. The relationship runs BOTH
ways:
  * one mode, many strategies — the 1-minute Scalper hosts the VWAP-ATR
    Pull-back and Volume-Burst Momentum;
  * one strategy, many modes — the Candlestick Engine declares params for all
    three, because a Bullish Engulfing is the same geometry on 1m and on daily.
A strategy therefore registers `params_by_mode`, and callers get a BoundStrategy
(a def resolved to one mode) so `.params` is never ambiguous.

    ADDING A STRATEGY
    -----------------
    1. Create   strategies/<your_strategy>.py   — one file per strategy.
    2. Write    fn(df, params, session_open) -> Optional[Signal]
       reading only pre-computed indicator columns from enrich().
    3. Add its StrategyParams to config.py, one per mode you support
       (risk_reward and risk_per_trade are enforced, not advisory).
    4. register(StrategyDef(key=..., name=..., params_by_mode={...}, fn=...)).
    Nothing else changes: strategies/ is auto-imported at the bottom of this
    file, so the dropdown, engine and backtester pick the new file up with no
    edit to app.py, engine.py or backtester.py.

Indicators are implemented directly in pandas/numpy rather than pandas-ta so the
system runs on any numpy version without dependency breakage. If you prefer
pandas-ta / TA-Lib, swap the bodies of the _indicator functions — the signatures
are stable.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import time as dtime
from typing import Callable, Optional

import numpy as np
import pandas as pd

from config import (INTRADAY_PARAMS, SCALPER_BURST_PARAMS, SCALPER_VWAP_PARAMS,
                    SWING_PARAMS, Mode, StrategyParams, add_minutes,
                    params_for_mode)


# --------------------------------------------------------------------------- #
#  Indicator primitives  (input: OHLCV columns open, high, low, close, volume)
# --------------------------------------------------------------------------- #
def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period).mean()


def vwap(df: pd.DataFrame) -> pd.Series:
    """Session VWAP. Resets each calendar day so intraday VWAP is meaningful."""
    typical = (df["high"] + df["low"] + df["close"]) / 3.0
    tpv = typical * df["volume"]
    if isinstance(df.index, pd.DatetimeIndex):
        day = df.index.normalize()
        cum_tpv = tpv.groupby(day).cumsum()
        cum_vol = df["volume"].groupby(day).cumsum()
    else:
        cum_tpv = tpv.cumsum()
        cum_vol = df["volume"].cumsum()
    return cum_tpv / cum_vol.replace(0, np.nan)


def macd(series: pd.Series, fast: int, slow: int, signal: int):
    macd_line = ema(series, fast) - ema(series, slow)
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def rsi(series: pd.Series, period: int) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def bollinger(series: pd.Series, period: int, num_std: float):
    mid = sma(series, period)
    std = series.rolling(period).std(ddof=0)
    upper = mid + num_std * std
    lower = mid - num_std * std
    return upper, mid, lower


def atr(df: pd.DataFrame, period: int) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


# --------------------------------------------------------------------------- #
#  Indicator enrichment — adds all columns a strategy might read.
# --------------------------------------------------------------------------- #
def enrich(df: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
    out = df.copy()
    close = out["close"]

    # Intraday indicators
    out["vwap"] = vwap(out)
    m, s, h = macd(close, params.macd_fast, params.macd_slow, params.macd_signal)
    out["macd"], out["macd_signal"], out["macd_hist"] = m, s, h

    # Swing indicators
    out["ema_trend"] = ema(close, params.ema_trend)
    # Fast dynamic trend filter (Volume Burst)
    out["ema_fast"] = ema(close, params.ema_fast)
    out["rsi"] = rsi(close, params.rsi_period)
    ub, mb, lb = bollinger(close, params.bb_period, params.bb_std)
    out["bb_upper"], out["bb_mid"], out["bb_lower"] = ub, mb, lb
    out["vol_sma"] = sma(out["volume"], params.vol_sma)

    out["atr"] = atr(out, params.atr_period)
    return out


# --------------------------------------------------------------------------- #
#  Signal object
# --------------------------------------------------------------------------- #
@dataclass
class Signal:
    # "BUY" (long) or "SELL" (short). Intraday/Swing emit BUY only; the Scalper is
    # explicitly two-sided (params.allow_short). Everything downstream — sizing,
    # exits, PnL — reads `side` rather than assuming long.
    side: str
    entry_price: float
    stop_loss: float
    target: float
    reason: str

    @property
    def is_long(self) -> bool:
        return self.side == "BUY"

    @property
    def risk_reward(self) -> float:
        risk = abs(self.entry_price - self.stop_loss)
        reward = abs(self.target - self.entry_price)
        return reward / risk if risk else 0.0


def _crossed_above(series: pd.Series, other: pd.Series) -> bool:
    """True if `series` crossed above `other` on the latest bar."""
    if len(series) < 2:
        return False
    return series.iloc[-2] <= other.iloc[-2] and series.iloc[-1] > other.iloc[-1]


# --------------------------------------------------------------------------- #
#  Entry logic
# --------------------------------------------------------------------------- #
def intraday_signal(df: pd.DataFrame, params: StrategyParams) -> Optional[Signal]:
    """
    Long entry (Intraday, 15m):
      Price above VWAP  AND  MACD line crosses above its Signal line.
    Stop = ATR-based (recent structure); Target = 1:2 RR.
    """
    if len(df) < max(params.macd_slow, params.atr_period) + 2:
        return None
    last = df.iloc[-1]
    price_above_vwap = last["close"] > last["vwap"]
    macd_cross = _crossed_above(df["macd"], df["macd_signal"])
    if not (price_above_vwap and macd_cross):
        return None

    entry = float(last["close"])
    atr_val = float(last["atr"]) if not np.isnan(last["atr"]) else entry * 0.005
    stop = entry - params.atr_sl_mult * atr_val
    if stop >= entry:
        return None
    target = entry + params.risk_reward * (entry - stop)   # enforces 1:2
    return Signal("BUY", entry, stop, target,
                  "Price>VWAP + MACD bullish cross")


def swing_signal(df: pd.DataFrame, params: StrategyParams) -> Optional[Signal]:
    """
    Long entry (Swing, Daily). The Bollinger reclaim/bounce is the TRIGGER event;
    trend, momentum and volume are confirming filters. (Requiring the RSI-cross
    and the band reclaim to land on the exact same bar almost never happens, so
    momentum is treated as a state — RSI in breakout territory above 50.)
      Trend:      close > 200 EMA
      Momentum:   RSI above 50 (breakout territory)
      Volatility: close reclaims the middle band, or bounces off the lower band  <-- trigger
      Volume:     entry candle volume strictly > 20-period volume SMA
    Stop = ATR-based; Target = 1:3 RR.
    """
    if len(df) < params.ema_trend + 2:
        return None
    last = df.iloc[-1]
    prev = df.iloc[-2]

    trend_ok = last["close"] > last["ema_trend"]
    momentum_ok = last["rsi"] > params.rsi_breakout            # RSI > 50 (state)
    # Trigger event: reclaiming the middle band from below, or bouncing off lower.
    volatility_trigger = (
        (prev["close"] <= prev["bb_mid"] and last["close"] > last["bb_mid"])
        or (prev["low"] <= prev["bb_lower"] and last["close"] > prev["close"])
    )
    volume_ok = last["volume"] > last["vol_sma"]

    if not (trend_ok and momentum_ok and volatility_trigger and volume_ok):
        return None

    entry = float(last["close"])
    atr_val = float(last["atr"]) if not np.isnan(last["atr"]) else entry * 0.01
    stop = entry - params.atr_sl_mult * atr_val
    if stop >= entry:
        return None
    target = entry + params.risk_reward * (entry - stop)   # enforces 1:3
    return Signal("BUY", entry, stop, target,
                  "Trend+Momentum+Volatility+Volume aligned")


# --------------------------------------------------------------------------- #
#  Scalper (1-minute VWAP + ATR) — see scalping.md
# --------------------------------------------------------------------------- #
def _atr_in_normal_range(df: pd.DataFrame, params: StrategyParams) -> bool:
    """Volatility check (scalping.md §1): the ATR must sit in a 'normal' band for
    this instrument before we trust it to size a stop. Referencing the ATR's own
    rolling median makes the band self-calibrating per instrument, so GOLD and
    SBIN are judged on their own volatility rather than one hardcoded number.
    Filters out both dead tape and erratic spikes."""
    a = df["atr"].dropna()
    if len(a) < max(10, params.atr_period + 2):
        return False
    med = float(a.tail(params.atr_median_window).median())
    cur = float(a.iloc[-1])
    if not np.isfinite(med) or med <= 0 or not np.isfinite(cur) or cur <= 0:
        return False
    return params.atr_norm_low * med <= cur <= params.atr_norm_high * med


def _past_entry_window(df: pd.DataFrame, params: StrategyParams,
                       session_open: Optional[dtime]) -> bool:
    """Time filter (scalping.md §4): ignore the first N minutes of the session.
    `session_open` comes from the instrument's segment, so equity (09:15 -> 09:30)
    and MCX (09:00 -> 09:15) each skip their own open."""
    if session_open is None or not params.entry_skip_minutes:
        return True
    if not isinstance(df.index, pd.DatetimeIndex) or df.empty:
        return True
    return df.index[-1].time() >= add_minutes(session_open, params.entry_skip_minutes)


def _hybrid_stop(df: pd.DataFrame, entry: float, atr_val: float,
                 params: StrategyParams, is_long: bool) -> float:
    """Hybrid stop = the STRICTER of a volatility stop and a structural stop.

      Volatility: entry -/+ atr_sl_mult × ATR — self-calibrating to how much this
                  instrument is currently moving.
      Structural: the lowest low (long) / highest high (short) of the last
                  struct_lookback candles — a level the market actually respected.

    For a long the stop is the LOWER of the two (min), so it sits below both the
    volatility band AND the recent swing low; for a short it's the HIGHER (max).
    Taking the further of the two keeps the stop from landing inside noise the
    structure has already shown price can reach. Tick-rounding happens later, at
    order time, where the instrument's tick_size is known (keeps this function
    instrument-agnostic — Immutable Rule #3)."""
    n = max(int(params.struct_lookback), 1)
    if is_long:
        vol_stop = entry - params.atr_sl_mult * atr_val
        structural = float(df["low"].tail(n).min())
        return min(vol_stop, structural)
    vol_stop = entry + params.atr_sl_mult * atr_val
    structural = float(df["high"].tail(n).max())
    return max(vol_stop, structural)


def scalper_signal(df: pd.DataFrame, params: StrategyParams,
                   session_open: Optional[dtime] = None) -> Optional[Signal]:
    """
    VWAP-ATR scalp on 1-minute candles. Two-sided.

      Long : price consistently ABOVE VWAP -> pull-back touches/dips below VWAP
             -> bullish candle closes back above VWAP with high > previous high.
      Short: mirrored (below VWAP, pull-back rises to/above it, bearish candle
             closes with low < previous low).

    Stop is the HYBRID of 1.5×ATR(7) and the 10-bar structural extreme (see
    _hybrid_stop); TP mirrors that final stop distance for a hard 1:1.
    """
    need = max(params.atr_period, params.context_bars,
               params.pullback_lookback) + 2
    if len(df) < need:
        return None
    last, prev = df.iloc[-1], df.iloc[-2]
    # Guard the trigger bar's own indicators. _atr_in_normal_range() reads the
    # last non-NaN ATR, which is not necessarily this bar's, so check it here —
    # a NaN would slip past the `atr_val <= 0` test and poison the stop.
    if not (np.isfinite(last.get("vwap", np.nan))
            and np.isfinite(last.get("atr", np.nan))):
        return None
    if not _past_entry_window(df, params, session_open):
        return None
    if params.use_atr_gate and not _atr_in_normal_range(df, params):
        return None

    atr_val = float(last["atr"])
    entry = float(last["close"])
    if atr_val <= 0 or entry <= 0:
        return None

    # Trend context: how much of the recent window sat on each side of VWAP.
    ctx = df.tail(params.context_bars)
    frac_above = float((ctx["close"] > ctx["vwap"]).mean())
    frac_below = float((ctx["close"] < ctx["vwap"]).mean())
    # The pull-back must have happened BEFORE the trigger bar, so exclude it.
    pull = df.iloc[-(params.pullback_lookback + 1):-1]

    # -- Long ---------------------------------------------------------------- #
    if (frac_above >= params.context_min_frac
            and last["close"] > last["vwap"]                  # back above VWAP
            and bool((pull["low"] <= pull["vwap"]).any())     # pulled back to it
            and last["close"] > last["open"]                  # bullish candle
            and last["high"] > prev["high"]):                 # momentum breakout
        stop = _hybrid_stop(df, entry, atr_val, params, is_long=True)
        if stop >= entry:
            return None
        target = entry + params.risk_reward * (entry - stop)  # enforces 1:1
        return Signal("BUY", entry, stop, target,
                      f"VWAP pull-back + bullish breakout (ATR {atr_val:.2f})")

    # -- Short --------------------------------------------------------------- #
    if (params.allow_short
            and frac_below >= params.context_min_frac
            and last["close"] < last["vwap"]                  # back below VWAP
            and bool((pull["high"] >= pull["vwap"]).any())    # pulled up to it
            and last["close"] < last["open"]                  # bearish candle
            and last["low"] < prev["low"]):                   # momentum breakdown
        stop = _hybrid_stop(df, entry, atr_val, params, is_long=False)
        if stop <= entry:
            return None
        target = entry - params.risk_reward * (stop - entry)  # enforces 1:1
        if target <= 0:
            return None
        return Signal("SELL", entry, stop, target,
                      f"VWAP pull-back + bearish breakdown (ATR {atr_val:.2f})")

    return None


# --------------------------------------------------------------------------- #
#  Volume-Burst Momentum (1-minute) — coil + volume-confirmed breakout
# --------------------------------------------------------------------------- #
def _consolidation_range(df: pd.DataFrame, params: StrategyParams
                         ) -> Optional[tuple[float, float, int]]:
    """Find the "coil" — a cluster of small-bodied candles sitting immediately
    BEFORE the trigger bar — and return (range_high, range_low, n_bars).

    "Small body" is measured against ATR rather than a fixed number of rupees, so
    the same definition works on a ₹800 bank stock and a ₹1.4L gold contract.
    Longest cluster wins (5 down to 3): a wider coil is a stronger base, and its
    range is the level the breakout must actually clear.
    """
    atr_val = float(df["atr"].iloc[-1])
    if not np.isfinite(atr_val) or atr_val <= 0:
        return None
    max_body = params.small_body_atr * atr_val
    for n in range(params.consolidation_max, params.consolidation_min - 1, -1):
        if len(df) < n + 2:
            continue
        cluster = df.iloc[-(n + 1):-1]          # excludes the trigger bar
        bodies = (cluster["close"] - cluster["open"]).abs()
        if bool((bodies <= max_body).all()):
            return (float(cluster["high"].max()), float(cluster["low"].min()), n)
    return None


def volume_burst_signal(df: pd.DataFrame, params: StrategyParams,
                        session_open: Optional[dtime] = None) -> Optional[Signal]:
    """
    Volume-Burst Momentum. Two-sided.

      Long : close > 20 EMA, price has been coiling (3-5 small-bodied candles),
             and this candle CLOSES above the coil's high on volume greater than
             the average of the previous 10 candles.
      Short: mirrored (below the 20 EMA, closing below the coil's low).

    The volume surge is what separates a real momentum shift from a fake-out —
    without it this is just a range breakout.

    Stop is the HYBRID of 1.5×ATR(7) and the 10-bar structural extreme (see
    _hybrid_stop); TP mirrors that final stop distance for a hard 1:1.
    """
    need = max(params.ema_fast, params.atr_period, params.vol_avg_period,
               params.consolidation_max) + 2
    if len(df) < need:
        return None
    last = df.iloc[-1]
    if not (np.isfinite(last.get("ema_fast", np.nan))
            and np.isfinite(last.get("atr", np.nan))):
        return None
    if not _past_entry_window(df, params, session_open):
        return None
    if params.use_atr_gate and not _atr_in_normal_range(df, params):
        return None

    atr_val = float(last["atr"])
    entry = float(last["close"])
    if atr_val <= 0 or entry <= 0:
        return None

    # Volume burst: this candle must beat the mean of the PREVIOUS n candles
    # (excluding itself — including it would dilute the very spike we're testing).
    prev_vol = df["volume"].iloc[-(params.vol_avg_period + 1):-1]
    avg_vol = float(prev_vol.mean())
    if not np.isfinite(avg_vol) or avg_vol <= 0:
        return None
    vol = float(last["volume"])
    if vol <= avg_vol:
        return None

    coil = _consolidation_range(df, params)
    if coil is None:
        return None
    coil_hi, coil_lo, n_bars = coil

    # -- Long ---------------------------------------------------------------- #
    if last["close"] > last["ema_fast"] and entry > coil_hi:
        stop = _hybrid_stop(df, entry, atr_val, params, is_long=True)
        if stop >= entry:
            return None
        target = entry + params.risk_reward * (entry - stop)   # enforces 1:1
        return Signal("BUY", entry, stop, target,
                      f"Volume burst: {n_bars}-bar coil broken above "
                      f"{coil_hi:.2f} on {vol / avg_vol:.1f}x avg volume")

    # -- Short --------------------------------------------------------------- #
    if (params.allow_short and last["close"] < last["ema_fast"]
            and entry < coil_lo):
        stop = _hybrid_stop(df, entry, atr_val, params, is_long=False)
        if stop <= entry:
            return None
        target = entry - params.risk_reward * (stop - entry)    # enforces 1:1
        if target <= 0:
            return None
        return Signal("SELL", entry, stop, target,
                      f"Volume burst: {n_bars}-bar coil broken below "
                      f"{coil_lo:.2f} on {vol / avg_vol:.1f}x avg volume")

    return None


# --------------------------------------------------------------------------- #
#  Strategy registry
#
#  A Mode fixes the TIMEFRAME (and therefore the data feed); it does not fix the
#  strategy. Several strategies can share a mode — the 1-minute Scalper hosts both
#  the VWAP pull-back and the Volume Burst. The UI lists what's registered for the
#  chosen mode and the engine trades whichever key it's handed.
#
#  To add a strategy: write fn(df, params, session_open) -> Optional[Signal],
#  add its StrategyParams in config.py, and register() it below. Nothing else in
#  the system needs to change — that's the point of the registry.
# --------------------------------------------------------------------------- #
SignalFn = Callable[[pd.DataFrame, StrategyParams, Optional[dtime]],
                    Optional[Signal]]


@dataclass(frozen=True)
class StrategyDef:
    """A strategy as REGISTERED — potentially valid on several timeframes.

    `params_by_mode` is the declaration of which modes a strategy supports, and
    with what settings on each. Most strategies name exactly one mode: the VWAP
    pull-back is meaningless on daily candles. Some are timeframe-agnostic — a
    Bullish Engulfing is the same geometry on 1m and 1d — and those name all
    three. There is no separate "modes" list to fall out of sync: a mode is
    supported iff there are params for it, because a mode without params could
    not be traded anyway.
    """
    key: str                 # stable id, stored on trades and used by the UI
    name: str                # human label for the dropdown
    params_by_mode: dict[Mode, StrategyParams]
    fn: SignalFn
    summary: str             # one line shown under the dropdown

    @property
    def modes(self) -> tuple[Mode, ...]:
        return tuple(self.params_by_mode)

    def supports(self, mode: Mode) -> bool:
        return mode in self.params_by_mode

    def bind(self, mode: Mode) -> "BoundStrategy":
        return BoundStrategy(key=self.key, name=self.name, mode=mode,
                             params=self.params_by_mode[mode], fn=self.fn,
                             summary=self.summary)


@dataclass(frozen=True)
class BoundStrategy:
    """A StrategyDef resolved to ONE mode, so `.params` is unambiguous.

    Everything downstream (engine, backtester, UI) is handed one of these rather
    than a StrategyDef, which is why none of them need to know that a strategy can
    span modes at all — they still just read .key/.name/.params/.fn.
    """
    key: str
    name: str
    mode: Mode
    params: StrategyParams
    fn: SignalFn
    summary: str


_REGISTRY: dict[str, StrategyDef] = {}


def register(sd: StrategyDef) -> StrategyDef:
    """Add a strategy to the registry. Idempotent by key, so a module that gets
    imported twice re-registers rather than duplicating."""
    for mode, p in sd.params_by_mode.items():
        if p.mode != mode:
            raise ValueError(
                f"{sd.key}: params filed under {mode.value} declare "
                f"mode={p.mode.value}. The engine reads params.timeframe to pick "
                f"the feed, so a mismatch here trades the wrong candles.")
    _REGISTRY[sd.key] = sd
    return sd


def _no_session(fn) -> SignalFn:
    """Adapt a fn(df, params) to the uniform registry signature. Intraday and
    Swing have no session-time filter, so they simply ignore session_open."""
    def wrapped(df: pd.DataFrame, params: StrategyParams,
                session_open: Optional[dtime] = None) -> Optional[Signal]:
        return fn(df, params)
    return wrapped


register(StrategyDef(
    key="intraday_vwap_macd", name="VWAP + MACD cross",
    params_by_mode={Mode.INTRADAY: INTRADAY_PARAMS},
    fn=_no_session(intraday_signal),
    summary="Long when price is above VWAP and MACD crosses up. 1:2 RR, long only.",
))
register(StrategyDef(
    key="swing_trend_momentum", name="Trend + Momentum + Volatility + Volume",
    params_by_mode={Mode.SWING: SWING_PARAMS}, fn=_no_session(swing_signal),
    summary="Long on a Bollinger reclaim above the 200 EMA with RSI>50 and "
            "above-average volume. 1:3 RR, long only.",
))
register(StrategyDef(
    key="scalp_vwap_atr", name="VWAP-ATR Pull-back",
    params_by_mode={Mode.SCALPER: SCALPER_VWAP_PARAMS}, fn=scalper_signal,
    summary="Waits for price to pull back to VWAP, then breaks out with "
            "momentum. SL/TP = 1.0×ATR(7), 1:1 RR, long+short. Selective.",
))
register(StrategyDef(
    key="scalp_volume_burst", name="Volume-Burst Momentum",
    params_by_mode={Mode.SCALPER: SCALPER_BURST_PARAMS}, fn=volume_burst_signal,
    summary="Fires when a 3-5 candle coil breaks on above-average volume, in "
            "the direction of the 20 EMA. SL/TP = 0.8×ATR(7), 1:1 RR, "
            "long+short. Triggers more often than the pull-back.",
))

# The strategy each mode runs when the caller doesn't name one. Keeps every
# existing call site (which only knows about Mode) working unchanged.
_DEFAULT_BY_MODE = {
    Mode.INTRADAY: "intraday_vwap_macd",
    Mode.SWING: "swing_trend_momentum",
    Mode.SCALPER: "scalp_vwap_atr",
}


def strategies_for_mode(mode: Mode) -> list[BoundStrategy]:
    """Everything tradable on this timeframe, ready to run. This is what fills the
    UI dropdown, so registering a strategy is all it takes to make it selectable."""
    return [s.bind(mode) for s in _REGISTRY.values() if s.supports(mode)]


def get_strategy(key: str, mode: Optional[Mode] = None) -> Optional[BoundStrategy]:
    """Look a strategy up by key. `mode` disambiguates multi-mode strategies; with
    it omitted the strategy binds to the first mode it declared, which is exact
    for the single-mode majority."""
    sd = _REGISTRY.get(key)
    if sd is None:
        return None
    if mode is None:
        return sd.bind(sd.modes[0])
    return sd.bind(mode) if sd.supports(mode) else None


def default_strategy(mode: Mode) -> BoundStrategy:
    return _REGISTRY[_DEFAULT_BY_MODE[mode]].bind(mode)


def resolve_strategy(mode: Mode, key: str = "") -> BoundStrategy:
    """The strategy named by `key`, falling back to the mode's default.

    A key that does not SUPPORT this mode is ignored rather than honoured: running
    a 1-minute strategy against 15-minute candles would silently trade the wrong
    timeframe, which is worse than not honouring the request. A strategy that
    declares the mode is bound to that mode's own params, so the same logic runs
    at 1:1 on the Scalper and 1:3 on Swing without the caller arranging anything.
    """
    sd = _REGISTRY.get(key)
    if sd is not None and sd.supports(mode):
        return sd.bind(mode)
    return default_strategy(mode)


def run_strategy(sd: BoundStrategy, df: pd.DataFrame,
                 session_open: Optional[dtime] = None) -> Optional[Signal]:
    """Enrich with indicators, then evaluate one strategy."""
    return sd.fn(enrich(df, sd.params), sd.params, session_open)


def generate_signal(df: pd.DataFrame, mode: Mode,
                    session_open: Optional[dtime] = None,
                    strategy_key: str = "") -> Optional[Signal]:
    """Dispatch to a strategy after enriching with indicators. With no
    strategy_key this runs the mode's default, preserving old behaviour."""
    return run_strategy(resolve_strategy(mode, strategy_key), df, session_open)


# --------------------------------------------------------------------------- #
#  Position sizing — the risk core. Never exceed the per-trade risk cap.
# --------------------------------------------------------------------------- #
def position_size(
    total_capital: float,
    signal: Signal,
    params: StrategyParams,
    lot_size: int = 1,
    contract_multiplier: int = 1,
    max_leverage: Optional[float] = None,
    account_capital: Optional[float] = None,
) -> tuple[int, float]:
    """
    Returns (quantity, risk_amount).

    Quantity is the SMALLEST of these independent limits:

      1. Risk limit     — risk_capital / (|entry - stop| * contract_multiplier)
      2. Notional limit — (total_capital * max_leverage) / (entry * multiplier)
      3. Capital limit  — (account_capital * max_capital_per_trade_pct)
                          / (entry * multiplier)   [only if the pct is set]

    Limit 3 caps the CAPITAL a single trade deploys (its notional) at a fixed %
    of the ACCOUNT, independent of risk. Risk sizing bounds the loss; it does NOT
    bound how much cash the position commits — a tight stop can buy a quantity
    whose value swallows the whole account while risking only 1%. Limit 3 is what
    stops that. It uses `account_capital` (the full account) when provided so
    "20% of account" means the same no matter what is already open; callers that
    omit it fall back to `total_capital`.

    risk_capital is params.risk_per_trade × total_capital, optionally capped by a
    fixed rupee ceiling (params.risk_per_trade_cash) — min() of the two, so the
    per-mode % stays the hard maximum. Pass the AVAILABLE capital (total minus
    margin already committed to open positions) as total_capital and limit 2
    doubles as the "can the account fund this?" guard.

    `max_leverage` defaults to the mode's ceiling; callers that know the
    instrument should pass config.max_leverage_for(segment, params), since equity
    (~5x MIS) and MCX futures (~15x) fund very differently.

    `abs()` makes limit 1 direction-agnostic: a short's stop sits above its entry,
    and it must risk the same cash as the equivalent long.

    contract_multiplier converts a price move into rupees (1 for equity). For the
    Scalper |entry - stop| is exactly 1.0 x ATR, so limit 1 is scalping.md's
    `Quantity = Risk_Amount / (ATR * Contract_Multiplier)` — the formula that
    holds cash risk constant no matter what volatility does.

    Limit 2 exists because limit 1 alone is unbounded as the stop tightens: a
    1-minute ATR under a rupee sizes crores of notional against lakhs of capital.
    That position is correct on paper risk and impossible to fund — the broker
    rejects it, or worse, fills it. Capping notional is what makes ATR sizing
    safe on fast timeframes.

    For commodities quantity is rounded down to whole lots. The realised risk is
    recomputed from the rounded quantity so the reported risk never overstates.
    """
    per_unit_risk = abs(signal.entry_price - signal.stop_loss)
    if per_unit_risk <= 0 or not np.isfinite(per_unit_risk):
        return 0, 0.0
    mult = max(int(contract_multiplier), 1)
    rupees_per_unit = per_unit_risk * mult
    # Risk budget: the per-mode % of capital, OPTIONALLY capped by a fixed cash
    # amount. The fixed figure is a CEILING, never an override — taking the min
    # guarantees the immutable per-mode % (Rule #1) is the hard upper bound, so a
    # ₹2000 setting still yields only 1% (₹1000) on ₹1L of scalper capital.
    pct_risk = total_capital * params.risk_per_trade
    if params.risk_per_trade_cash and params.risk_per_trade_cash > 0:
        risk_capital = min(params.risk_per_trade_cash, pct_risk)
    else:
        risk_capital = pct_risk
    qty_by_risk = risk_capital / rupees_per_unit

    lev = params.max_leverage if max_leverage is None else max_leverage
    raw_qty = qty_by_risk
    if signal.entry_price > 0 and lev > 0:
        qty_by_notional = (total_capital * lev) / (signal.entry_price * mult)
        raw_qty = min(raw_qty, qty_by_notional)

    # Limit 3: cap the capital this single trade deploys at a % of the ACCOUNT.
    # Uses the full account capital when the caller supplies it (the engine sizes
    # against AVAILABLE capital, but "20% of account" must not shrink as capital
    # gets tied up), else falls back to the capital passed in.
    cap_pct = getattr(params, "max_capital_per_trade_pct", 0.0)
    if cap_pct and cap_pct > 0 and signal.entry_price > 0:
        cap_base = account_capital if account_capital is not None else total_capital
        qty_by_capital = (cap_base * cap_pct) / (signal.entry_price * mult)
        raw_qty = min(raw_qty, qty_by_capital)

    # raw_qty is ALREADY a lot count: every limit above divides the rupee budget by
    # (price|risk × contract_multiplier), and contract_multiplier is the PER-LOT
    # point value (₹ per ₹1 move for one whole contract), so the quotient counts
    # LOTS — for equity that is shares, since its multiplier and lot_size are 1.
    #
    # We deliberately do NOT multiply by lot_size. `quantity` is carried as a lot
    # count everywhere downstream (broker order, margin, notional, PnL) because the
    # broker's own margin/order APIs treat MCX quantity that way (quantity=1 => one
    # lot). Multiplying by lot_size here — with contract_multiplier ALSO applied
    # downstream — double-counted contract size and inflated PnL/notional/risk by
    # lot_size for every contract whose lot_size != 1.
    qty = int(raw_qty)                       # whole lots (equity: whole shares)
    realized_risk = qty * rupees_per_unit
    return max(qty, 0), realized_risk


# --------------------------------------------------------------------------- #
#  Plugin loading — every module in strategies/ is imported for its register()
#  side effect. Dropping a file in that folder is the ONLY step needed to make a
#  strategy appear in the UI, the engine and the backtester.
#
#  This runs last on purpose. A plugin does `from strategy import register, ...`,
#  so by the time it executes this module must be fully defined — importing from
#  the bottom guarantees that and keeps the cycle harmless.
# --------------------------------------------------------------------------- #
def _load_strategy_plugins() -> list[str]:
    """Import strategies/*.py. Returns the names that failed, so a broken plugin
    is reported rather than silently absent — a strategy that quietly vanishes
    from the dropdown is a much worse failure than a loud import error."""
    import importlib
    import pkgutil
    from pathlib import Path

    failed: list[str] = []
    pkg_dir = Path(__file__).resolve().parent / "strategies"
    if not pkg_dir.is_dir():
        return failed
    for info in pkgutil.iter_modules([str(pkg_dir)]):
        # Leading underscore = a shared helper, not a strategy.
        if info.name.startswith("_"):
            continue
        try:
            importlib.import_module(f"strategies.{info.name}")
        except Exception as exc:                       # noqa: BLE001
            failed.append(f"{info.name}: {exc}")
    return failed


PLUGIN_ERRORS: list[str] = _load_strategy_plugins()
