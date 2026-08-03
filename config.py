"""
config.py
Central configuration: environment loading, the tradable instrument universe
(NSE equity + MCX commodities), market hours, and strategy parameters.

Nothing here talks to a broker or a database — it is pure configuration so the
rest of the system can stay decoupled (Immutable Rule #3).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import time
from enum import Enum

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:  # dotenv is optional; env can be set by the OS instead
    pass


# --------------------------------------------------------------------------- #
#  Enums for the two "axes" of the system
# --------------------------------------------------------------------------- #
class Mode(str, Enum):
    INTRADAY = "Intraday"
    SWING = "Swing"
    SCALPER = "Scalper"      # aggressive 1-minute VWAP-ATR scalping


class Environment(str, Enum):
    PAPER = "Paper"
    LIVE = "Live"


class Broker(str, Enum):
    UPSTOX = "Upstox"
    DHAN = "Dhan"
    ZERODHA = "Zerodha"
    KOTAK = "Kotak Neo"
    SIMULATED = "Simulated"   # used automatically when no credentials exist


class Segment(str, Enum):
    EQUITY = "NSE_EQUITY"
    MCX = "MCX_COMMODITY"


# --------------------------------------------------------------------------- #
#  Environment variables
# --------------------------------------------------------------------------- #
def _env(key: str, default: str = "") -> str:
    return os.getenv(key, default).strip()


MONGO_URI = _env("MONGO_URI", "mongodb://localhost:27017/")
MONGO_DB_NAME = _env("MONGO_DB_NAME", "trading_bot")

# Connection-pool ceiling for the single shared MongoClient (mongo_client.py).
# pymongo's own default is 100 sockets, each TLS-wrapped to Atlas — far more
# than this workload needs and a real cost on a 1 GB server. A handful of
# concurrent users and one engine thread never exceed a pool of 5.
try:
    MONGO_MAX_POOL_SIZE = max(1, int(_env("MONGO_MAX_POOL_SIZE", "5")))
except ValueError:
    MONGO_MAX_POOL_SIZE = 5

UPSTOX_SANDBOX_TOKEN = _env("UPSTOX_SANDBOX_TOKEN")
UPSTOX_LIVE_ACCESS_TOKEN = _env("UPSTOX_LIVE_ACCESS_TOKEN")
UPSTOX_LIVE_API_KEY = _env("UPSTOX_LIVE_API_KEY")
UPSTOX_LIVE_SECRET = _env("UPSTOX_LIVE_SECRET")

DHAN_CLIENT_ID = _env("DHAN_CLIENT_ID")
DHAN_ACCESS_TOKEN = _env("DHAN_ACCESS_TOKEN")

# Zerodha (Kite Connect). ZERODHA_ACCESS_TOKEN is a DAILY token (Kite sessions
# expire ~06:00 IST every day) generated via the login flow in kite_auth.py /
# the sidebar's "Zerodha Token" panel — API_KEY/SECRET are the app credentials
# needed to run that exchange, not trading credentials themselves.
ZERODHA_API_KEY = _env("ZERODHA_API_KEY")
ZERODHA_API_SECRET = _env("ZERODHA_API_SECRET")
ZERODHA_ACCESS_TOKEN = _env("ZERODHA_ACCESS_TOKEN")

KOTAK_NEO_CONSUMER_KEY = _env("KOTAK_NEO_CONSUMER_KEY")
KOTAK_NEO_CONSUMER_SECRET = _env("KOTAK_NEO_CONSUMER_SECRET")
KOTAK_NEO_ACCESS_TOKEN = _env("KOTAK_NEO_ACCESS_TOKEN")

try:
    TOTAL_CAPITAL = float(_env("TOTAL_CAPITAL", "100000") or "100000")
except ValueError:
    TOTAL_CAPITAL = 100_000.0


# --------------------------------------------------------------------------- #
#  Instrument universe
#  `instrument_key` is the Upstox V3 style key; adapt per broker in broker_api.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Instrument:
    symbol: str            # human friendly name shown in the UI
    segment: Segment
    instrument_key: str    # broker feed subscription key (Upstox format shown)
    lot_size: int = 1      # commodities trade in lots; equity lot_size = 1
    tick_size: float = 0.05
    reference_price: float = 100.0   # only used to seed the simulated feed
    # Units of the underlying per 1 quoted price unit. Equity = 1 (₹1 move on 1
    # share = ₹1). MCX futures are quoted per small unit but contract a larger
    # one (GOLD quotes ₹/10g on a 1kg contract => 100), so a ₹1 price move is
    # worth ₹100. Scalper sizing divides by this so cash risk stays constant
    # (scalping.md: Quantity = Risk_Amount / (ATR * Contract_Multiplier)).
    contract_multiplier: int = 1


# NSE equities (cash) — the Nifty 100 universe ------------------------------- #
# Real, ISIN-based Upstox instrument_keys for the full Nifty 100 (Nifty 50 +
# Nifty Next 50), generated into nifty100_instruments.py by
# tools/refresh_nifty100.py. Imported HERE — after Instrument and Segment are
# defined — so the round-trip import (that module does `from config import
# Instrument, Segment`) resolves without a circular-import error. config stays
# the single source of the tradable universe.
#
# ISIN-based keys are stable (they don't expire like MCX futures), so this list
# only needs regenerating when index membership changes. Falls back to a small
# built-in set if the generated module is missing.
try:
    from nifty100_instruments import NIFTY100_INSTRUMENTS
    EQUITY_INSTRUMENTS = NIFTY100_INSTRUMENTS
except Exception:  # generated module absent — keep the bot runnable
    EQUITY_INSTRUMENTS = [
        Instrument("RELIANCE", Segment.EQUITY, "NSE_EQ|INE002A01018", 1, 0.05, 2900.0),
        Instrument("TCS",      Segment.EQUITY, "NSE_EQ|INE467B01029", 1, 0.05, 3850.0),
        Instrument("INFY",     Segment.EQUITY, "NSE_EQ|INE009A01021", 1, 0.05, 1550.0),
        Instrument("HDFCBANK", Segment.EQUITY, "NSE_EQ|INE040A01034", 1, 0.05, 1650.0),
        Instrument("SBIN",     Segment.EQUITY, "NSE_EQ|INE062A01020", 1, 0.05, 820.0),
    ]

# MCX commodities ------------------------------------------------------------ #
# REAL front-month futures instrument_keys pulled from the Upstox MCX instrument
# master, with real lot/tick sizes. Both FULL-size and MINI/MICRO contracts are
# included — the mini contracts (GOLDM, CRUDEOILM, SILVERM, SILVERMIC, NATGASMINI)
# track the same underlying but tie up a fraction of the margin, which is the
# whole point of trading them (CRUDEOILM ≈ ₹22k/lot vs CRUDEOIL ≈ ₹2.5L/lot).
#
# MCX futures keys EXPIRE. Regenerate them each expiry with tools/refresh_mcx.py,
# which downloads the live master and rolls every root to its nearest active
# future automatically, writing mcx_instruments.py. Mirroring the equity pattern,
# that generated module is preferred when present; the inline list below is the
# runnable fallback (kept current as of the last refresh).
#
# NOTE on live sizing: MCX futures carry a contract multiplier (e.g. GOLD is
# quoted per 10g on a 1kg contract). Intraday/Swing size on raw price distance,
# which is exact for equities only. The Scalper honours contract_multiplier, so
# its cash risk is correct for commodities too.
#
# ⚠️ VERIFY BEFORE LIVE COMMODITY TRADING: lot_size rounds the order quantity,
# while contract_multiplier converts a price move into rupees. Both are taken
# straight from the Upstox master (lot_size / qty_multiplier). Confirm against
# your broker's contract spec and margin before going live.
try:
    from mcx_instruments import MCX_INSTRUMENTS
except Exception:  # generated module absent — keep the bot runnable
    MCX_INSTRUMENTS = [
        #          symbol         segment      instrument_key    lot  tick  ref price   multiplier
        # --- Full-size contracts ---
        Instrument("GOLD",        Segment.MCX, "MCX_FO|466583", 1,    1.0,  142419.0,  100),   # 1 kg, quoted ₹/10g
        Instrument("CRUDEOIL",    Segment.MCX, "MCX_FO|520702", 100,  1.0,  7580.0,    100),   # 100 barrels, quoted ₹/barrel
        Instrument("NATURALGAS",  Segment.MCX, "MCX_FO|538685", 1250, 0.10, 279.7,     1250),  # 1250 mmBtu, quoted ₹/mmBtu
        Instrument("SILVER",      Segment.MCX, "MCX_FO|471725", 30,   1.0,  223320.0,  30),    # 30 kg, quoted ₹/kg
        # --- Mini / micro contracts (fractional size => fractional margin) ---
        Instrument("GOLDM",       Segment.MCX, "MCX_FO|555922", 100,  1.0,  142419.0,  10),    # 100 g, quoted ₹/10g
        Instrument("CRUDEOILM",   Segment.MCX, "MCX_FO|520703", 10,   1.0,  7580.0,    10),    # 10 barrels, quoted ₹/barrel
        Instrument("NATGASMINI",  Segment.MCX, "MCX_FO|538686", 250,  0.10, 279.7,     250),   # 250 mmBtu, quoted ₹/mmBtu
        Instrument("SILVERM",     Segment.MCX, "MCX_FO|471726", 5,    1.0,  223320.0,  5),     # 5 kg, quoted ₹/kg
        Instrument("SILVERMIC",   Segment.MCX, "MCX_FO|488788", 1,    1.0,  223320.0,  1),     # 1 kg, quoted ₹/kg
    ]

ALL_INSTRUMENTS = EQUITY_INSTRUMENTS + MCX_INSTRUMENTS
INSTRUMENTS_BY_SYMBOL = {i.symbol: i for i in ALL_INSTRUMENTS}


# --------------------------------------------------------------------------- #
#  MCX margin — hardcoded per-lot figures (user-provided).
#
#  Commodity margin is NOT a formula (notional ÷ leverage is nowhere near right):
#  it is the exchange's SPAN + Exposure margin plus SEBI peak-margin. For the
#  PAPER trading bot we deliberately use these FIXED per-lot figures rather than a
#  live broker call — they are the user's real broker-side reference values, they
#  are stable enough for paper simulation, and they avoid depending on a live
#  token being present. LIVE trading still prefers the broker's own margin API and
#  only falls back to this table (see engine._mcx_margin / broker.required_margin).
#
#  Values are rupees of margin for ONE lot (1 contract), taken as the mid-point of
#  the user-supplied broker ranges (e.g. CRUDEOIL ₹2.40–2.55L => ₹2.475L).
#
#  ⚠️ Real margins drift daily with volatility and SEBI peak-margin rules. Revisit
#  these periodically against your broker's margin calculator.
MCX_MARGIN_PER_LOT = {
    # --- Full-size contracts ---           # 1-lot size          user range
    "GOLD":       1_325_000.0,   # 1 kg (1000 g)      ₹13.00–13.50 L
    "SILVER":     1_075_000.0,   # 30 kg              ₹10.50–11.00 L
    "CRUDEOIL":     247_500.0,   # 100 barrels        ₹2.40–2.55 L
    "NATURALGAS":    55_000.0,   # 1250 mmBtu         ₹52–58 k
    # --- Mini / micro contracts ---
    "GOLDM":        132_500.0,   # 100 g              ₹1.30–1.35 L
    "SILVERM":      180_000.0,   # 5 kg               ₹1.75–1.85 L
    "CRUDEOILM":     24_750.0,   # 10 barrels         ₹24.0–25.5 k
    "NATGASMINI":    11_000.0,   # 250 mmBtu          ₹10.5–11.6 k
    "SILVERMIC":     36_000.0,   # 1 kg (1/30 of SILVER; not in user table)
}


def mcx_margin_per_lot(symbol: str) -> float:
    """Approximate fallback margin for ONE lot of an MCX symbol, or 0.0 if unknown.
    Used only when the live broker margin is unavailable (see the note above)."""
    return float(MCX_MARGIN_PER_LOT.get(symbol, 0.0))


def instruments_for_segment(segment: Segment) -> list[Instrument]:
    return [i for i in ALL_INSTRUMENTS if i.segment == segment]


# --------------------------------------------------------------------------- #
#  Market hours (IST). MCX stays open into the night — this is the whole point
#  of the commodity addition, so the engine must respect the later close.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class MarketHours:
    open_t: time
    close_t: time

    def is_open(self, now_t: time) -> bool:
        return self.open_t <= now_t <= self.close_t


# NSE equity: 09:15 - 15:30 IST
EQUITY_HOURS = MarketHours(time(9, 15), time(15, 30))
# MCX: 09:00 - 23:30 IST (winter session runs to 23:55; using 23:30 as a safe cutoff)
MCX_HOURS = MarketHours(time(9, 0), time(23, 30))


def market_hours_for_segment(segment: Segment) -> MarketHours:
    return MCX_HOURS if segment == Segment.MCX else EQUITY_HOURS


# Notional leverage a segment realistically supports, i.e. 1 / margin_rate.
#
# EQUITY is deliberately pinned to 1x — NO LEVERAGE. Position notional can never
# exceed the (available) cash backing it, so the account trades like a delivery /
# cash-and-carry book even intraday. This is a risk choice, not a broker limit:
# MIS would allow ~5x, but we decline it. Consequence: one position ties up its
# full notional as committed capital (see the engine's available-capital tracker).
#
# MCX futures keep ~15x (≈6-7% SPAN+ELM margin), because a GOLD contract carries
# ₹1.4cr notional — at 1x it could never be funded, so commodities would silently
# stop trading. 15x reflects the real margin a broker posts against the contract.
# These are conservative approximations, NOT your broker's actual numbers — they
# vary by broker, by scrip and by SEBI peak-margin rules.
#
# ⚠️ VERIFY THESE AGAINST YOUR BROKER BEFORE LIVE TRADING. They bound how large a
# position the bot will take; setting them too high invites margin calls.
SEGMENT_MAX_LEVERAGE = {Segment.EQUITY: 1.0, Segment.MCX: 15.0}


def max_leverage_for(segment: Segment, params: "StrategyParams") -> float:
    """Effective notional cap = the stricter of what the segment supports and
    what the mode allows. Swing holds overnight (delivery, 1x) so its mode cap
    wins everywhere; intraday modes defer to the segment."""
    return min(params.max_leverage, SEGMENT_MAX_LEVERAGE.get(segment, 1.0))


def add_minutes(t: time, minutes: int) -> time:
    """Wall-clock arithmetic (no date), for session filters like 'skip the first
    15 minutes'. Segment-aware by construction: equity opens 09:15 so it skips to
    09:30, MCX opens 09:00 so it skips to 09:15."""
    total = (t.hour * 60 + t.minute + minutes) % (24 * 60)
    return time(total // 60, total % 60)


# --------------------------------------------------------------------------- #
#  Strategy parameters — one place, enforcing the Immutable Risk Rules.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class StrategyParams:
    mode: Mode
    timeframe: str
    risk_per_trade: float      # fraction of capital risked per trade
    risk_reward: float         # reward : risk ratio (the "2" or "3" in 1:2 / 1:3)
    # Ceiling on position NOTIONAL, as a multiple of total capital.
    #
    # Risk-based sizing alone is not enough: qty = risk_budget / stop_distance, so
    # a tight stop implies a huge quantity. On 1-minute bars the ATR can be well
    # under a rupee, which sizes crores of notional against lakhs of capital —
    # correct on risk, impossible to actually fund.
    #
    # This is the MODE's ceiling; the effective cap is the stricter of this and
    # the segment's (see max_leverage_for). Swing sets 1.0 because overnight
    # positions are delivery; intraday modes leave the real limit to the segment.
    max_leverage: float = 1.0
    # Ceiling on the CAPITAL a single trade may deploy, as a fraction of the
    # ACCOUNT (not available capital). This is independent of the risk cap:
    # risk-based sizing controls how much you LOSE if the stop hits, but says
    # nothing about how much cash the position commits. A tight stop on a
    # high-priced stock can size a quantity whose notional swallows the whole
    # account while still risking only 1%. This caps that: notional per trade
    # <= account × this fraction. 0.20 => at most 20% of the account in one
    # name (~5 concurrent positions). 0 disables it (unlimited, the old
    # behaviour). Applied as a THIRD limit in position_size, min() with the rest.
    max_capital_per_trade_pct: float = 0.0
    # Intraday indicator params
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    # Swing indicator params
    ema_trend: int = 200
    ema_fast: int = 20          # fast dynamic trend filter (Volume Burst)
    rsi_period: int = 14
    rsi_breakout: float = 50.0
    bb_period: int = 20
    bb_std: float = 2.0
    vol_sma: int = 20
    atr_period: int = 14
    atr_sl_mult: float = 1.5   # volatility stop distance = atr_sl_mult * ATR

    # -- Hybrid stop-loss + fixed-cash risk (Scalper) ------------------------ #
    # A fixed CASH amount to risk per trade (e.g. ₹2000). Kept as a CEILING, not
    # an override: sizing risks min(risk_per_trade_cash, capital × risk_per_trade),
    # so the per-mode % (Immutable Rule #1) can never be exceeded. On ₹1L capital a
    # 1% scalper cap (₹1000) still wins over ₹2000; the fixed figure only bites
    # once capital is large enough that the % would otherwise risk more. 0 disables
    # it => pure percentage sizing (Intraday/Swing keep their existing behaviour).
    risk_per_trade_cash: float = 0.0
    # Structural stop look-back. The hybrid stop is the STRICTER of the volatility
    # stop (above) and market structure: the lowest low (longs) / highest high
    # (shorts) of the last N candles, so the stop sits beyond a real swing point
    # rather than at an arbitrary ATR multiple. Only the Scalper's VWAP pull-back
    # and Volume-Burst use it; every other strategy ignores it.
    struct_lookback: int = 10

    # -- Scalper-only knobs (left inert for Intraday/Swing) ------------------ #
    allow_short: bool = False       # Intraday/Swing stay long-only by design
    entry_skip_minutes: int = 0     # ignore the first N min of the session
    max_hold_minutes: int = 0       # 0 = no time-based exit
    # Bars that must CLOSE after an entry or exit before the same symbol may be
    # traded again. Without this the bot re-enters the identical setup off the
    # identical candle the instant a position closes — with a time exit that
    # becomes a loss loop (enter, time-exit, re-enter, repeat). 1 = require at
    # least one fresh bar, which also blocks trading on a stalled feed.
    reentry_cooldown_bars: int = 1
    pullback_lookback: int = 5      # bars scanned for the VWAP pull-back
    context_bars: int = 10          # bars used to judge "consistently above VWAP"
    context_min_frac: float = 0.6   # fraction of them that must be on that side
    use_atr_gate: bool = False      # require ATR to sit in a "normal" band
    atr_median_window: int = 50     # window for the ATR "normal range" reference
    atr_norm_low: float = 0.5       # ATR must be >= this * median ATR
    atr_norm_high: float = 2.0      # ATR must be <= this * median ATR (skip spikes)
    # -- Volume Burst knobs -------------------------------------------------- #
    vol_avg_period: int = 10        # breakout volume must beat this many bars' mean
    consolidation_min: int = 3      # a "coil" is at least this many small candles
    consolidation_max: int = 5      # ...and at most this many
    small_body_atr: float = 0.5     # a body <= this * ATR counts as "small"
    use_limit_entry: bool = False   # enter with a limit inside the spread
    limit_offset_ticks: float = 1.0  # how far inside the spread a limit entry sits
    # -- Candlestick engine knobs (plan.md Phase 1) -------------------------- #
    # Only the TRADING knobs live here. The pattern geometry (what counts as a
    # doji, a hammer, an engulfing) is intrinsic to the pattern definition, not a
    # tuning dial, so it stays in strategies/candlestick_engine.py.
    cs_min_score: float = 3.0       # weighted pattern evidence needed to trade
    cs_trend_lookback: int = 10     # bars used to judge the trend BEFORE a pattern
    cs_sl_buffer_atr: float = 0.25  # stop sits this far beyond the pattern extreme
    cs_min_sl_atr: float = 0.5      # ...but never closer to entry than this
    cs_max_sl_atr: float = 3.0      # ...and never further than this


INTRADAY_PARAMS = StrategyParams(
    mode=Mode.INTRADAY, timeframe="15m",
    # 1% max loss per trade (₹1,000 on a ₹1L account). Scales with capital, and
    # stays well inside the 2% ceiling Immutable Rule #1 forbids exceeding.
    risk_per_trade=0.01, risk_reward=2.0,   # Hard 1:2
    max_leverage=15.0,                      # defer to the segment's real cap
    max_capital_per_trade_pct=0.20,         # <=20% of the account per trade
)
SWING_PARAMS = StrategyParams(
    mode=Mode.SWING, timeframe="1d",
    risk_per_trade=0.03, risk_reward=3.0,   # Hard 1:3, max 3% — Immutable Rule #1
    max_leverage=1.0,                       # positions held overnight = delivery
)
# -- Scalper (1-minute) strategies ------------------------------------------ #
# Risk per trade is deliberately 1% — HALF the Intraday cap. The Immutable Rules
# name only Intraday (2%) and Swing (3%), so this mode picks its own number; at
# scalping frequency a 2% risk compounds into ruin far faster than it does at
# 15m, and "Aggressive" here refers to trade frequency, not to risk per trade.
# 2% is the ceiling scalping strategies must never cross.

# 1) VWAP-ATR pull-back (scalping.md). Hard 1:1, but the stop is now HYBRID:
# the stricter of 1.5×ATR(7) and the 10-bar structural low/high (struct_lookback),
# so it sits beyond a real swing point. TP mirrors the final stop distance => 1:1.
# Risk per trade is a fixed ₹2000 ceiling, still bounded by the 1% cap above.
SCALPER_VWAP_PARAMS = StrategyParams(
    mode=Mode.SCALPER, timeframe="1m",
    risk_per_trade=0.01, risk_reward=1.0,
    atr_period=7, atr_sl_mult=1.5,
    risk_per_trade_cash=2000.0, struct_lookback=10,
    allow_short=True,           # the strategy is explicitly two-sided
    entry_skip_minutes=15,      # skip the open's price discovery
    max_hold_minutes=7,         # bail out of stagnant trades
    use_limit_entry=True,       # slippage would eat a 1:1 edge
    use_atr_gate=True,          # its spec calls for a volatility check
    max_leverage=15.0,          # defer to the segment; without ANY cap a
)                               # sub-rupee ATR sizes crores against lakhs

# 2) Volume-Burst momentum. Coil (3-5 small candles) breaks with a volume surge.
# Hard 1:1 with the same HYBRID stop as the VWAP strategy: the stricter of
# 1.5×ATR(7) and the 10-bar structural extreme. (This supersedes the earlier fixed
# 0.8×ATR stop — the structural leg now anchors the stop to the coil's own low/high
# instead of a bare volatility multiple.) No ATR band gate: its spec doesn't ask
# for one, and gating would undo the "trigger more often" intent.
SCALPER_BURST_PARAMS = StrategyParams(
    mode=Mode.SCALPER, timeframe="1m",
    risk_per_trade=0.01, risk_reward=1.0,
    atr_period=7, atr_sl_mult=1.5,
    risk_per_trade_cash=2000.0, struct_lookback=10,
    ema_fast=20,
    vol_avg_period=10,
    consolidation_min=3, consolidation_max=5, small_body_atr=0.5,
    allow_short=True,
    entry_skip_minutes=15,
    max_hold_minutes=7,
    use_limit_entry=True,
    use_atr_gate=False,
    max_leverage=15.0,
)

# Back-compat alias: the default Scalper params.
SCALPER_PARAMS = SCALPER_VWAP_PARAMS


# --------------------------------------------------------------------------- #
#  Candlestick engine (plan.md Phase 1) — one param set PER TIMEFRAME.
#
#  This strategy is registered against all three modes, so it needs three param
#  sets. The pattern logic is identical in each; what changes is what the mode
#  demands of it. Risk and RR are NOT the strategy's choice — they are fixed per
#  mode by Immutable Rule #1, and the engine enforces whatever is set here.
# --------------------------------------------------------------------------- #
CANDLE_SCALPER_PARAMS = StrategyParams(
    mode=Mode.SCALPER, timeframe="1m",
    risk_per_trade=0.01, risk_reward=1.0,
    atr_period=7,
    # Same fixed-cash sizing as the other Scalper strategies (bounded by the 1%
    # cap). The STOP here stays pattern-based (cs_* knobs below) — the hybrid
    # ATR/structural stop is specific to the VWAP and Volume-Burst signals.
    risk_per_trade_cash=2000.0,
    allow_short=True,
    entry_skip_minutes=15,
    max_hold_minutes=7,
    use_limit_entry=True,
    use_atr_gate=True,          # a 1-minute candle "pattern" in dead tape is noise
    max_leverage=15.0,
    # A 1-minute candle carries the least information of any bar the bot trades,
    # so it must clear the HIGHEST evidence bar. One medium pattern is not a trade
    # here; it takes a high-strength multi-candle formation.
    cs_min_score=4.0,
    cs_trend_lookback=10,
)
CANDLE_INTRADAY_PARAMS = StrategyParams(
    mode=Mode.INTRADAY, timeframe="15m",
    # 1% max loss per trade (₹1,000 on a ₹1L account), inside the 2% ceiling.
    risk_per_trade=0.01, risk_reward=2.0,   # Hard 1:2
    max_capital_per_trade_pct=0.20,         # <=20% of the account per trade
    atr_period=14,
    allow_short=True,           # MIS permits shorting, and half of the pattern
    max_leverage=15.0,          # library is bearish — long-only would discard it
    # Raised from 3.0 (the bare "one high-strength single-candle pattern"
    # floor) to 6.0 on request — now needs roughly two agreeing patterns (e.g.
    # a high 2-candle + a medium 2-candle, 3.75+2.5=6.25) rather than one
    # marginal single-candle hit, to cut down on low-conviction entries.
    cs_min_score=3.0,
    cs_trend_lookback=10,
)
CANDLE_SWING_PARAMS = StrategyParams(
    mode=Mode.SWING, timeframe="1d",
    risk_per_trade=0.03, risk_reward=3.0,   # Hard 1:3, max 3% — Immutable Rule #1
    atr_period=14,
    # Long only, and NOT for the usual "by design" reason: a Swing position is
    # held overnight, which in the NSE cash segment means delivery, and you cannot
    # take delivery of a short. Bearish patterns are still detected — they simply
    # can't be traded in this mode.
    allow_short=False,
    max_leverage=1.0,           # delivery = unleveraged
    cs_min_score=3.0,
    cs_trend_lookback=20,       # a daily "trend" deserves a longer look-back
)


def params_for_mode(mode: Mode) -> StrategyParams:
    """Default params for a mode. A mode can host SEVERAL strategies (see the
    registry in strategy.py) — this returns the default one's params."""
    if mode == Mode.SWING:
        return SWING_PARAMS
    if mode == Mode.SCALPER:
        return SCALPER_VWAP_PARAMS
    return INTRADAY_PARAMS


# --------------------------------------------------------------------------- #
#  Credential presence helpers — lets the engine auto-pick Simulated broker.
# --------------------------------------------------------------------------- #
def has_upstox_sandbox() -> bool:
    return bool(UPSTOX_SANDBOX_TOKEN)


def has_upstox_live() -> bool:
    return bool(UPSTOX_LIVE_ACCESS_TOKEN)


def has_dhan() -> bool:
    return bool(DHAN_CLIENT_ID and DHAN_ACCESS_TOKEN)


def has_zerodha() -> bool:
    return bool(ZERODHA_API_KEY and ZERODHA_ACCESS_TOKEN)


def has_kotak() -> bool:
    return bool(KOTAK_NEO_ACCESS_TOKEN)


def reload_tokens() -> None:
    """
    Re-read broker tokens from the environment / .env and update the module
    globals in place. Called after the UI refreshes a broker token so the
    running process picks up the new token without a restart.
    """
    global UPSTOX_SANDBOX_TOKEN, UPSTOX_LIVE_ACCESS_TOKEN, UPSTOX_LIVE_API_KEY
    global UPSTOX_LIVE_SECRET, DHAN_CLIENT_ID, DHAN_ACCESS_TOKEN
    global ZERODHA_API_KEY, ZERODHA_API_SECRET, ZERODHA_ACCESS_TOKEN
    global KOTAK_NEO_ACCESS_TOKEN
    try:
        from dotenv import load_dotenv as _ld
        _ld(override=True)
    except Exception:
        pass
    UPSTOX_SANDBOX_TOKEN = _env("UPSTOX_SANDBOX_TOKEN")
    UPSTOX_LIVE_ACCESS_TOKEN = _env("UPSTOX_LIVE_ACCESS_TOKEN")
    UPSTOX_LIVE_API_KEY = _env("UPSTOX_LIVE_API_KEY")
    UPSTOX_LIVE_SECRET = _env("UPSTOX_LIVE_SECRET")
    DHAN_CLIENT_ID = _env("DHAN_CLIENT_ID")
    DHAN_ACCESS_TOKEN = _env("DHAN_ACCESS_TOKEN")
    ZERODHA_API_KEY = _env("ZERODHA_API_KEY")
    ZERODHA_API_SECRET = _env("ZERODHA_API_SECRET")
    ZERODHA_ACCESS_TOKEN = _env("ZERODHA_ACCESS_TOKEN")
    KOTAK_NEO_ACCESS_TOKEN = _env("KOTAK_NEO_ACCESS_TOKEN")


# Local storage fallback location (used when MongoDB is unreachable)
LOCAL_DB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(LOCAL_DB_DIR, exist_ok=True)
