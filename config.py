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
from datetime import datetime, time, timedelta, timezone
from enum import Enum
from typing import Optional

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
    #: Declared ahead of any instruments existing, so the category below is a
    #: TOTAL mapping from day one. Adding real crypto instruments later is then
    #: purely a config change — no reporting, storage or UI work follows it.
    CRYPTO = "CRYPTO"


class Category(str, Enum):
    """The asset class a trade belongs to, for book-keeping and reporting.

    Distinct from Segment on purpose: a segment is an EXCHANGE VENUE (NSE cash,
    MCX futures) and there may be several per asset class, while a category is
    what you actually want to see a P&L line for. Stored on every trade so the
    split survives an instrument being reclassified or a venue being added.
    """
    EQUITY = "Equity"
    COMMODITY = "Commodity"
    CRYPTO = "Crypto"


SEGMENT_CATEGORY: dict[Segment, Category] = {
    Segment.EQUITY: Category.EQUITY,
    Segment.MCX: Category.COMMODITY,
    Segment.CRYPTO: Category.CRYPTO,
}


def category_for_segment(segment) -> str:
    """Category name for a Segment or its raw string value.

    Accepts a bare string because it is called with `trade["segment"]` from
    stored documents, which are plain JSON. An unrecognised segment falls back
    to Equity rather than raising: a trade that cannot be categorised must
    still appear in the book, and being in the wrong bucket is recoverable
    where vanishing from the totals is not.
    """
    if isinstance(segment, Segment):
        return SEGMENT_CATEGORY[segment].value
    try:
        return SEGMENT_CATEGORY[Segment(str(segment))].value
    except (ValueError, KeyError):
        return Category.EQUITY.value


def category_of_trade(trade: dict) -> str:
    """A stored trade's category, derived from `segment` when the trade
    predates the field. Every read path goes through this, so a document
    written before categories existed is never left uncategorised even if the
    one-off backfill has not run."""
    stored = (trade or {}).get("category")
    if stored:
        return str(stored)
    return category_for_segment((trade or {}).get("segment", ""))


ALL_CATEGORIES: tuple[str, ...] = tuple(c.value for c in Category)


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

# --------------------------------------------------------------------------- #
#  AI Auditor (ai_auditor/) — an ON-DEMAND, READ-ONLY review of how the bot has
#  traded. These are the only credentials it uses, and they reach nothing but
#  the chosen LLM endpoint. Absent keys simply disable that provider in the UI.
# --------------------------------------------------------------------------- #
AI_AUDITOR_PROVIDER = _env("AI_AUDITOR_PROVIDER", "openrouter")
OPENROUTER_API_KEY = _env("OPENROUTER_API_KEY")
#: Preferred OpenRouter model. Left blank on purpose: with no explicit choice
#: the auditor walks OPENROUTER_MODEL_CHAIN below, so it keeps working when a
#: model id is retired — which they are, regularly.
OPENROUTER_MODEL = _env("OPENROUTER_MODEL", "")
#: Ordered fallback chain, strongest first. Tried in turn until one answers; a
#: model that no longer exists on OpenRouter is skipped rather than fatal.
#: Model ids change — check https://openrouter.ai/models and override this in
#: .env rather than editing code.
OPENROUTER_MODEL_CHAIN = [
    m.strip() for m in _env(
        "OPENROUTER_MODEL_CHAIN",
        "anthropic/claude-sonnet-4.5,"
        "openai/gpt-5,"
        "google/gemini-2.5-pro,"
        "anthropic/claude-3.7-sonnet,"
        "deepseek/deepseek-r1"
    ).split(",") if m.strip()
]
GEMINI_API_KEY = _env("GEMINI_API_KEY")
GEMINI_MODEL = _env("GEMINI_MODEL", "gemini-2.5-pro")
#: Fall back to the OTHER provider when the first one cannot produce a report.
#: The audit is a manual, occasional action — finishing on the second provider
#: beats making the operator notice and retry.
AI_AUDITOR_FALLBACK = _env("AI_AUDITOR_FALLBACK", "true").lower() not in (
    "false", "0", "no")
try:
    AI_AUDITOR_MAX_TOKENS = max(512, int(_env("AI_AUDITOR_MAX_TOKENS", "8000")))
except ValueError:
    AI_AUDITOR_MAX_TOKENS = 8000
try:
    AI_AUDITOR_TIMEOUT_SECONDS = max(10, int(_env("AI_AUDITOR_TIMEOUT_SECONDS", "120")))
except ValueError:
    AI_AUDITOR_TIMEOUT_SECONDS = 120

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
    # Contract expiry as "YYYY-MM-DD", for DERIVATIVES only — "" for equity,
    # which never expires. Written by tools/refresh_mcx.py.
    #
    # This exists because an expired MCX key does not degrade, it DIES: Upstox
    # answers UDAPI100011 "Invalid Instrument key" and the symbol silently
    # stops producing data. Carrying the date lets the bot warn while the
    # contract is still tradable instead of leaving you to notice the gap.
    expiry: str = ""


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
    # LAST-RESORT SNAPSHOT, refreshed 2026-08-05. These keys EXPIRE, so this
    # list rots: it is only reached when mcx_instruments.py is missing or
    # fails to import, and by then the dates below are probably stale too.
    # Every entry carries its expiry so the engine's startup check reports
    # exactly which of them have died rather than trading a silent void.
    #
    # If you find yourself here, the fix is `python tools/refresh_mcx.py`.
    MCX_INSTRUMENTS = [
        #          symbol         segment      instrument_key    lot  tick  ref price   mult  expiry
        # --- Full-size contracts ---
        Instrument("GOLD",        Segment.MCX, "MCX_FO|483079", 1,    1.0,  142419.0,  100,  expiry="2026-10-05"),   # 1 kg, quoted ₹/10g
        Instrument("CRUDEOIL",    Segment.MCX, "MCX_FO|560977", 100,  1.0,  7580.0,    100,  expiry="2026-08-19"),   # 100 barrels, quoted ₹/barrel
        Instrument("NATURALGAS",  Segment.MCX, "MCX_FO|561496", 1250, 0.10, 279.7,     1250, expiry="2026-08-26"),   # 1250 mmBtu, quoted ₹/mmBtu
        Instrument("SILVER",      Segment.MCX, "MCX_FO|471725", 30,   1.0,  223320.0,  30,   expiry="2026-09-04"),   # 30 kg, quoted ₹/kg
        # --- Mini / micro contracts (fractional size => fractional margin) ---
        Instrument("GOLDM",       Segment.MCX, "MCX_FO|563946", 100,  1.0,  142419.0,  10,   expiry="2026-09-04"),   # 100 g, quoted ₹/10g
        Instrument("CRUDEOILM",   Segment.MCX, "MCX_FO|560978", 10,   1.0,  7580.0,    10,   expiry="2026-08-19"),   # 10 barrels, quoted ₹/barrel
        Instrument("NATGASMINI",  Segment.MCX, "MCX_FO|561497", 250,  0.10, 279.7,     250,  expiry="2026-08-26"),   # 250 mmBtu, quoted ₹/mmBtu
        Instrument("SILVERM",     Segment.MCX, "MCX_FO|471726", 5,    1.0,  223320.0,  5,    expiry="2026-08-31"),   # 5 kg, quoted ₹/kg
        Instrument("SILVERMIC",   Segment.MCX, "MCX_FO|488788", 1,    1.0,  223320.0,  1,    expiry="2026-08-31"),   # 1 kg, quoted ₹/kg
    ]

ALL_INSTRUMENTS = EQUITY_INSTRUMENTS + MCX_INSTRUMENTS
INSTRUMENTS_BY_SYMBOL = {i.symbol: i for i in ALL_INSTRUMENTS}


# --------------------------------------------------------------------------- #
#  MCX margin — hardcoded per-lot figures (user-provided).
#
#  BACKTEST ONLY. The live and paper engines no longer read this table at all —
#  engine._mcx_margin asks the BROKER for the real figure in both environments
#  and refuses the trade if it can't get one, rather than sizing a real position
#  off a number typed in months ago.
#
#  It survives here for the one job it is still honest at: backtesting. A
#  historical run cannot ask for today's margin — margin in August tells you
#  nothing about what the exchange demanded in March — so an approximate,
#  stable per-lot figure is the best available input and its inaccuracy is
#  bounded and obvious.
#
#  Commodity margin is NOT a formula (notional ÷ leverage is nowhere near right):
#  it is the exchange's SPAN + Exposure margin plus SEBI peak-margin.
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


#: Warn this many days before a derivative contract expires. Crude oil and
#: natural gas roll MONTHLY, metals every two months, so a week is enough
#: notice to refresh without nagging for most of the contract's life.
EXPIRY_WARN_DAYS = 7


def days_to_expiry(inst: Instrument) -> Optional[int]:
    """Whole days until this contract expires, or None if it never does
    (equity) or the date is unparseable."""
    if not inst.expiry:
        return None
    try:
        exp = datetime.strptime(inst.expiry, "%Y-%m-%d").date()
    except ValueError:
        return None
    return (exp - now_ist().date()).days


def expiring_soon(instruments: list[Instrument],
                  within_days: int = EXPIRY_WARN_DAYS
                  ) -> list[tuple[Instrument, int]]:
    """(instrument, days_left) for every contract at or past `within_days`,
    soonest first. A NEGATIVE days_left means it has already expired — its key
    is dead and that symbol cannot produce data at all."""
    out = []
    for inst in instruments:
        left = days_to_expiry(inst)
        if left is not None and left <= within_days:
            out.append((inst, left))
    return sorted(out, key=lambda pair: pair[1])


# --------------------------------------------------------------------------- #
#  The clock. EVERY wall-clock decision in this system is IST — market hours,
#  candle timestamps (Upstox candles are tz-converted to Asia/Kolkata and then
#  stripped of tzinfo in data_feed), log stamps, session-open offsets.
#
#  Never use datetime.now() for those: it returns the SERVER's local clock, and
#  the deployed box runs UTC. That made is_open() compare 09:15-15:30 against a
#  UTC time, so the bot read "market CLOSED" through the entire real session and
#  never opened a position — while the same code on an IST laptop traded fine.
#  now_ist() derives IST from UTC, so behaviour is identical on both.
#
#  India has no DST and has been a fixed UTC+05:30 offset since 1945, so a fixed
#  offset is used rather than a zoneinfo lookup (no tzdata dependency needed on
#  slim containers or Windows).
# --------------------------------------------------------------------------- #
IST = timezone(timedelta(hours=5, minutes=30))


def now_ist() -> datetime:
    """Current IST wall-clock time as a NAIVE datetime, independent of server TZ."""
    return datetime.now(IST).replace(tzinfo=None)


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


# --------------------------------------------------------------------------- #
#  Intraday square-off
#
#  An INTRADAY position must not survive the session. Left alone it either gets
#  auto-squared by the broker at whatever price the close happens to print, or
#  — worse in the cash segment — turns into a delivery the account never
#  intended to fund. So the bot closes everything itself, at a time it picks,
#  while there is still liquidity to get out on.
#
#  15:09 for equity: ~20 minutes before the 15:30 close, comfortably ahead of
#  the closing auction and the last-minute spread widening, and ahead of most
#  brokers' own auto-square-off (typically 15:15-15:20) so the exit happens on
#  OUR terms rather than theirs.
#
#  MCX runs to 23:30, so it gets its own, later cutoff — using 15:09 there
#  would cut the commodity session off in the middle of its most active hours.
#
#  This fires REGARDLESS of profit or loss: it is a time stop, not a decision.
#  SWING is deliberately absent — it holds overnight by design, and applying an
#  end-of-day exit to it would silently convert it into an intraday strategy.
# --------------------------------------------------------------------------- #
DEFAULT_SQUARE_OFF = {
    Segment.EQUITY: time(15, 9),
    Segment.MCX: time(23, 15),
}

#: Modes whose positions must be flat by the end of the session. Swing is NOT
#: here, and must never be added — see above.
SQUARE_OFF_MODES = (Mode.INTRADAY, Mode.SCALPER)


def default_square_off(segment: Segment) -> time:
    return DEFAULT_SQUARE_OFF.get(segment, time(15, 9))


def parse_clock(value: str) -> Optional[time]:
    """"HH:MM" -> time, or None for empty/invalid. Shared by the admin
    square-off setting and anything else taking a wall-clock string."""
    text = (value or "").strip()
    if not text:
        return None
    try:
        hh, mm = text.split(":")
        return time(int(hh), int(mm))
    except Exception:
        return None


def square_off_time_for(segment: Segment, mode: Mode,
                        override: str = "") -> Optional[time]:
    """When positions in this instrument must be flat, or None if the mode
    holds overnight (Swing) — in which case no square-off applies at all."""
    if mode not in SQUARE_OFF_MODES:
        return None
    return parse_clock(override) or default_square_off(segment)


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
#  Risk : reward
#
#  INTRADAY_RR_NOTE — Intraday's default moved from 1:2 to 1:1 (owner's call,
#  2026-08-04). This is a deliberate amendment to the "Hard 1:2" rule that
#  CLAUDE.md previously stated; the doc has been updated to match, so the code
#  and the rule agree. What did NOT change is the risk CAP: risk_per_trade
#  stays at 1% (ceiling 2%), because RR only moves the TARGET — it never
#  affects position size, which is risk_budget / stop_distance.
#
#  Worth knowing what 1:1 costs: at 1:2 the break-even win rate is ~33%; at 1:1
#  it is >50% before brokerage and slippage. Backtest before trusting it live.
#
#  RR is now selectable per mode by admin (admin_config.ModeConfig.risk_reward);
#  0.0 there means "use the strategy's own value declared below".
# --------------------------------------------------------------------------- #
#: Risk:reward values admin may pick, as reward-per-1-unit-of-risk. A fixed list
#: rather than a free number field — it keeps the UI honest and stops a typo
#: like 0.1 silently turning every trade into a 10:1 loser.
RR_CHOICES: tuple[float, ...] = (1.0, 1.5, 2.0, 2.5, 3.0)


def rr_label(rr: float) -> str:
    """Format an RR for display: 1.0 -> "1:1", 1.5 -> "1:1.5"."""
    return f"1:{rr:g}"


def is_valid_rr(rr: float) -> bool:
    """0 is valid and means 'inherit the strategy's own RR'."""
    return rr == 0.0 or rr in RR_CHOICES


# --------------------------------------------------------------------------- #
#  Signal score (pattern-evidence threshold)
#
#  How much weighted evidence a setup must carry before it is traded — see
#  StrategyParams.cs_min_score and strategies/candlestick_engine.py. Raising it
#  trades LESS but with more agreement behind each entry; lowering it trades
#  MORE and takes weaker setups. It is admin-tunable per mode so the threshold
#  can be tested against a real market without editing code.
#
#  The scale comes from the pattern weights: STRENGTH_WEIGHT (weak 1.0 /
#  medium 2.0 / high 3.0) x SPAN_WEIGHT (1-candle 1.0 ... 5-candle 1.75), so
#  ONE pattern is worth 1.0-5.25 and several agreeing ones sum. Useful
#  landmarks: 1.0 = any single weak pattern (very loose), 3.0 = one
#  high-strength single-candle pattern, ~6.0 = roughly two agreeing patterns.
#
#  This moves ENTRY SELECTIVITY only. It has no effect on position size, the
#  risk cap, or the stop — Immutable Rule #1 is untouched by any value here.
# --------------------------------------------------------------------------- #
MIN_SCORE_MIN, MIN_SCORE_MAX = 0.5, 20.0


def is_valid_min_score(score: float) -> bool:
    """0 is valid and means 'inherit the strategy's own threshold'."""
    return score == 0.0 or MIN_SCORE_MIN <= score <= MIN_SCORE_MAX


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
    #: PER-RUN candlestick pattern allow-list. Empty (the default, and what
    #: every live run uses) means "defer to the saved allow-list in
    #: pattern_config.py" — so the live bot keeps reading its dashboard
    #: setting and nothing here changes its behaviour.
    #:
    #: A non-empty value OVERRIDES that saved setting for this run only. It
    #: exists so a backtest can try a pattern set WITHOUT editing what the live
    #: bot is trading — the same relationship `risk_reward` and `cs_min_score`
    #: already have with their admin overrides. A tuple, not a list, because
    #: StrategyParams is frozen and gets copied with dataclasses.replace().
    allowed_patterns: tuple[str, ...] = ()
    #: Ignore the SAVED dashboard pattern filter entirely for this run.
    #:
    #: Needed by the combination search: its screening pass must see every
    #: pattern the engine can emit, or it can only ever "discover" patterns
    #: that were already switched on — which is not discovery at all. Default
    #: False everywhere else, so the live bot and ordinary backtests keep
    #: reading the saved filter exactly as before.
    ignore_pattern_filter: bool = False


INTRADAY_PARAMS = StrategyParams(
    mode=Mode.INTRADAY, timeframe="15m",
    # 1% max loss per trade (₹1,000 on a ₹1L account). Scales with capital, and
    # stays well inside the 2% ceiling Immutable Rule #1 forbids exceeding.
    risk_per_trade=0.01, risk_reward=1.0,   # 1:1 — see INTRADAY_RR_NOTE
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
    risk_per_trade=0.01, risk_reward=1.0,   # 1:1 — see INTRADAY_RR_NOTE
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
    global OPENROUTER_API_KEY, OPENROUTER_MODEL, GEMINI_API_KEY, GEMINI_MODEL
    global AI_AUDITOR_PROVIDER
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
    AI_AUDITOR_PROVIDER = _env("AI_AUDITOR_PROVIDER", "openrouter")
    OPENROUTER_API_KEY = _env("OPENROUTER_API_KEY")
    OPENROUTER_MODEL = _env("OPENROUTER_MODEL", "")
    GEMINI_API_KEY = _env("GEMINI_API_KEY")
    GEMINI_MODEL = _env("GEMINI_MODEL", "gemini-2.5-pro")


# Local storage fallback location (used when MongoDB is unreachable)
LOCAL_DB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(LOCAL_DB_DIR, exist_ok=True)
