"""
config.py
Central configuration: environment loading, the tradable instrument universe
(NSE equity + MCX commodities), market hours, and strategy parameters.

Nothing here talks to a broker or a database — it is pure configuration so the
rest of the system can stay decoupled (Immutable Rule #3).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from datetime import datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo
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
    #: US cash equities (NYSE/NASDAQ), traded through Alpaca. Its bars and its
    #: clock are EXCHANGE-LOCAL (America/New_York), not IST — see US_TZ below
    #: for why that choice is forced rather than stylistic.
    US_EQUITY = "US_EQUITY"


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
    # A US share is still equity for P&L reporting; the venue differs, the
    # asset class does not. (Currency does differ — see US_CURRENCY.)
    Segment.US_EQUITY: Category.EQUITY,
}


#: Currency each segment is denominated in. NEVER aggregate across two of
#: these — a ₹ P&L and a $ P&L summed into one number is silently wrong.
SEGMENT_CURRENCY: dict[Segment, str] = {
    Segment.EQUITY: "INR", Segment.MCX: "INR", Segment.CRYPTO: "INR",
    Segment.US_EQUITY: "USD",
}


def currency_for_segment(segment) -> str:
    """Currency code for a Segment or its raw string value."""
    try:
        return SEGMENT_CURRENCY.get(Segment(segment), "INR")
    except Exception:
        return "INR"


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

# US equities (Alpaca) ------------------------------------------------------- #
# A US instrument_key is simply its ticker — there is no ISIN-style lookup and
# nothing expires, so unlike MCX this list does not rot. Regenerate with
# tools/refresh_us.py when you want a different universe; it is NOT required to
# get started. Absent module => empty list, and every US code path below is
# then simply unreachable, leaving the Indian bot byte-identical.
try:
    from us_instruments import US_INSTRUMENTS
except Exception:
    US_INSTRUMENTS = []

ALL_INSTRUMENTS = EQUITY_INSTRUMENTS + MCX_INSTRUMENTS + US_INSTRUMENTS
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
#  Exchange-local clocks.
#
#  Every Indian segment is quoted, stored and reasoned about in IST, so one
#  global now_ist() sufficed. A US session cannot join that scheme: 09:30-16:00
#  New York is roughly 19:00-02:30 IST, which CROSSES MIDNIGHT, and
#  MarketHours.is_open() is a plain `open <= now <= close` comparison that is
#  false for every minute of a wrapping session. US DST makes it worse — the
#  IST offset moves by an hour twice a year while India never shifts.
#
#  So each segment keeps its OWN wall clock, and its bars are stored in that
#  same local time. In New York terms the session is 09:30-16:00 on one
#  calendar day, which means:
#     * is_open() works unchanged, with no wrap-around special case,
#     * the square-off comparison works unchanged,
#     * strategy.vwap()'s per-calendar-day reset lands on the real session
#       boundary instead of splitting it at IST midnight.
#  Nothing about the Indian path changes: now_for_segment() returns exactly
#  now_ist() for every pre-existing segment.
# --------------------------------------------------------------------------- #
US_TZ = ZoneInfo("America/New_York")

#: Trades in this segment are denominated in this currency. Reporting and any
#: cross-account aggregation must not mix them — see the US_EQUITY note in
#: cost_model / the integration guide.


def now_for_segment(segment) -> datetime:
    """Current NAIVE wall-clock time at the segment's own exchange.

    IST for every Indian segment (identical to now_ist(), so existing callers
    are unaffected); New York local time for US equities.
    """
    if segment == Segment.US_EQUITY:
        return datetime.now(US_TZ).replace(tzinfo=None)
    return now_ist()


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


# US regular session: 09:30 - 16:00 America/New_York. Expressed in NEW YORK
# local time, which is what now_for_segment(US_EQUITY) returns — so DST is
# handled by the zoneinfo database rather than by arithmetic here.
US_EQUITY_HOURS = MarketHours(time(9, 30), time(16, 0))


def market_hours_for_segment(segment: Segment) -> MarketHours:
    if segment == Segment.MCX:
        return MCX_HOURS
    if segment == Segment.US_EQUITY:
        return US_EQUITY_HOURS
    return EQUITY_HOURS


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
    # 15:45 New York. Chosen to mirror NSE's semantics EXACTLY rather than to
    # match its clock: 15:09 IST is passed by NSE's final 15-minute bar (which
    # starts 15:15), so the square-off always fires on the last bar. The US
    # session's final 15-minute bar starts at 15:45, so a 15:50 cutoff would be
    # passed by NO bar at all and the square-off would silently never run,
    # letting intraday positions drift out of the session. 15:45 restores the
    # "closed on the final bar" behaviour, and live it still leaves 15 minutes
    # before the 16:00 close.
    Segment.US_EQUITY: time(15, 45),
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


# --------------------------------------------------------------------------- #
#  "When is this session, in MY time?"
#
#  The engine reasons entirely in exchange-local time (see now_for_segment), and
#  it should: that is what makes is_open(), the square-off and the VWAP day
#  reset correct without special cases. But the OPERATOR is in India, and for a
#  US instrument the exchange clock tells them nothing useful — 15:45 New York
#  is the middle of the night in Mumbai, and WHICH night-time hour it is moves
#  by one when US daylight saving flips (India has no DST, so the offset is
#  4h30 in US summer and 5h30... no: 9h30 and 10h30 behind IST respectively).
#
#  These helpers exist purely to answer that, and are display-only — nothing in
#  the trading path reads them.
# --------------------------------------------------------------------------- #
def _to_ist(t: time, tz, on: Optional["date_cls"] = None) -> tuple[time, int]:
    """`t` in timezone `tz` -> (IST time, day offset).

    The day offset is 0 when the moment lands on the same IST calendar day and
    1 when it spills past midnight — which is the normal case for a US close
    seen from India, and exactly the fact a trader needs to know.
    """
    from datetime import date as _date, datetime as _dt
    day = on or _date.today()
    local = _dt.combine(day, t, tz)
    ist = local.astimezone(IST)
    return ist.time(), (ist.date() - local.date()).days


def session_windows(segment: Segment, mode: Mode = Mode.INTRADAY,
                    override: str = "", on=None) -> dict:
    """Open / close / square-off for a segment, in BOTH clocks.

    DST-aware: it converts on a real DATE, so a US session correctly reads
    19:00 IST in summer and 20:00 IST in winter rather than a fixed guess.
    """
    hours = market_hours_for_segment(segment)
    tz = US_TZ if segment == Segment.US_EQUITY else IST
    sq = square_off_time_for(segment, mode, override)

    def pair(t):
        if t is None:
            return None
        ist_t, off = _to_ist(t, tz, on)
        return {"local": t.strftime("%H:%M"),
                "ist": ist_t.strftime("%H:%M"),
                "ist_next_day": bool(off)}

    return {
        "segment": segment.value,
        "timezone": "America/New_York" if segment == Segment.US_EQUITY
                    else "Asia/Kolkata",
        "open": pair(hours.open_t),
        "close": pair(hours.close_t),
        "square_off": pair(sq),
        "currency": currency_for_segment(segment),
    }


def session_summary_ist(segment: Segment, mode: Mode = Mode.INTRADAY,
                        override: str = "") -> str:
    """One line an Indian operator can act on, e.g.
    "US Equity 19:00-01:30 IST (next day), square-off 01:15 IST"."""
    w = session_windows(segment, mode, override)
    if segment != Segment.US_EQUITY:
        sq = f", square-off {w['square_off']['ist']}" if w["square_off"] else ""
        return f"{w['open']['ist']}-{w['close']['ist']} IST{sq}"
    nxt = " (next day)" if w["close"]["ist_next_day"] else ""
    sq = ""
    if w["square_off"]:
        sq_nxt = " next day" if w["square_off"]["ist_next_day"] else ""
        sq = f", square-off {w['square_off']['ist']} IST{sq_nxt}"
    return (f"{w['open']['ist']}-{w['close']['ist']} IST{nxt}{sq} "
            f"({w['open']['local']}-{w['close']['local']} New York)")


def square_off_time_for(segment: Segment, mode: Mode,
                        override: str = "") -> Optional[time]:
    """When positions in this instrument must be flat, or None if the mode
    holds overnight (Swing) — in which case no square-off applies at all."""
    if mode not in SQUARE_OFF_MODES:
        return None
    return parse_clock(override) or default_square_off(segment)


def entry_cutoff_for(segment: Segment, mode: Mode, params,
                     square_off_override: str = "") -> Optional[time]:
    """Latest wall-clock time at which a NEW position may be opened, or None
    when the strategy sets no cutoff (`entry_cutoff_before_close = 0`, the
    default) or the mode holds overnight.

    DERIVED FROM THE FLAT-OUT, not configured as a clock, because the thing it
    protects against is a trade with no runway: an entry taken twenty minutes
    before everything is squared off cannot reach its target, so it is a
    brokerage donation with a chart attached. Measured on a real Intraday log,
    every position opened on the last tradeable bar lost money — six for six —
    on a combined +Rs24 of gross movement.

    Because it is derived, moving the square-off time moves this with it, which
    is the correct coupling: the runway a trade needs does not change just
    because the day was shortened.
    """
    minutes = int(getattr(params, "entry_cutoff_before_close", 0) or 0)
    if minutes <= 0:
        return None
    flat_by = square_off_time_for(segment, mode, square_off_override)
    if flat_by is None:                       # Swing holds overnight
        return None
    return (datetime.combine(datetime(2000, 1, 1).date(), flat_by)
            - timedelta(minutes=minutes)).time()


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
SEGMENT_MAX_LEVERAGE = {Segment.EQUITY: 1.0, Segment.MCX: 15.0,
                        # Same deliberate no-leverage choice as NSE cash.
                        Segment.US_EQUITY: 1.0}


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
    # -- Stop-distance band (percent of ENTRY price) ------------------------ #
    #  The stop is still derived from ATR exactly as before; these only BOUND
    #  the answer. Both default to 0.0 = no bound, so every existing strategy
    #  takes the identical code path it did before this existed.
    #
    #  WHY A BAND. ATR is a volatility reading, not a tradeability check. On a
    #  quiet day it can put the stop 0.2% away, inside the spread and the
    #  noise; on an event day it can put it 4% away, which pushes a 1:1 target
    #  4% away too — a move the session usually will not deliver, so the trade
    #  resolves neither way and dies at the square-off. Measured on a real
    #  Intraday log, risk distances ran from 0.64% to 3.9% of price on ONE
    #  symbol.
    #
    #  THE RR IS PRESERVED. Clamping moves the stop and the target together,
    #  so a 1:1 setup stays 1:1 — see strategy.clamp_signal_risk.
    #
    #  SIZING FOLLOWS, AS IT ALWAYS DOES. qty = risk_budget / stop_distance,
    #  so a clamped-tighter stop buys a BIGGER position for the same rupee
    #  risk. Immutable Rule #1 is untouched — risk per trade is unchanged —
    #  but notional, and therefore cost, rises. That is the trade being made,
    #  and it is why this is measured rather than assumed.
    #: Widest the stop may sit from entry, in PERCENT. 0.0 = no cap.
    max_stop_pct: float = 0.0
    #: Tightest the stop may sit from entry, in PERCENT. 0.0 = no floor.
    #: Guards the other end: a stop inside the spread is a guaranteed stop-out.
    min_stop_pct: float = 0.0

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
    # -- Session-anchored Opening Range knobs (Commodity.md) ----------------- #
    #  Every one of these is INERT AT ITS DEFAULT. `orb_minutes = 0` switches
    #  the whole opening-range apparatus off, and `partial_exit_fraction = 0.0`
    #  switches the scale-out off, so a strategy that does not set them takes
    #  the identical code path it did before this existed. Only the CRUDEOIL
    #  strategy sets them today.
    #
    #  The anchor is a TIMEZONE + wall-clock time, never a fixed IST string:
    #  09:00 America/New_York is 18:30 IST under EDT but 19:30 IST under EST,
    #  so a hardcoded IST anchor silently builds the opening range on the wrong
    #  bars for roughly five months of the year.
    #: MULTI-TIMEFRAME bias filter, in minutes. 0 (the default) = single
    #: timeframe, which is what every pre-existing strategy is and stays.
    #:
    #: The higher timeframe is RESAMPLED from the base candles inside the
    #: strategy rather than fetched as a second feed. That is a deliberate
    #: trade: one subscription, one candle stream, no second websocket to keep
    #: in sync, and — the part that matters — a backtest replays the identical
    #: derivation, so the higher timeframe cannot silently differ between live
    #: and test. It must be an exact multiple of the base timeframe.
    htf_minutes: int = 0
    #: How many completed higher-timeframe bars set the bias, and how much of
    #: them must agree. Ignored entirely when htf_minutes is 0.
    htf_bars: int = 3
    orb_anchor_tz: str = ""          # "" = no session anchor (day-anchored)
    orb_anchor_hhmm: str = ""        # wall-clock time IN orb_anchor_tz, "HH:MM"
    orb_minutes: int = 0             # 0 = no opening range; feature off
    #: The opening range must be a sane multiple of current ATR before it is
    #: worth trading. A collapsed range gives instant whipsaw breakouts; a
    #: blown-out one means the move already happened inside the window and the
    #: resulting stop would be enormous. 0 disables either bound.
    orb_range_min_atr: float = 0.0
    orb_range_max_atr: float = 0.0
    #: Stop opening NEW positions this many minutes before the segment's
    #: square-off, so an entry always has room to resolve. 0 = no cutoff.
    entry_cutoff_before_close: int = 0
    # -- Scale-out / runner management (Commodity.md §9.2-9.5) --------------- #
    #: Fraction of the position booked when the first target prints. 0.0 (the
    #: default) = no partial: the whole position closes at the target exactly
    #: as it always has.
    #:
    #: A ONE-LOT position NEVER partials regardless of this value — there is
    #: nothing to split, so it runs to the full target and closes there. The
    #: engine enforces that, not this number.
    partial_exit_fraction: float = 0.0
    #: After the partial is booked, the remainder's stop moves to entry plus
    #: this many ticks (long; minus for a short) — the friction buffer that
    #: makes the runner genuinely risk-free after ROUND-TRIP costs rather than
    #: merely break-even before them.
    #:
    #: ONLY MEANINGFUL PER INSTRUMENT. Round-trip cost is mostly proportional
    #: to turnover, so the tick count that covers it depends entirely on the
    #: price and the tick grid: measured against cost_model, a ~Rs1L intraday
    #: equity round trip is 7 ticks of a Rs250 share and 92 ticks of a Rs3,200
    #: one. Set this only when a fixed grid distance is genuinely what is
    #: meant (CRUDEOIL does); otherwise set the bps field below.
    breakeven_buffer_ticks: float = 0.0
    #: The same buffer in BASIS POINTS OF ENTRY PRICE, added to the tick term.
    #: This is the portable unit — ~14.3 bps covers an intraday equity round
    #: trip and ~12.5 bps an MCX crude one, near enough the same number that
    #: one setting serves every book. 0.0 (the default) = no proportional
    #: term, i.e. exactly the behaviour before this field existed.
    breakeven_buffer_bps: float = 0.0
    #: Runner target, as a multiple of the ORIGINAL risk distance. The
    #: remainder is managed by the trail; this is the backstop beyond it.
    runner_rr_mult: float = 0.0
    #: Trail the remainder after the partial, independently of the per-symbol
    #: opt-in trail. False = the remainder simply holds to its break-even stop
    #: or the runner target.
    trail_remainder: bool = False
    #: LOCK THE FIRST TARGET. After the partial, the runner's stop is floored
    #: at TP1 instead of at break-even, so the 1R the trade already earned can
    #: never be handed back: price going above TP1 and returning to it squares
    #: the runner off there, and only a move on to TP2 pays more.
    #:
    #: Strictly tighter than the break-even buffer — it does not compete with
    #: it, it supersedes it, because TP1 is always the further of the two into
    #: profit. Default False, so this is inert for every existing strategy.
    #:
    #: NOT OBVIOUSLY BETTER, which is why it is a knob: a runner floored at
    #: TP1 books ~1R reliably but dies on any pullback, while one floored at
    #: break-even risks that 1R for the chance of 2R+. Measure it.
    lock_first_target: bool = False
    #: WHEN the runner's stop is promoted to the first target, as a fraction
    #: of the way from TP1 to TP2. 0.0 = at once (Approach 2b); 0.5 = only
    #: once price reaches the midpoint of TP1..TP2 (Approach 2c), leaving the
    #: runner on break-even + friction until then so it is allowed to breathe.
    #: Inert unless lock_first_target is on, and superseded by the ATR trigger
    #: below whenever that is set.
    lock_trigger_frac: float = 0.0
    #: The same trigger in ATR: promote once price is this many ATR past TP1.
    #: 0.0 = use the fraction instead. A fraction is a share of a distance the
    #: strategy picked; ATR is a reading of what the instrument is actually
    #: doing today, so it means the same thing across symbols and regimes.
    #: See exit_manager.ExitParams.lock_trigger_atr_mult for the multiple
    #: that is worth choosing and the one that quietly disables the lock.
    lock_trigger_atr_mult: float = 0.0
    #: Trail the FULL position from ENTRY, with no partial at all — the other
    #: profit-booking approach ("Approach 1": let the whole size run behind an
    #: ATR chandelier instead of banking half at 1R).
    #:
    #: Default False, so this is inert for every existing strategy: with it off
    #: and `partial_exit_fraction = 0`, a position is the plain fixed-RR trade
    #: it has always been. The trail distance is `atr_sl_mult` — the same
    #: multiple the entry stop was built from — unless the symbol overrides it
    #: (symbol_config.SymbolRules.trail_mult).
    trail_from_entry: bool = False
    #: Chandelier trail distance, in ATR. 0.0 (the default) = INHERIT
    #: `atr_sl_mult`, which is what the trail used before this field existed,
    #: so nothing changes for a strategy that leaves it alone.
    #:
    #: It is a SEPARATE field because `atr_sl_mult` also sets the ENTRY stop:
    #: sweeping the trail distance through `atr_sl_mult` would move the entry
    #: stop with it, and the sweep would then be measuring two changes at once
    #: instead of one. Per-symbol rules still win over both.
    trail_atr_mult: float = 0.0


# --------------------------------------------------------------------------- #
#  INTRADAY_CAPITAL_CAP_NOTE — the 20%-of-account notional cap was DISABLED on
#  2026-08-29 (owner's call). Every Intraday StrategyParams below now sets
#  max_capital_per_trade_pct=0.0, which strategy.position_size reads as "no
#  capital cap"; the limit was previously 0.20.
#
#  WHY: measured on real fills, the cap — not risk_per_trade — was what sized
#  every equity trade. On a Rs1L account it allowed ~17 shares of a Rs1,140
#  stock, so realised risk was 0.04-0.12% against the 1% budget, and the flat
#  Rs20/order brokerage made the round trip cost 0.34% of notional instead of
#  0.14%. Removing the cap restores full-capital sizing and cuts that drag ~2.4x.
#
#  WHAT DID NOT CHANGE — Immutable Rule #1 is untouched. position_size still
#  takes the MIN of three limits, and both survivors still bind:
#    * risk_per_trade (1%, ceiling 2%) — with a wide stop this binds first
#      (measured: 86 shares, Rs993 = 0.99% of a Rs1L account).
#    * the notional/leverage limit — equity is pinned to 1x by
#      SEGMENT_MAX_LEVERAGE, so a position can never exceed the account.
#
#  THE REAL TRADE-OFF — the cap was what let several positions run at once
#  (5 symbols x 20% = 100%). With it off, the first signal of the day can take
#  the whole account at 1x and later symbols size to qty=0 and are skipped.
#  This concentrates the book by design. Backtests already ignored this cap
#  (backtester.py drops it), so live sizing and backtest sizing now agree.
# --------------------------------------------------------------------------- #
INTRADAY_PARAMS = StrategyParams(
    mode=Mode.INTRADAY, timeframe="15m",
    # 1% max loss per trade (₹1,000 on a ₹1L account). Scales with capital, and
    # stays well inside the 2% ceiling Immutable Rule #1 forbids exceeding.
    risk_per_trade=0.01, risk_reward=1.0,   # 1:1 — see INTRADAY_RR_NOTE
    max_leverage=15.0,                      # defer to the segment's real cap
    max_capital_per_trade_pct=0.0,          # cap OFF - see INTRADAY_CAPITAL_CAP_NOTE
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
    max_capital_per_trade_pct=0.0,          # cap OFF - see INTRADAY_CAPITAL_CAP_NOTE
    atr_period=14,
    allow_short=True,           # MIS permits shorting, and half of the pattern
    max_leverage=15.0,          # library is bearish — long-only would discard it
    # -- MEASURED SETTINGS (2026-09-02) ------------------------------------ #
    #  Backtested over 8 Nifty names, Jan-Aug 2026, Rs2L, net of costs. The
    #  strategy ran at -30.53% with the old defaults and -0.38% with these,
    #  and the entire difference is trade COUNT: 2,890 positions cost
    #  Rs615,246 in brokerage and slippage, 173 positions cost Rs40,526.
    #  Gross was positive throughout; the strategy was being eaten by its own
    #  turnover, not by its direction.
    #
    #  Evidence per setting is in the CLAUDE.md changelog. None of these
    #  touches Immutable Rule #1 — risk per trade is still 1%.

    #: Pattern evidence needed to trade. BASE FLOOR ONLY (2026-10-07, owner's
    #: request): the admin selects the real threshold per run from the panel's
    #: Min score control (ModeConfig.min_score / the Broadcast tab), which
    #: overrides this. 3.0 = one high-strength single-candle pattern; the
    #: measured 7.0 (two agreeing patterns, +29 of the 30 points in the
    #: 2026-09-02 backtest) is now a panel choice, not the default.
    cs_min_score=3.0,
    cs_trend_lookback=10,
    #: Entry cutoff REMOVED (2026-10-07, owner's request) — was 190 (= no new
    #: entries after 11:59, measured +0.96pp). Entries now run to the normal
    #: session end; the 15:09 square-off still flattens everything.
    entry_cutoff_before_close=0,
    #: Floor the ATR stop at 0.8% of price. COUNTER-INTUITIVE and measured
    #: both ways: WIDENING stops pays (+0.19pp) because it converts -Rs1,205
    #: stop-outs into -Rs65 sideways exits. Capping them does the reverse and
    #: costs ~2pp, which is why max_stop_pct is deliberately left at 0.
    min_stop_pct=0.8,
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

# --------------------------------------------------------------------------- #
#  CRUDEOIL — US-session Opening Range Breakout (Commodity.md)
#
#  MCX crude barely moves on its own in the Indian morning; it moves when the
#  US session opens and NYMEX starts trading. So this strategy ignores the
#  09:00 IST MCX open entirely, anchors its VWAP and its opening range to
#  09:00 New York, and trades the break of that range.
#
#  Timeframe is 15m because Mode.INTRADAY's feed is 15m (data_feed.TF_MINUTES);
#  the opening range is therefore the first TWO bars after the anchor. Filed
#  under INTRADAY (not a new Mode) on purpose: INTRADAY is already in
#  SQUARE_OFF_MODES and already square-offs MCX at 23:15, which is exactly the
#  behaviour this strategy needs, so it inherits it rather than re-declaring it.
#
#  SIZING IS NOT SET HERE. Commodities size at the FIXED lot count the admin
#  puts next to the symbol in the sidebar (engine._mcx_fixed_size) — the same
#  path every other MCX trade takes. risk_per_trade below is the ceiling that
#  path is checked against, never a second sizing rule.
# --------------------------------------------------------------------------- #
CRUDEOIL_PARAMS = StrategyParams(
    mode=Mode.INTRADAY, timeframe="15m",
    # 1% risk ceiling — well inside the 2% Immutable Rule #1 forbids crossing.
    # Crude gaps harder than equity, so it gets the tighter of the two.
    risk_per_trade=0.01,
    risk_reward=1.5,            # in RR_CHOICES; admin/per-symbol may override
    atr_period=14, atr_sl_mult=1.5,
    allow_short=True,           # crude is genuinely two-sided
    use_atr_gate=True,          # skip dead tape and volatility spikes
    atr_median_window=50, atr_norm_low=0.5, atr_norm_high=2.0,
    context_bars=10, context_min_frac=0.6,
    use_limit_entry=True,       # bounds slippage, which bounds realised risk
    limit_offset_ticks=2.0,
    max_hold_minutes=0,         # no time stop — an ORB trade needs room
    reentry_cooldown_bars=3,
    max_leverage=25.0,          # MCX MIS margin ~4.5% => ~22x
    # -- multi-timeframe ----------------------------------------------------- #
    # Signal on the 15m base bar, bias from the 60m structure resampled off it.
    # Commodity.md specified 5m/15m; the base timeframe here is 15m because
    # Mode.INTRADAY's feed is 15m (data_feed.TF_MINUTES), so the SAME 1:4 ratio
    # is preserved one rung up rather than pretending to a 5m feed that does
    # not exist. Whatever the base becomes, htf_minutes must stay a multiple.
    htf_minutes=60, htf_bars=3,
    # -- the opening range itself ------------------------------------------- #
    orb_anchor_tz="America/New_York",
    orb_anchor_hhmm="09:00",    # = 18:30 IST (EDT) / 19:30 IST (EST)
    orb_minutes=30,             # two 15-minute bars
    orb_range_min_atr=0.3, orb_range_max_atr=3.0,
    entry_cutoff_before_close=45,   # no new entries after ~22:30 IST
    # -- scale-out (only ever applies to a MULTI-lot position) -------------- #
    partial_exit_fraction=0.5,
    breakeven_buffer_ticks=3.0,
    runner_rr_mult=3.0,
    trail_remainder=True,
)


# --------------------------------------------------------------------------- #
#  Crudeoil Pipeline bridge params (crudeoil_pipeline/, newcrudeoil.md)
#
#  These configure the PLATFORM side of the bridge only — the feed timeframe,
#  the risk ceiling the engine enforces, and the ATR inputs handed across.
#  Everything else the pipeline does (session anchoring, event overlays, the
#  cost gate, regime classification) is governed by
#  crudeoil_pipeline/config/settings.yaml, which stays that package's single
#  source of truth. Two config files is deliberate: the package must remain
#  runnable and testable with no WelthWest code present at all.
# --------------------------------------------------------------------------- #
CRUDEOIL_PIPELINE_PARAMS = StrategyParams(
    mode=Mode.INTRADAY, timeframe="15m",
    risk_per_trade=0.01,        # 1% — inside the 2% Immutable Rule #1 ceiling
    risk_reward=2.0,            # the pipeline's momentum target, in R
    atr_period=14, atr_sl_mult=1.25,
    allow_short=True,
    max_leverage=25.0,
    reentry_cooldown_bars=3,
)


# --------------------------------------------------------------------------- #
#  EXIT STYLES — the two profit-booking approaches, as CONFIGURATION.
#
#  Both are the same lifecycle (exit_manager.py): the strategy's ATR stop and
#  target are untouched, and everything AFTER the first target is what these
#  select. Nothing here is on by default — `params_for_mode` and every
#  StrategyParams below keep the plain fixed-RR exit unless a caller asks for
#  one of these by name (the backtest form, or an admin per-mode override).
#
#      "strategy"      whatever the strategy itself declares. The default, and
#                      the only style that can leave CRUDEOIL's own scale-out
#                      in place.
#      "fixed"         the BASELINE to measure against: no partial, no trail,
#                      exit strictly on the ATR stop / target / time.
#      "trail_full"    Approach 1 — no partial; trail the WHOLE position from
#                      entry behind `trail_atr_mult` x ATR.
#      "partial_trail" Approach 2 — book `partial_exit_fraction` at the first
#                      target, move the stop to break-even + buffer, trail the
#                      runner. `runner_rr_mult = 0` means the trail alone
#                      decides the runner's exit.
#      "partial_lock"  Approach 2b — the same partial, but the runner's stop is
#                      floored at TP1 rather than break-even, and it holds for
#                      a hard TP2 instead of trailing. Above TP1 and back to
#                      TP1 = out with the 1R kept; through to TP2 = out at 2R.
#                      Books more of its winners at exactly 1R and gives back
#                      nothing; the cost is every runner that would have gone
#                      to 2R+ after dipping through TP1 on the way.
#      "partial_ladder" Approach 2c — the same partial and the same TP2, but a
#                      TWO-STAGE stop. The runner starts on break-even +
#                      friction and is promoted to TP1 only once price reaches
#                      the MIDPOINT of TP1..TP2. It buys back exactly what 2b
#                      gives away — the runner that dips and recovers — and
#                      pays for it with the 1R that 2b would have banked on
#                      the ones that simply stall.
#
#  Which one is right is an EMPIRICAL question per instrument (see the MFE
#  study in the implementation brief) — that is exactly why they are runtime
#  options rather than an edit to a strategy file.
# --------------------------------------------------------------------------- #
EXIT_STYLES: tuple[str, ...] = ("strategy", "fixed", "trail_full",
                                "partial_trail", "partial_lock",
                                "partial_ladder")

#: Break-even buffer used by the managed styles, in basis points of entry.
#:
#: MEASURED, not chosen, and measured against the RUNNER rather than the whole
#: position — the runner is what this stop protects, and it is the smaller
#: half, so the flat per-order brokerage is spread over fewer shares. Against
#: cost_model, a half-size intraday equity runner's own round trip costs
#: ~19 bps at every price from Rs120 to Rs5,000 (the percentage heads are
#: price-invariant and the flat Rs20/order is ~8 bps of a Rs50k leg); an MCX
#: crude runner's costs ~12.5. 20 bps clears both, so a runner handed this stop
#: is genuinely free after costs rather than merely level before them — which
#: is the entire purpose of the buffer.
#:
#: The one place it can still fall marginally short is a very cheap share,
#: where the 0.05 tick grid is too coarse to express the level: the stop is
#: floored onto the grid (looser side, the convention everywhere here), which
#: on a Rs120 share can give back ~4 bps. Rounding it the other way would
#: protect more but contradicts the rounding rule the initial stop and the
#: trail both follow, and 4 bps of a typical 60 bps stop is not worth the
#: inconsistency.
#:
#: NOTE WHAT THIS COSTS: 20 bps against a typical 0.6% intraday stop is a third
#: of R. Protecting the runner properly is not free, and that is a reason to
#: MEASURE Approach 2 against the baseline rather than assume it wins.
#:
#: Re-derive it if the cost model's rates change.
BREAKEVEN_BUFFER_BPS = 20.0

#: TP2 for the locked-runner style, as a multiple of the original risk.
#:
#: MEASURED, like the buffer: analysis/mfe_study.py over 8 Nifty names puts
#: 30-47% of Intraday winners past 2R and only 6-24% past 3R, and the median
#: winner's excursion at ~1.6R. A 2R second target is therefore reachable by a
#: real share of runners; a 3R one mostly is not, and a runner held for a level
#: it never reaches just rides back down into its own stop.
RUNNER_TARGET_RR = 2.0

#: Where the runner must reach before its stop is promoted to TP1, as a
#: fraction of the TP1..TP2 span. Now only the FALLBACK for when no ATR is
#: available — LOCK_TRIGGER_ATR below is what normally decides.
LOCK_TRIGGER_FRAC = 0.5

#: The promotion trigger in ATR: the runner must travel this many ATR past TP1
#: before its stop is moved up to TP1.
#:
#: 1.5 IS AT THE EDGE OF THE SPAN, AND THAT IS THE POINT. The stop is itself
#: ATR-derived (`atr_sl_mult = 1.5`), so with a 1:1 target the whole TP1..TP2
#: span is about 1.5 x ATR — a 1.5 multiple therefore puts the trigger on TP2,
#: and the lock almost never fires before the target does.
#:
#: MEASURED, and it inverts the reason this setting was added. Over 8 Nifty
#: names, Jan-Aug 2026, 173 positions, mean net:
#:
#:     0.25 x ATR  -0.38%   lock fires on 60% of runners
#:     0.50 x ATR  -0.39%                  44%
#:     0.75 x ATR  -0.31%                  37%
#:     1.00 x ATR  -0.28%                  23%
#:     1.50 x ATR  -0.19%                  10%     <- shipped
#:     3.00 x ATR  -0.19%                   0%
#:     no lock     -0.19%                   0%
#:
#: Monotonic: the MORE the lock fires, the worse the result, and 1.5 scores
#: exactly what switching the lock off entirely scores. Locking the runner at
#: TP1 does not protect a gain on this strategy, it truncates the 25% of
#: runners that would have reached TP2. The value is kept as a trigger rather
#: than turned into `lock_first_target = False` so the machinery stays
#: available for a strategy whose runners behave differently — but on this one
#: it is deliberately set where it does nothing.
LOCK_TRIGGER_ATR = 1.5

#: Fraction of the position booked at the first target for the ladder style.
#:
#: 0.30 rather than a half, and measured: on the same 173 positions, booking
#: 30% nets -0.28% against -0.42% for 50%. Two reasons, both real — a bigger
#: runner is what the two-stage stop exists to carry, and the larger remainder
#: spreads the flat per-order fee over more shares (on a 100-share position
#: the runner's round trip falls from ~Rs1.04 to ~Rs0.77 a share).
LADDER_PARTIAL_FRACTION = 0.30

#: Style -> the StrategyParams fields it pins. Absent fields are left alone.
_EXIT_STYLE_FIELDS: dict[str, dict] = {
    "strategy": {},
    "fixed": dict(partial_exit_fraction=0.0, trail_remainder=False,
                  trail_from_entry=False),
    # Both managed styles pin the break-even buffer in BPS, not ticks. A tick
    # count only ever describes the one instrument it was tuned on (see
    # StrategyParams.breakeven_buffer_ticks); 15 bps clears a round trip on
    # intraday equity (~14.3) and on MCX crude (~12.5) alike, so the same
    # style means the same thing on every symbol these run against.
    "trail_full": dict(partial_exit_fraction=0.0, breakeven_buffer_ticks=0.0,
                       breakeven_buffer_bps=BREAKEVEN_BUFFER_BPS,
                       runner_rr_mult=0.0, trail_remainder=False,
                       trail_from_entry=True),
    "partial_trail": dict(partial_exit_fraction=0.5,
                          breakeven_buffer_ticks=0.0,
                          breakeven_buffer_bps=BREAKEVEN_BUFFER_BPS,
                          runner_rr_mult=0.0, lock_first_target=False,
                          trail_remainder=True, trail_from_entry=False),
    # Approach 2b. The buffer is still declared even though the TP1 floor
    # always supersedes it: it is what the runner falls back to if a future
    # change ever makes TP1 unreachable, and leaving it at 0 would look like a
    # decision to run the runner at bare break-even.
    #
    # runner_rr_mult = 2.0, not 3.0: measured on the MFE study, ~30-47% of
    # Intraday winners that reach 1R go on to 2R but only ~6-24% reach 3R, so
    # a 3R second target is a level most runners would simply never see.
    "partial_lock": dict(partial_exit_fraction=0.5,
                         breakeven_buffer_ticks=0.0,
                         breakeven_buffer_bps=BREAKEVEN_BUFFER_BPS,
                         runner_rr_mult=RUNNER_TARGET_RR,
                         lock_first_target=True, lock_trigger_frac=0.0,
                         trail_remainder=False, trail_from_entry=False),
    # Approach 2c. Identical to 2b except WHEN the stop is promoted: at the
    # midpoint of TP1..TP2 rather than at TP1 itself.
    "partial_ladder": dict(partial_exit_fraction=LADDER_PARTIAL_FRACTION,
                           breakeven_buffer_ticks=0.0,
                           breakeven_buffer_bps=BREAKEVEN_BUFFER_BPS,
                           runner_rr_mult=RUNNER_TARGET_RR,
                           lock_first_target=True,
                           lock_trigger_atr_mult=LOCK_TRIGGER_ATR,
                           lock_trigger_frac=LOCK_TRIGGER_FRAC,
                           trail_remainder=False, trail_from_entry=False),
}


def is_valid_exit_style(style: str) -> bool:
    """Whether `style` names one of EXIT_STYLES. Blank counts as valid and
    means "strategy" — an unset field must never be an error."""
    return (style or "strategy").strip().lower() in _EXIT_STYLE_FIELDS


def apply_exit_style(params: StrategyParams, style: str,
                     trail_atr_mult: float = 0.0,
                     partial_exit_fraction: float = -1.0,
                     runner_rr_mult: float = -1.0) -> StrategyParams:
    """Return `params` with one of EXIT_STYLES applied. Never mutates.

    Only the exit knobs move. `atr_sl_mult`, `risk_reward`, `risk_per_trade`
    and every sizing input are passed through untouched, so a style change can
    never alter the entry, the initial stop/target, or the position size
    (Immutable Rule #1).

    The three numeric arguments are the sweep handles: -1.0 (and 0.0 for the
    trail) means "leave whatever the style chose". They are applied AFTER the
    style so a walk-forward can move one knob at a time.
    """
    style = (style or "strategy").strip().lower()
    if style not in _EXIT_STYLE_FIELDS:
        raise ValueError(
            f"Unknown exit style {style!r}; use one of {', '.join(EXIT_STYLES)}.")
    fields = dict(_EXIT_STYLE_FIELDS[style])
    if trail_atr_mult and trail_atr_mult > 0:
        fields["trail_atr_mult"] = float(trail_atr_mult)
    if partial_exit_fraction is not None and partial_exit_fraction >= 0:
        # 0.0 is a MEANINGFUL value here (no partial), hence the -1 sentinel.
        if not 0.0 <= partial_exit_fraction < 1.0:
            raise ValueError("partial_exit_fraction must be in [0, 1).")
        fields["partial_exit_fraction"] = float(partial_exit_fraction)
    if runner_rr_mult is not None and runner_rr_mult >= 0:
        fields["runner_rr_mult"] = float(runner_rr_mult)
    if not fields:
        return params
    return replace(params, **fields)


# --------------------------------------------------------------------------- #
#  INTRADAY, under each of the two profit-booking approaches.
#
#  Written out as constants rather than left implicit in EXIT_STYLES so the
#  exact settings Intraday runs under each approach are readable in one place —
#  and so a test can assert on them, which is what stops the styles drifting
#  away from what was measured.
#
#  NOTHING HERE IS A DEFAULT. `params_for_mode(Mode.INTRADAY)` still returns
#  INTRADAY_PARAMS, i.e. the plain fixed 1:1 exit, and these are reached only
#  when a run or a mode explicitly selects the style by name. That is
#  deliberate: which approach is right is an empirical question per book (see
#  analysis/mfe_study.py), and the baseline is what they must beat NET of the
#  costs the extra exit adds.
#
#  What is Intraday-specific about them:
#    * The break-even buffer is in BPS, not ticks. Intraday trades equity
#      across a 10x price range, where a tick count that is right for a Rs250
#      share is off by 13x on a Rs3,200 one.
#    * `runner_rr_mult = 0` — the trail alone exits the runner, with no hard
#      second target. Measured on the MFE study, only ~30% of Intraday winners
#      that reach 1R go on to 2R and ~6% to 3R, so a fixed 3R runner target is
#      a level most runners will simply never reach; the trail books what the
#      move actually gave.
#    * The trail distance is `atr_sl_mult` (1.5xATR(14) on 15m bars) — the same
#      multiple the entry stop was built from, so the trail starts out exactly
#      as far from price as the risk the trade was sized against.
# --------------------------------------------------------------------------- #

#: Approach 1 — no partial; trail the WHOLE position from entry.
INTRADAY_TRAIL_FULL_PARAMS = apply_exit_style(INTRADAY_PARAMS, "trail_full")

#: Approach 2 — book half at the first target, trail the runner from
#: break-even + friction.
INTRADAY_PARTIAL_TRAIL_PARAMS = apply_exit_style(INTRADAY_PARAMS,
                                                 "partial_trail")

#: Approach 2b — book half at TP1, floor the runner's stop AT TP1, and hold it
#: for TP2 (2R). The variant that refuses to give back the 1R it earned.
INTRADAY_PARTIAL_LOCK_PARAMS = apply_exit_style(INTRADAY_PARAMS,
                                                "partial_lock")

#: Approach 2c — the same, but the runner sits on break-even + friction until
#: it reaches the midpoint of TP1..TP2, and only then is promoted to TP1.
INTRADAY_PARTIAL_LADDER_PARAMS = apply_exit_style(INTRADAY_PARAMS,
                                                  "partial_ladder")


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
