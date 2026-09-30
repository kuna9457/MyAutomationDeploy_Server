"""
api/schemas.py
Pydantic request/response shapes for the FastAPI layer. These are pure
transport types — they carry the exact same fields app.py already reads off
its sidebar widgets, nothing more, so the engine/strategy/broker layers
receive identical inputs to what they get from Streamlit today.
"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class StartBotRequest(BaseModel):
    environment: str          # "Paper" | "Live"
    # strategy_key/segments/symbols/mcx_lots are ADMIN-ONLY fields: for a
    # client (see api/routers/bot.py) they're ignored entirely and resolved
    # server-side from admin_config instead, so all of them are optional here
    # — a client's request simply omits them.
    # `mode` is the one exception: a client MAY send it, but it is checked
    # against admin_config.available_client_modes() before use.
    mode: str = ""             # "Intraday" | "Swing" | "Scalper"
    strategy_key: str = ""
    segments: list[str] = []  # ["NSE_EQUITY", "MCX_COMMODITY"]
    symbols: list[str] = []   # instrument symbols within those segments
    capital: float
    broker: Optional[str] = None          # required when environment == "Live"
    mcx_lots: dict[str, int] = {}
    #: ADMIN-ONLY, like the fields above. Risk:reward for this run; 0 = the
    #: strategy's own. A client's value is ignored — theirs comes from the
    #: admin's saved ModeConfig, so they cannot widen or narrow their target.
    risk_reward: float = 0.0
    #: ADMIN-ONLY. Signal-score threshold for this run; 0 = the strategy's
    #: own. Decides how much pattern evidence an entry needs. A client's
    #: value is ignored — theirs comes from the admin's saved ModeConfig.
    min_score: float = 0.0
    #: ADMIN-ONLY. End-of-session flat-out; "" = the segment default.
    square_off_time: str = ""
    square_off_enabled: bool = True
    #: ADMIN-ONLY. How an OPEN position is managed after its first target —
    #: one of config.EXIT_STYLES. "strategy" = whatever the strategy declares,
    #: which is what every request sent before this field existed means.
    #: Backtest a style (POST /backtest/run takes the same names) before
    #: running it: the two share exit_manager, so the numbers transfer.
    #: A client's value is ignored — theirs comes from admin's ModeConfig.
    exit_style: str = "strategy"
    #: ADMIN-ONLY. Chandelier trail distance in ATR; 0 = the strategy's own.
    trail_atr_mult: float = 0.0
    #: Widest the ATR-derived stop may sit from entry, in PERCENT of price.
    #: 0 = the strategy's own, unbounded. The target moves with it, so the
    #: reward:risk is unchanged. NOTE the trade-off before using it: a tighter
    #: stop converts sideways square-off exits (worth about -Rs65 each) into
    #: stop-outs (about -Rs1,200 each), so fewer square-offs is NOT
    #: automatically better. Backtest before switching it on.
    max_stop_pct: float = 0.0
    #: Tightest the stop may sit, in PERCENT. 0 = no floor.
    min_stop_pct: float = 0.0


class RiskLimitsRequest(BaseModel):
    capital_allocated: float = 0.0
    max_daily_loss_cash: float = 0.0
    max_daily_loss_pct: float = 0.0
    max_trades_per_day: int = 0
    max_qty_per_trade: int = 0
    intraday_leverage: float = 1.0


class BacktestRequest(BaseModel):
    ticker: str
    mode: str                 # "Intraday" | "Swing" | "Scalper"
    strategy_key: str = ""
    start: str
    end: str
    initial_capital: float = 100_000.0
    #: 0 = the strategy's own RR. Lets you compare 1:1 against 1:2 on the same
    #: symbol and window before committing the change to live.
    risk_reward: float = 0.0
    #: 0 = the strategy's own threshold. The point of exposing it here is to
    #: measure a score change on real history BEFORE putting it in front of
    #: the market — same symbol, same window, one variable moved.
    min_score: float = 0.0
    #: BACKTEST-ONLY entry filters (backtester.TradeFilters). All empty/BOTH =
    #: unrestricted, which is the original behaviour exactly. These gate
    #: ENTRIES only; an open position is always managed to its exit.
    #: Weekday numbers, Monday = 0.
    trade_days: list[int] = []
    #: IST hours 0-23; hour H covers H:00-H:59.
    trade_hours: list[int] = []
    #: "BOTH" | "BUY" | "SELL"
    side: str = "BOTH"
    #: PER-RUN candlestick pattern allow-list. EMPTY = fall back to the saved
    #: dashboard filter, so an untouched form measures what the live bot
    #: actually trades. Non-empty overrides it for this run only and never
    #: writes anything back.
    patterns: list[str] = []
    #: 0 = the mode's own bar size (15 for Intraday). Only meaningful for
    #: Intraday — Swing (daily) and Scalper (native 1m) ignore it. Lets you
    #: compare a 5m base against the saved 15m before committing the change
    #: to live, the same way risk_reward/min_score already let you.
    timeframe_minutes: int = 0
    #: How a position is managed AFTER its first target — one of
    #: config.EXIT_STYLES. "strategy" (the default) measures exactly what the
    #: strategy declares, so an untouched form is the run it was before this
    #: existed. The others let the two profit-booking approaches be compared
    #: on the same symbol and window with one variable moved:
    #:   "fixed"         — baseline: fixed ATR stop/target, nothing managed
    #:   "trail_full"    — Approach 1: trail the whole position from entry
    #:   "partial_trail" — Approach 2: book part at 1R, trail the runner
    exit_style: str = "strategy"
    #: Chandelier trail distance in ATR. 0 = the strategy's own. Kept separate
    #: from atr_sl_mult on purpose: that one also sets the ENTRY stop, so
    #: sweeping the trail through it would move two things at once.
    trail_atr_mult: float = 0.0
    #: Fraction booked at the first target. -1 = whatever the style chose.
    #: 0 is meaningful (no partial), hence the sentinel.
    partial_exit_fraction: float = -1.0
    #: Runner target in multiples of the original risk. -1 = the style's own;
    #: 0 means no hard runner target, i.e. the trail alone decides the exit.
    runner_rr_mult: float = -1.0
    #: Widest the ATR-derived stop may sit from entry, in PERCENT of price.
    #: 0 = the strategy's own, unbounded. The target moves with it, so the
    #: reward:risk is unchanged. NOTE the trade-off before using it: a tighter
    #: stop converts sideways square-off exits (worth about -Rs65 each) into
    #: stop-outs (about -Rs1,200 each), so fewer square-offs is NOT
    #: automatically better. Backtest before switching it on.
    max_stop_pct: float = 0.0
    #: Tightest the stop may sit, in PERCENT. 0 = no floor.
    min_stop_pct: float = 0.0
    #: HOLD PAST THE SESSION. False (the default) = every existing request,
    #: unchanged: an Intraday/Scalper position is squared off at the segment
    #: flat-out (15:09 equity) whatever its P&L. True removes that flat-out
    #: AND the entry cutoff derived from it, so a position closes only on its
    #: own stop, target, trail or max-hold.
    #:
    #: BACKTEST-ONLY, and deliberately absent from AdminConfigRequest: this
    #: answers "did the signal have edge when given room?", it does not
    #: configure the live bot. A held-overnight cash position is DELIVERY —
    #: it cannot be short, and it pays delivery STT/stamp that the backtest's
    #: INTRADAY_EQUITY cost model does not charge, so a net figure from this
    #: flag reads better than the same trades would have.
    hold_overnight: bool = False
    #: Drop the strategy's late-entry gate (entry_cutoff_before_close — 11:59
    #: for Candlestick Intraday). SEPARATE from hold_overnight on purpose: this
    #: changes how many trades are taken, that one changes how they close.
    #: Setting both in one run moves two variables and attributes neither.
    ignore_entry_cutoff: bool = False


class BulkBacktestRequest(BaseModel):
    """The same strategy and window across a bucket of symbols, ranked."""
    tickers: list[str]
    mode: str
    strategy_key: str = ""
    start: str
    end: str
    initial_capital: float = 100_000.0
    risk_reward: float = 0.0
    min_score: float = 0.0
    trade_days: list[int] = []
    trade_hours: list[int] = []
    side: str = "BOTH"
    #: Held constant across every symbol in the bucket, so the ranking compares
    #: symbols and nothing else. Empty = fall back to the saved dashboard
    #: filter, exactly as in a single run.
    patterns: list[str] = []
    #: 0 = the mode's own bar size. See BacktestRequest.timeframe_minutes.
    timeframe_minutes: int = 0
    #: How an OPEN position is managed after its first target — one of
    #: config.EXIT_STYLES. "strategy" (the default) is the plain fixed exit
    #: for every strategy that declares no management of its own, so a bulk
    #: run left at the default measures a DIFFERENT bot from a single-symbol
    #: run with a style selected. Pass the same style you intend to trade.
    exit_style: str = "strategy"
    #: Chandelier trail distance in ATR; 0 = the strategy's own.
    trail_atr_mult: float = 0.0
    #: Fraction booked at the first target; -1 = the style's own.
    partial_exit_fraction: float = -1.0
    #: Runner target in R; -1 = the style's own, 0 = trail only.
    runner_rr_mult: float = -1.0
    #: Bounds on the ATR stop, in PERCENT of price. 0 = the strategy's own.
    max_stop_pct: float = 0.0
    min_stop_pct: float = 0.0
    #: HOLD PAST THE SESSION. False (the default) = every existing request,
    #: unchanged: an Intraday/Scalper position is squared off at the segment
    #: flat-out (15:09 equity) whatever its P&L. True removes that flat-out
    #: AND the entry cutoff derived from it, so a position closes only on its
    #: own stop, target, trail or max-hold.
    #:
    #: BACKTEST-ONLY, and deliberately absent from AdminConfigRequest: this
    #: answers "did the signal have edge when given room?", it does not
    #: configure the live bot. A held-overnight cash position is DELIVERY —
    #: it cannot be short, and it pays delivery STT/stamp that the backtest's
    #: INTRADAY_EQUITY cost model does not charge, so a net figure from this
    #: flag reads better than the same trades would have.
    hold_overnight: bool = False
    #: Drop the strategy's late-entry gate (entry_cutoff_before_close — 11:59
    #: for Candlestick Intraday). SEPARATE from hold_overnight on purpose: this
    #: changes how many trades are taken, that one changes how they close.
    #: Setting both in one run moves two variables and attributes neither.
    ignore_entry_cutoff: bool = False


class RRSweepRequest(BaseModel):
    """The same backtest run once per risk:reward in a ladder.

    Everything except `risk_reward` is held constant, so the resulting table
    isolates RR as the single variable. Bounds are enforced server-side
    (backtester.rr_sweep_values) — a step of 0.001 would otherwise queue
    thousands of full simulations.
    """
    ticker: str
    mode: str
    strategy_key: str = ""
    start: str
    end: str
    initial_capital: float = 100_000.0
    #: First RR to test (reward per 1 unit of risk, so 1.0 is 1:1).
    rr_start: float = 1.0
    #: How much to add each step.
    rr_step: float = 0.25
    #: Last RR to test, inclusive.
    rr_end: float = 3.0
    #: 0 = the strategy's own threshold. Held constant across the sweep.
    min_score: float = 0.0
    #: Held constant across the sweep, like every other input except RR.
    patterns: list[str] = []
    #: 0 = the mode's own bar size. See BacktestRequest.timeframe_minutes.
    timeframe_minutes: int = 0
    #: How an OPEN position is managed after its first target — one of
    #: config.EXIT_STYLES. "strategy" (the default) is the plain fixed exit
    #: for every strategy that declares no management of its own, so a bulk
    #: run left at the default measures a DIFFERENT bot from a single-symbol
    #: run with a style selected. Pass the same style you intend to trade.
    exit_style: str = "strategy"
    #: Chandelier trail distance in ATR; 0 = the strategy's own.
    trail_atr_mult: float = 0.0
    #: Fraction booked at the first target; -1 = the style's own.
    partial_exit_fraction: float = -1.0
    #: Runner target in R; -1 = the style's own, 0 = trail only.
    runner_rr_mult: float = -1.0
    #: Bounds on the ATR stop, in PERCENT of price. 0 = the strategy's own.
    max_stop_pct: float = 0.0
    min_stop_pct: float = 0.0
    #: HOLD PAST THE SESSION. False (the default) = every existing request,
    #: unchanged: an Intraday/Scalper position is squared off at the segment
    #: flat-out (15:09 equity) whatever its P&L. True removes that flat-out
    #: AND the entry cutoff derived from it, so a position closes only on its
    #: own stop, target, trail or max-hold.
    #:
    #: BACKTEST-ONLY, and deliberately absent from AdminConfigRequest: this
    #: answers "did the signal have edge when given room?", it does not
    #: configure the live bot. A held-overnight cash position is DELIVERY —
    #: it cannot be short, and it pays delivery STT/stamp that the backtest's
    #: INTRADAY_EQUITY cost model does not charge, so a net figure from this
    #: flag reads better than the same trades would have.
    hold_overnight: bool = False
    #: Drop the strategy's late-entry gate (entry_cutoff_before_close — 11:59
    #: for Candlestick Intraday). SEPARATE from hold_overnight on purpose: this
    #: changes how many trades are taken, that one changes how they close.
    #: Setting both in one run moves two variables and attributes neither.
    ignore_entry_cutoff: bool = False


class WatchlistSaveRequest(BaseModel):
    name: str
    symbols: list[str]


class BrokerCredentialsRequest(BaseModel):
    """A client's OWN broker-app credentials. Write-only: there is no response
    model that carries these back, by design."""
    api_key: str
    api_secret: str


class UpstoxExchangeRequest(BaseModel):
    code: str


class ZerodhaExchangeRequest(BaseModel):
    request_token: str


class CreateClientRequest(BaseModel):
    username: str
    password: str
    #: REQUIRED: where this client's forgot-password code is sent. Mandatory
    #: at creation because an account without one can never self-serve a
    #: reset — it silently becomes admin's problem forever. `display_name`
    #: stays optional; it is cosmetic, this is not.
    email: str
    display_name: str = ""


class SetEmailRequest(BaseModel):
    email: str


class ChangePasswordRequest(BaseModel):
    """Signed-in password change. The current password is required so a
    hijacked session cannot lock the real owner out of their own account."""
    current_password: str
    new_password: str


class PasswordResetRequest(BaseModel):
    """Step 1 of forgot-password: mail a one-time code.

    Identifies the account by USERNAME, not email — one mailbox may hold more
    than one login, and the address alone would be ambiguous.
    """
    username: str


class PasswordResetConfirm(BaseModel):
    """Step 2: redeem the code and set the new password."""
    username: str
    code: str
    new_password: str


class SetStatusRequest(BaseModel):
    status: str  # "active" | "disabled"


class SetPasswordRequest(BaseModel):
    password: str


class AdminConfigRequest(BaseModel):
    """One mode's client-facing config. `mode` names WHICH mode is being
    saved — other modes' entries are left untouched."""
    mode: str
    strategy_key: str
    segments: list[str]
    symbols: list[str]
    mcx_lots: dict[str, int] = {}
    #: Risk:reward for this mode. 0 = inherit the strategy's own. Validated
    #: against config.RR_CHOICES in the route, so a client of the API can't
    #: post an arbitrary ratio.
    risk_reward: float = 0.0
    #: Signal-score threshold for this mode. 0 = inherit the strategy's own.
    #: Range-checked in the route (config.is_valid_min_score).
    min_score: float = 0.0
    #: End-of-session flat-out. "" = the segment default (15:09 equity).
    square_off_time: str = ""
    square_off_enabled: bool = True
    #: How an OPEN position is managed after its first target, for this mode —
    #: one of config.EXIT_STYLES, validated in the route. "strategy" = inherit
    #: whatever the chosen strategy declares, which is what every config saved
    #: before this field existed reads as.
    exit_style: str = "strategy"
    #: Chandelier trail distance in ATR. 0 = the strategy's own.
    trail_atr_mult: float = 0.0
    #: Widest the ATR-derived stop may sit from entry, in PERCENT of price.
    #: 0 = the strategy's own, unbounded. The target moves with it, so the
    #: reward:risk is unchanged. NOTE the trade-off before using it: a tighter
    #: stop converts sideways square-off exits (worth about -Rs65 each) into
    #: stop-outs (about -Rs1,200 each), so fewer square-offs is NOT
    #: automatically better. Backtest before switching it on.
    max_stop_pct: float = 0.0
    #: Tightest the stop may sit, in PERCENT. 0 = no floor.
    min_stop_pct: float = 0.0


class RangeResetRequest(BaseModel):
    """Delete trades in a date range — the surgical alternative to wiping a
    whole environment.

    `confirm` defaults to FALSE, which makes the request a PREVIEW: it reports
    exactly what would be deleted and removes nothing. Nothing is destroyed
    until the same request comes back with confirm=true, so a wipe can never
    happen from a single mis-click or a half-filled form.
    """
    environment: str          # "Paper" | "Live"
    start: str                # "YYYY-MM-DD", inclusive
    end: str                  # "YYYY-MM-DD", inclusive
    #: "" = every category. Lets you drop just the simulated commodity trades
    #: while keeping equity history from the same days.
    category: str = ""
    #: "" = the admin's own book. A username scopes the delete to that client.
    username: str = ""
    confirm: bool = False


class ClientModesRequest(BaseModel):
    modes: list[str]          # subset of admin_config.CLIENT_SELECTABLE_MODES


class SymbolConfigRequest(BaseModel):
    """One instrument's own settings within one mode (symbol_config.py).

    EVERY field defaults to its "not configured" value, so an omitted field
    means "leave this at the strategy's behaviour" — and a body of all
    defaults is treated as a reset, deleting the entry rather than storing an
    inert one. The route validates via symbol_config.validate() before writing.
    """
    mode: str                          # "Intraday" | "Swing" | "Scalper"
    symbol: str
    #: Weekdays new entries may open on, 0=Mon .. 6=Sun. [] = every day.
    trade_days: list[int] = []
    #: HOURS (0-23 IST) new entries may open in. Hour H covers H:00-H:59, so
    #: gaps are expressible: [9,10,11,15] skips 12:00-14:59. [] = no filter.
    trade_hours: list[int] = []
    #: "HH:MM" IST. "" = the segment's own session open/close.
    start_time: str = ""
    end_time: str = ""
    #: 0 = inherit the mode/strategy RR. Validated against config.RR_CHOICES.
    risk_reward: float = 0.0
    #: Close an open position when the window ends. Default off.
    square_off_at_end: bool = False
    #: Trail this symbol's stop behind the best price reached. Default off.
    trail_enabled: bool = False
    #: ATR multiple the trail sits behind the peak. 0 = inherit the strategy's
    #: own atr_sl_mult. Ignored entirely unless trail_enabled.
    trail_atr_mult: float = 0.0


class StrategyGroupPayload(BaseModel):
    """One strategy and the stocks dropped onto it (strategy_groups.py)."""
    strategy_key: str
    symbols: list[str] = []
    mcx_lots: dict[str, int] = {}
    #: 0 = use the strategy's own. Validated against config.RR_CHOICES.
    risk_reward: float = 0.0
    #: 0 = use the strategy's own.
    min_score: float = 0.0
    enabled: bool = True


class PatternConfigRequest(BaseModel):
    """One candlestick strategy's pattern allow-list, for one mode.

    `enabled=False` OR an empty `allowed` means no filtering at all — the
    strategy behaves exactly as it did before the feature existed. Empty-while-
    enabled is deliberately a no-op rather than "block everything", so building
    a list up cannot accidentally halt trading.
    """
    strategy_key: str
    mode: str
    enabled: bool = False
    allowed: list[str] = []


class StrategyGroupsRequest(BaseModel):
    """The WHOLE board for one mode. Replaces rather than merges — moving a
    stock between strategies edits two groups at once."""
    mode: str
    groups: list[StrategyGroupPayload] = []


class PresetSaveRequest(BaseModel):
    """A named snapshot of the whole Controls sidebar (presets.py).

    Purely a transport shape — every field mirrors one sidebar widget, and the
    route hands them straight to presets.validate(). Saving is inert: it never
    starts, stops or reconfigures a running bot.
    """
    name: str
    environment: str = "Paper"
    mode: str = "Intraday"
    broker: str = ""
    strategy_key: str = ""
    segments: list[str] = ["NSE_EQUITY"]
    symbols: list[str] = []
    capital: float = 100_000.0
    min_score: float = 0.0
    mcx_lots: dict[str, int] = {}
    #: symbol -> that symbol's settings for `mode`, as SymbolConfigRequest's
    #: fields. Omitted entirely, the preset simply carries no per-symbol
    #: customisation and loading it clears the mode's settings.
    symbol_configs: dict[str, dict] = {}
