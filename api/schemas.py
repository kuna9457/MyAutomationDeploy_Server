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
    display_name: str = ""


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
    #: "HH:MM" IST. "" = the segment's own session open/close.
    start_time: str = ""
    end_time: str = ""
    #: 0 = inherit the mode/strategy RR. Validated against config.RR_CHOICES.
    risk_reward: float = 0.0
    #: Close an open position when the window ends. Default off.
    square_off_at_end: bool = False


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
    mcx_lots: dict[str, int] = {}
    #: symbol -> that symbol's settings for `mode`, as SymbolConfigRequest's
    #: fields. Omitted entirely, the preset simply carries no per-symbol
    #: customisation and loading it clears the mode's settings.
    symbol_configs: dict[str, dict] = {}
