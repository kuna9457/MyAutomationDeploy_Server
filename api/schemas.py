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


class ClientModesRequest(BaseModel):
    modes: list[str]          # subset of admin_config.CLIENT_SELECTABLE_MODES
