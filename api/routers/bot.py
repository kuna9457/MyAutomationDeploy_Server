"""
api/routers/bot.py
Start/stop/status/close-position — the exact operations app.py's sidebar
"Start Bot"/"Stop Bot" buttons and per-trade "Close" buttons perform, just
reached over HTTP instead of a Streamlit rerun. TradingEngine itself
(engine.py) is called with identical arguments; nothing about sizing,
signals, or order placement changes.

Each logged-in user gets their own TradingEngine, keyed by username in
engine_registry — Phase 1's one shared admin bot and Phase 2's many
concurrent client bots are the same code path.

role="client" is handled differently at start: strategy/segments/instruments
are NEVER taken from the request body — they're read server-side from
admin_config, so a client can't select or bypass what they trade even by
editing the request. `mode` IS a client choice, but only from the set admin
enabled (admin_config.available_client_modes()); anything else is rejected
rather than silently substituted, and the strategy/instruments that come
with it are still the admin's, looked up per mode. Their own broker access
token (saved via /broker/*/exchange) is required for Live and resolved
server-side too, never accepted as input.
"""
from __future__ import annotations

import admin_config
import config
import user_manager
from api import engine_registry
from api.auth import CurrentUser, get_current_user
from api.schemas import StartBotRequest
from config import Broker, Environment, Mode
from db_manager import DBManager
from engine import TradingEngine
from fastapi import APIRouter, Depends, HTTPException

router = APIRouter(prefix="/bot", tags=["bot"])

# One long-lived manager for this router, matching trades.py/admin_users.py.
# `reset_environment` below used to construct a throwaway DBManager per request
# and never close it, leaking a client and its monitor threads on every call.
_db = DBManager()

_BROKER_LABELS = {"Upstox": Broker.UPSTOX, "Dhan": Broker.DHAN,
                  "Zerodha": Broker.ZERODHA, "Kotak Neo": Broker.KOTAK}
# Only brokers with a per-client OAuth token flow wired up (api/routers/broker.py)
# can be used by a CLIENT for Live trading. Dhan/Kotak are admin-only until
# their own per-client credential flow exists (frontend_migration_plan.md §8).
_CLIENT_LIVE_BROKERS = {"Upstox", "Zerodha"}


def _token_is_live(broker: str, token: str, api_key: str) -> bool:
    """Read-only validity probe before a LIVE start. Network failures are
    treated as "live" so a transient blip can't block a start — the broker
    itself is still the final authority on every order."""
    try:
        if broker == "Upstox":
            import upstox_auth
            res = upstox_auth.check_token(token)
        else:
            import kite_auth
            res = kite_auth.check_token(token, api_key)
    except Exception:
        return True
    if res.get("ok"):
        return True
    return "Network error" in (res.get("error") or "")


@router.post("/start")
def start_bot(req: StartBotRequest, user: CurrentUser = Depends(get_current_user)):
    try:
        environment = Environment(req.environment)
    except ValueError as exc:
        raise HTTPException(400, f"Invalid environment: {exc}")

    access_token = ""
    broker_api_key = ""

    if user.role == "client":
        allowed = admin_config.available_client_modes()
        if not allowed:
            raise HTTPException(
                400, "Trading hasn't been configured yet — ask your admin to "
                     "set a strategy and instruments before you can start.")
        # An omitted mode means "the only one on offer"; with several enabled
        # the client must name one rather than get an arbitrary default.
        requested = req.mode or (allowed[0] if len(allowed) == 1 else "")
        if requested not in allowed:
            raise HTTPException(
                400, f"Pick a trading mode to run. Available to you: "
                     f"{', '.join(allowed)}.")
        mode_cfg = admin_config.get_mode_config(requested)
        mode = Mode(requested)
        strategy_key = mode_cfg.strategy_key
        symbols = mode_cfg.symbols
        mcx_lots = mode_cfg.mcx_lots
        if environment == Environment.LIVE:
            if req.broker not in _CLIENT_LIVE_BROKERS:
                raise HTTPException(
                    400, f"Live trading is only available via "
                         f"{'/'.join(_CLIENT_LIVE_BROKERS)} for client accounts.")
            cred_key, cred_secret = user_manager.get_broker_credentials(
                user.user_id, req.broker)
            if not (cred_key and cred_secret):
                raise HTTPException(
                    400, f"Add your {req.broker} API key and secret first "
                         "(Broker Credentials panel).")
            access_token = user_manager.get_broker_token(user.user_id, req.broker)
            if not access_token:
                raise HTTPException(
                    400, f"Connect your {req.broker} account first (see the "
                         "Broker panel below).")
            # Pre-flight: catch the expired-overnight case here, with an
            # actionable message, instead of at the first order attempt.
            if not _token_is_live(req.broker, access_token, cred_key):
                raise HTTPException(
                    400, f"Your {req.broker} session has expired — reconnect "
                         "your broker, then start again.")
            broker_api_key = cred_key
    else:
        try:
            mode = Mode(req.mode)
        except ValueError as exc:
            raise HTTPException(400, f"Invalid mode: {exc}")
        strategy_key = req.strategy_key
        symbols = req.symbols
        mcx_lots = req.mcx_lots

    selected = [config.INSTRUMENTS_BY_SYMBOL[s] for s in symbols
               if s in config.INSTRUMENTS_BY_SYMBOL]
    if not selected:
        raise HTTPException(400, "Select at least one instrument.")

    broker_choice = Broker.SIMULATED
    if environment == Environment.LIVE:
        if req.broker not in _BROKER_LABELS:
            raise HTTPException(400, "A valid broker is required for Live trading.")
        broker_choice = _BROKER_LABELS[req.broker]

    existing = engine_registry.get_engine(user.username)
    if existing and existing.state.running:
        existing.stop()

    eng = TradingEngine(environment, mode, broker_choice, selected, req.capital,
                        strategy_key=strategy_key, mcx_lots=mcx_lots,
                        user_id=user.username, broker_access_token=access_token,
                        broker_api_key=broker_api_key)
    try:
        eng.start()
    except RuntimeError as exc:
        raise HTTPException(400, str(exc))
    engine_registry.set_engine(user.username, eng)
    return {"ok": True, "strategy": eng.strategy.name, "broker": eng.broker.name}


@router.post("/stop")
def stop_bot(user: CurrentUser = Depends(get_current_user)):
    stopped = engine_registry.stop_engine(user.username)
    return {"ok": stopped}


@router.get("/status")
def bot_status(user: CurrentUser = Depends(get_current_user)):
    eng = engine_registry.get_engine(user.username)
    if eng is None:
        return {"started": False}
    snap = eng.state.snapshot()
    out = {
        "started": True,
        "environment": eng.environment.value,
        "mode": eng.mode.value,
        "total_capital": eng.total_capital,
        **snap,
    }
    # A client is told which phase is running, never which strategy runs it
    # (same reasoning as /config/client-modes). Admin still gets both.
    if user.role != "client":
        out["strategy"] = {"key": eng.strategy.key, "name": eng.strategy.name}
    return out


@router.get("/broker-positions")
def broker_positions(user: CurrentUser = Depends(get_current_user)):
    eng = engine_registry.get_engine(user.username)
    if eng is None:
        return []
    return eng.broker_positions()


@router.post("/positions/{symbol}/close")
def close_position(symbol: str, user: CurrentUser = Depends(get_current_user)):
    eng = engine_registry.get_engine(user.username)
    if eng is None:
        raise HTTPException(404, "Bot not started.")
    closed = eng.close_position(symbol)
    return {"ok": closed}


@router.post("/broker-positions/{symbol}/close")
def close_broker_position(symbol: str, quantity: int, side: str,
                          user: CurrentUser = Depends(get_current_user)):
    eng = engine_registry.get_engine(user.username)
    if eng is None:
        raise HTTPException(404, "Bot not started.")
    ok, msg = eng.close_broker_position(symbol, quantity, side)
    return {"ok": ok, "message": msg}


@router.post("/reset")
def reset_portfolio(environment: str, user: CurrentUser = Depends(get_current_user)):
    """Mirrors app.py's Danger Zone reset: uses the running engine when it
    owns this environment (clears the live dashboard too), otherwise wipes
    storage directly via DBManager — same two paths as the Streamlit tab.
    Always scoped to the caller's own trades (user_id) — a client can never
    reset another account's book, including the admin's."""
    try:
        env = Environment(environment)
    except ValueError:
        raise HTTPException(400, "Invalid environment.")
    eng = engine_registry.get_engine(user.username)
    if eng is not None and eng.environment == env:
        stats = eng.reset_portfolio()
    else:
        stats = _db.reset_environment(env, user_id=user.username)
    return stats
