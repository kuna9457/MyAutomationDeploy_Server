"""
api/routers/bot.py
Start/stop/status/close-position — the exact operations app.py's sidebar
"Start Bot"/"Stop Bot" buttons and per-trade "Close" buttons perform, just
reached over HTTP instead of a Streamlit rerun. TradingEngine itself
(engine.py) is called with identical arguments; nothing about sizing,
signals, or order placement changes.

Each logged-in user gets their own TradingEngine, keyed by username in
engine_registry. That engine is an EXECUTION ACCOUNT: what to trade is
decided once by a shared StrategyRunner (strategy_runner.py) and broadcast to
every account, which then sizes it against its own capital and punches it
through its own broker.

role="client" therefore decides NOTHING about the trade. mode, strategy,
segments, instruments, risk:reward and per-symbol windows are ALL read
server-side from admin_config — `req.mode` is ignored outright rather than
validated, so it cannot be steered from the request body any more than the
instrument list can. Their own broker access token (saved via
/broker/*/exchange) is required for Live and resolved server-side too, never
accepted as input.

What a client DOES control is their own money: environment (Paper/Live),
total capital, allocated capital and their risk guardrails. That is the whole
client surface — see /platform-signals for how they watch the platform trade
before starting their own bot.
"""
from __future__ import annotations

import admin_config
import config
import strategy
import strategy_runner
import symbol_config
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
        # A client does not choose WHAT to trade — admin does. `req.mode` is
        # ignored outright rather than validated, so the mode cannot be
        # steered from the request body any more than the strategy or the
        # instrument list can. The client's bot only sizes and punches.
        requested = admin_config.active_client_mode()
        if not requested:
            raise HTTPException(
                400, "Trading hasn't been configured yet — ask your admin to "
                     "set a strategy and instruments before you can start.")
        mode_cfg = admin_config.get_mode_config(requested)
        mode = Mode(requested)
        strategy_key = mode_cfg.strategy_key
        symbols = mode_cfg.symbols
        mcx_lots = mode_cfg.mcx_lots
        # A client's RR comes from admin's saved config for this mode, never
        # from the request body — same rule as strategy/instruments.
        risk_reward = mode_cfg.risk_reward
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
        risk_reward = req.risk_reward
        if not config.is_valid_rr(risk_reward):
            raise HTTPException(
                400, f"risk_reward {risk_reward:g} is not offered. Pick one of "
                     f"{', '.join(config.rr_label(c) for c in config.RR_CHOICES)}, "
                     f"or 0 to use the strategy's own.")

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

    # Per-symbol settings (trading days / entry window / that symbol's own RR)
    # are read SERVER-SIDE from admin's saved config for this mode, never from
    # the request body — the same rule as strategy/instruments, so a client
    # cannot widen their own trading window. Only symbols with settings that
    # actually change something come back; everything else runs untouched.
    rules = symbol_config.rules_for(mode.value, [i.symbol for i in selected])

    eng = TradingEngine(environment, mode, broker_choice, selected, req.capital,
                        strategy_key=strategy_key, mcx_lots=mcx_lots,
                        user_id=user.username, broker_access_token=access_token,
                        broker_api_key=broker_api_key, risk_reward=risk_reward,
                        symbol_rules=rules)
    try:
        eng.start()
    except RuntimeError as exc:
        # StartupBlocked carries a second, client-safe message; a plain
        # RuntimeError has none and falls through to str(exc) as before. The
        # role check lives HERE rather than in the engine so engine.py stays
        # unaware of who is asking — it only states the two framings.
        detail = str(exc)
        if user.role == "client":
            detail = getattr(exc, "client_message", "") or detail
        raise HTTPException(400, detail)
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


def _client_runner_key() -> str:
    """The runner key a CLIENT resolves to, derived purely from admin's saved
    config — no engine required.

    This is what lets a client watch the platform's signals before starting:
    the decisions are already being made under this key by whoever started
    first, so we can look them up without the client owning anything. It
    mirrors exactly what TradingEngine.__init__ computes, so a client who
    then presses Start joins the very runner they were watching.
    """
    mode_name = admin_config.active_client_mode()
    if not mode_name or not strategy_runner.replication_enabled():
        return ""
    mode_cfg = admin_config.get_mode_config(mode_name)
    mode = Mode(mode_name)
    bound = strategy.resolve_strategy(mode, mode_cfg.strategy_key)
    rr = mode_cfg.risk_reward or bound.params.risk_reward
    instruments = [config.INSTRUMENTS_BY_SYMBOL[s] for s in mode_cfg.symbols
                   if s in config.INSTRUMENTS_BY_SYMBOL]
    if not instruments:
        return ""
    token = config.UPSTOX_LIVE_ACCESS_TOKEN or config.UPSTOX_SANDBOX_TOKEN
    return strategy_runner.runner_key(mode, bound.key, instruments, rr, token)


@router.get("/platform-signals")
def platform_signals(user: CurrentUser = Depends(get_current_user)):
    """Signals the platform is generating right now, whether or not THIS
    account has started its bot.

    A client who logs in mid-session sees the trades being taken around them
    immediately, instead of a blank screen until they press Start. Carries no
    quantity — that is per-account and only exists once an account actually
    takes the trade — and no strategy identity, matching what
    /config/client-modes already withholds from clients.

    `running` is deliberately about the CALLER's own bot, not the platform's:
    watching is not trading, and the dashboard must not imply otherwise.
    """
    eng = engine_registry.get_engine(user.username)
    running = bool(eng and eng.state.running)
    if eng is not None and getattr(eng, "_runner", None) is not None:
        key = eng._runner_key
    elif user.role == "client":
        key = _client_runner_key()
    else:
        key = ""
    runner = strategy_runner.get(key) if key else None
    return {
        "running": running,
        "live": runner is not None,
        "mode": runner.mode.value if runner is not None else
                (admin_config.active_client_mode() if user.role == "client" else ""),
        "signals": runner.recent_signals() if runner is not None else [],
    }


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
