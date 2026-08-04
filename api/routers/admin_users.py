"""
api/routers/admin_users.py
Admin-only: create/manage client accounts, and roll up how each client is
doing (running state + PnL) across both environments. This is the "how
does admin make a client an ID" flow from frontend_migration_plan.md §6 —
admin sets a username/password here, tells the client directly, the client
logs in with it.
"""
from __future__ import annotations

import os

import admin_config
import config
import presets
import symbol_config
import user_manager
from admin_config import BotConfig
from api import engine_registry
from api.auth import CurrentUser, require_admin
from api.schemas import (AdminConfigRequest, ClientModesRequest,
                         CreateClientRequest, PresetSaveRequest,
                         SetPasswordRequest, SetStatusRequest,
                         SymbolConfigRequest)
from config import Environment, Mode
from db_manager import DBManager
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse

router = APIRouter(prefix="/admin", tags=["admin"], dependencies=[Depends(require_admin)])

_db = DBManager()


# -- client accounts ----------------------------------------------------------- #
@router.post("/users")
def create_client(req: CreateClientRequest):
    if len(req.password) < 6:
        raise HTTPException(400, "Password must be at least 6 characters.")
    try:
        user = user_manager.create_user(
            req.username, req.password, role="client",
            display_name=req.display_name)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    return {k: v for k, v in user.items() if k not in ("password_hash", "broker_tokens")}


@router.get("/users")
def list_clients():
    return user_manager.list_users(role="client")


@router.put("/users/{user_id}/status")
def set_client_status(user_id: str, req: SetStatusRequest):
    if req.status not in ("active", "disabled"):
        raise HTTPException(400, "status must be 'active' or 'disabled'.")
    user = user_manager.set_status(user_id, req.status)
    if user is None:
        raise HTTPException(404, "User not found.")
    # A disabled client's running bot must stop immediately — status alone
    # (checked on every request in get_current_user) blocks new API calls,
    # but a bot already running in the background would otherwise keep
    # trading with no one able to stop it.
    if req.status == "disabled":
        engine_registry.stop_engine(user["username"])
    return {k: v for k, v in user.items() if k not in ("password_hash", "broker_tokens")}


@router.put("/users/{user_id}/password")
def reset_client_password(user_id: str, req: SetPasswordRequest):
    if len(req.password) < 6:
        raise HTTPException(400, "Password must be at least 6 characters.")
    user = user_manager.set_password(user_id, req.password)
    if user is None:
        raise HTTPException(404, "User not found.")
    return {"ok": True}


# -- overview -------------------------------------------------------------------- #
@router.get("/clients-overview")
def clients_overview():
    rows = []
    for client in user_manager.list_users(role="client"):
        username = client["username"]
        eng = engine_registry.get_engine(username)
        running = bool(eng and eng.state.running)
        paper_pnl = _db.analytics_summary(Environment.PAPER, user_id=username)["total_pnl"]
        live_pnl = _db.analytics_summary(Environment.LIVE, user_id=username)["total_pnl"]
        rows.append({
            "user_id": client["user_id"],
            "username": username,
            "display_name": client.get("display_name", username),
            "status": client.get("status", "active"),
            "created_at": client.get("created_at", ""),
            "running": running,
            "environment": eng.environment.value if eng else None,
            "broker": eng.broker.name if (eng and eng.broker) else None,
            "paper_total_pnl": paper_pnl,
            "live_total_pnl": live_pnl,
            # Booleans only — you can see who is set up without any endpoint
            # ever carrying a key, secret or token (list_users strips those).
            "broker_connected": sorted(
                b for b in ("Upstox", "Zerodha")
                if user_manager.credential_summary(client["user_id"], b)["has_token"]),
            "credentials_configured": sorted(
                b for b in ("Upstox", "Zerodha")
                if user_manager.credential_summary(client["user_id"], b)["configured"]),
        })
    return rows


# -- one client's own bot statistics ------------------------------------------- #
# Trades are stored keyed by username (engine's user_id — see bot.py), so that
# is what identifies a client here. Everything below reuses db_manager's
# existing user_id filter; no query is unscoped, so "admin looks at a client"
# can never accidentally return the whole book.

def _client_username(username: str) -> str:
    user = user_manager.get_user(username)
    if user is None:
        raise HTTPException(404, f"No such user: {username}")
    return user["username"]


def _environment(environment: str) -> Environment:
    try:
        return Environment(environment)
    except ValueError:
        raise HTTPException(400, "environment must be 'Paper' or 'Live'.")


@router.get("/clients/{username}/stats")
def client_stats(username: str, environment: str = "Paper"):
    """Everything the drill-down panel needs in one round trip: headline
    summary, per-day PnL rows, and the trade list — all scoped to this one
    client."""
    uname = _client_username(username)
    env = _environment(environment)
    trades = _db.get_trades(env, user_id=uname)
    daily = _db.daily_pnl(env, user_id=uname)
    eng = engine_registry.get_engine(uname)
    return {
        "username": uname,
        "environment": env.value,
        "running": bool(eng and eng.state.running),
        "summary": _db.analytics_summary(env, user_id=uname),
        "daily_pnl": daily.to_dict("records") if not daily.empty else [],
        "trades": trades.to_dict("records") if not trades.empty else [],
    }


@router.get("/clients/{username}/export")
def client_export(username: str, environment: str = "Paper"):
    """The same Excel export a client can pull for themselves, for one client."""
    uname = _client_username(username)
    path = _db.export_excel(_environment(environment), user_id=uname)
    if not path or not os.path.exists(path):
        raise HTTPException(500, "Export failed.")
    return FileResponse(
        path, filename=os.path.basename(path),
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


# -- the shared strategy/instrument config every client trades ----------------- #
@router.get("/config")
def get_bot_config() -> BotConfig:
    return admin_config.get_config()


@router.put("/config")
def set_bot_config(req: AdminConfigRequest) -> BotConfig:
    """Save the strategy/instruments clients get when they pick `req.mode`.
    Saving Intraday leaves Scalper's own selection untouched, and vice versa."""
    payload = req.model_dump()
    mode = payload.pop("mode")
    if mode not in admin_config.CLIENT_SELECTABLE_MODES:
        raise HTTPException(
            400, f"Clients can only be given {', '.join(admin_config.CLIENT_SELECTABLE_MODES)} "
                 f"— {mode!r} is admin-only.")
    rr = float(payload.get("risk_reward") or 0.0)
    if not config.is_valid_rr(rr):
        raise HTTPException(
            400, f"risk_reward {rr:g} is not offered. Pick one of "
                 f"{', '.join(config.rr_label(c) for c in config.RR_CHOICES)}, "
                 f"or 0 to use the strategy's own.")
    return admin_config.set_mode_config(mode, **payload)


# -- per-symbol settings (symbol_config.py) ------------------------------------ #
# ADMIN-ONLY, like everything else in this router: these decide when and at what
# RR an instrument trades, so a client must not be able to set them for
# themselves — they arrive at the engine server-side from here, never from a
# client's own /bot/start body.

def _valid_mode(mode: str) -> str:
    try:
        return Mode(mode).value
    except ValueError:
        raise HTTPException(
            400, f"Invalid mode {mode!r}. Use one of "
                 f"{', '.join(m.value for m in Mode)}.")


@router.get("/symbol-config")
def get_symbol_configs(mode: str):
    """Every instrument with custom settings under this mode, as
    {symbol: {...}}. Symbols absent from the map run the plain strategy."""
    from dataclasses import asdict as _asdict
    return {sym: _asdict(cfg)
            for sym, cfg in symbol_config.get_all(_valid_mode(mode)).items()}


@router.put("/symbol-config")
def set_symbol_config(req: SymbolConfigRequest):
    """Save one instrument's settings for one mode. Other symbols and other
    modes are untouched, and a body that is entirely defaults RESETS the
    symbol (the entry is deleted) rather than storing an inert one."""
    mode = _valid_mode(req.mode)
    if req.symbol not in config.INSTRUMENTS_BY_SYMBOL:
        raise HTTPException(400, f"Unknown instrument: {req.symbol}.")
    try:
        cfg = symbol_config.validate(symbol_config.SymbolConfig(
            trade_days=req.trade_days, start_time=req.start_time,
            end_time=req.end_time, risk_reward=req.risk_reward,
            square_off_at_end=req.square_off_at_end))
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    symbol_config.set_symbol(mode, req.symbol, cfg)
    return get_symbol_configs(mode)


@router.delete("/symbol-config/{mode}/{symbol}")
def delete_symbol_config(mode: str, symbol: str):
    """Drop one instrument's settings — it goes back to trading exactly as the
    strategy alone dictates."""
    mode = _valid_mode(mode)
    symbol_config.delete_symbol(mode, symbol)
    return get_symbol_configs(mode)


# -- saved Controls presets (presets.py) --------------------------------------- #
# A preset is INERT: saving one never touches a running bot, and loading one
# only repopulates the sidebar and restores that mode's per-symbol settings.
# Nothing here starts, stops or reconfigures an engine.

@router.get("/presets")
def list_presets():
    """Every saved setup as {name: preset}, for the picker."""
    from dataclasses import asdict as _asdict
    return {name: _asdict(p) for name, p in presets.load_all().items()}


@router.put("/presets")
def save_preset(req: PresetSaveRequest):
    """Save (or overwrite) the whole Controls sidebar under a name."""
    payload = req.model_dump()
    name = payload.pop("name")
    try:
        presets.save(presets.clean_name(name), presets.validate(
            presets.ControlPreset(**payload)))
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    return list_presets()


@router.post("/presets/{name}/load")
def load_preset(name: str):
    """Restore a saved setup. Its per-symbol settings are written back for the
    preset's OWN mode (replacing that mode's current ones — see presets.apply);
    the rest is returned for the sidebar to repopulate itself with. The bot is
    not started: that stays a separate, deliberate action."""
    preset = presets.apply(name)
    if preset is None:
        raise HTTPException(404, f"No preset named {name!r}.")
    from dataclasses import asdict as _asdict
    return _asdict(preset)


@router.delete("/presets/{name}")
def delete_preset(name: str):
    presets.delete(name)
    return list_presets()


@router.put("/config/client-modes")
def set_client_modes(req: ClientModesRequest) -> BotConfig:
    """Which modes appear in the client's Start Bot picker. Enabling a mode
    that has no instruments saved yet is allowed but won't surface to clients
    until it does (admin_config.available_client_modes)."""
    unknown = [m for m in req.modes if m not in admin_config.CLIENT_SELECTABLE_MODES]
    if unknown:
        raise HTTPException(
            400, f"Not selectable by clients: {', '.join(unknown)}. "
                 f"Allowed: {', '.join(admin_config.CLIENT_SELECTABLE_MODES)}.")
    return admin_config.set_client_modes(req.modes)
