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
import mailer
import pandas as pd
import pattern_config
import presets
import strategy
import strategy_groups
import symbol_config
import user_manager
from admin_config import BotConfig
from api import engine_registry
from api.auth import CurrentUser, require_admin
from api.schemas import (AdminConfigRequest, ClientModesRequest,
                         CreateClientRequest, PresetSaveRequest,
                         RangeResetRequest, SetEmailRequest,
                         PatternConfigRequest, SetPasswordRequest,
                         SetStatusRequest, StrategyGroupsRequest,
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
    """Create a client login. Username, password AND email are all required.

    Email is mandatory rather than nice-to-have: it is the only way the
    client can ever recover their own account. Allowing it to be skipped
    produced accounts that looked fine until the day someone forgot their
    password, at which point the only route left was admin resetting it by
    hand — so the requirement is enforced here, at the one place accounts are
    created, instead of chased later.
    """
    if len(req.password) < 6:
        raise HTTPException(400, "Password must be at least 6 characters.")
    email = mailer.normalise(req.email)
    if not email:
        raise HTTPException(
            400, "An email address is required — it's how this client resets "
                 "their own password if they forget it.")
    if not mailer.looks_valid(email):
        raise HTTPException(400, "That doesn't look like a valid email address.")
    try:
        user = user_manager.create_user(
            req.username, req.password, role="client",
            display_name=req.display_name, email=email)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    return {k: v for k, v in user.items() if k not in ("password_hash", "broker_tokens")}


@router.put("/users/{user_id}/email")
def set_client_email(user_id: str, req: SetEmailRequest):
    """Set where a client's forgot-password code is sent. Without one they
    cannot self-serve a reset — only the admin reset below works for them."""
    email = mailer.normalise(req.email)
    if email and not mailer.looks_valid(email):
        raise HTTPException(400, "That doesn't look like a valid email address.")
    user = user_manager.set_email(user_id, email)
    if user is None:
        raise HTTPException(404, "User not found.")
    return {"ok": True, "email": email, "email_masked": mailer.mask(email)}


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
            # Full address, not masked: admin set it and needs to correct
            # typos — a wrong address here silently breaks that client's
            # ability to ever reset their own password.
            "email": mailer.optional_recipient(client.get("email")),
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
    summary, per-day PnL rows, the per-strategy breakdown of those same days,
    and the trade list — all scoped to this one client."""
    uname = _client_username(username)
    env = _environment(environment)
    trades = _db.get_trades(env, user_id=uname)
    daily = _db.daily_pnl(env, user_id=uname)
    by_strategy = _db.strategy_summary(env, user_id=uname)
    daily_by_strategy = _db.daily_strategy_pnl(env, user_id=uname)
    eng = engine_registry.get_engine(uname)

    def _rows(df):
        return df.to_dict("records") if not df.empty else []

    return {
        "username": uname,
        "environment": env.value,
        "running": bool(eng and eng.state.running),
        "summary": _db.analytics_summary(env, user_id=uname),
        "daily_pnl": _rows(daily),
        "strategy_pnl": _rows(by_strategy),
        "daily_strategy_pnl": _rows(daily_by_strategy),
        "trades": _rows(trades),
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
    score = float(payload.get("min_score") or 0.0)
    if not config.is_valid_min_score(score):
        raise HTTPException(
            400, f"min_score {score:g} is out of range. Use "
                 f"{config.MIN_SCORE_MIN:g}-{config.MIN_SCORE_MAX:g}, or 0 to "
                 f"use the strategy's own.")
    cutoff = str(payload.get("square_off_time") or "")
    if cutoff and config.parse_clock(cutoff) is None:
        raise HTTPException(
            400, f"square_off_time {cutoff!r} is not a valid time — use HH:MM "
                 f"(24-hour), or leave blank for the segment default.")
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
            trade_days=req.trade_days, trade_hours=req.trade_hours,
            start_time=req.start_time, end_time=req.end_time,
            risk_reward=req.risk_reward,
            square_off_at_end=req.square_off_at_end,
            trail_enabled=req.trail_enabled,
            trail_atr_mult=req.trail_atr_mult))
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


# -- targeted trade deletion ---------------------------------------------------- #
@router.get("/trades/by-category")
def admin_by_category(environment: str = "Paper", username: str = ""):
    """Category-wise P&L for one account (blank = admin's own book)."""
    return _db.category_summary(_environment(environment),
                                user_id=_client_username(username) if username
                                else "admin")


@router.post("/trades/reset-range")
def reset_trade_range(req: RangeResetRequest):
    """Delete trades whose trading day falls in [start, end].

    Two-step by construction: with `confirm=false` (the default) this is a
    PREVIEW that counts what would go, broken down by day, and deletes
    nothing. Only a second call with confirm=true actually removes anything.

    Built for cleaning out trades punched against simulated or bad data
    without losing the real history around them — `/bot/reset` is
    all-or-nothing, this is scoped to a date range and optionally one
    category.

    Admin-only (the whole router is), and IRREVERSIBLE once confirmed.
    """
    env = _environment(req.environment)
    category = (req.category or "").strip()
    if category and category not in config.ALL_CATEGORIES:
        raise HTTPException(
            400, f"Unknown category {category!r}. Use one of "
                 f"{', '.join(config.ALL_CATEGORIES)}, or leave blank for all.")
    owner = _client_username(req.username) if req.username else "admin"
    try:
        result = _db.reset_range(
            env, req.start, req.end, user_id=owner,
            category=category or None, dry_run=not req.confirm)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    result["environment"] = env.value
    result["category"] = category or "All"
    result["username"] = owner
    return result


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


# --------------------------------------------------------------------------- #
#  Candlestick pattern allow-list (pattern_config.py)
# --------------------------------------------------------------------------- #
@router.get("/pattern-catalogue")
def pattern_catalogue():
    """Every pattern the engine can emit, plus which strategies can be
    filtered. The list is a picker convenience — the filter matches on the name
    a hit actually carries, so a pattern added to the engine later still works
    the moment its name is allow-listed."""
    return {"patterns": list(pattern_config.PATTERN_CATALOGUE),
            "strategies": list(pattern_config.FILTERABLE_STRATEGIES)}


@router.get("/pattern-config")
def get_pattern_config(mode: str):
    from dataclasses import asdict as _asdict
    return {k: _asdict(v)
            for k, v in pattern_config.get_all(_valid_mode(mode)).items()}


@router.put("/pattern-config")
def set_pattern_config(req: PatternConfigRequest):
    """Save one (strategy, mode) allow-list.

    Rejects a strategy that does not go through pattern detection: a filter on
    a VWAP strategy would be a control that silently does nothing, which is
    worse than no control.
    """
    mode = _valid_mode(req.mode)
    if req.strategy_key not in pattern_config.FILTERABLE_STRATEGIES:
        raise HTTPException(
            400, f"{req.strategy_key} does not use candlestick patterns. "
                 f"Filterable: {', '.join(pattern_config.FILTERABLE_STRATEGIES)}.")
    saved = pattern_config.set_rules(
        req.strategy_key, mode,
        pattern_config.PatternRules(enabled=req.enabled, allowed=req.allowed))
    from dataclasses import asdict as _asdict
    return {req.strategy_key: _asdict(saved)}


@router.get("/pattern-stats")
def pattern_stats(environment: str = "Paper", strategy: str = "", mode: str = ""):
    """Realised PnL attributed back to individual patterns.

    A trade's `entry_reason` records the COMBINATION that fired ("Bullish
    Engulfing, Hammer"), so a trade counts toward every pattern in it. That
    overlap is why `solo_*` is reported separately: those are the trades where
    a pattern fired ALONE, and they are the only clean read on that pattern by
    itself. Judge on `solo` where the sample allows, and treat the attributed
    figures as a hint about company a pattern keeps.
    """
    try:
        env = Environment(environment)
    except ValueError:
        raise HTTPException(400, "environment must be 'Paper' or 'Live'.")

    df = _db.get_trades(env)
    if df.empty or "status" not in df.columns:
        return {"patterns": [], "trades": 0}
    df = df[df["status"] == "CLOSED"]
    if strategy:
        df = df[df.get("strategy", "") == strategy]
    if mode:
        df = df[df.get("mode", "") == mode]
    if df.empty:
        return {"patterns": [], "trades": 0}

    agg: dict[str, dict] = {}
    for _, t in df.iterrows():
        names = [p.strip() for p in
                 str(t.get("entry_reason", "")).split(" (")[0].split(",")
                 if p.strip()]
        if not names:
            continue
        pnl = float(pd.to_numeric(t.get("realized_pnl"), errors="coerce") or 0.0)
        solo = len(names) == 1
        for n in names:
            a = agg.setdefault(n, {"trades": 0, "pnl": 0.0, "wins": 0,
                                   "solo_trades": 0, "solo_pnl": 0.0,
                                   "solo_wins": 0})
            a["trades"] += 1
            a["pnl"] += pnl
            a["wins"] += 1 if pnl > 0 else 0
            if solo:
                a["solo_trades"] += 1
                a["solo_pnl"] += pnl
                a["solo_wins"] += 1 if pnl > 0 else 0

    out = []
    for name, a in agg.items():
        out.append({
            "pattern": name,
            "trades": a["trades"], "pnl": round(a["pnl"], 2),
            "win_rate": round(100.0 * a["wins"] / a["trades"], 1),
            "avg_pnl": round(a["pnl"] / a["trades"], 2),
            "solo_trades": a["solo_trades"],
            "solo_pnl": round(a["solo_pnl"], 2),
            "solo_win_rate": (round(100.0 * a["solo_wins"] / a["solo_trades"], 1)
                              if a["solo_trades"] else 0.0),
        })
    out.sort(key=lambda r: r["pnl"], reverse=True)
    return {"patterns": out, "trades": int(len(df))}


@router.get("/strategy-groups")
def get_strategy_groups(mode: str):
    """This mode's strategy board: which strategy trades which stocks. An
    empty list means no board — the bot runs the single strategy picked in the
    sidebar, exactly as before this existed."""
    from dataclasses import asdict as _asdict
    return [_asdict(g) for g in strategy_groups.get_all(_valid_mode(mode))]


@router.put("/strategy-groups")
def set_strategy_groups(req: StrategyGroupsRequest):
    """Replace this mode's whole board in one write.

    Wholesale rather than per-group because dragging a stock from one strategy
    to another changes TWO groups at once; saving them separately could persist
    half a move.
    """
    mode = _valid_mode(req.mode)
    known = {s.key for s in strategy.strategies_for_mode(Mode(mode))}
    incoming = [strategy_groups.StrategyGroup(
        strategy_key=g.strategy_key, symbols=g.symbols, mcx_lots=g.mcx_lots,
        risk_reward=g.risk_reward, min_score=g.min_score, enabled=g.enabled)
        for g in req.groups]
    unsupported = [g.strategy_key for g in incoming
                   if g.strategy_key and g.strategy_key not in known]
    if unsupported:
        # resolve_strategy() silently falls back to the mode default for a
        # key that doesn't support the mode, so an unchecked board could run a
        # strategy nobody chose. Rejected here instead.
        raise HTTPException(
            400, f"Not available in {mode}: {', '.join(unsupported)}.")
    try:
        clean = strategy_groups.validate(incoming, mode)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    strategy_groups.replace_mode(mode, clean)
    return get_strategy_groups(mode)


@router.put("/config/auto-start-clients")
def set_auto_start_clients(enabled: bool) -> BotConfig:
    """Whether admin's Start Bot also publishes that run to clients and starts
    them (admin_config.auto_start_clients). Off leaves the client config and
    every client bot exactly as they are."""
    return admin_config.set_auto_start_clients(enabled)


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
