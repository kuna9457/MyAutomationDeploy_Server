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
import capital_ledger
import config
import risk_manager
import strategy
import strategy_groups
import strategy_runner
import symbol_config
import user_manager
from api import engine_registry
from api.auth import CurrentUser, get_current_user
from api.schemas import StartBotRequest
from config import Broker, Environment, Mode, Segment
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

#: Notional capital an auto-started client runs on in PAPER when they have no
#: allocation of their own. Paper risks nothing real, so a stand-in is fine
#: here; the LIVE path deliberately has no equivalent and skips the account
#: instead of inventing a size for real money.
_PAPER_FALLBACK_CAPITAL = 100_000.0


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


def _start_board(owner: str, groups: list, environment: Environment, mode: Mode,
                 broker_choice: Broker, capital: float, access_token: str,
                 broker_api_key: str, square_off_time: str,
                 square_off_enabled: bool) -> list[TradingEngine]:
    """Start ONE engine per strategy group, sharing a capital ledger.

    Each engine is the ordinary single-strategy engine — same construction,
    same decision path. The board only decides HOW MANY there are and what each
    one trades; the ledger is what makes them behave as one account (shared
    wallet, one open trade per stock). See capital_ledger.py.

    All-or-nothing: if any group fails to start, the ones already started are
    stopped again before the error propagates. A half-started board would leave
    some stocks trading and others not, with the capital ledger describing a
    set of engines that no longer matches what is running.
    """
    ledger = capital_ledger.CapitalLedger(owner)
    started: list[TradingEngine] = []

    # ONE market-data socket for the whole board. Every group on this mode
    # reads identical candles, and the broker refuses a second concurrent
    # market-data connection on the same token — without this the first group
    # gets the WebSocket and the rest silently fall back to REST polling.
    # Warmed on the UNION up front so each runner finds a superset and simply
    # increments the ref count instead of restarting the socket per group.
    board_instruments: dict[str, object] = {}
    for g in groups:
        for s in g.symbols:
            inst = config.INSTRUMENTS_BY_SYMBOL.get(s)
            if inst is not None:
                board_instruments.setdefault(s, inst)
    feed_token = (config.UPSTOX_LIVE_ACCESS_TOKEN
                  or config.UPSTOX_SANDBOX_TOKEN)
    warm_key = ""
    if board_instruments:
        warm_key = strategy_runner.prewarm_feed(
            mode, feed_token, list(board_instruments.values()))

    try:
        for g in groups:
            selected = [config.INSTRUMENTS_BY_SYMBOL[s] for s in g.symbols
                        if s in config.INSTRUMENTS_BY_SYMBOL]
            if not selected:
                continue
            rules = symbol_config.rules_for(mode.value,
                                            [i.symbol for i in selected])
            eng = TradingEngine(
                environment, mode, broker_choice, selected, capital,
                strategy_key=g.strategy_key, mcx_lots=g.mcx_lots,
                user_id=owner, broker_access_token=access_token,
                broker_api_key=broker_api_key, risk_reward=g.risk_reward,
                min_score=g.min_score, square_off_time=square_off_time,
                square_off_enabled=square_off_enabled, symbol_rules=rules)
            eng.group_key = g.strategy_key
            # Attached BEFORE start so the very first tick already sizes
            # against the shared wallet rather than the full ceiling.
            ledger.attach(eng)
            eng.start()
            started.append(eng)
    except Exception:
        for eng in started:
            try:
                eng.stop()
            except Exception:
                pass
        raise
    finally:
        # Hand back the warm-up reference now that every runner holds its own.
        # In the failure path above this is what lets the socket close instead
        # of leaking a feed nobody is reading.
        if warm_key:
            strategy_runner.release_feed(warm_key)
    return started


def _client_broker(user_id: str) -> tuple[str, str, str]:
    """(broker_name, access_token, api_key) for the LIVE broker this client
    has actually connected, or ("", "", "") if none is usable.

    Probes only the brokers with a per-client OAuth flow, in a fixed order, and
    requires BOTH stored credentials and a token the broker still accepts — the
    same three checks a client's own Start Bot makes, so an auto-start can
    never get further than a manual one would.
    """
    for name in sorted(_CLIENT_LIVE_BROKERS):
        cred_key, cred_secret = user_manager.get_broker_credentials(user_id, name)
        if not (cred_key and cred_secret):
            continue
        token = user_manager.get_broker_token(user_id, name)
        if not token or not _token_is_live(name, token, cred_key):
            continue
        return name, token, cred_key
    return "", "", ""


def _start_one_client(account: dict, environment: Environment) -> str:
    """Start ONE client's bot on the config admin just published. Returns ""
    on success, or a short human reason it was skipped.

    Never raises: this runs inside a fan-out where one client's expired token
    or rejected broker must not stop the others, exactly as Immutable Rule #4
    requires of every per-account call.

    The engine is built from `admin_config` and from THIS account's own
    capital and own broker token. Nothing about size is copied from admin —
    qty is always risk_budget / stop_distance against the client's own
    capital, so a smaller account gets a smaller position, never admin's.
    """
    user_id = account.get("user_id") or ""
    username = account.get("username") or ""
    if not user_id or not username:
        return "account record incomplete"
    if account.get("status") != "active":
        return "account is not active"

    mode_name = admin_config.active_client_mode()
    if not mode_name:
        return "no client mode configured"
    mode_cfg = admin_config.get_mode_config(mode_name)
    selected = [config.INSTRUMENTS_BY_SYMBOL[s] for s in mode_cfg.symbols
                if s in config.INSTRUMENTS_BY_SYMBOL]
    if not selected:
        return "no tradable instruments in the published config"

    broker_choice = Broker.SIMULATED
    access_token = broker_api_key = ""
    # LIVE-only: real money needs this account's OWN capital ceiling and its
    # OWN broker session. Neither is inferred from admin — a client whose
    # capital has never been set is SKIPPED rather than started against a
    # number nobody chose for them.
    # TWO DIFFERENT KEYS, deliberately — this is not a typo. Broker
    # credentials are stored against the account's `user_id`, but risk limits
    # are stored against its USERNAME (risk.py keys on user.username, and
    # TradingEngine is constructed with user_id=username, so that is what
    # engine._live_risk_check will look up at run time). Reading capital by
    # user_id here would find nothing and skip every client as "no capital".
    limits = risk_manager.get_limits(username)
    capital = float(limits.capital_allocated or 0.0)
    if environment == Environment.LIVE:
        if capital <= 0:
            return "no capital allocated (set it in their Risk panel)"
        broker_name, access_token, broker_api_key = _client_broker(user_id)
        if not broker_name:
            return "no connected broker, or its session has expired"
        broker_choice = _BROKER_LABELS[broker_name]
    elif capital <= 0:
        # Paper risks nothing real, so a default is safe here in a way it
        # would never be above.
        capital = _PAPER_FALLBACK_CAPITAL

    existing = engine_registry.get_engine(username)
    if existing and existing.state.running:
        return "already running"

    rules = symbol_config.rules_for(mode_name, [i.symbol for i in selected])
    try:
        eng = TradingEngine(
            environment, Mode(mode_name), broker_choice, selected, capital,
            strategy_key=mode_cfg.strategy_key, mcx_lots=mode_cfg.mcx_lots,
            user_id=username, broker_access_token=access_token,
            broker_api_key=broker_api_key, risk_reward=mode_cfg.risk_reward,
            min_score=mode_cfg.min_score,
            square_off_time=mode_cfg.square_off_time,
            square_off_enabled=mode_cfg.square_off_enabled,
            symbol_rules=rules)
        eng.start()
    except Exception as exc:
        return str(exc)[:160] or "failed to start"
    engine_registry.set_engine(username, eng)
    return ""


def _fan_out_to_clients(environment: Environment) -> dict:
    """Start every eligible client. Returns a summary for admin's toast.

    Sequential on purpose. The parallel dispatch in strategy_runner exists so
    ENTRIES land together across accounts; this is a one-off setup step where
    each start does its own broker handshake, and doing them together would
    buy nothing while making a partial failure much harder to report.
    """
    accounts = user_manager.list_users(role="client")
    started, skipped = [], []
    for acct in accounts:
        reason = _start_one_client(acct, environment)
        if reason:
            skipped.append({"username": acct.get("username", "?"),
                            "reason": reason})
        else:
            started.append(acct.get("username", "?"))
    return {"total": len(accounts), "started": started, "skipped": skipped}


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
        min_score = mode_cfg.min_score
        square_off_time = mode_cfg.square_off_time
        square_off_enabled = mode_cfg.square_off_enabled
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
        min_score = req.min_score
        square_off_time = req.square_off_time
        square_off_enabled = req.square_off_enabled
        if square_off_time and config.parse_clock(square_off_time) is None:
            raise HTTPException(
                400, f"square_off_time {square_off_time!r} is not a valid "
                     f"time — use HH:MM (24-hour), or leave blank for the "
                     f"segment default.")
        if not config.is_valid_min_score(min_score):
            raise HTTPException(
                400, f"min_score {min_score:g} is out of range. Use "
                     f"{config.MIN_SCORE_MIN:g}-{config.MIN_SCORE_MAX:g}, or 0 "
                     f"to use the strategy's own.")
        if not config.is_valid_rr(risk_reward):
            raise HTTPException(
                400, f"risk_reward {risk_reward:g} is not offered. Pick one of "
                     f"{', '.join(config.rr_label(c) for c in config.RR_CHOICES)}, "
                     f"or 0 to use the strategy's own.")

    # A strategy board supplies its own instruments per group, so the sidebar's
    # flat list is not required — and is ignored entirely — when one exists.
    # Resolved here (before the guard below) so a board start never trips a
    # check that only applies to the single-strategy path.
    board = strategy_groups.enabled_groups(mode.value) if user.role == "admin" else []

    selected = [config.INSTRUMENTS_BY_SYMBOL[s] for s in symbols
               if s in config.INSTRUMENTS_BY_SYMBOL]
    if not selected and not board:
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

    # STRATEGY BOARD: several strategies, each with its own stocks, running at
    # once (strategy_groups.py). Taken only when a board actually exists for
    # this mode — with none configured, `groups` is empty and everything below
    # is the original single-strategy path, untouched.
    groups = board
    if groups:
        try:
            engines = _start_board(
                user.username, groups, environment, mode, broker_choice,
                req.capital, access_token, broker_api_key, square_off_time,
                square_off_enabled)
        except RuntimeError as exc:
            detail = str(exc)
            if user.role == "client":
                detail = getattr(exc, "client_message", "") or detail
            raise HTTPException(400, detail)
        if not engines:
            raise HTTPException(400, "No strategy group has tradable instruments.")
        engine_registry.set_engines(user.username, engines)
        return {"ok": True, "groups": [e.group_key for e in engines],
                "broker": engines[0].broker.name}

    eng = TradingEngine(environment, mode, broker_choice, selected, req.capital,
                        strategy_key=strategy_key, mcx_lots=mcx_lots,
                        user_id=user.username, broker_access_token=access_token,
                        broker_api_key=broker_api_key, risk_reward=risk_reward,
                        min_score=min_score, square_off_time=square_off_time,
                        square_off_enabled=square_off_enabled,
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

    out = {"ok": True, "strategy": eng.strategy.name, "broker": eng.broker.name}

    # ADMIN ONLY, and only AFTER admin's own engine is confirmed running:
    # publish this exact run as the config clients follow, then start them on
    # it. Ordering matters — if admin's own start had failed above we raised
    # already, so clients are never started on a run that did not itself
    # survive its own pre-flight checks.
    if (user.role == "admin" and admin_config.auto_start_clients()
            and mode.value in admin_config.CLIENT_SELECTABLE_MODES):
        published = admin_config.publish_run(
            mode.value,
            strategy_key=eng.strategy.key,
            symbols=[i.symbol for i in selected],
            segments=sorted({i.segment.value for i in selected}),
            mcx_lots={i.symbol: eng.mcx_lots.get(i.symbol, 1)
                      for i in selected if i.segment == Segment.MCX},
            # The RESOLVED values, not the request's: admin may have left
            # these at 0 ("use the strategy's own"), and clients must run the
            # same numbers admin is running, not the same blanks.
            risk_reward=eng.params.risk_reward,
            min_score=eng.params.cs_min_score,
            square_off_time=square_off_time,
            square_off_enabled=square_off_enabled)
        if published:
            out["clients"] = _fan_out_to_clients(environment)
    return out


@router.post("/stop")
def stop_bot(user: CurrentUser = Depends(get_current_user)):
    stopped = engine_registry.stop_engine(user.username)
    return {"ok": stopped}


def _merge_board(engines: list[TradingEngine]) -> dict:
    """One dashboard view of several strategy engines.

    Position/quote maps are keyed by symbol and the ledger guarantees only one
    engine holds a given symbol, so unioning them cannot lose or double-count a
    position. Money is SUMMED — it is one account across the board — while the
    infrastructure fields (feed, broker, storage) are taken from the primary,
    since every group shares the same feed, broker and store.

    `groups` carries the per-strategy breakdown that the flat view cannot
    express: which strategy holds what, and what each has made today.
    """
    primary = engines[0].state.snapshot()
    if len(engines) == 1:
        return primary

    merged = dict(primary)
    positions, quotes, signals, logs = {}, {}, [], []
    day = realized = unrealized = 0.0
    groups = []
    for eng in engines:
        snap = eng.state.snapshot()
        positions.update(snap.get("open_positions") or {})
        quotes.update(snap.get("live_quotes") or {})
        signals.extend(snap.get("last_signals") or [])
        # Prefixed so a line can be traced back to the strategy that wrote it —
        # without this the merged log reads as one bot contradicting itself.
        logs.extend(f"[{eng.group_key}] {line}" for line in (snap.get("log") or []))
        day += float(snap.get("day_pnl") or 0.0)
        realized += float(snap.get("realized_pnl") or 0.0)
        unrealized += float(snap.get("unrealized_pnl") or 0.0)
        groups.append({
            "strategy_key": eng.strategy.key,
            "strategy_name": eng.strategy.name,
            "running": bool(snap.get("running")),
            "symbols": [i.symbol for i in eng.instruments],
            "open": sorted((snap.get("open_positions") or {}).keys()),
            "day_pnl": float(snap.get("day_pnl") or 0.0),
            "risk_reward": float(eng.params.risk_reward),
        })

    merged.update({
        "running": any(g["running"] for g in groups),
        "open_positions": positions,
        "live_quotes": quotes,
        # Newest first across the whole board, not just whichever group's
        # buffer happened to be longest.
        "last_signals": signals[-60:],
        "log": logs[-400:],
        "day_pnl": day,
        "realized_pnl": realized,
        "unrealized_pnl": unrealized,
        "groups": groups,
    })
    return merged


@router.get("/status")
def bot_status(user: CurrentUser = Depends(get_current_user)):
    engines = engine_registry.get_engines(user.username)
    if not engines:
        return {"started": False}
    eng = engines[0]
    snap = _merge_board(engines)
    out = {
        "started": True,
        "environment": eng.environment.value,
        "mode": eng.mode.value,
        "total_capital": eng.total_capital,
        **snap,
    }
    # A client is told which phase is running, never which strategy runs it
    # (same reasoning as /config/client-modes). Admin still gets both.
    run_config = dict(eng.run_config)
    if user.role != "client":
        out["strategy"] = {"key": eng.strategy.key, "name": eng.strategy.name}
    elif run_config:
        # Same withholding applied to the snapshot, or the panel would leak
        # via run_config exactly what `strategy` is being kept back above.
        # The tuning knobs go too: they describe HOW the strategy decides,
        # which is the same secret by another name. What a client keeps is
        # what governs their own money and their own session.
        for secret in ("strategy", "atr_sl_mult", "atr_period", "min_score",
                       "entry_skip_minutes", "use_limit_entry"):
            run_config.pop(secret, None)
    out["run_config"] = run_config
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
    score = mode_cfg.min_score or bound.params.cs_min_score
    instruments = [config.INSTRUMENTS_BY_SYMBOL[s] for s in mode_cfg.symbols
                   if s in config.INSTRUMENTS_BY_SYMBOL]
    if not instruments:
        return ""
    token = config.UPSTOX_LIVE_ACCESS_TOKEN or config.UPSTOX_SANDBOX_TOKEN
    return strategy_runner.runner_key(
        mode, bound.key, instruments, rr, token, score,
        f"{mode_cfg.square_off_time}|{mode_cfg.square_off_enabled}")


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


@router.get("/broker-protection")
def broker_protection(user: CurrentUser = Depends(get_current_user)):
    """The SL/TP orders actually resting at the broker right now. Read-only,
    scoped to the caller's own engine (hence their own broker token), so this
    can never surface another account's orders."""
    eng = engine_registry.get_engine(user.username)
    if eng is None:
        return []
    return eng.broker_protection()


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
