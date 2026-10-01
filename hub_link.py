"""
hub_link.py
The NODE half of the fleet. Runs only when NODE_MODE=worker — on a client's own
server, the one with the static IP their broker app is locked to.

What a node is
--------------
The ordinary backend with the ordinary client UI, and one difference: its
TradingEngines get their market view from the admin HUB instead of from a
market-data socket of their own.

    hub ──wss──>  HubLink ──frame──>  HubRunner ──begin_tick/on_tick/
                                                  on_signal/on_exit──>  TradingEngine
                                                                        (unmodified)

Everything after that line is the code that has always run: the engine sizes the
signal against THIS account's capital and risk limits (Immutable Rules #1, #4),
places the order through THIS account's broker token from THIS machine's IP,
and manages the open position with exit_manager off the prices the hub relays.
The hub never sends a quantity and the node never sends a credential.

HubRunner is duck-typed to exactly what TradingEngine touches of a
StrategyRunner (see strategy_runner.set_runner_class), so engine.py needed one
line to accept it.

Configuration (env, on the node only)
-------------------------------------
    NODE_MODE=worker
    HUB_URL=wss://admin.example.com        (or the full .../fleet/ws)
    NODE_ID=node-1a2b3c4d                  (from the hub's Fleet panel)
    NODE_SECRET=...                        (shown once, when the node was added)
    ALLOW_INSECURE_HUB=false               (true only for local testing: ws://)
"""
from __future__ import annotations

import asyncio
import json
import os
import queue
import threading
import time
import weakref
from collections import OrderedDict
from datetime import datetime
from typing import Any, Optional

import config
import mongo_client
import symbol_config
from config import Environment, Instrument, Mode, now_for_segment, now_ist
from fleet_protocol import (MAX_SIGNAL_AGE_SECONDS, PROTOCOL_VERSION,
                            BroadcastConfig, FrameAge, db_fingerprint,
                            signal_from_dict, tick_from_dict)

NODE_VERSION = "1"

#: No frame from the hub for this long means the link is unusable even if the
#: socket still looks open. Polls arrive every 3s (0.5s for the Scalper).
STALE_LINK_SECONDS = 30.0
REPORT_EVERY_SECONDS = 5.0
#: Signal ids remembered, so a duplicate frame can never enter twice.
SEEN_SIGNALS_KEEP = 500


def is_worker() -> bool:
    return (os.getenv("NODE_MODE", "") or "").strip().lower() == "worker"


def _env(name: str) -> str:
    return (os.getenv(name, "") or "").strip()


def hub_ws_url() -> str:
    """HUB_URL normalised to the websocket endpoint. Accepts the bare host,
    an http(s) URL or the full ws(s) URL."""
    url = _env("HUB_URL")
    if not url:
        return ""
    if url.startswith("https://"):
        url = "wss://" + url[len("https://"):]
    elif url.startswith("http://"):
        url = "ws://" + url[len("http://"):]
    elif "://" not in url:
        url = "wss://" + url
    url = url.rstrip("/")
    if not url.endswith("/fleet/ws"):
        url += "/fleet/ws"
    return url


# --------------------------------------------------------------------------- #
#  The runner an engine sits behind on a node
# --------------------------------------------------------------------------- #
class _HubFeed:
    """Just enough of a MarketDataFeed for what the engine asks of one."""

    def __init__(self, link: "HubLink"):
        self._link = link

    def status(self) -> str:
        return self._link.status_text()

    def data_problems(self) -> list:
        return []

    def stop(self) -> None:
        pass


class HubRunner:
    """Stands in for strategy_runner.StrategyRunner on a node.

    Takes the same constructor arguments (the engine builds it through
    strategy_runner.runner_class()) but decides NOTHING: it replays the hub's
    events into its accounts in the order the hub's runner produced them —
    begin, then per-instrument tick / exit / signal — so a node's engine sees
    exactly the call sequence a local runner would have given it.
    """

    def __init__(self, key: str, mode: Mode, strategy, params,
                 instruments: list[Instrument], feed_token: str = "",
                 poll_seconds: float = 3.0, symbol_rules: Optional[dict] = None,
                 square_off_time: str = "", square_off_enabled: bool = True):
        self.key = key
        self.mode = mode
        self.strategy = strategy
        self.params = params
        self.instruments = instruments
        self.symbol_rules = dict(symbol_rules or {})
        self.square_off_time = square_off_time
        self.square_off_enabled = square_off_enabled
        self._by_symbol = {i.symbol: i for i in instruments}
        self._accounts: list = []
        self._lock = threading.Lock()
        self.running = False
        self.feed = _HubFeed(link())
        self._seen: OrderedDict[str, None] = OrderedDict()
        self._recent_signals: list[dict] = []
        self._last_price: dict[str, float] = {}
        self._squared_off: set[str] = set()
        self._squared_off_day = ""

    # -- the StrategyRunner surface an engine uses --------------------------- #
    def subscribe(self, account) -> None:
        with self._lock:
            if account not in self._accounts:
                self._accounts.append(account)

    def unsubscribe(self, account) -> int:
        with self._lock:
            if account in self._accounts:
                self._accounts.remove(account)
            return len(self._accounts)

    def account_count(self) -> int:
        with self._lock:
            return len(self._accounts)

    def ensure_started(self) -> None:
        if not self.running:
            self.start()

    def start(self) -> None:
        self.running = True
        link().register(self)

    def stop(self) -> None:
        self.running = False
        link().unregister(self)

    def feed_is_simulated(self) -> bool:
        """True when the engine must NOT place real orders: the hub link is
        down (no prices at all) or the hub itself is on simulated data. Shares
        the engine's existing refusal — StartupBlocked — rather than adding a
        second safety path."""
        return link().feed_unusable()

    def status(self) -> str:
        return self.feed.status()

    def tick(self) -> None:
        """Engines call this only in tests; frames drive a node."""

    def recent_signals(self) -> list[dict]:
        with self._lock:
            return list(self._recent_signals)

    # -- replay -------------------------------------------------------------- #
    def _snapshot(self) -> list:
        with self._lock:
            return list(self._accounts)

    @staticmethod
    def _call(account, fn) -> bool:
        """One account's handling, isolated exactly as the hub's runner does it:
        a broker raising here must not stop the other accounts or the frame."""
        try:
            return bool(fn(account))
        except Exception as exc:
            try:
                account.push_log(f"⚠️ error handling hub event: {exc}")
            except Exception:
                pass
            return False

    def handle_events(self, events: list[dict], frame_age: float,
                      ack) -> None:
        accounts = self._snapshot()
        if not accounts:
            return
        for ev in events:
            kind = ev.get("t")
            if kind == "begin":
                try:
                    now_dt = datetime.fromisoformat(ev["now_dt"])
                except Exception:
                    now_dt = now_ist()
                status = f"Hub · {ev.get('feed_status', '')}"
                for a in accounts:
                    self._call(a, lambda x: x.begin_tick(now_dt, status))
            elif kind == "tick":
                tick = tick_from_dict(ev)
                if tick is None or tick.instrument.symbol not in self._by_symbol:
                    continue
                self._last_price[tick.instrument.symbol] = tick.live_price
                for a in accounts:
                    self._call(a, lambda x: x.on_tick(tick))
            elif kind == "exit":
                inst = self._by_symbol.get(ev.get("symbol", ""))
                if inst is None:
                    continue
                price, reason = float(ev["price"]), str(ev.get("reason", ""))
                for a in accounts:
                    self._call(a, lambda x: x.on_exit(inst, price, reason))
            elif kind == "signal":
                self._handle_signal(ev, accounts, frame_age, ack)

    def _handle_signal(self, ev: dict, accounts: list, frame_age: float,
                       ack) -> None:
        sid = str(ev.get("signal_id", ""))
        sev = signal_from_dict(ev)
        if sev is None or sev.instrument.symbol not in self._by_symbol:
            return
        if sid in self._seen:
            return                                  # duplicate frame
        self._seen[sid] = None
        while len(self._seen) > SEEN_SIGNALS_KEEP:
            self._seen.popitem(last=False)
        if frame_age > MAX_SIGNAL_AGE_SECONDS:
            # Late is worse than absent: the price the decision was made at is
            # gone. Exits are never dropped this way.
            note = f"arrived {frame_age:.1f}s late — skipped, not entered"
            for a in accounts:
                a.push_log(f"⏭️ {sev.instrument.symbol}: signal {note}.")
            ack({"t": "ack", "signal_id": sid, "entered": False, "note": note})
            return
        self._record(sev)
        results = [self._call(a, lambda x: x.on_signal(sev)) for a in accounts]
        ack({"t": "ack", "signal_id": sid, "entered": any(results),
             "note": "" if any(results) else "no account entered"})

    def _record(self, sev) -> None:
        s = sev.signal
        rules = self.symbol_rules.get(sev.instrument.symbol)
        rr = (rules.risk_reward if rules is not None and rules.risk_reward > 0
              else self.params.risk_reward)
        dist = abs(s.entry_price - s.stop_loss)
        target = (s.entry_price + rr * dist if s.side == "BUY"
                  else s.entry_price - rr * dist)
        row = {"time": now_ist().strftime("%H:%M:%S"),
               "symbol": sev.instrument.symbol,
               "segment": sev.instrument.segment.value, "side": s.side,
               "entry": round(s.entry_price, 2), "stop": round(s.stop_loss, 2),
               "target": round(target, 2), "rr": round(rr, 2),
               "reason": s.reason}
        with self._lock:
            self._recent_signals.insert(0, row)
            self._recent_signals = self._recent_signals[:50]

    # -- the link is down ---------------------------------------------------- #
    def offline_safety(self) -> None:
        """Called while the hub link is unusable. The ONE thing a node must not
        depend on the hub for is being flat at the close: an intraday position
        held overnight is auto-squared by the broker at the auction price, or
        becomes a delivery the account never funded. So when the session's own
        square-off time passes, flatten on this machine's clock.

        Stops and targets keep working — they rest at the broker — but nothing
        TRAILS or books partials without prices, which is the cost of the hub
        being unreachable, and is why the link loss is reported loudly."""
        if not self.square_off_enabled:
            return
        today = now_ist().strftime("%Y-%m-%d")
        if self._squared_off_day != today:
            self._squared_off_day = today
            self._squared_off.clear()
        for inst in self.instruments:
            if inst.symbol in self._squared_off:
                continue
            cutoff = config.square_off_time_for(inst.segment, self.mode,
                                                self.square_off_time)
            if cutoff is None:
                continue
            if now_for_segment(inst.segment).time() < cutoff:
                continue
            price = self._last_price.get(inst.symbol)
            if price is None:
                continue
            self._squared_off.add(inst.symbol)
            reason = f"SQUARE-OFF ({cutoff.strftime('%H:%M')}, hub unreachable)"
            for a in self._snapshot():
                if self._call(a, lambda x: x.holds(inst.symbol)):
                    self._call(a, lambda x: x.on_exit(inst, price, reason))


# --------------------------------------------------------------------------- #
#  The link
# --------------------------------------------------------------------------- #
class HubLink:
    def __init__(self):
        self.url = hub_ws_url()
        self.node_id = _env("NODE_ID")
        self.secret = _env("NODE_SECRET")
        self.connected = False
        self.last_frame_at = 0.0
        self.last_error = ""
        self.feed_simulated = False
        self.session_active = False
        self.last_summary: dict = {}
        #: The client account(s) this server trades for — one client, one
        #: server. Set by the hub in every session frame; empty until then, so
        #: a server that has not heard from its hub starts nobody.
        self.assigned: list[str] = []
        #: The broadcast being followed, held in memory only (client_run).
        self._run: Optional[BroadcastConfig] = None
        self._rules: dict = {}
        self._applied: Optional[dict] = None
        self._runners: "weakref.WeakSet[HubRunner]" = weakref.WeakSet()
        self._age = FrameAge()
        self._inbox: "queue.Queue[dict]" = queue.Queue(maxsize=2000)
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._ws: Any = None
        self._stop = threading.Event()
        self._started = False
        self._apply_lock = threading.Lock()

    # -- registry of runners the frames are replayed into -------------------- #
    def register(self, runner: HubRunner) -> None:
        self._runners.add(runner)

    def unregister(self, runner: HubRunner) -> None:
        self._runners.discard(runner)

    # -- state --------------------------------------------------------------- #
    def link_usable(self) -> bool:
        return self.connected and (time.time() - self.last_frame_at) < STALE_LINK_SECONDS

    def feed_unusable(self) -> bool:
        return (not self.link_usable()) or self.feed_simulated

    def status_text(self) -> str:
        if not self.connected:
            return "🔴 Hub disconnected"
        if not self.link_usable():
            return "🟠 Hub link stale"
        return "🟢 Hub link live" + (" (simulated data)" if self.feed_simulated else "")

    # -- lifecycle ----------------------------------------------------------- #
    def start(self) -> None:
        if self._started:
            return
        problem = self._config_problem()
        if problem:
            self.last_error = problem
            print(f"[hub_link] NOT connecting: {problem}")
            return
        self._started = True
        for name, target in (("hub-link", self._thread_main),
                             ("hub-process", self._process_loop),
                             ("hub-report", self._report_loop)):
            threading.Thread(target=target, daemon=True, name=name).start()
        print(f"[hub_link] Node {self.node_id} dialling {self.url}")

    def _config_problem(self) -> str:
        if not (self.url and self.node_id and self.secret):
            return "HUB_URL, NODE_ID and NODE_SECRET must all be set."
        insecure_ok = (_env("ALLOW_INSECURE_HUB").lower() in ("1", "true", "yes"))
        if self.url.startswith("ws://") and not insecure_ok:
            return ("HUB_URL is not TLS (ws://). The node secret would cross "
                    "the network in clear text. Use wss://, or set "
                    "ALLOW_INSECURE_HUB=true for local testing only.")
        return ""

    def stop(self) -> None:
        self._stop.set()

    # -- socket -------------------------------------------------------------- #
    def _thread_main(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._connect_forever())

    async def _connect_forever(self) -> None:
        try:
            from websockets.asyncio.client import connect
            header_kw = "additional_headers"
        except ImportError:                          # websockets < 13
            from websockets import connect           # type: ignore
            header_kw = "extra_headers"
        backoff = 1.0
        while not self._stop.is_set():
            try:
                headers = {"Authorization": f"Bearer {self.node_id}:{self.secret}"}
                async with connect(self.url, max_size=8 * 1024 * 1024,
                                   **{header_kw: headers}) as ws:
                    self._ws = ws
                    self.connected = True
                    self.last_frame_at = time.time()
                    self.last_error = ""
                    backoff = 1.0
                    await ws.send(json.dumps({
                        "t": "hello", "version": f"{NODE_VERSION}/p{PROTOCOL_VERSION}"}))
                    async for raw in ws:
                        self._on_raw(raw)
            except Exception as exc:
                self.last_error = f"{type(exc).__name__}: {exc}"[:200]
            finally:
                self.connected = False
                self._ws = None
            await asyncio.sleep(min(backoff, 30.0))
            backoff *= 2

    def _on_raw(self, raw: Any) -> None:
        try:
            msg = json.loads(raw)
        except ValueError:
            return
        if not isinstance(msg, dict):
            return
        self.last_frame_at = time.time()
        if "hub_ts" in msg:
            msg["_age"] = self._age.observe(msg["hub_ts"])
        try:
            self._inbox.put_nowait(msg)
        except queue.Full:
            # Drop the oldest: stale prices are worth less than fresh ones, and
            # the queue only fills when the engines fall behind the hub.
            try:
                self._inbox.get_nowait()
                self._inbox.put_nowait(msg)
            except Exception:
                pass

    def send(self, frame: dict) -> bool:
        loop, ws = self._loop, self._ws
        if not (self.connected and loop and ws):
            return False
        try:
            asyncio.run_coroutine_threadsafe(ws.send(json.dumps(frame, default=str)),
                                             loop)
            return True
        except Exception:
            return False

    # -- processing (its own thread, so the socket stays responsive while a
    #    broker call inside an engine takes seconds) ------------------------- #
    def _process_loop(self) -> None:
        while not self._stop.is_set():
            try:
                msg = self._inbox.get(timeout=1.0)
            except queue.Empty:
                self._watchdog()
                continue
            try:
                self._dispatch(msg)
            except Exception as exc:
                self.last_error = f"frame error: {exc}"[:200]

    def _watchdog(self) -> None:
        if self.link_usable():
            return
        for runner in list(self._runners):
            try:
                runner.offline_safety()
            except Exception:
                pass

    def _dispatch(self, msg: dict) -> None:
        kind = msg.get("t")
        if kind == "poll":
            age = float(msg.get("_age", 0.0))
            events = msg.get("events") or []
            for runner in list(self._runners):
                runner.handle_events(events, age, self.send)
        elif kind == "session":
            self._on_session(msg)
        elif kind == "control":
            self._on_control(msg)
        # "ping" needs nothing: receiving it already refreshed last_frame_at.

    # -- session ------------------------------------------------------------- #
    def _on_session(self, msg: dict) -> None:
        self.feed_simulated = bool(msg.get("feed_simulated"))
        # Whom this server serves — one client, one server. Updated on EVERY
        # session frame, active or not; a client taken off this server has
        # their bot stopped here rather than left trading from the wrong IP.
        assigned = [str(u) for u in (msg.get("clients") or []) if u]
        dropped = [u for u in self.assigned if u not in assigned]
        self.assigned = assigned
        if dropped:
            self._stop_clients("this server is no longer assigned to you",
                               only=dropped)
        if not msg.get("active"):
            if self.session_active or self._applied is not None:
                self._stop_clients("the hub stopped the broadcast")
            self.session_active = False
            self._applied = None
            self._run = None
            self._rules = {}
            return
        cfg = BroadcastConfig.from_dict(msg.get("config") or {})
        reason = cfg.validate()
        if reason:
            # A node never trusts the wire: a config its own validators reject
            # is not run, and the reason goes back in the next report.
            self.last_error = f"broadcast config rejected: {reason}"
            return
        with self._apply_lock:
            new = {"config": cfg.to_dict(),
                   "symbols": msg.get("symbol_settings") or {}}
            if self._applied is not None and self._applied != new:
                # The hub restarted a DIFFERENT run without a stop in between.
                # _start_one_client says "already running" for a live engine, so
                # without this the old engine would keep trading the old config.
                self._stop_clients("the hub changed the broadcast")
            # Active BEFORE _apply, not after: _apply starts the client through
            # bot._start_one_client, which reads the run via client_run ->
            # current_run(), and that answers "none" until session_active is
            # set. Setting it afterwards made every broadcast start fail with
            # "no client mode configured" (found on the first live run; the
            # unit tests stubbed _start_assigned out, so they never saw it).
            self._applied = new
            self.session_active = True
            self._apply(cfg, new["symbols"])

    def _apply(self, cfg: BroadcastConfig, symbol_settings: dict) -> None:
        """Hold the hub's run IN MEMORY (client_run reads it from here), then
        start this server's assigned client on it.

        Nothing is written to admin_config or symbol_config. The fleet shares
        one database, so those ARE the admin's own saved settings — writing the
        run into them (as the first version did) overwrote the admin's client
        defaults and deleted their per-stock settings outside the broadcast."""
        rules = {}
        for sym, raw in (symbol_settings or {}).items():
            try:
                sc = symbol_config.validate(symbol_config.SymbolConfig(
                    **{k: v for k, v in (raw or {}).items()
                       if k in symbol_config.SymbolConfig.__dataclass_fields__}))
                r = symbol_config.to_rules(str(sym), sc)
            except Exception:
                continue          # one bad row must not stop the run starting
            if r is not None:
                rules[str(sym)] = r
        self._run = cfg
        self._rules = rules
        self.last_summary = self._start_assigned(Environment(cfg.environment))

    def _start_assigned(self, environment: Environment) -> dict:
        """Start ONLY this server's client(s), through the ordinary client start
        path. Same summary shape as bot._fan_out_to_clients, so the hub shows
        why a client is not trading in the same words."""
        from api.routers import bot
        import user_manager
        accounts = {u.get("username"): u
                    for u in user_manager.list_users(role="client")}
        started, skipped = [], []
        if not self.assigned:
            skipped.append({"username": "-",
                            "reason": "no client is assigned to this server"})
        for username in self.assigned:
            acct = accounts.get(username)
            if acct is None:
                skipped.append({"username": username,
                                "reason": "no such client account"})
                continue
            reason = bot._start_one_client(acct, environment)
            if reason:
                skipped.append({"username": username, "reason": reason})
            else:
                started.append(username)
        return {"total": len(self.assigned), "started": started,
                "skipped": skipped}

    # -- what client_run reads ---------------------------------------------- #
    def current_run(self) -> Optional[BroadcastConfig]:
        """The broadcast this server is following, or None when none is."""
        return self._run if self.session_active else None

    def current_rules(self) -> dict:
        return dict(self._rules) if self.session_active else {}

    def _client_usernames(self) -> list[str]:
        """EVERY client account in the (shared) database. Only for sweeping —
        stop and flatten reach any client bot found on this machine, which is
        the safe direction. Starting and reporting use `self.assigned`."""
        import user_manager
        return [u.get("username") for u in user_manager.list_users(role="client")
                if u.get("username")]

    def _stop_clients(self, why: str, only: Optional[list[str]] = None) -> None:
        from api import engine_registry
        for username in (only if only is not None else self._client_usernames()):
            try:
                if engine_registry.stop_engine(username):
                    eng = engine_registry.get_engine(username)
                    if eng is not None:
                        eng.state.push_log(f"Bot stopped — {why}.")
            except Exception:
                pass

    def _on_control(self, msg: dict) -> None:
        from api import engine_registry
        action = msg.get("action")
        if action == "flatten":
            for username in self._client_usernames():
                for eng in engine_registry.get_engines(username):
                    with eng.state.lock:
                        held = list(eng.state.open_positions)
                    for symbol in held:
                        try:
                            eng.close_position(symbol)
                        except Exception:
                            pass
        elif action == "stop":
            # The admin disabled this server's client: their bot stops now,
            # not at the next broadcast change.
            self._stop_clients("your account was disabled", only=list(self.assigned))
        elif action == "resync" and self.session_active and self._applied:
            cfg = BroadcastConfig.from_dict(self._applied["config"])
            with self._apply_lock:
                self._apply(cfg, self._applied["symbols"])

    # -- reporting ----------------------------------------------------------- #
    def build_report(self) -> dict:
        from api import engine_registry
        from api.routers import bot

        clients, positions = [], {}
        day = realized = unrealized = 0.0
        running, environment, broker = False, "", ""
        # Where trades are ACTUALLY going right now. Read off the live engine's
        # DBManager, not from config: DBManager drops to a local file the moment
        # a Mongo call fails mid-session, and from then on this server's trades
        # exist only on this server. The hub reads every client's history from
        # ITS database, so that must be reported, not discovered later from a
        # client who looks like they stopped trading.
        db_backend = ""
        # Only this server's own client. Every account in the shared database
        # is visible here, and reporting them would tell the hub that this
        # server trades clients it does not.
        for username in list(self.assigned):
            engines = engine_registry.get_engines(username)
            if not engines:
                continue
            snap = bot._merge_board(engines)
            running = running or bool(snap.get("running"))
            environment = environment or engines[0].environment.value
            broker = broker or str(snap.get("broker_name", ""))
            day += float(snap.get("day_pnl") or 0.0)
            realized += float(snap.get("realized_pnl") or 0.0)
            unrealized += float(snap.get("unrealized_pnl") or 0.0)
            opened = snap.get("open_positions") or {}
            for eng in engines:
                backend = getattr(getattr(eng, "db", None), "backend", "")
                if backend and (not db_backend or backend != "MongoDB"):
                    db_backend = backend
            clients.append({"username": username,
                            "running": bool(snap.get("running")),
                            "environment": engines[0].environment.value,
                            "broker": str(snap.get("broker_name", "")),
                            "day_pnl": float(snap.get("day_pnl") or 0.0),
                            "open": sorted(opened)})
            for sym, pos in opened.items():
                positions.setdefault(sym, []).append({
                    "client": username, "side": pos.get("side"),
                    "quantity": pos.get("quantity"),
                    "entry_price": pos.get("entry_price")})
        if not db_backend:
            # No engine yet: report the connection state without attempting one.
            db_backend = "MongoDB" if mongo_client.is_connected() else "Local JSON"
        return {"t": "report", "running": running, "environment": environment,
                "db_backend": db_backend, "db_fingerprint": db_fingerprint(),
                "broker": broker, "clients": clients, "positions": positions,
                "day_pnl": day, "realized_pnl": realized,
                "unrealized_pnl": unrealized, "hub_link": self.status_text(),
                "last_error": self.last_error,
                "last_start": self.last_summary}

    def _report_loop(self) -> None:
        while not self._stop.is_set():
            time.sleep(REPORT_EVERY_SECONDS)
            if not self.connected:
                continue
            try:
                self.send(self.build_report())
            except Exception as exc:
                self.last_error = f"report error: {exc}"[:200]


# --------------------------------------------------------------------------- #
#  Process singleton
# --------------------------------------------------------------------------- #
_link: Optional[HubLink] = None
_link_lock = threading.Lock()


def link() -> HubLink:
    global _link
    with _link_lock:
        if _link is None:
            _link = HubLink()
        return _link


def start_if_worker() -> bool:
    """Called once at API startup. Does nothing on a hub, so the hub's behaviour
    is exactly what it was."""
    if not is_worker():
        return False
    import strategy_runner
    strategy_runner.set_runner_class(HubRunner)
    link().start()
    return True
