"""
fleet_broadcast.py
The HUB half of the fleet: one broadcast session that decides, and pushes what
it decided to every connected client node.

                 admin's own bot                 BROADCAST (this module)
                 ───────────────                 ───────────────────────
    engine_registry / TradingEngine              FleetHub  (module singleton)
    admin's broker, admin's capital              its own StrategyRunner
    started/stopped from the Controls panel      RemoteAccount per node
                                                 started/stopped from Broadcast

The two share exactly one thing: the market-data SOCKET, because the broker
refuses a second one (strategy_runner's feed pool). Everything else is separate
— separate runner key, separate state, separate Start/Stop — so starting or
stopping a broadcast can never start, stop or alter the admin's own bot, and
the reverse.

The runner is the ordinary strategy_runner.StrategyRunner. A client node is
just another Account on it (RemoteAccount), exactly as a local TradingEngine
is. Nothing about how a signal is decided changes.

A broadcast stays ON until the admin stops it. It is not tied to admin's own
bot and not to any node being connected: with nobody listening it keeps
deciding (the ObserverAccount keeps the runner polling), so admin still sees
the signals being generated and a node that connects late joins mid-session.
"""
from __future__ import annotations

import asyncio
import json
import os
import threading
import time
from collections import deque
from dataclasses import asdict, replace
from datetime import datetime, timezone
from typing import Any, Optional

import config
import mcx_rollover
import config_store
import fleet_registry
import mongo_client
import strategy_runner
import symbol_config
from config import Mode, now_ist
from fleet_protocol import (PROTOCOL_VERSION, BroadcastConfig, db_fingerprint,
                            db_problem, signal_to_dict, tick_to_dict)
from strategy import resolve_strategy
from strategy_runner import SignalEvent, TickEvent

_STATE_KEY = "fleet_broadcast"
#: The runner's key. Constant, and prefixed so it can never equal a key an
#: admin or client TradingEngine derives (those start with the mode name).
RUNNER_KEY = "fleet-broadcast"

#: How long after forwarding a signal we keep believing the node holds that
#: symbol while waiting for its ack / next report.
PENDING_TTL_SECONDS = 20.0
#: Un-sent frames a slow node may have queued before we start dropping POLL
#: frames for it. Signals and exits are never dropped.
MAX_QUEUED_FRAMES = 200


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in ("0", "false", "no", "off")


# --------------------------------------------------------------------------- #
#  One connected node
# --------------------------------------------------------------------------- #
class NodeConnection:
    """A node's WebSocket, callable from the runner's worker THREADS.

    FastAPI serves the socket on the event loop; the runner broadcasts from a
    thread pool. `send` is the bridge: it schedules the write on the loop and
    returns immediately, so a slow or dead node can never stall the runner —
    the same isolation rule strategy_runner._broadcast enforces per account.
    """

    def __init__(self, node: dict, ws: Any, loop: asyncio.AbstractEventLoop):
        self.node = node
        self.node_id: str = node["node_id"]
        self.ws = ws
        self.loop = loop
        self.connected = True
        self.connected_at = time.time()
        self.last_seen = time.time()
        self.version = ""
        self.report: dict = {}
        self._queued = 0
        self._lock = threading.Lock()

    def send(self, frame: dict, droppable: bool = False) -> bool:
        if not self.connected:
            return False
        with self._lock:
            if droppable and self._queued >= MAX_QUEUED_FRAMES:
                return False
            self._queued += 1
        try:
            fut = asyncio.run_coroutine_threadsafe(
                self.ws.send_text(json.dumps(frame, default=str)), self.loop)
        except Exception:
            self.connected = False
            with self._lock:
                self._queued -= 1
            return False

        def _done(f) -> None:
            with self._lock:
                self._queued -= 1
            if f.cancelled() or f.exception() is not None:
                self.connected = False

        fut.add_done_callback(_done)
        return True

    def close(self) -> None:
        self.connected = False
        try:
            asyncio.run_coroutine_threadsafe(self.ws.close(), self.loop)
        except Exception:
            pass


# --------------------------------------------------------------------------- #
#  Accounts the broadcast runner feeds
# --------------------------------------------------------------------------- #
class ObserverAccount:
    """Keeps the runner polling when no node is connected, and records what the
    runner says. Holds nothing and enters nothing, so it never affects a
    decision: StrategyRunner.tick simply returns early with zero accounts, which
    would leave the admin looking at a broadcast that silently decides nothing."""

    def __init__(self, log: deque):
        self._log = log

    def begin_tick(self, now_dt, feed_status) -> None: ...
    def on_tick(self, ev) -> bool: return False
    def on_signal(self, ev) -> bool: return False
    def on_exit(self, instrument, price, reason) -> bool: return False
    def holds(self, symbol: str) -> bool: return False

    def push_log(self, msg: str) -> None:
        self._log.appendleft(f"[{now_ist().strftime('%H:%M:%S')}] {msg}")


class RemoteAccount:
    """A client node, seen by the runner as one more execution account.

    It implements the six-method Account protocol and turns each call into an
    event in an ordered frame. It never sizes, never holds a position and never
    touches a broker — it forwards a DECISION. The node that receives it sizes
    against its own capital and its own risk limits (Immutable Rules #1, #4).

    Ticks are BUFFERED and flushed once per poll (end_tick). A signal or exit
    flushes immediately, carrying every tick buffered before it in the SAME
    ordered frame — so a node always sees "price, then decision", never the
    reverse, and an entry is never waiting for the end of a long poll.
    """

    def __init__(self, conn: NodeConnection, log: deque):
        self.conn = conn
        self._log = log
        self._buf: list[dict] = []
        self._lock = threading.Lock()
        self._seq = 0
        #: symbol -> monotonic time we forwarded a signal for it.
        self._pending: dict[str, float] = {}
        self._sig_n = 0

    # -- helpers ------------------------------------------------------------- #
    def _flush_locked(self, droppable: bool) -> bool:
        if not self._buf:
            return True
        self._seq += 1
        frame = {"t": "poll", "seq": self._seq, "hub_ts": time.time(),
                 "events": self._buf}
        self._buf = []
        return self.conn.send(frame, droppable=droppable)

    def _signal_id(self, inst, bar_ts) -> str:
        self._sig_n += 1
        return f"{RUNNER_KEY}|{inst.symbol}|{bar_ts}|{self._sig_n}"

    # -- Account protocol ---------------------------------------------------- #
    def begin_tick(self, now_dt: datetime, feed_status: str) -> None:
        if not self.conn.connected:
            return
        with self._lock:
            self._buf.append({"t": "begin", "now_dt": now_dt.isoformat(),
                              "feed_status": feed_status})

    def on_tick(self, ev: TickEvent) -> bool:
        if not self.conn.connected:
            return False
        with self._lock:
            self._buf.append(tick_to_dict(ev))
        return False

    def on_signal(self, ev: SignalEvent) -> bool:
        """Forward the decision NOW. Returns True when it was sent — the node
        may still decline it (no capital, risk halt), and says so in its ack.
        True is what lets the runner arm its re-entry cooldown and its
        reference exit exactly as it would for a local account that entered."""
        if not self.conn.connected:
            return False
        with self._lock:
            sid = self._signal_id(ev.instrument, ev.bar_ts)
            self._buf.append(signal_to_dict(ev, sid))
            sent = self._flush_locked(droppable=False)
            if sent:
                self._pending[ev.instrument.symbol] = time.monotonic()
        return sent

    def on_exit(self, instrument, price: float, reason: str) -> bool:
        if not self.conn.connected:
            return False
        held = self.holds(instrument.symbol)
        with self._lock:
            self._buf.append({"t": "exit", "symbol": instrument.symbol,
                              "price": float(price), "reason": reason})
            self._flush_locked(droppable=False)
            self._pending.pop(instrument.symbol, None)
        return held

    def holds(self, symbol: str) -> bool:
        """Does this node hold `symbol`? From its last report, plus a short
        grace for a signal it has not had time to ack — the runner's reference
        exit asks this to decide whether anyone still needs the trade managed,
        and answering "no" in that gap would drop the exit for a live entry."""
        if symbol in (self.conn.report.get("positions") or {}):
            return True
        sent = self._pending.get(symbol)
        return sent is not None and (time.monotonic() - sent) < PENDING_TTL_SECONDS

    def push_log(self, msg: str) -> None:
        self._log.appendleft(f"[{now_ist().strftime('%H:%M:%S')}] {msg}")

    def end_tick(self) -> bool:
        if not self.conn.connected:
            return False
        with self._lock:
            return self._flush_locked(droppable=True)

    # -- from the node ------------------------------------------------------- #
    def on_ack(self, msg: dict) -> None:
        sym = str(msg.get("signal_id", "")).split("|")[1:2]
        if not msg.get("entered"):
            if sym:
                self._pending.pop(sym[0], None)
            self._log.appendleft(
                f"[{now_ist().strftime('%H:%M:%S')}] "
                f"{self.conn.node['name']}: signal not taken"
                f"{' — ' + msg['note'] if msg.get('note') else ''}")


# --------------------------------------------------------------------------- #
#  Building the run — the SAME overrides TradingEngine.__init__ applies
# --------------------------------------------------------------------------- #
def resolve_run(cfg: BroadcastConfig):
    """(bound strategy, params, instruments, symbol_rules, poll_seconds).

    This mirrors the parameter overrides in TradingEngine.__init__ step for
    step, because the hub's runner must decide with the very parameters a
    TradingEngine given this config would — the node builds its engine from the
    same config, and the two halves deciding different things is the failure
    this whole design exists to prevent. tests/test_fleet.py pins the two
    together field by field, so a change to one that forgets the other fails a
    test instead of quietly diverging.
    """
    mode = Mode(cfg.mode)
    strategy = resolve_strategy(mode, cfg.strategy_key)
    params = strategy.params
    if cfg.risk_reward and cfg.risk_reward > 0:
        params = replace(params, risk_reward=float(cfg.risk_reward))
    if cfg.min_score and cfg.min_score > 0:
        params = replace(params, cs_min_score=float(cfg.min_score))
    exit_style = (cfg.exit_style or "strategy").strip().lower()
    params = config.apply_exit_style(params, exit_style,
                                     trail_atr_mult=cfg.trail_atr_mult)
    if cfg.max_stop_pct and cfg.max_stop_pct > 0:
        params = replace(params, max_stop_pct=float(cfg.max_stop_pct))
    if cfg.min_stop_pct and cfg.min_stop_pct > 0:
        params = replace(params, min_stop_pct=float(cfg.min_stop_pct))
    if params is not strategy.params:
        strategy = replace(strategy, params=params)
    instruments = [config.INSTRUMENTS_BY_SYMBOL[s] for s in cfg.symbols
                   if s in config.INSTRUMENTS_BY_SYMBOL]
    # Same MCX roll TradingEngine.__init__ applies, so hub and node resolve
    # the same contract for the same symbol.
    instruments = mcx_rollover.current(instruments)
    rules = symbol_config.rules_for(mode.value, [i.symbol for i in instruments])
    poll = 0.5 if mode == Mode.SCALPER else 3.0
    return strategy, params, instruments, rules, poll


def feed_token() -> str:
    """The market-data token — the admin's, exactly as TradingEngine._feed_token
    reads it, so the broadcast shares the admin's socket instead of asking the
    broker for a second one."""
    return config.UPSTOX_LIVE_ACCESS_TOKEN or config.UPSTOX_SANDBOX_TOKEN


# --------------------------------------------------------------------------- #
#  The hub
# --------------------------------------------------------------------------- #
class FleetHub:
    def __init__(self):
        self._lock = threading.RLock()
        self.active = False
        self.config: Optional[BroadcastConfig] = None
        self.started_at = ""
        self.runner: Optional[strategy_runner.StrategyRunner] = None
        self.log: deque[str] = deque(maxlen=200)
        self.observer = ObserverAccount(self.log)
        self.connections: dict[str, NodeConnection] = {}
        self.accounts: dict[str, RemoteAccount] = {}
        #: Per-symbol settings FROZEN at start, sent to nodes so their engines
        #: run the same RR / trail / window rules the hub's runner decided with.
        self._symbol_settings: dict[str, dict] = {}
        self._pinger: Optional[threading.Thread] = None

    # -- session ------------------------------------------------------------- #
    def start(self, cfg: BroadcastConfig) -> None:
        """Begin broadcasting. Raises ValueError for a config that cannot run
        and RuntimeError when it must not — never half-starts."""
        reason = cfg.validate()
        if reason:
            raise ValueError(reason)
        with self._lock:
            if self.active:
                raise RuntimeError(
                    "A broadcast is already running. Stop it before starting "
                    "another — clients follow exactly one run at a time.")
            strategy, params, instruments, rules, poll = resolve_run(cfg)
            token = feed_token()
            runner = strategy_runner.acquire(
                RUNNER_KEY,
                lambda: strategy_runner.StrategyRunner(
                    key=RUNNER_KEY, mode=Mode(cfg.mode), strategy=strategy,
                    params=params, instruments=instruments, feed_token=token,
                    poll_seconds=poll, symbol_rules=rules,
                    square_off_time=cfg.square_off_time,
                    square_off_enabled=cfg.square_off_enabled))
            runner.subscribe(self.observer)
            try:
                runner.ensure_started()
                # Same refusal TradingEngine.start makes: never place real
                # orders off invented prices. Here "place" means telling real
                # client accounts to.
                if cfg.environment == "Live" and runner.feed_is_simulated():
                    raise RuntimeError(
                        "Live broadcast blocked: real market data is "
                        "unavailable, so clients would be told to trade on "
                        "simulated prices. Refresh the Upstox market-data "
                        "token and try again.")
            except Exception:
                strategy_runner.release(RUNNER_KEY, self.observer)
                raise
            self.runner = runner
            self.config = cfg
            self.active = True
            self.started_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
            self._symbol_settings = {
                sym: asdict(c)
                for sym, c in symbol_config.get_all(cfg.mode).items()
                if sym in cfg.symbols}
            self._persist()
            self.log.appendleft(
                f"[{now_ist().strftime('%H:%M:%S')}] Broadcast started — "
                f"{cfg.environment} / {cfg.mode} / {strategy.name} / "
                f"{len(instruments)} instruments.")
            for conn in list(self.connections.values()):
                self._attach(conn)
            self._send_session_all()

    def stop(self) -> None:
        with self._lock:
            if not self.active and self.runner is None:
                return
            self.active = False
            self._persist()
            self._send_session_all()
            for node_id, acct in list(self.accounts.items()):
                strategy_runner.release(RUNNER_KEY, acct)
            self.accounts.clear()
            strategy_runner.release(RUNNER_KEY, self.observer)
            self.runner = None
            self.log.appendleft(
                f"[{now_ist().strftime('%H:%M:%S')}] Broadcast stopped.")

    def _persist(self) -> None:
        config_store.save(_STATE_KEY, {
            "active": self.active,
            "config": self.config.to_dict() if self.config else {},
            "started_at": self.started_at})

    def resume_if_active(self) -> str:
        """Re-start a broadcast the admin left ON across a hub restart. Without
        this a restart silently ends it: nodes still hold positions and would
        lose the prices that manage them. Returns "" or why it did not resume."""
        if not _env_flag("FLEET_AUTO_RESUME", True):
            return "auto-resume disabled"
        state = config_store.load(_STATE_KEY) or {}
        if not state.get("active") or not state.get("config"):
            return ""
        try:
            self.start(BroadcastConfig.from_dict(state["config"]))
            return ""
        except Exception as exc:
            self.log.appendleft(
                f"[{now_ist().strftime('%H:%M:%S')}] Could not resume the "
                f"broadcast after restart: {exc}")
            return str(exc)

    # -- frames -------------------------------------------------------------- #
    def session_frame(self, node_id: str = "") -> dict:
        runner = self.runner
        return {
            "t": "session", "v": PROTOCOL_VERSION, "hub_ts": time.time(),
            # The ONLY client(s) this server may trade (one client, one
            # server). Sent even when no broadcast is active, so a node always
            # knows whom it serves. Without it, every node would start every
            # client it can see in the shared database.
            "clients": fleet_registry.clients_of(node_id),
            "active": bool(self.active),
            "config": self.config.to_dict() if (self.active and self.config) else None,
            "symbol_settings": self._symbol_settings if self.active else {},
            "feed_simulated": bool(runner.feed_is_simulated()) if runner else False,
        }

    def _send_session_all(self) -> None:
        for conn in list(self.connections.values()):
            conn.send(self.session_frame(conn.node_id))

    # -- node lifecycle ------------------------------------------------------ #
    def _attach(self, conn: NodeConnection) -> None:
        """Subscribe this node to the live runner. Called with the lock held."""
        if self.runner is None or conn.node_id in self.accounts:
            return
        acct = RemoteAccount(conn, self.log)
        self.accounts[conn.node_id] = acct
        self.runner.subscribe(acct)

    def on_connect(self, conn: NodeConnection) -> None:
        with self._lock:
            old = self.connections.get(conn.node_id)
            if old is not None and old is not conn:
                # A reconnect beat the old socket's timeout. The newest wins.
                self._detach(old)
                old.close()
            self.connections[conn.node_id] = conn
            if self.active:
                self._attach(conn)
            conn.send(self.session_frame(conn.node_id))
            self.log.appendleft(
                f"[{now_ist().strftime('%H:%M:%S')}] Node connected: "
                f"{conn.node['name']} ({conn.node_id}).")
        self._ensure_pinger()

    def _detach(self, conn: NodeConnection) -> None:
        acct = self.accounts.pop(conn.node_id, None)
        if acct is not None and self.runner is not None:
            self.runner.unsubscribe(acct)

    def on_disconnect(self, conn: NodeConnection) -> None:
        conn.connected = False
        with self._lock:
            if self.connections.get(conn.node_id) is conn:
                self.connections.pop(conn.node_id, None)
                self._detach(conn)
                self.log.appendleft(
                    f"[{now_ist().strftime('%H:%M:%S')}] Node disconnected: "
                    f"{conn.node['name']} ({conn.node_id}).")

    def on_message(self, conn: NodeConnection, msg: dict) -> None:
        conn.last_seen = time.time()
        kind = msg.get("t")
        if kind == "hello":
            conn.version = str(msg.get("version", ""))[:40]
            try:
                fleet_registry.record_connect(conn.node_id, conn.version)
            except Exception:
                pass
        elif kind == "report":
            conn.report = msg
        elif kind == "ack":
            acct = self.accounts.get(conn.node_id)
            if acct is not None:
                acct.on_ack(msg)

    def _control(self, action: str, node_id: Optional[str]) -> int:
        sent = 0
        with self._lock:
            for nid, conn in self.connections.items():
                if node_id and nid != node_id:
                    continue
                if conn.send({"t": "control", "action": action,
                              "hub_ts": time.time()}):
                    sent += 1
        return sent

    def flatten(self, node_id: Optional[str] = None) -> int:
        """Ask node(s) to close every open position now. Returns how many were
        asked. The node closes through its own engine and its own broker."""
        return self._control("flatten", node_id)

    def stop_node_clients(self, node_id: str) -> int:
        """Stop the client bot on ONE server now (e.g. the client was
        disabled). Unlike disconnecting it, the server keeps its price feed, so
        a position still open there stays managed until it closes."""
        return self._control("stop", node_id)

    def resync(self, node_id: Optional[str] = None) -> int:
        """Ask node(s) to (re)start their client bots on the running broadcast.
        For a client who connected their broker AFTER the session began: the
        node only starts clients when a session arrives, not on a timer, so a
        client who stopped their own bot is never restarted behind their back."""
        if not self.active:
            return 0
        return self._control("resync", node_id)

    def disconnect(self, node_id: str) -> None:
        """Drop a node's connection (used when it is revoked or disabled)."""
        with self._lock:
            conn = self.connections.get(node_id)
        if conn is not None:
            self.on_disconnect(conn)
            conn.close()

    def _ensure_pinger(self) -> None:
        if self._pinger is not None and self._pinger.is_alive():
            return

        def _loop() -> None:
            while True:
                time.sleep(10.0)
                for conn in list(self.connections.values()):
                    conn.send({"t": "ping", "hub_ts": time.time()},
                              droppable=True)

        self._pinger = threading.Thread(target=_loop, daemon=True,
                                        name="fleet-pinger")
        self._pinger.start()

    # -- read side ----------------------------------------------------------- #
    def client_live(self, username: str) -> Optional[dict]:
        """What the connected nodes say about one client: which server they are
        on and whether their bot is running. None when no connected node has
        them. This is what makes the admin's client views fleet-aware — the hub
        has no TradingEngine for a client trading on a node, so asking
        engine_registry alone reports them as stopped."""
        with self._lock:
            for conn in self.connections.values():
                if not conn.connected:
                    continue
                for c in (conn.report.get("clients") or []):
                    if c.get("username") == username:
                        return {"node_id": conn.node_id,
                                "node_name": conn.node["name"],
                                "running": bool(c.get("running")),
                                "environment": c.get("environment") or None,
                                "broker": c.get("broker") or None,
                                "open": list(c.get("open") or []),
                                "day_pnl": float(c.get("day_pnl") or 0.0)}
        return None

    def node_of(self, username: str) -> Optional[str]:
        live = self.client_live(username)
        return live["node_name"] if live else None

    def status(self) -> dict:
        hub_fp = db_fingerprint()
        with self._lock:
            runner = self.runner
            cfg = self.config.to_dict() if self.config else None
            nodes = []
            registered = {n["node_id"]: n for n in fleet_registry.list_nodes()}
            for nid, rec in registered.items():
                conn = self.connections.get(nid)
                rep = (conn.report if conn else {}) or {}
                nodes.append({
                    "node_id": nid, "name": rec["name"],
                    # The one client this server trades for ("" = none: a
                    # server made before the binding existed trades nobody).
                    "username": rec.get("username", ""),
                    "enabled": rec.get("enabled", True),
                    "connected": bool(conn and conn.connected),
                    "seconds_since_seen": (round(time.time() - conn.last_seen, 1)
                                           if conn else None),
                    "version": (conn.version if conn else "") or rec.get("last_version", ""),
                    "last_connected_at": rec.get("last_connected_at", ""),
                    "created_at": rec.get("created_at", ""),
                    "running": bool(rep.get("running")),
                    "environment": rep.get("environment", ""),
                    "broker": rep.get("broker", ""),
                    "clients": rep.get("clients", []),
                    "positions": rep.get("positions", {}),
                    "day_pnl": rep.get("day_pnl", 0.0),
                    "realized_pnl": rep.get("realized_pnl", 0.0),
                    "unrealized_pnl": rep.get("unrealized_pnl", 0.0),
                    "hub_link": rep.get("hub_link", ""),
                    "last_error": rep.get("last_error", ""),
                    # {"started": [...], "skipped": [{"username","reason"}]} —
                    # why a client on that node is NOT trading, in words.
                    "last_start": rep.get("last_start", {}),
                    # Will this server's trades reach THIS database? A node that
                    # silently fell back to a local file, or points at another
                    # database, trades fine and is invisible here.
                    "db_backend": rep.get("db_backend", ""),
                    "db_problem": db_problem(
                        rep.get("db_backend", ""), rep.get("db_fingerprint", ""),
                        hub_fp) if (conn and conn.connected) else "",
                })
            return {
                "hub_db_backend": ("MongoDB" if mongo_client.is_connected()
                                   else "Local JSON"),
                "active": self.active, "config": cfg,
                "started_at": self.started_at,
                "feed": runner.status() if runner else "🔴 Not started",
                "feed_simulated": bool(runner.feed_is_simulated()) if runner else False,
                "signals": runner.recent_signals()[:30] if runner else [],
                "nodes": nodes,
                "connected": sum(1 for n in nodes if n["connected"]),
                "log": list(self.log)[:80],
            }


#: The one hub for this process. api/routers/fleet.py is the only caller.
hub = FleetHub()
