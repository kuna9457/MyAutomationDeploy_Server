"""The fleet: one hub deciding, client nodes executing.

What is pinned here, and why each one is the thing that would hurt if it broke:

  * NOTHING SIZE-SHAPED OR SECRET crosses the wire — a signal has no quantity,
    so each node sizes on its own capital (Immutable Rules #1 and #4).
  * The hub's runner decides with the SAME parameters a TradingEngine given the
    same config would (resolve_run vs engine.__init__). If they drift, hub and
    node decide different things and every "replicated" trade is fiction.
  * Frames replay on a node in the order the hub produced them, and a signal is
    never waiting for the end of a poll.
  * A late signal is dropped, a late EXIT never is, a duplicate never enters
    twice, and none of that depends on the two machines' clocks agreeing.
  * A broadcast is its own thing: it never appears in engine_registry, and
    starting/stopping it leaves the shared feed pool clean.
  * A node rejects a config its own validators refuse, and a Live broadcast
    refuses to start on invented prices.
  * One real WebSocket round trip, end to end, with the real router.

Every config write is redirected to an in-memory store: config_store otherwise
writes to whatever MongoDB the machine's .env points at.
"""
from __future__ import annotations

import json
import socket
import threading
import time
from datetime import datetime

import pytest

import admin_config
import config
import config_store
import engine as engine_mod
import fleet_broadcast
import fleet_registry
import hub_link
import strategy_runner
from config import Environment, Mode
from fleet_protocol import (MAX_SIGNAL_AGE_SECONDS, BroadcastConfig, FrameAge,
                            signal_from_dict, signal_to_dict, tick_from_dict,
                            tick_to_dict)
from data_feed import LiveQuote
from strategy import Signal
from strategy_runner import SignalEvent, TickEvent

SYMBOLS = [i.symbol for i in config.ALL_INSTRUMENTS
           if i.segment == config.Segment.EQUITY][:3]
assert len(SYMBOLS) == 3


@pytest.fixture(autouse=True)
def mem_store(monkeypatch):
    """No test may write to the developer's real database or data/ folder."""
    store: dict[str, dict] = {}
    monkeypatch.setattr(config_store, "load",
                        lambda name, default=None: json.loads(
                            json.dumps(store.get(name, dict(default or {})))))
    monkeypatch.setattr(config_store, "save",
                        lambda name, data: store.__setitem__(
                            name, json.loads(json.dumps(data))))
    # Never let a developer's real market-data token open a real socket.
    monkeypatch.setattr(fleet_broadcast, "feed_token", lambda: "")
    return store


@pytest.fixture(autouse=True)
def clean_pool():
    yield
    with strategy_runner._feed_lock:
        for entry in list(strategy_runner._feed_pool.values()):
            try:
                entry["feed"].stop()
            except Exception:
                pass
        strategy_runner._feed_pool.clear()
    strategy_runner._runners.pop(fleet_broadcast.RUNNER_KEY, None)


def _cfg(**kw) -> BroadcastConfig:
    base = dict(environment="Paper", mode="Intraday", symbols=list(SYMBOLS))
    base.update(kw)
    return BroadcastConfig(**base)


def _inst(symbol=SYMBOLS[0]):
    return config.INSTRUMENTS_BY_SYMBOL[symbol]


def _tick_event(symbol=SYMBOLS[0], price=100.0) -> TickEvent:
    q = LiveQuote(ltp=price, ts=datetime(2026, 9, 30, 10, 0, 0),
                  received_at=time.monotonic() - 1.5, bid=price - 0.05,
                  ask=price + 0.05, source="ws")
    return TickEvent(_inst(symbol), q, price, True,
                     __import__("pandas").Timestamp("2026-09-30 10:00:00"),
                     datetime(2026, 9, 30, 10, 0, 1), 1.25)


def _signal_event(symbol=SYMBOLS[0]) -> SignalEvent:
    sig = Signal("BUY", 100.0, 99.0, 101.0, "test setup")
    return SignalEvent(_inst(symbol), sig, _tick_event(symbol).quote,
                       __import__("pandas").Timestamp("2026-09-30 10:00:00"))


# --------------------------------------------------------------------------- #
#  Protocol
# --------------------------------------------------------------------------- #
def test_tick_round_trips_through_json():
    ev = _tick_event()
    back = tick_from_dict(json.loads(json.dumps(tick_to_dict(ev))))
    assert back.instrument.symbol == ev.instrument.symbol
    assert back.live_price == ev.live_price and back.atr == ev.atr
    assert back.market_open is True and back.bar_ts == ev.bar_ts
    assert back.now_dt == ev.now_dt
    assert (back.quote.bid, back.quote.ask) == (ev.quote.bid, ev.quote.ask)
    # The age is re-anchored on the receiving machine's clock, not copied.
    assert 1.4 < back.quote.age_seconds < 3.0


def test_signal_round_trips_and_carries_no_quantity_or_secret():
    ev = _signal_event()
    wire = json.loads(json.dumps(signal_to_dict(ev, "sid-1")))
    back = signal_from_dict(wire)
    assert (back.signal.side, back.signal.entry_price, back.signal.stop_loss,
            back.signal.target) == ("BUY", 100.0, 99.0, 101.0)
    # Sizing is the node's own job.
    text = json.dumps(wire).lower()
    for forbidden in ("quantity", "qty", "capital", "token", "secret", "api_key"):
        assert forbidden not in text


def test_unknown_symbol_is_ignored_not_raised():
    d = tick_to_dict(_tick_event())
    d["symbol"] = "NOT-A-SYMBOL"
    assert tick_from_dict(d) is None
    s = signal_to_dict(_signal_event(), "x")
    s["symbol"] = "NOT-A-SYMBOL"
    assert signal_from_dict(s) is None


@pytest.mark.parametrize("kw,needle", [
    (dict(environment="Demo"), "environment"),
    (dict(mode="Swing"), "cannot be broadcast"),
    (dict(symbols=["NOPE"]), "instrument"),
    (dict(exit_style="bogus"), "exit_style"),
    (dict(square_off_time="25:99"), "square_off_time"),
    (dict(min_stop_pct=2.0, max_stop_pct=1.0), "min_stop_pct"),
])
def test_config_validation_rejects(kw, needle):
    assert needle in _cfg(**kw).validate()


def test_a_normal_config_validates():
    assert _cfg().validate() == ""
    assert _cfg(mode="Scalper", exit_style="partial_trail",
                risk_reward=1.0).validate() == ""


def test_frame_age_ignores_clock_skew_but_sees_lateness():
    """A node whose clock is 10 minutes fast must neither reject everything nor
    accept a genuinely late signal."""
    skew = 600.0
    age = FrameAge()
    t = 1_000_000.0
    for i in range(10):                       # steady 50 ms network delay
        a = age.observe(hub_ts=t + i, node_now=t + i + skew + 0.05)
    assert a < 0.01
    late = age.observe(hub_ts=t + 20, node_now=t + 20 + skew + 0.05 + 7.0)
    assert late > MAX_SIGNAL_AGE_SECONDS


# --------------------------------------------------------------------------- #
#  The hub's runner decides with the parameters a TradingEngine would
# --------------------------------------------------------------------------- #
class _NoDB:
    def __init__(self, *a, **k):
        self.backend = "test"


@pytest.mark.parametrize("kw", [
    {},
    dict(risk_reward=2.0),
    dict(min_score=6.0),
    dict(exit_style="partial_trail"),
    dict(exit_style="trail_full", trail_atr_mult=2.5),
    dict(max_stop_pct=2.0, min_stop_pct=0.5),
    dict(mode="Scalper", risk_reward=1.0, exit_style="partial_lock"),
    dict(strategy_key="candlestick_phase1", min_score=7.0,
         min_stop_pct=0.8),
])
def test_hub_runner_params_match_a_real_trading_engine(monkeypatch, kw):
    monkeypatch.setattr(engine_mod, "DBManager", _NoDB)
    monkeypatch.setattr(engine_mod.strategy_runner, "acquire",
                        lambda key, factory: None)
    cfg = _cfg(**kw)
    strategy, params, instruments, rules, poll = fleet_broadcast.resolve_run(cfg)

    eng = engine_mod.TradingEngine(
        Environment.PAPER, Mode(cfg.mode), config.Broker.SIMULATED, instruments,
        100_000.0, strategy_key=cfg.strategy_key, risk_reward=cfg.risk_reward,
        min_score=cfg.min_score, exit_style=cfg.exit_style,
        trail_atr_mult=cfg.trail_atr_mult, max_stop_pct=cfg.max_stop_pct,
        min_stop_pct=cfg.min_stop_pct)

    assert params == eng.params
    assert strategy.params == eng.strategy.params
    assert strategy.key == eng.strategy.key
    assert poll == eng.poll_seconds


# --------------------------------------------------------------------------- #
#  Registry
# --------------------------------------------------------------------------- #
def test_registry_authenticates_and_never_stores_the_secret():
    rec, secret = fleet_registry.create_node("Client A")
    assert "secret_hash" not in rec and secret not in json.dumps(rec)
    assert fleet_registry.authenticate(rec["node_id"], secret)["name"] == "Client A"
    assert fleet_registry.authenticate(rec["node_id"], secret + "x") is None
    assert fleet_registry.authenticate("node-unknown", secret) is None
    assert all("secret_hash" not in n for n in fleet_registry.list_nodes())


def test_a_disabled_or_deleted_node_cannot_connect():
    rec, secret = fleet_registry.create_node("B")
    fleet_registry.set_enabled(rec["node_id"], False)
    assert fleet_registry.authenticate(rec["node_id"], secret) is None
    fleet_registry.set_enabled(rec["node_id"], True)
    assert fleet_registry.authenticate(rec["node_id"], secret) is not None
    assert fleet_registry.delete_node(rec["node_id"]) is True
    assert fleet_registry.authenticate(rec["node_id"], secret) is None


# --------------------------------------------------------------------------- #
#  Hub -> node loopback, no sockets
# --------------------------------------------------------------------------- #
class RecordingAccount:
    """What a node's TradingEngine looks like to the runner."""

    def __init__(self):
        self.calls: list[tuple] = []
        self.held: set[str] = set()
        self.enter = True

    def begin_tick(self, now_dt, feed_status):
        self.calls.append(("begin", feed_status))

    def on_tick(self, ev):
        self.calls.append(("tick", ev.instrument.symbol, ev.live_price))
        return False

    def on_signal(self, ev):
        self.calls.append(("signal", ev.instrument.symbol, ev.signal.side,
                           ev.signal.entry_price))
        if self.enter:
            self.held.add(ev.instrument.symbol)
        return self.enter

    def on_exit(self, inst, price, reason):
        self.calls.append(("exit", inst.symbol, price, reason))
        return inst.symbol in self.held

    def holds(self, symbol):
        return symbol in self.held

    def push_log(self, msg):
        self.calls.append(("log", msg))


class LoopbackConn:
    """A NodeConnection whose 'network' is a list — frames are JSON-encoded and
    decoded exactly as they would be on a socket."""

    def __init__(self, node_id="node-x"):
        self.node = {"node_id": node_id, "name": "Loopback"}
        self.node_id = node_id
        self.connected = True
        self.report: dict = {}
        self.frames: list[dict] = []

    def send(self, frame, droppable=False):
        if not self.connected:
            return False
        self.frames.append(json.loads(json.dumps(frame, default=str)))
        return True


def _node_runner(account, symbols=SYMBOLS):
    instruments = [_inst(s) for s in symbols]
    strategy, params, *_ = fleet_broadcast.resolve_run(_cfg(symbols=list(symbols)))
    runner = hub_link.HubRunner("k", Mode.INTRADAY, strategy, params, instruments)
    runner.subscribe(account)
    return runner


def _replay(runner, conn, age=0.0):
    acks = []
    for frame in conn.frames:
        runner.handle_events(frame["events"], age, acks.append)
    return acks


def test_events_replay_on_a_node_in_the_order_the_hub_produced_them():
    from collections import deque
    conn = LoopbackConn()
    remote = fleet_broadcast.RemoteAccount(conn, deque(maxlen=50))

    remote.begin_tick(datetime(2026, 9, 30, 10, 0, 1), "🟢 live")
    remote.on_tick(_tick_event(SYMBOLS[0], 100.0))
    remote.on_tick(_tick_event(SYMBOLS[1], 50.0))
    assert conn.frames == []                       # ticks buffer, nothing sent yet

    # The signal flushes NOW and carries the ticks buffered before it.
    assert remote.on_signal(_signal_event(SYMBOLS[1])) is True
    assert len(conn.frames) == 1
    kinds = [e["t"] for e in conn.frames[0]["events"]]
    assert kinds == ["begin", "tick", "tick", "signal"]

    remote.on_tick(_tick_event(SYMBOLS[2], 70.0))
    remote.end_tick()
    assert [e["t"] for e in conn.frames[1]["events"]] == ["tick"]

    node_acct = RecordingAccount()
    acks = _replay(_node_runner(node_acct), conn)
    names = [c[0] for c in node_acct.calls]
    assert names == ["begin", "tick", "tick", "signal", "tick"]
    assert node_acct.calls[3][1:] == (SYMBOLS[1], "BUY", 100.0)
    assert acks == [{"t": "ack", "signal_id": acks[0]["signal_id"],
                     "entered": True, "note": ""}]


def test_a_late_signal_is_dropped_but_a_late_exit_is_applied():
    from collections import deque
    conn = LoopbackConn()
    remote = fleet_broadcast.RemoteAccount(conn, deque(maxlen=50))
    remote.on_signal(_signal_event())
    remote.on_exit(_inst(), 99.5, "STOP-LOSS")

    acct = RecordingAccount()
    acct.held.add(SYMBOLS[0])
    acks = _replay(_node_runner(acct), conn, age=MAX_SIGNAL_AGE_SECONDS + 4)
    names = [c[0] for c in acct.calls]
    assert "signal" not in names                   # never entered late
    assert ("exit", SYMBOLS[0], 99.5, "STOP-LOSS") in acct.calls
    assert acks[0]["entered"] is False and "late" in acks[0]["note"]


def test_a_duplicate_signal_never_enters_twice():
    from collections import deque
    conn = LoopbackConn()
    remote = fleet_broadcast.RemoteAccount(conn, deque(maxlen=50))
    remote.on_signal(_signal_event())
    acct = RecordingAccount()
    runner = _node_runner(acct)
    _replay(runner, conn)
    _replay(runner, conn)                          # the same frame, redelivered
    assert [c[0] for c in acct.calls].count("signal") == 1


def test_one_failing_account_does_not_stop_the_others_on_a_node():
    from collections import deque
    conn = LoopbackConn()
    remote = fleet_broadcast.RemoteAccount(conn, deque(maxlen=50))
    remote.on_tick(_tick_event())
    remote.on_signal(_signal_event())

    class Boom(RecordingAccount):
        def on_signal(self, ev):
            raise RuntimeError("broker down")

    bad, good = Boom(), RecordingAccount()
    runner = _node_runner(bad)
    runner.subscribe(good)
    acks = _replay(runner, conn)
    assert any(c[0] == "signal" for c in good.calls)
    assert acks[0]["entered"] is True              # the healthy account entered
    assert any(c[0] == "log" and "broker down" in c[1] for c in bad.calls)


def test_remote_account_reports_holdings_from_node_report_and_pending():
    from collections import deque
    conn = LoopbackConn()
    remote = fleet_broadcast.RemoteAccount(conn, deque(maxlen=50))
    assert remote.holds(SYMBOLS[0]) is False
    remote.on_signal(_signal_event())              # sent, not yet acked
    assert remote.holds(SYMBOLS[0]) is True        # grace: don't drop the exit
    remote.on_ack({"signal_id": f"x|{SYMBOLS[0]}|t|1", "entered": False,
                   "note": "no capital"})
    assert remote.holds(SYMBOLS[0]) is False
    conn.report = {"positions": {SYMBOLS[1]: [{"client": "c"}]}}
    assert remote.holds(SYMBOLS[1]) is True


def test_a_disconnected_node_receives_nothing_and_blocks_nobody():
    from collections import deque
    conn = LoopbackConn()
    conn.connected = False
    remote = fleet_broadcast.RemoteAccount(conn, deque(maxlen=50))
    remote.begin_tick(datetime.now(), "x")
    assert remote.on_tick(_tick_event()) is False
    assert remote.on_signal(_signal_event()) is False
    assert remote.on_exit(_inst(), 1.0, "r") is False
    assert remote.end_tick() is False
    assert conn.frames == []


# --------------------------------------------------------------------------- #
#  The broadcast session is separate from the admin's own bot
# --------------------------------------------------------------------------- #
def test_start_and_stop_leave_the_admin_engine_registry_and_feed_pool_alone():
    from api import engine_registry
    before = dict(engine_registry._engines)
    hub = fleet_broadcast.FleetHub()
    hub.start(_cfg())
    try:
        assert hub.active and hub.status()["active"] is True
        runner = strategy_runner.get(fleet_broadcast.RUNNER_KEY)
        assert runner is not None and runner.running
        assert engine_registry._engines == before   # nothing of admin's touched
    finally:
        hub.stop()
    assert strategy_runner.get(fleet_broadcast.RUNNER_KEY) is None
    assert not hub.active
    assert strategy_runner._feed_pool == {}         # the socket was given back
    assert engine_registry._engines == before


def test_a_second_broadcast_cannot_start_over_a_running_one():
    hub = fleet_broadcast.FleetHub()
    hub.start(_cfg())
    try:
        with pytest.raises(RuntimeError, match="already running"):
            hub.start(_cfg(mode="Scalper"))
    finally:
        hub.stop()


def test_live_broadcast_refuses_to_run_on_simulated_prices():
    hub = fleet_broadcast.FleetHub()
    with pytest.raises(RuntimeError, match="market data"):
        hub.start(_cfg(environment="Live"))
    assert not hub.active
    assert strategy_runner.get(fleet_broadcast.RUNNER_KEY) is None
    assert strategy_runner._feed_pool == {}


def test_an_invalid_config_never_half_starts():
    hub = fleet_broadcast.FleetHub()
    with pytest.raises(ValueError):
        hub.start(_cfg(mode="Swing"))
    assert not hub.active and strategy_runner.get(fleet_broadcast.RUNNER_KEY) is None


def test_a_node_connecting_mid_session_joins_and_leaving_unsubscribes():
    hub = fleet_broadcast.FleetHub()
    hub.start(_cfg())
    try:
        conn = LoopbackConn("node-late")
        hub.on_connect(conn)
        assert conn.frames and conn.frames[0]["t"] == "session"
        assert conn.frames[0]["active"] is True
        assert conn.frames[0]["config"]["mode"] == "Intraday"
        runner = strategy_runner.get(fleet_broadcast.RUNNER_KEY)
        assert runner.account_count() == 2           # observer + the node
        hub.on_disconnect(conn)
        assert runner.account_count() == 1
    finally:
        hub.stop()


def test_stop_tells_every_node_the_session_is_over():
    hub = fleet_broadcast.FleetHub()
    hub.start(_cfg())
    conn = LoopbackConn("node-a")
    hub.on_connect(conn)
    hub.stop()
    last = conn.frames[-1]
    assert last["t"] == "session" and last["active"] is False
    assert last["config"] is None


def test_a_running_broadcast_resumes_after_a_hub_restart(mem_store):
    hub = fleet_broadcast.FleetHub()
    hub.start(_cfg(risk_reward=2.0))
    # The process dies without stop(); a fresh hub reads the persisted state.
    strategy_runner.release(fleet_broadcast.RUNNER_KEY, hub.observer)
    with strategy_runner._feed_lock:
        strategy_runner._feed_pool.clear()
    fresh = fleet_broadcast.FleetHub()
    assert fresh.resume_if_active() == ""
    try:
        assert fresh.active and fresh.config.risk_reward == 2.0
    finally:
        fresh.stop()


def test_a_stopped_broadcast_does_not_resume(mem_store):
    hub = fleet_broadcast.FleetHub()
    hub.start(_cfg())
    hub.stop()
    fresh = fleet_broadcast.FleetHub()
    assert fresh.resume_if_active() == ""
    assert not fresh.active


# --------------------------------------------------------------------------- #
#  Feed pool: extending it must not strand a runner that already borrowed it
# --------------------------------------------------------------------------- #
def test_extending_the_shared_feed_repoints_runners_already_on_it():
    a, b = _inst(SYMBOLS[0]), _inst(SYMBOLS[1])
    strategy, params, *_ = fleet_broadcast.resolve_run(_cfg(symbols=[SYMBOLS[0]]))
    runner = strategy_runner.StrategyRunner(
        key="pool-test", mode=Mode.INTRADAY, strategy=strategy, params=params,
        instruments=[a], feed_token="", poll_seconds=3.0)
    strategy_runner._runners["pool-test"] = runner
    try:
        runner.start()
        first = runner.feed
        # Someone else needs a symbol this socket does not carry.
        feed2, _ = strategy_runner.acquire_feed(Mode.INTRADAY, "", [b])
        assert feed2 is not first
        assert runner.feed is feed2                 # not left on the dead one
    finally:
        runner.stop()
        strategy_runner._runners.pop("pool-test", None)


# --------------------------------------------------------------------------- #
#  The node turns a session into the ordinary client start path
# --------------------------------------------------------------------------- #
def _session(active=True, cfg=None, clients=("c1",), **extra):
    return {"t": "session", "active": active, "hub_ts": time.time(),
            "clients": list(clients),
            "config": (cfg or _cfg()).to_dict() if active else None,
            "symbol_settings": extra.get("symbol_settings", {}),
            "feed_simulated": extra.get("feed_simulated", False)}


@pytest.fixture
def node(monkeypatch):
    link = hub_link.HubLink()
    calls = {"fan_out": [], "stopped": 0}

    def fake_start(env):
        calls["fan_out"].append(env)
        return {"total": 1, "started": ["c1"], "skipped": []}

    monkeypatch.setattr(link, "_start_assigned", fake_start)
    monkeypatch.setattr(link, "_stop_clients",
                        lambda why, only=None: calls.__setitem__(
                            "stopped", calls["stopped"] + 1))
    return link, calls


def test_a_session_is_held_in_memory_and_starts_the_client(node):
    link, calls = node
    link._on_session(_session(cfg=_cfg(risk_reward=2.0, exit_style="partial_trail",
                                       environment="Paper")))
    assert calls["fan_out"] == [Environment.PAPER]
    run = link.current_run()
    assert run.risk_reward == 2.0 and run.exit_style == "partial_trail"
    assert run.symbols == SYMBOLS
    assert link.session_active is True
    assert link.last_summary["started"] == ["c1"]


def test_a_session_never_writes_the_admins_shared_settings(node, mem_store):
    """The fleet shares ONE database: admin_config and symbol_config on a node
    ARE the admin's own saved settings. The first version wrote the broadcast
    into them — overwriting the admin's client defaults and deleting their
    per-stock settings outside the broadcast."""
    import symbol_config
    admin_config.set_mode_config("Intraday", risk_reward=3.0, symbols=["ADMIN1"])
    symbol_config.set_symbol("Intraday", "OTHER", symbol_config.validate(
        symbol_config.SymbolConfig(risk_reward=2.0)))
    before = json.dumps(mem_store, sort_keys=True)

    link, _ = node
    link._on_session(_session(cfg=_cfg(risk_reward=1.0), symbol_settings={
        SYMBOLS[0]: {"risk_reward": 2.0}}))

    assert json.dumps(mem_store, sort_keys=True) == before
    assert admin_config.get_mode_config("Intraday").risk_reward == 3.0
    assert "OTHER" in symbol_config.get_all("Intraday")


def test_an_identical_session_redelivered_does_not_restart_clients(node):
    link, calls = node
    link._on_session(_session())
    link._on_session(_session())                   # e.g. a reconnect
    assert calls["stopped"] == 0


def test_a_changed_session_without_a_stop_restarts_clients_first(node):
    link, calls = node
    link._on_session(_session(cfg=_cfg(risk_reward=1.0)))
    link._on_session(_session(cfg=_cfg(risk_reward=2.0)))
    assert calls["stopped"] == 1                   # old engine would say "already running"
    assert link.current_run().risk_reward == 2.0


def test_an_inactive_session_stops_the_clients(node):
    link, calls = node
    link._on_session(_session())
    link._on_session(_session(active=False))
    assert calls["stopped"] == 1 and link.session_active is False


def test_a_node_rejects_a_config_its_own_validators_refuse(node):
    link, calls = node
    bad = _session()
    bad["config"]["mode"] = "Swing"
    link._on_session(bad)
    assert calls["fan_out"] == [] and link.session_active is False
    assert "rejected" in link.last_error


def test_per_symbol_settings_travel_with_the_session(node):
    link, calls = node
    link._on_session(_session(symbol_settings={
        SYMBOLS[0]: {"risk_reward": 2.0, "trade_days": [0, 1]},
        SYMBOLS[1]: {"not_a_field": 1}}))          # all-default -> dropped
    rules = link.current_rules()
    assert rules[SYMBOLS[0]].risk_reward == 2.0
    assert SYMBOLS[1] not in rules


def test_live_start_is_blocked_on_a_node_when_the_hub_link_is_down():
    """engine.start() refuses Live when feed_is_simulated() — a node with no
    hub link has no prices at all, which is the same refusal."""
    link = hub_link.HubLink()
    assert link.feed_unusable() is True            # never connected
    link.connected, link.last_frame_at = True, time.time()
    assert link.feed_unusable() is False
    link.feed_simulated = True                     # the hub is on invented data
    assert link.feed_unusable() is True
    link.feed_simulated = False
    link.last_frame_at = time.time() - 120         # socket open, hub silent
    assert link.feed_unusable() is True


def test_node_refuses_to_send_its_secret_in_clear_text(monkeypatch):
    monkeypatch.setenv("HUB_URL", "ws://hub.example.com")
    monkeypatch.setenv("NODE_ID", "node-1")
    monkeypatch.setenv("NODE_SECRET", "s")
    monkeypatch.delenv("ALLOW_INSECURE_HUB", raising=False)
    assert "TLS" in hub_link.HubLink()._config_problem()
    monkeypatch.setenv("ALLOW_INSECURE_HUB", "true")
    assert hub_link.HubLink()._config_problem() == ""
    monkeypatch.setenv("HUB_URL", "https://hub.example.com/")
    assert hub_link.hub_ws_url() == "wss://hub.example.com/fleet/ws"


def test_hub_url_normalisation(monkeypatch):
    for raw, want in [("hub.example.com", "wss://hub.example.com/fleet/ws"),
                      ("wss://h/fleet/ws", "wss://h/fleet/ws"),
                      ("http://localhost:8000", "ws://localhost:8000/fleet/ws")]:
        monkeypatch.setenv("HUB_URL", raw)
        assert hub_link.hub_ws_url() == want


def test_offline_safety_squares_off_on_the_nodes_own_clock(monkeypatch):
    """With the hub unreachable, the one thing a node must still do by itself is
    be flat at the close."""
    import hub_link as hl
    acct = RecordingAccount()
    acct.held.add(SYMBOLS[0])
    runner = _node_runner(acct)
    runner._last_price[SYMBOLS[0]] = 99.0
    monkeypatch.setattr(hl, "now_for_segment",
                        lambda seg: datetime(2026, 9, 30, 15, 25))
    runner.offline_safety()
    exits = [c for c in acct.calls if c[0] == "exit"]
    assert len(exits) == 1 and "hub unreachable" in exits[0][3]
    runner.offline_safety()                        # once per symbol per day
    assert len([c for c in acct.calls if c[0] == "exit"]) == 1


def test_offline_safety_does_nothing_before_the_cutoff(monkeypatch):
    import hub_link as hl
    acct = RecordingAccount()
    acct.held.add(SYMBOLS[0])
    runner = _node_runner(acct)
    runner._last_price[SYMBOLS[0]] = 99.0
    monkeypatch.setattr(hl, "now_for_segment",
                        lambda seg: datetime(2026, 9, 30, 11, 0))
    runner.offline_safety()
    assert [c for c in acct.calls if c[0] == "exit"] == []


# --------------------------------------------------------------------------- #
#  A real WebSocket, the real router, the real node link
# --------------------------------------------------------------------------- #
def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait(cond, timeout=20.0, step=0.1):
    end = time.time() + timeout
    while time.time() < end:
        if cond():
            return True
        time.sleep(step)
    return False


@pytest.fixture
def live_hub(monkeypatch):
    import uvicorn
    from api.main import app
    port = _free_port()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port,
                                           log_level="error", lifespan="on"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    assert _wait(lambda: server.started, 15), "hub did not start"
    yield port
    fleet_broadcast.hub.stop()
    for nid in list(fleet_broadcast.hub.connections):
        fleet_broadcast.hub.disconnect(nid)
    server.should_exit = True
    thread.join(timeout=10)


def _node_link(monkeypatch, port, node_id, secret):
    monkeypatch.setenv("HUB_URL", f"ws://127.0.0.1:{port}")
    monkeypatch.setenv("NODE_ID", node_id)
    monkeypatch.setenv("NODE_SECRET", secret)
    monkeypatch.setenv("ALLOW_INSECURE_HUB", "true")
    return hub_link.HubLink()


def test_end_to_end_over_a_real_websocket(live_hub, monkeypatch):
    rec, secret = fleet_registry.create_node("E2E node", username="e2e-client")
    link = _node_link(monkeypatch, live_hub, rec["node_id"], secret)
    started = []
    monkeypatch.setattr(link, "_start_assigned",
                        lambda env: started.append(list(link.assigned)) or
                        {"total": 1, "started": list(link.assigned), "skipped": []})

    acct = RecordingAccount()
    runner = _node_runner(acct)
    link.register(runner)
    link.start()
    hub = fleet_broadcast.hub

    assert _wait(lambda: rec["node_id"] in hub.connections), link.last_error
    assert link.connected

    # Its one client arrives over the wire before any broadcast exists.
    assert _wait(lambda: link.assigned == ["e2e-client"]), link.assigned

    hub.start(_cfg())                              # simulated feed, Paper
    assert _wait(lambda: link.session_active), "node never got the session"
    assert started == [["e2e-client"]]             # it started exactly its client
    # Prices flow hub -> node and reach the account the engine would be.
    assert _wait(lambda: any(c[0] == "tick" for c in acct.calls), 20), acct.calls
    assert any(c[0] == "begin" for c in acct.calls)

    # The node reports back; the hub can see it.
    assert _wait(lambda: hub.connections[rec["node_id"]].report.get("t") == "report", 15)
    status = hub.status()
    assert status["connected"] == 1 and status["nodes"][0]["connected"] is True

    # Stop ends it for the node too.
    hub.stop()
    assert _wait(lambda: not link.session_active), "node kept running after stop"
    link.stop()


def test_a_wrong_secret_never_connects(live_hub, monkeypatch):
    rec, secret = fleet_registry.create_node("Intruder target")
    link = _node_link(monkeypatch, live_hub, rec["node_id"], secret + "WRONG")
    link.start()
    time.sleep(2.0)
    assert link.connected is False
    assert rec["node_id"] not in fleet_broadcast.hub.connections
    assert link.last_error                          # it was refused, and said so
    link.stop()


def test_admin_endpoints_need_an_admin_token(live_hub):
    import urllib.error
    import urllib.request
    for method, path in [("GET", "/fleet/broadcast/status"),
                         ("POST", "/fleet/broadcast/stop"),
                         ("GET", "/fleet/pnl?environment=Paper"),
                         ("GET", "/fleet/nodes")]:
        req = urllib.request.Request(f"http://127.0.0.1:{live_hub}{path}",
                                     method=method,
                                     data=b"" if method == "POST" else None)
        with pytest.raises(urllib.error.HTTPError) as err:
            urllib.request.urlopen(req, timeout=5)
        assert err.value.code == 401
