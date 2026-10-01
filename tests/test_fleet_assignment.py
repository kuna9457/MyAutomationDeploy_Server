"""One client, one server.

A client account and their server are created TOGETHER (POST /admin/users),
and from then on:

  * a server trades ONLY its own client — the shared database shows every node
    every client, and without the binding every node started every client
    (duplicate bots, orders from the wrong IP);
  * the hub never runs a client that has a server, and a node refuses a client
    that is not its own — whichever address they log in at;
  * a node holds the broadcast IN MEMORY and never writes the admin's shared
    settings (client_run is where every client-side reader now looks).
"""
from __future__ import annotations

import json
import time

import pytest
from fastapi import HTTPException

import admin_config
import client_run
import config
import config_store
import fleet_broadcast
import fleet_registry
import hub_link
from config import Environment
from fleet_protocol import BroadcastConfig

SYMBOLS = [i.symbol for i in config.ALL_INSTRUMENTS
           if i.segment == config.Segment.EQUITY][:3]


@pytest.fixture(autouse=True)
def mem_store(monkeypatch):
    store: dict[str, dict] = {}
    monkeypatch.setattr(config_store, "load",
                        lambda name, default=None: json.loads(
                            json.dumps(store.get(name, dict(default or {})))))
    monkeypatch.setattr(config_store, "save",
                        lambda name, data: store.__setitem__(
                            name, json.loads(json.dumps(data))))
    return store


def _cfg(**kw) -> BroadcastConfig:
    base = dict(environment="Paper", mode="Intraday", symbols=list(SYMBOLS))
    base.update(kw)
    return BroadcastConfig(**base)


def _session(clients=("c1",), active=True, cfg=None, symbol_settings=None):
    return {"t": "session", "active": active, "hub_ts": time.time(),
            "clients": list(clients),
            "config": (cfg or _cfg()).to_dict() if active else None,
            "symbol_settings": symbol_settings or {}, "feed_simulated": False}


@pytest.fixture
def worker(monkeypatch):
    """This process as a fleet NODE, whose link is `link`."""
    link = hub_link.HubLink()
    monkeypatch.setattr(hub_link, "is_worker", lambda: True)
    monkeypatch.setattr(hub_link, "link", lambda: link)
    return link


class _Recorder:
    def __init__(self):
        self.calls: list = []

    def __call__(self, *a, **k):
        self.calls.append((a, k))
        return 1


# --------------------------------------------------------------------------- #
#  Registry
# --------------------------------------------------------------------------- #
def test_a_server_is_bound_to_exactly_one_client():
    rec, _ = fleet_registry.create_node("Rahul", username="rahul")
    assert rec["username"] == "rahul"
    assert fleet_registry.node_for_client("rahul")["node_id"] == rec["node_id"]
    assert fleet_registry.clients_of(rec["node_id"]) == ["rahul"]
    with pytest.raises(ValueError, match="already has a server"):
        fleet_registry.create_node("Rahul again", username="rahul")


def test_an_unbound_server_trades_nobody():
    rec, _ = fleet_registry.create_node("Old server")
    assert fleet_registry.clients_of(rec["node_id"]) == []
    assert fleet_registry.node_for_client("") is None


def test_regenerating_the_secret_kills_the_old_one():
    rec, old = fleet_registry.create_node("S", username="c1")
    new = fleet_registry.regenerate_secret(rec["node_id"])
    assert new and new != old
    assert fleet_registry.authenticate(rec["node_id"], old) is None
    assert fleet_registry.authenticate(rec["node_id"], new) is not None
    assert fleet_registry.regenerate_secret("node-unknown") is None


# --------------------------------------------------------------------------- #
#  The hub tells each server whom it serves
# --------------------------------------------------------------------------- #
def test_each_server_is_told_only_its_own_client():
    a, _ = fleet_registry.create_node("A", username="c1")
    b, _ = fleet_registry.create_node("B", username="c2")
    legacy, _ = fleet_registry.create_node("Legacy")
    hub = fleet_broadcast.FleetHub()
    assert hub.session_frame(a["node_id"])["clients"] == ["c1"]
    assert hub.session_frame(b["node_id"])["clients"] == ["c2"]
    assert hub.session_frame(legacy["node_id"])["clients"] == []
    # Sent even with no broadcast running, so a node always knows whom it serves.
    assert hub.session_frame(a["node_id"])["active"] is False


# --------------------------------------------------------------------------- #
#  The node starts ONLY its own client
# --------------------------------------------------------------------------- #
def test_a_node_starts_only_its_assigned_client(monkeypatch):
    import user_manager
    from api.routers import bot
    monkeypatch.setattr(user_manager, "list_users", lambda role=None: [
        {"user_id": "1", "username": "c1", "status": "active"},
        {"user_id": "2", "username": "c2", "status": "active"},
        {"user_id": "3", "username": "c3", "status": "active"}])
    started = []
    monkeypatch.setattr(bot, "_start_one_client",
                        lambda acct, env: started.append(acct["username"]) or "")
    link = hub_link.HubLink()
    link._on_session(_session(clients=["c2"]))
    assert started == ["c2"]                       # never c1 or c3
    assert link.last_summary["started"] == ["c2"]


def test_a_node_with_no_client_starts_nobody_and_says_why(monkeypatch):
    import user_manager
    from api.routers import bot
    monkeypatch.setattr(user_manager, "list_users", lambda role=None: [
        {"user_id": "1", "username": "c1", "status": "active"}])
    monkeypatch.setattr(bot, "_start_one_client",
                        lambda acct, env: pytest.fail("must start nobody"))
    link = hub_link.HubLink()
    link._on_session(_session(clients=[]))
    assert "no client is assigned" in link.last_summary["skipped"][0]["reason"]


def test_a_missing_account_is_reported_not_crashed(monkeypatch):
    import user_manager
    monkeypatch.setattr(user_manager, "list_users", lambda role=None: [])
    link = hub_link.HubLink()
    link._on_session(_session(clients=["ghost"]))
    assert link.last_summary["skipped"] == [
        {"username": "ghost", "reason": "no such client account"}]


def test_a_client_taken_off_a_server_is_stopped_there(monkeypatch):
    link = hub_link.HubLink()
    link.assigned = ["c1"]
    stopped = []
    monkeypatch.setattr(link, "_stop_clients",
                        lambda why, only=None: stopped.append(only))
    link._on_session(_session(clients=[], active=False))
    assert stopped[0] == ["c1"]
    assert link.assigned == []


def test_disabling_stops_only_this_servers_client(monkeypatch):
    link = hub_link.HubLink()
    link.assigned = ["c1"]
    stopped = []
    monkeypatch.setattr(link, "_stop_clients",
                        lambda why, only=None: stopped.append((why, only)))
    link._on_control({"t": "control", "action": "stop"})
    assert stopped == [("your account was disabled", ["c1"])]


# --------------------------------------------------------------------------- #
#  A node never writes the admin's shared settings
# --------------------------------------------------------------------------- #
def test_client_run_on_a_node_reads_the_broadcast_from_memory(worker, monkeypatch,
                                                              mem_store):
    monkeypatch.setattr(worker, "_start_assigned",
                        lambda env: {"total": 1, "started": ["c1"], "skipped": []})
    admin_config.set_mode_config("Intraday", risk_reward=3.0, symbols=["ADMIN"])
    before = json.dumps(mem_store, sort_keys=True)

    worker._on_session(_session(cfg=_cfg(risk_reward=2.0, exit_style="partial_trail"),
                                symbol_settings={SYMBOLS[0]: {"risk_reward": 1.5}}))

    assert client_run.active_mode() == "Intraday"
    assert client_run.available_modes() == ["Intraday"]
    mc = client_run.mode_config("Intraday")
    assert (mc.risk_reward, mc.exit_style, mc.symbols) == (2.0, "partial_trail", SYMBOLS)
    assert client_run.rules_for("Intraday", SYMBOLS)[SYMBOLS[0]].risk_reward == 1.5
    # ...and the admin's own settings are exactly as they were.
    assert json.dumps(mem_store, sort_keys=True) == before
    assert admin_config.get_mode_config("Intraday").risk_reward == 3.0

    worker._on_session(_session(active=False))
    assert client_run.active_mode() == "" and client_run.available_modes() == []


def test_client_run_on_the_hub_is_exactly_admin_config(monkeypatch):
    monkeypatch.setattr(hub_link, "is_worker", lambda: False)
    admin_config.publish_run("Intraday", symbols=list(SYMBOLS), risk_reward=2.0)
    assert client_run.active_mode() == admin_config.active_client_mode() == "Intraday"
    assert client_run.mode_config("Intraday") == admin_config.get_mode_config("Intraday")
    assert client_run.available_modes() == admin_config.available_client_modes()


def test_client_modes_on_a_node_offer_the_running_broadcast(worker, monkeypatch):
    from api.routers import config_router
    monkeypatch.setattr(worker, "_start_assigned",
                        lambda env: {"total": 0, "started": [], "skipped": []})
    assert config_router.list_client_modes() == []      # nothing running yet
    worker._on_session(_session(cfg=_cfg(risk_reward=2.0)))
    modes = config_router.list_client_modes()
    assert [m["key"] for m in modes] == ["Intraday"]
    assert modes[0]["risk_reward"] == 2.0 and modes[0]["instrument_count"] == 3
    assert "strategy" not in json.dumps(modes)          # still withheld from clients


# --------------------------------------------------------------------------- #
#  The REAL start path — nothing stubbed between the session and the engine
# --------------------------------------------------------------------------- #
def test_a_session_really_starts_the_assigned_client_through_the_real_path(
        worker, monkeypatch):
    """The unit tests above replace _start_assigned, which is exactly how a
    live-only bug got through: _on_session marked the session active AFTER
    starting the client, so _start_one_client read 'no run' and refused with
    'no client mode configured'. This drives the real chain end to end and
    stops only at the broker/engine boundary."""
    import risk_manager
    import user_manager
    from api import engine_registry
    from api.routers import bot

    built = []

    class FakeEngine:
        def __init__(self, environment, mode, broker_choice, instruments, capital,
                     **kw):
            built.append({"env": environment, "mode": mode, "capital": capital,
                          "symbols": [i.symbol for i in instruments], **kw})
            self.state = type("S", (), {"running": False})()

        def start(self):
            self.state.running = True

    monkeypatch.setattr(bot, "TradingEngine", FakeEngine)
    monkeypatch.setattr(engine_registry, "get_engine", lambda u: None)
    monkeypatch.setattr(engine_registry, "set_engine", lambda u, e: None)
    monkeypatch.setattr(risk_manager, "get_limits",
                        lambda u: type("L", (), {"capital_allocated": 0.0})())
    monkeypatch.setattr(user_manager, "list_users", lambda role=None: [
        {"user_id": "id1", "username": "c1", "status": "active"},
        {"user_id": "id2", "username": "c2", "status": "active"}])

    worker._on_session(_session(
        clients=["c1"], cfg=_cfg(risk_reward=2.0, exit_style="partial_trail",
                                 min_score=6.0)))

    assert worker.last_summary["skipped"] == [], worker.last_summary
    assert worker.last_summary["started"] == ["c1"]
    assert len(built) == 1                         # c2 is not this server's client
    eng = built[0]
    assert eng["user_id"] == "c1" and eng["symbols"] == SYMBOLS
    assert eng["env"] == Environment.PAPER and eng["capital"] > 0
    # The run's settings reached the engine, from memory.
    assert (eng["risk_reward"], eng["exit_style"], eng["min_score"]) == (
        2.0, "partial_trail", 6.0)


# --------------------------------------------------------------------------- #
#  Whichever address a client logs in at, their bot runs only on their server
# --------------------------------------------------------------------------- #
def test_the_hub_refuses_to_run_a_client_that_has_a_server(monkeypatch):
    from api.auth import CurrentUser
    from api.routers import bot
    from api.schemas import StartBotRequest
    monkeypatch.setattr(hub_link, "is_worker", lambda: False)
    fleet_registry.create_node("Mumbai", username="c1")
    with pytest.raises(HTTPException) as err:
        bot.start_bot(StartBotRequest(environment="Paper", capital=100000),
                      CurrentUser(user_id="id1", username="c1", role="client"))
    assert err.value.status_code == 400
    assert "your own server" in err.value.detail


def test_a_node_refuses_a_client_that_is_not_its_own(worker):
    from api.auth import CurrentUser
    from api.routers import bot
    from api.schemas import StartBotRequest
    worker.assigned = ["c1"]
    with pytest.raises(HTTPException) as err:
        bot.start_bot(StartBotRequest(environment="Paper", capital=100000),
                      CurrentUser(user_id="id2", username="c2", role="client"))
    assert "isn't set up for your account" in err.value.detail
    assert bot._fleet_refusal("c1") == ""
    assert bot._fleet_refusal("c2") != ""


def test_a_client_without_a_server_on_a_plain_hub_is_unaffected(monkeypatch):
    from api.routers import bot
    monkeypatch.setattr(hub_link, "is_worker", lambda: False)
    assert bot._fleet_refusal("someone") == ""


# --------------------------------------------------------------------------- #
#  The admin side: a server is created WITH its client
# --------------------------------------------------------------------------- #
@pytest.fixture
def admin(monkeypatch):
    from api.routers import admin_users as au
    users: dict[str, dict] = {}

    def create_user(username, password, role="client", display_name="", email=""):
        if any(u["username"] == username for u in users.values()):
            raise ValueError(f"Username '{username}' already exists.")
        u = {"user_id": f"id-{username}", "username": username, "role": role,
             "status": "active", "display_name": display_name or username,
             "email": email, "password_hash": "x"}
        users[u["user_id"]] = u
        return u

    monkeypatch.setattr(au.user_manager, "create_user", create_user)
    monkeypatch.setattr(au.user_manager, "get_user_by_id", lambda uid: users.get(uid))
    return au, users


def test_creating_a_client_creates_their_server(admin):
    from api.schemas import CreateClientRequest
    au, _ = admin
    out = au.create_client(CreateClientRequest(
        username="rahul", password="secret1", display_name="Rahul",
        email="rahul@example.com"))
    server = out["server"]
    assert server["node_id"].startswith("node-") and server["secret"]
    assert server["name"] == "Rahul"
    assert fleet_registry.node_for_client("rahul")["node_id"] == server["node_id"]
    # The secret authenticates exactly that server, and is not stored in clear.
    assert fleet_registry.authenticate(server["node_id"], server["secret"])
    assert server["secret"] not in json.dumps(fleet_registry.list_nodes())
    assert "password_hash" not in out


def test_a_second_server_for_the_same_client_is_refused(admin):
    from api.schemas import CreateClientRequest
    au, _ = admin
    out = au.create_client(CreateClientRequest(
        username="c1", password="secret1", email="c1@example.com"))
    with pytest.raises(HTTPException) as err:
        au.create_client_server(out["user_id"])
    assert err.value.status_code == 400


def test_an_older_client_can_be_given_a_server(admin):
    au, users = admin
    users["id-old"] = {"user_id": "id-old", "username": "old", "role": "client",
                       "status": "active", "display_name": "Old"}
    server = au.create_client_server("id-old")
    assert server["secret"] and fleet_registry.node_for_client("old")


def test_regenerating_a_clients_secret_disconnects_the_old_session(admin, monkeypatch):
    from api.schemas import CreateClientRequest
    au, _ = admin
    dropped = _Recorder()
    monkeypatch.setattr(au.fleet_hub, "disconnect", dropped)
    out = au.create_client(CreateClientRequest(
        username="c1", password="secret1", email="c1@example.com"))
    old = out["server"]["secret"]
    new = au.regenerate_client_server_secret(out["user_id"])["secret"]
    node_id = out["server"]["node_id"]
    assert fleet_registry.authenticate(node_id, old) is None
    assert fleet_registry.authenticate(node_id, new) is not None
    assert dropped.calls[0][0] == (node_id,)


def test_disabling_a_client_stops_their_bot_on_their_server(admin, monkeypatch):
    from api.schemas import CreateClientRequest, SetStatusRequest
    au, users = admin
    out = au.create_client(CreateClientRequest(
        username="c1", password="secret1", email="c1@example.com"))
    monkeypatch.setattr(au.user_manager, "set_status",
                        lambda uid, s: {**users[uid], "status": s})
    monkeypatch.setattr(au.engine_registry, "stop_engine", lambda u: False)
    stopped = _Recorder()
    monkeypatch.setattr(au.fleet_hub, "stop_node_clients", stopped)
    au.set_client_status(out["user_id"], SetStatusRequest(status="disabled"))
    assert stopped.calls[0][0] == (out["server"]["node_id"],)


def test_the_clients_list_shows_each_clients_server(admin, monkeypatch):
    from api.schemas import CreateClientRequest
    au, users = admin
    out = au.create_client(CreateClientRequest(
        username="c1", password="secret1", email="c1@example.com"))
    node_id = out["server"]["node_id"]

    class Conn:
        connected = True

    class Hub:
        connections = {node_id: Conn()}

        def client_live(self, username):
            return None

    monkeypatch.setattr(au, "fleet_hub", Hub())
    monkeypatch.setattr(au.engine_registry, "get_engine", lambda u: None)
    monkeypatch.setattr(au.user_manager, "list_users",
                        lambda role=None: list(users.values()))
    monkeypatch.setattr(au.user_manager, "credential_summary",
                        lambda uid, b: {"has_token": False, "configured": False})
    monkeypatch.setattr(au, "_db", type("D", (), {
        "analytics_summary": staticmethod(lambda env, user_id=None: {"total_pnl": 0.0})})())
    row = au.clients_overview()[0]
    assert row["server"]["node_id"] == node_id
    assert row["server"]["connected"] is True


# --------------------------------------------------------------------------- #
#  The Broadcast tab no longer makes servers
# --------------------------------------------------------------------------- #
def test_broadcast_cannot_create_a_free_standing_server():
    from api.routers import fleet
    assert not hasattr(fleet, "create_node")
    assert not any(getattr(r, "path", "") == "/fleet/nodes"
                   and "POST" in getattr(r, "methods", set())
                   for r in fleet.router.routes)


def test_a_clients_server_cannot_be_deleted_from_broadcast():
    from api.routers import fleet
    owned, _ = fleet_registry.create_node("Mine", username="c1")
    with pytest.raises(HTTPException) as err:
        fleet.delete_node(owned["node_id"])
    assert "Clients tab" in err.value.detail
    assert fleet_registry.get_node(owned["node_id"]) is not None


def test_a_leftover_unbound_server_can_be_removed():
    from api.routers import fleet
    orphan, _ = fleet_registry.create_node("Leftover")
    assert fleet.delete_node(orphan["node_id"]) == {"ok": True}
    assert fleet_registry.get_node(orphan["node_id"]) is None
