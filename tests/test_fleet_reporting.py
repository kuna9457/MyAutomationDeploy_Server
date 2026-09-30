"""Seeing what the clients are making, when they trade on other servers.

The admin reads every client's trades from ONE database (each engine tags its
trades with its own user_id). That already worked for the admin's per-client
views; what the fleet adds, and what is pinned here:

  * the roll-up across clients is right — admin's own trades excluded, today vs
    all-time separated, open trades not counted as PnL — and cannot differ from
    the per-client numbers because it uses the same aggregate;
  * a node that is NOT writing to this database (fell back to a local file, or
    points at another cluster) is reported as such instead of looking idle;
  * the client views no longer call a client trading on a node "stopped".
"""
from __future__ import annotations

import time

import pandas as pd
import pytest

import config
import config_store
import fleet_broadcast
import fleet_registry
import hub_link
from fleet_protocol import db_fingerprint, db_problem
from fleet_reporting import client_pnl_report

TODAY = "2026-09-30"
YESTERDAY = "2026-09-29"


@pytest.fixture(autouse=True)
def mem_store(monkeypatch):
    store: dict[str, dict] = {}
    import json
    monkeypatch.setattr(config_store, "load",
                        lambda name, default=None: json.loads(
                            json.dumps(store.get(name, dict(default or {})))))
    monkeypatch.setattr(config_store, "save",
                        lambda name, data: store.__setitem__(
                            name, json.loads(json.dumps(data))))
    return store


def _trade(user, day, status="CLOSED", pnl=0.0, ticker="AAA"):
    return {"trade_id": f"{user}-{day}-{ticker}-{pnl}-{status}",
            "timestamp": f"{day}T10:00:00", "user_id": user, "status": status,
            "realized_pnl": pnl if status == "CLOSED" else None,
            "ticker": ticker}


def _df(rows):
    # Same shape DBManager.get_trades hands back: object dtype, None for nulls.
    return pd.DataFrame(rows).astype(object).where(lambda d: d.notna(), None)


CLIENTS = [{"username": "c1", "display_name": "Client One", "status": "active"},
           {"username": "c2", "display_name": "Client Two", "status": "active"},
           {"username": "c3", "display_name": "Client Three", "status": "active"}]


def _rows():
    return [
        _trade("c1", TODAY, pnl=100.0, ticker="AAA"),
        _trade("c1", TODAY, pnl=-40.0, ticker="BBB"),
        _trade("c1", TODAY, status="OPEN", ticker="CCC"),
        _trade("c1", YESTERDAY, pnl=50.0, ticker="AAA"),
        _trade("c2", TODAY, pnl=-30.0, ticker="AAA"),
        _trade("admin", TODAY, pnl=999.0, ticker="AAA"),        # not a client
    ]


def _live(u):
    return {"c1": {"node_name": "Mumbai", "running": True,
                   "environment": "Paper"}}.get(u)


# --------------------------------------------------------------------------- #
#  The roll-up
# --------------------------------------------------------------------------- #
def test_rollup_totals_per_client_and_excludes_the_admin():
    rep = client_pnl_report(_df(_rows()), CLIENTS, _live, TODAY)
    by = {r["username"]: r for r in rep["clients"]}

    c1 = by["c1"]
    assert c1["total_pnl"] == 110.0                  # 100 - 40 + 50, OPEN adds nothing
    assert c1["today_pnl"] == 60.0                   # yesterday's +50 is not today
    assert (c1["trades"], c1["closed"], c1["open"], c1["wins"]) == (4, 3, 1, 2)
    assert c1["win_rate"] == pytest.approx(66.67, abs=0.01)
    assert by["c2"]["total_pnl"] == -30.0 and by["c2"]["today_pnl"] == -30.0

    t = rep["totals"]
    assert t["total_pnl"] == 80.0                    # admin's +999 is NOT in here
    assert t["today_pnl"] == 30.0
    assert (t["clients"], t["trades"], t["closed"], t["open"]) == (3, 5, 4, 1)
    assert t["win_rate"] == 50.0                     # 2 wins of 4 closed


def test_a_client_with_no_trades_is_listed_with_zeros():
    rep = client_pnl_report(_df(_rows()), CLIENTS, _live, TODAY)
    c3 = next(r for r in rep["clients"] if r["username"] == "c3")
    assert c3["total_pnl"] == 0.0 and c3["trades"] == 0 and c3["win_rate"] == 0.0


def test_rows_are_ranked_by_total_pnl_and_carry_their_server():
    rep = client_pnl_report(_df(_rows()), CLIENTS, _live, TODAY)
    assert [r["username"] for r in rep["clients"]] == ["c1", "c3", "c2"]
    c1 = rep["clients"][0]
    assert c1["node"] == "Mumbai" and c1["running"] is True
    assert rep["clients"][1]["node"] is None and rep["clients"][1]["running"] is False


def test_daily_series_sums_clients_only_newest_first():
    rep = client_pnl_report(_df(_rows()), CLIENTS, _live, TODAY)
    assert [d["date"] for d in rep["daily"]] == [TODAY, YESTERDAY]
    today = rep["daily"][0]
    assert today["pnl"] == 30.0                      # 100 - 40 - 30, admin excluded
    assert today["clients"] == 2
    assert rep["daily"][1]["pnl"] == 50.0


def test_legacy_rows_without_a_user_id_belong_to_admin_not_a_client():
    legacy = {"trade_id": "old", "timestamp": f"{TODAY}T09:00:00",
              "status": "CLOSED", "realized_pnl": 500.0, "ticker": "AAA"}
    rep = client_pnl_report(_df(_rows() + [legacy]), CLIENTS, _live, TODAY)
    assert rep["totals"]["total_pnl"] == 80.0


@pytest.mark.parametrize("empty", [None, pd.DataFrame()])
def test_no_trades_at_all_is_zeros_not_a_crash(empty):
    rep = client_pnl_report(empty, CLIENTS, _live, TODAY)
    assert rep["totals"]["total_pnl"] == 0.0 and rep["daily"] == []
    assert len(rep["clients"]) == 3


def test_no_clients_is_an_empty_report():
    rep = client_pnl_report(_df(_rows()), [], _live, TODAY)
    assert rep["clients"] == [] and rep["daily"] == []
    assert rep["totals"]["clients"] == 0


def test_rollup_uses_the_same_aggregate_as_the_per_client_stats():
    """A client's total here must be exactly what their own stats panel
    computes (DBManager.analytics_summary over the same rows)."""
    from db_manager import DBManager
    df = _df(_rows())
    rep = client_pnl_report(df, CLIENTS, _live, TODAY)
    c1 = next(r for r in rep["clients"] if r["username"] == "c1")
    per_client = DBManager._perf_row(df[df["user_id"] == "c1"])
    assert c1["total_pnl"] == per_client["Realized PnL (₹)"]
    assert c1["win_rate"] == per_client["Win Rate %"]


def test_pnl_endpoint_wires_the_database_clients_and_fleet_together(monkeypatch):
    from api.routers import fleet as fleet_router
    from fastapi import HTTPException
    seen = {}

    class StubDB:
        def get_trades(self, env):
            seen["env"] = env.value
            return _df(_rows())

    monkeypatch.setattr(fleet_router, "_trades_db", lambda: StubDB())
    monkeypatch.setattr(fleet_router.user_manager, "list_users",
                        lambda role=None: CLIENTS if role == "client" else [])
    monkeypatch.setattr(fleet_router.hub, "client_live", _live)
    monkeypatch.setattr(fleet_router.DBManager, "today_key",
                        staticmethod(lambda: TODAY))

    out = fleet_router.fleet_pnl("Live")
    assert seen["env"] == "Live" and out["environment"] == "Live"
    assert out["totals"]["total_pnl"] == 80.0
    assert out["clients"][0]["node"] == "Mumbai"

    with pytest.raises(HTTPException) as err:
        fleet_router.fleet_pnl("Demo")
    assert err.value.status_code == 400


# --------------------------------------------------------------------------- #
#  Are this node's trades landing in MY database?
# --------------------------------------------------------------------------- #
def _fp(monkeypatch, uri, db="trading_bot"):
    monkeypatch.setattr(config, "MONGO_URI", uri)
    monkeypatch.setattr(config, "MONGO_DB_NAME", db)
    return db_fingerprint()


def test_fingerprint_ignores_credentials_and_host_order(monkeypatch):
    a = _fp(monkeypatch, "mongodb://alice:pw1@h1:27017,h2:27017/?replicaSet=rs")
    b = _fp(monkeypatch, "mongodb://bob:OTHER@h2:27017,h1:27017/trading_bot")
    assert a == b


def test_fingerprint_differs_by_host_and_by_database(monkeypatch):
    base = _fp(monkeypatch, "mongodb://u:p@cluster-a:27017")
    assert _fp(monkeypatch, "mongodb://u:p@cluster-b:27017") != base
    assert _fp(monkeypatch, "mongodb://u:p@cluster-a:27017", db="other") != base


def test_fingerprint_never_contains_a_secret_and_never_resolves_dns(monkeypatch):
    fp = _fp(monkeypatch,
             "mongodb+srv://user:S3CR3T@no-such-cluster.invalid/trading_bot")
    assert "S3CR3T" not in fp and len(fp) == 12   # returned without a DNS lookup


def test_a_default_localhost_node_does_not_match_a_remote_hub(monkeypatch):
    hub_fp = _fp(monkeypatch, "mongodb+srv://u:p@cluster0.example.net")
    node_fp = _fp(monkeypatch, "mongodb://localhost:27017/")
    assert "DIFFERENT" in db_problem("MongoDB", node_fp, hub_fp)


def test_db_problem_verdicts():
    assert db_problem("MongoDB", "abc", "abc") == ""
    assert "NOT writing to MongoDB" in db_problem("Local JSON", "abc", "abc")
    assert "DIFFERENT" in db_problem("MongoDB", "abc", "xyz")
    # A node that has not reported yet (or an older build) is unknown, not bad.
    assert db_problem("", "", "abc") == ""


# --------------------------------------------------------------------------- #
#  The hub reports it, per server
# --------------------------------------------------------------------------- #
class Conn:
    def __init__(self, node_id, name, report, connected=True):
        self.node = {"node_id": node_id, "name": name}
        self.node_id = node_id
        self.connected = connected
        self.report = report
        self.last_seen = time.time()
        self.version = "1/p1"

    def send(self, frame, droppable=False):
        return True


def _hub_with(monkeypatch, reports):
    """A FleetHub whose registry holds one node per report, all connected."""
    monkeypatch.setattr(fleet_broadcast.mongo_client, "is_connected", lambda: True)
    hub = fleet_broadcast.FleetHub()
    ids = []
    for name, rep, connected in reports:
        rec, _ = fleet_registry.create_node(name)
        hub.connections[rec["node_id"]] = Conn(rec["node_id"], name, rep, connected)
        ids.append(rec["node_id"])
    return hub, ids


def test_hub_flags_a_node_that_fell_back_to_local_json(monkeypatch):
    fp = db_fingerprint()
    hub, _ = _hub_with(monkeypatch, [
        ("Healthy", {"db_backend": "MongoDB", "db_fingerprint": fp}, True),
        ("Fell back", {"db_backend": "Local JSON", "db_fingerprint": fp}, True),
        ("Wrong DB", {"db_backend": "MongoDB", "db_fingerprint": "deadbeef0000"}, True),
        ("Offline", {"db_backend": "Local JSON", "db_fingerprint": "x"}, False),
    ])
    nodes = {n["name"]: n for n in hub.status()["nodes"]}
    assert nodes["Healthy"]["db_problem"] == ""
    assert "NOT writing to MongoDB" in nodes["Fell back"]["db_problem"]
    assert "DIFFERENT" in nodes["Wrong DB"]["db_problem"]
    assert nodes["Offline"]["db_problem"] == ""        # nothing to say while offline
    assert hub.status()["hub_db_backend"] == "MongoDB"


def test_client_live_finds_the_server_a_client_is_on(monkeypatch):
    hub, _ = _hub_with(monkeypatch, [
        ("Mumbai", {"clients": [{"username": "c1", "running": True,
                                 "environment": "Live", "broker": "Zerodha",
                                 "open": ["AAA"], "day_pnl": 42.5}]}, True),
        ("Delhi", {"clients": [{"username": "c2", "running": False}]}, True),
    ])
    live = hub.client_live("c1")
    assert live["node_name"] == "Mumbai" and live["running"] is True
    assert (live["environment"], live["broker"]) == ("Live", "Zerodha")
    assert live["open"] == ["AAA"] and live["day_pnl"] == 42.5
    assert hub.client_live("c2")["running"] is False
    assert hub.client_live("nobody") is None
    assert hub.node_of("c1") == "Mumbai"


def test_client_live_ignores_a_disconnected_node(monkeypatch):
    hub, _ = _hub_with(monkeypatch, [
        ("Gone", {"clients": [{"username": "c1", "running": True}]}, False)])
    assert hub.client_live("c1") is None


# --------------------------------------------------------------------------- #
#  The admin's client views stop calling node clients "stopped"
# --------------------------------------------------------------------------- #
class _Eng:
    class state:
        running = True

    class environment:
        value = "Paper"

    class broker:
        name = "Simulated"


class _StubHub:
    def __init__(self, live):
        self._live = live
        self.connections: dict = {}

    def client_live(self, username):
        return self._live


def test_overview_shows_a_node_client_as_running_on_its_server(monkeypatch):
    from api.routers import admin_users as au
    monkeypatch.setattr(au.engine_registry, "get_engine", lambda u: None)
    monkeypatch.setattr(au, "fleet_hub", _StubHub(
        {"node_name": "Mumbai", "running": True, "environment": "Live",
         "broker": "Zerodha"}))
    monkeypatch.setattr(au.user_manager, "list_users", lambda role=None: [
        {"user_id": "id1", "username": "c1", "display_name": "C One",
         "status": "active", "created_at": "", "email": ""}])
    monkeypatch.setattr(au.user_manager, "credential_summary",
                        lambda uid, b: {"has_token": False, "configured": False})
    monkeypatch.setattr(au, "_db", type("D", (), {
        "analytics_summary": staticmethod(lambda env, user_id=None: {"total_pnl": 12.5})})())
    row = au.clients_overview()[0]
    assert row["running"] is True and row["node"] == "Mumbai"
    assert (row["environment"], row["broker"]) == ("Live", "Zerodha")
    assert row["paper_total_pnl"] == 12.5


def test_a_local_engine_still_wins_over_the_fleet(monkeypatch):
    from api.routers import admin_users as au
    monkeypatch.setattr(au.engine_registry, "get_engine", lambda u: _Eng())
    monkeypatch.setattr(au, "fleet_hub", _StubHub({"node_name": "X", "running": False}))
    view = au._live_view("c1")
    assert view == {"running": True, "environment": "Paper",
                    "broker": "Simulated", "node": None}


def test_a_client_nobody_reports_is_stopped(monkeypatch):
    from api.routers import admin_users as au
    monkeypatch.setattr(au.engine_registry, "get_engine", lambda u: None)
    monkeypatch.setattr(au, "fleet_hub", _StubHub(None))
    assert au._live_view("c1") == {"running": False, "environment": None,
                                   "broker": None, "node": None}


def test_the_hub_never_starts_a_client_that_has_its_own_server(monkeypatch):
    """Shared database => the hub sees every client. A client with a server
    must only ever run there, from their own static IP."""
    from api.routers import bot
    from config import Environment
    fleet_registry.create_node("Mumbai", username="c1")
    monkeypatch.setattr(bot, "TradingEngine",
                        lambda *a, **k: pytest.fail("must not build an engine"))
    reason = bot._start_one_client(
        {"user_id": "id1", "username": "c1", "status": "active"},
        Environment.PAPER)
    assert "Mumbai" in reason


def test_a_client_with_no_server_is_left_to_the_normal_start_path(monkeypatch):
    from api.routers import bot
    from config import Environment
    monkeypatch.setattr(bot.admin_config, "active_client_mode", lambda: "")
    # Falls through to the existing checks untouched.
    assert bot._start_one_client(
        {"user_id": "id1", "username": "c1", "status": "active"},
        Environment.PAPER) == "no client mode configured"


# --------------------------------------------------------------------------- #
#  The node's report carries what the hub needs
# --------------------------------------------------------------------------- #
class _FakeState:
    lock = __import__("threading").RLock()

    def __init__(self, running=True):
        self._running = running

    def snapshot(self):
        return {"running": self._running, "broker_name": "Zerodha",
                "day_pnl": 25.0, "realized_pnl": 20.0, "unrealized_pnl": 5.0,
                "open_positions": {"AAA": {"side": "BUY", "quantity": 10,
                                           "entry_price": 100.0}}}


class _FakeEngine:
    def __init__(self, backend):
        self.state = _FakeState()
        self.db = type("DB", (), {"backend": backend})()

        class Env:
            value = "Live"
        self.environment = Env()


@pytest.mark.parametrize("backend", ["MongoDB", "Local JSON"])
def test_node_report_says_where_trades_are_really_going(monkeypatch, backend):
    from api import engine_registry
    monkeypatch.setattr(engine_registry, "get_engines",
                        lambda u: [_FakeEngine(backend)])
    link = hub_link.HubLink()
    link.assigned = ["c1"]
    rep = link.build_report()
    assert rep["db_backend"] == backend
    assert rep["db_fingerprint"] == db_fingerprint()
    c1 = rep["clients"][0]
    assert (c1["username"], c1["running"]) == ("c1", True)
    assert (c1["environment"], c1["broker"]) == ("Live", "Zerodha")
    assert c1["day_pnl"] == 25.0 and c1["open"] == ["AAA"]
    assert rep["positions"]["AAA"][0]["client"] == "c1"


def test_one_engine_off_mongo_makes_the_whole_node_report_off_mongo(monkeypatch):
    """A node with several clients is only as good as its worst one."""
    from api import engine_registry
    engines = {"c1": [_FakeEngine("MongoDB")], "c2": [_FakeEngine("Local JSON")]}
    monkeypatch.setattr(engine_registry, "get_engines", lambda u: engines[u])
    link = hub_link.HubLink()
    link.assigned = ["c1", "c2"]
    assert link.build_report()["db_backend"] == "Local JSON"


def test_a_node_reports_only_its_own_client(monkeypatch):
    """Every account in the shared database is visible to a node; reporting
    others would tell the hub this server trades clients it does not."""
    from api import engine_registry
    monkeypatch.setattr(engine_registry, "get_engines",
                        lambda u: [_FakeEngine("MongoDB")])
    link = hub_link.HubLink()
    link.assigned = ["c1"]
    assert [c["username"] for c in link.build_report()["clients"]] == ["c1"]


def test_an_idle_node_reports_the_connection_state_without_dialling(monkeypatch):
    from api import engine_registry
    monkeypatch.setattr(engine_registry, "get_engines", lambda u: [])
    monkeypatch.setattr(hub_link.mongo_client, "is_connected", lambda: False)
    monkeypatch.setattr(hub_link.mongo_client, "get_client",
                        lambda: pytest.fail("status must not open a connection"))
    assert hub_link.HubLink().build_report()["db_backend"] == "Local JSON"
