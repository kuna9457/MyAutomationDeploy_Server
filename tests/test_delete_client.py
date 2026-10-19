"""Admin deleting a client account (DELETE /admin/users/{id})."""
from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import commission
import risk_manager
import user_manager
from api.auth import CurrentUser, require_admin
from api.routers import admin_users as routes


class FakeDB:
    def __init__(self):
        self.open_live = []
        self.resets = []

    def get_open_trades(self, env, user_id=None):
        return list(self.open_live)

    def reset_environment(self, env, user_id=None):
        assert user_id, "must never wipe a whole environment"
        self.resets.append((env.value, user_id))
        return {"trades_removed": 3}


@pytest.fixture
def world(monkeypatch):
    users = {"u1": {"user_id": "u1", "username": "tester", "role": "client",
                    "status": "disabled"},
             "a1": {"user_id": "a1", "username": "admin", "role": "admin",
                    "status": "active"}}
    deleted = []
    monkeypatch.setattr(user_manager, "get_user_by_id", lambda i: users.get(i))
    monkeypatch.setattr(user_manager, "delete_user",
                        lambda i: deleted.append(i) or users.pop(i, None) is not None)
    db = FakeDB()
    monkeypatch.setattr(routes, "_db", db)
    running = {"v": False}
    monkeypatch.setattr(routes, "_live_view", lambda u: {"running": running["v"]})
    monkeypatch.setattr(routes.engine_registry, "stop_engine", lambda u: False)
    nodes = {"tester": {"node_id": "n1"}}
    removed_nodes = []
    monkeypatch.setattr(routes.fleet_registry, "node_for_client", lambda u: nodes.get(u))
    monkeypatch.setattr(routes.fleet_registry, "delete_node",
                        lambda n: removed_nodes.append(n) or True)
    monkeypatch.setattr(routes.fleet_hub, "disconnect", lambda n: None)
    invoices = []
    monkeypatch.setattr(commission, "delete_client_invoices",
                        lambda u: invoices.append(u) or 2)
    monkeypatch.setattr(risk_manager, "forget", lambda u: None)

    app = FastAPI()
    app.include_router(routes.router)
    app.dependency_overrides[require_admin] = lambda: CurrentUser(
        user_id="a1", username="admin", role="admin")
    return {"http": TestClient(app), "users": users, "db": db, "deleted": deleted,
            "running": running, "nodes": removed_nodes, "invoices": invoices}


def test_deletes_account_server_trades_and_invoices(world):
    res = world["http"].delete("/admin/users/u1")
    assert res.status_code == 200, res.text
    assert res.json() == {"username": "tester", "server_removed": True,
                          "trades_removed": 6, "invoices_removed": 2}
    assert world["deleted"] == ["u1"]
    assert world["db"].resets == [("Paper", "tester"), ("Live", "tester")]
    assert world["nodes"] == ["n1"] and world["invoices"] == ["tester"]


def test_can_keep_the_history(world):
    res = world["http"].delete("/admin/users/u1?delete_data=false")
    assert res.status_code == 200
    assert world["db"].resets == [] and world["invoices"] == []
    assert world["deleted"] == ["u1"]


def test_must_be_disabled_first(world):
    world["users"]["u1"]["status"] = "active"
    assert world["http"].delete("/admin/users/u1").status_code == 400
    assert world["deleted"] == []


def test_refused_while_running_or_holding_a_live_position(world):
    world["running"]["v"] = True
    assert world["http"].delete("/admin/users/u1").status_code == 409
    world["running"]["v"] = False
    world["db"].open_live = [{"trade_id": "t"}]
    assert world["http"].delete("/admin/users/u1").status_code == 409
    assert world["deleted"] == [] and world["db"].resets == []


def test_an_admin_account_cannot_be_deleted(world):
    assert world["http"].delete("/admin/users/a1").status_code == 404
    assert world["http"].delete("/admin/users/nope").status_code == 404
    assert world["deleted"] == []


def test_user_manager_never_deletes_an_admin(monkeypatch, tmp_path):
    monkeypatch.setattr(user_manager, "_db", None)
    monkeypatch.setattr(user_manager, "_USERS_PATH", str(tmp_path / "users.json"))
    user_manager._write_local([
        {"user_id": "a1", "username": "admin", "role": "admin"},
        {"user_id": "c1", "username": "c", "role": "client"}])
    assert user_manager.delete_user("a1") is False
    assert user_manager.delete_user("c1") is True
    assert [u["user_id"] for u in user_manager._read_local()] == ["a1"]
