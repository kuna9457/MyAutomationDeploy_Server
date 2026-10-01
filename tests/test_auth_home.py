"""GET /auth/home — which of several servers is a user's own.

One client UI fronts every client server, and on a shared database a login
succeeds on all of them; the UI keeps the one that answers home=True. Getting
this wrong lands a client on someone else's server, where Start is refused
(or, on the hub, where their bot would run from the wrong IP).
"""
from __future__ import annotations

import json

import pytest

import config_store
import fleet_registry
import hub_link
from api.auth import CurrentUser


@pytest.fixture(autouse=True)
def mem_store(monkeypatch):
    store: dict[str, dict] = {}
    monkeypatch.setattr(config_store, "load",
                        lambda name, default=None: json.loads(
                            json.dumps(store.get(name, dict(default or {})))))
    monkeypatch.setattr(config_store, "save",
                        lambda name, data: store.__setitem__(
                            name, json.loads(json.dumps(data))))


def _home(user):
    from api.main import auth_home
    return auth_home(user)["home"]


CLIENT = CurrentUser(user_id="1", username="rishav", role="client")
OTHER = CurrentUser(user_id="2", username="amit", role="client")
ADMIN = CurrentUser(user_id="0", username="admin", role="admin")


def test_a_node_is_home_only_for_its_own_client(monkeypatch):
    node, _ = fleet_registry.create_node("Rishav", username="rishav")
    monkeypatch.setattr(hub_link, "is_worker", lambda: True)
    monkeypatch.setenv("NODE_ID", node["node_id"])
    assert _home(CLIENT) is True
    assert _home(OTHER) is False
    assert _home(ADMIN) is False


def test_a_node_knows_its_client_without_a_hub_connection(monkeypatch):
    """Read from the shared registry, not the live link — a node that has not
    reached its hub must still route its own client correctly."""
    node, _ = fleet_registry.create_node("Rishav", username="rishav")
    monkeypatch.setattr(hub_link, "is_worker", lambda: True)
    monkeypatch.setenv("NODE_ID", node["node_id"])
    fresh = hub_link.HubLink()                        # never connected to a hub
    monkeypatch.setattr(hub_link, "link", lambda: fresh)
    assert fresh.assigned == []
    assert _home(CLIENT) is True


def test_the_hub_is_not_home_for_a_client_with_a_server(monkeypatch):
    fleet_registry.create_node("Rishav", username="rishav")
    monkeypatch.setattr(hub_link, "is_worker", lambda: False)
    assert _home(CLIENT) is False          # must use their own server's IP
    assert _home(OTHER) is True            # no server: the hub is their home
    assert _home(ADMIN) is True


def test_an_unconfigured_node_is_home_for_nobody(monkeypatch):
    monkeypatch.setattr(hub_link, "is_worker", lambda: True)
    monkeypatch.delenv("NODE_ID", raising=False)
    assert _home(CLIENT) is False
