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
# Imported HERE, before any test patches hub_link.is_worker: api.main decides at
# import time whether to mount the hub's /fleet routes, and importing it for the
# first time inside a "worker" patch would build an app without them for every
# later test in the session.
from api import main as api_main  # noqa: F401
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


# --------------------------------------------------------------------------- #
#  A client's server never serves the admin
# --------------------------------------------------------------------------- #
class _Form:
    def __init__(self, username, password="pw"):
        self.username, self.password = username, password


def _login(monkeypatch, user, worker):
    from api import main
    monkeypatch.setattr(hub_link, "is_worker", lambda: worker)
    monkeypatch.setattr(main, "authenticate", lambda u, p: user)
    monkeypatch.setattr(main, "create_access_token", lambda u: "tok")
    return main.login(_Form(user.username))


def test_a_client_server_refuses_an_admin_login(monkeypatch):
    """Found in production: an admin panel pointed at a client's server logged
    in (shared database), ran there, and its broker panels asked for keys that
    are correctly absent on a client's machine."""
    from fastapi import HTTPException
    with pytest.raises(HTTPException) as err:
        _login(monkeypatch, ADMIN, worker=True)
    assert err.value.status_code == 403            # right password, wrong server
    assert "hub" in err.value.detail


def test_a_client_server_still_logs_in_its_client(monkeypatch):
    assert _login(monkeypatch, CLIENT, worker=True).access_token == "tok"


def test_the_hub_logs_in_the_admin(monkeypatch):
    assert _login(monkeypatch, ADMIN, worker=False).access_token == "tok"


def _session(monkeypatch, role, worker):
    """Validate a real signed token for `role` through get_current_user."""
    import jwt as pyjwt
    from api import auth
    monkeypatch.setattr(hub_link, "is_worker", lambda: worker)
    monkeypatch.setattr(auth.user_manager, "get_user_by_id",
                        lambda uid: {"user_id": uid, "status": "active"})
    monkeypatch.setattr(auth.user_manager, "token_version", lambda u: 0)
    token = pyjwt.encode({"sub": "someone", "uid": "u1", "role": role, "tv": 0},
                         auth.JWT_SECRET, algorithm=auth.JWT_ALGORITHM)
    return auth.get_current_user(token)


def test_an_admin_session_already_open_on_a_client_server_is_ended(monkeypatch):
    from fastapi import HTTPException
    with pytest.raises(HTTPException) as err:
        _session(monkeypatch, "admin", worker=True)
    assert err.value.status_code == 401            # the UI drops it and re-logs in
    assert "hub" in err.value.detail


def test_sessions_that_belong_are_untouched(monkeypatch):
    assert _session(monkeypatch, "client", worker=True).role == "client"
    assert _session(monkeypatch, "admin", worker=False).role == "admin"


def test_an_unconfigured_node_is_home_for_nobody(monkeypatch):
    monkeypatch.setattr(hub_link, "is_worker", lambda: True)
    monkeypatch.delenv("NODE_ID", raising=False)
    assert _home(CLIENT) is False
