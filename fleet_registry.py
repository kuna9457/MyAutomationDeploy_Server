"""
fleet_registry.py
The hub's list of client nodes: who may connect, and whether they currently may.

One document in config_store (Mongo when reachable, local JSON otherwise — the
same durability as admin_config and risk limits), so a redeploy does not
orphan the fleet.

A node's secret is generated HERE, shown to the admin exactly once, and only
its SHA-256 is stored. SHA-256 is enough because the secret is 256 bits of
randomness (not a human password), and it is sent over TLS in the WebSocket
handshake — there is nothing to brute-force and nothing useful in a leaked
database row. Revoking a node is deleting its row; its next handshake fails.

Nothing here touches trading. It answers one question: "is this caller one of
my servers?"
"""
from __future__ import annotations

import hashlib
import hmac
import secrets
import threading
from datetime import datetime, timezone
from typing import Optional

import config_store

_KEY = "fleet_nodes"
_lock = threading.Lock()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _hash(secret: str) -> str:
    return hashlib.sha256((secret or "").encode("utf-8")).hexdigest()


def _load() -> dict:
    return (config_store.load(_KEY) or {}).get("nodes", {}) or {}


def _save(nodes: dict) -> None:
    config_store.save(_KEY, {"nodes": nodes})


def _public(rec: dict) -> dict:
    """A record without anything secret — safe to hand to a router."""
    return {k: v for k, v in rec.items() if k != "secret_hash"}


def create_node(name: str, username: str = "") -> tuple[dict, str]:
    """Register a node. Returns (public record, plaintext secret). The secret
    is not recoverable afterwards — regenerate_secret() issues a new one.

    `username` binds the server to ONE client account: one client, one server.
    A node trades ONLY the client it is bound to — without that binding every
    node sees every client in the shared database and starts them all, i.e.
    duplicate bots on the wrong IPs. Raises ValueError if that client already
    has a server. A node with no username (made before this binding existed)
    trades nobody."""
    name = (name or "").strip()[:60] or "Unnamed node"
    username = (username or "").strip()
    secret = secrets.token_urlsafe(32)
    with _lock:
        nodes = _load()
        if username and any(r.get("username") == username for r in nodes.values()):
            raise ValueError(f"Client '{username}' already has a server.")
        node_id = f"node-{secrets.token_hex(4)}"
        while node_id in nodes:
            node_id = f"node-{secrets.token_hex(4)}"
        rec = {"node_id": node_id, "name": name, "username": username,
               "enabled": True, "secret_hash": _hash(secret),
               "created_at": _now(), "last_connected_at": "", "last_version": ""}
        nodes[node_id] = rec
        _save(nodes)
    return _public(rec), secret


def node_for_client(username: str) -> Optional[dict]:
    """The server a client is bound to, or None."""
    if not username:
        return None
    with _lock:
        for rec in _load().values():
            if rec.get("username") == username:
                return _public(rec)
    return None


def clients_of(node_id: str) -> list[str]:
    """The client account(s) this node may trade — at most one today."""
    with _lock:
        rec = _load().get(node_id or "")
    name = (rec or {}).get("username") or ""
    return [name] if name else []


def regenerate_secret(node_id: str) -> Optional[str]:
    """Issue a new secret for an existing server (the old one stops working
    immediately). For a secret that was lost — it is only ever shown once.
    Returns the new plaintext secret, or None for an unknown node."""
    secret = secrets.token_urlsafe(32)
    with _lock:
        nodes = _load()
        if node_id not in nodes:
            return None
        nodes[node_id]["secret_hash"] = _hash(secret)
        _save(nodes)
    return secret


def list_nodes() -> list[dict]:
    with _lock:
        return [_public(r) for r in _load().values()]


def get_node(node_id: str) -> Optional[dict]:
    with _lock:
        rec = _load().get(node_id)
    return _public(rec) if rec else None


def authenticate(node_id: str, secret: str) -> Optional[dict]:
    """The node's public record if (node_id, secret) is valid AND the node is
    enabled, else None. Constant-time on the secret; the same None for an
    unknown id, a wrong secret and a disabled node, so a caller cannot probe
    which node ids exist."""
    with _lock:
        rec = _load().get(node_id or "")
    candidate = _hash(secret)
    stored = (rec or {}).get("secret_hash") or _hash(secrets.token_hex(16))
    ok = hmac.compare_digest(candidate, stored)
    if rec is None or not ok or not rec.get("enabled", True):
        return None
    return _public(rec)


def set_enabled(node_id: str, enabled: bool) -> bool:
    with _lock:
        nodes = _load()
        if node_id not in nodes:
            return False
        nodes[node_id]["enabled"] = bool(enabled)
        _save(nodes)
    return True


def delete_node(node_id: str) -> bool:
    with _lock:
        nodes = _load()
        if node_id not in nodes:
            return False
        nodes.pop(node_id)
        _save(nodes)
    return True


def record_connect(node_id: str, version: str) -> None:
    """Stamp a successful handshake. Written on connect only — never per
    frame — so a busy fleet is not a write load."""
    with _lock:
        nodes = _load()
        if node_id in nodes:
            nodes[node_id]["last_connected_at"] = _now()
            nodes[node_id]["last_version"] = str(version)[:40]
            _save(nodes)
