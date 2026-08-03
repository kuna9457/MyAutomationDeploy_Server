"""
mongo_client.py
One process-wide MongoClient, shared by every DBManager instance, by
user_manager, and by anything else that needs Mongo.

Why this exists
---------------
`MongoClient` is already an internally thread-safe connection *pool*; the
driver is explicitly designed to be created once per process and shared. The
codebase previously built a fresh one per `DBManager()` — at import in
`api/routers/trades.py` and `api/routers/admin_users.py`, again per running
engine, and again in `user_manager`. Each of those carried its own pool (the
pymongo default ceiling is 100 sockets) plus its own background topology
monitor threads. On a 1 GB server that is pure waste.

Behaviour is deliberately unchanged from the old per-instance connect:
  * A *successful* connection is cached and reused forever.
  * A *failed* connection is NOT cached, so the next caller retries exactly as
    a fresh `DBManager()` used to. That keeps the existing recovery path where
    an engine started after Mongo comes back can still reach it.
  * Any failure still returns None, and callers fall back to local JSON.
"""
from __future__ import annotations

import threading
from typing import Any, Optional

import config

_lock = threading.Lock()
_client: Optional[Any] = None


def get_client() -> Optional[Any]:
    """The shared MongoClient, or None when Mongo is unreachable.

    Connects on first call. Safe to call from any thread and as often as you
    like — after the first success it is just a dict lookup behind a lock.
    """
    global _client
    with _lock:
        if _client is not None:
            return _client
        try:
            from pymongo import MongoClient  # type: ignore
            client = MongoClient(
                config.MONGO_URI,
                serverSelectionTimeoutMS=1500,
                maxPoolSize=config.MONGO_MAX_POOL_SIZE,
                minPoolSize=0,           # don't hold sockets open while idle
                maxIdleTimeMS=60_000,    # reap idle sockets after a minute
            )
            client.admin.command("ping")   # force a real connection test
            _client = client
            print(f"[mongo] Connected to MongoDB "
                  f"(shared client, maxPoolSize={config.MONGO_MAX_POOL_SIZE}).")
        except Exception as exc:
            # Not cached — the next caller retries, matching the old behaviour.
            print(f"[mongo] MongoDB unavailable ({exc}); using local JSON.")
            return None
        return _client


def get_db() -> Optional[Any]:
    """The shared database handle, or None when Mongo is unreachable."""
    client = get_client()
    return None if client is None else client[config.MONGO_DB_NAME]
