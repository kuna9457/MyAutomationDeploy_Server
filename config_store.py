"""
config_store.py
Durable key/value storage for the small CONFIGURATION documents the app keeps
outside the trade database: admin's client defaults, per-symbol settings,
saved Controls presets, watchlists and live risk limits.

Why this exists
---------------
Those five stores each wrote a JSON file under `config.LOCAL_DB_DIR`
(`backend/data/`) and nothing else — while users and trades went to MongoDB.
On a host with an ephemeral filesystem (Render, and any container without a
mounted disk) that asymmetry is a real bug, not a preference:

    every redeploy wipes backend/data/, so `admin_bot_config.json` vanishes
    while the client ACCOUNTS in Mongo survive. `available_client_modes()`
    then returns nothing, `/config/client-modes` returns [], and the client's
    Start Bot button goes disabled with "No trading modes have been set up for
    you yet" — even though admin had configured it and nothing visibly changed.

So these documents now persist exactly the way users and trades already do:
MongoDB when it is reachable, local JSON when it is not.

Behaviour
---------
  * `load()` prefers Mongo; if Mongo has no document yet but a local file
    exists, that file is MIGRATED up on first read. An existing deployment
    therefore keeps its current configuration without anyone re-entering it.
  * `save()` writes to Mongo when available AND always mirrors to the local
    file, so a Mongo outage degrades to the old behaviour rather than losing
    the write, and a local-only setup is unchanged.
  * Every failure is non-fatal: storage problems must never stop the bot from
    starting. A read that fails everywhere returns the caller's default.

Callers hand over plain JSON-serialisable dicts and get plain dicts back —
this module knows nothing about what any of them mean.
"""
from __future__ import annotations

import json
import os
import threading
from typing import Any, Optional

import config
import mongo_client

#: One Mongo collection holds every config document, keyed by `name`. These
#: are a handful of small singleton documents, so a collection each would add
#: indexes and connections for no benefit.
COLLECTION = "app_config"

_lock = threading.Lock()


def _path(name: str) -> str:
    return os.path.join(config.LOCAL_DB_DIR, f"{name}.json")


def _collection() -> Optional[Any]:
    db = mongo_client.get_db()
    return None if db is None else db[COLLECTION]


# --------------------------------------------------------------------------- #
#  Local JSON — the fallback, and always kept in step as a mirror
# --------------------------------------------------------------------------- #
def _read_local(name: str) -> Optional[dict]:
    try:
        with open(_path(name), "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def _write_local(name: str, data: dict) -> None:
    try:
        os.makedirs(config.LOCAL_DB_DIR, exist_ok=True)
        with open(_path(name), "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)
    except Exception as exc:
        print(f"[config_store] local write of {name} failed ({exc}).")


# --------------------------------------------------------------------------- #
#  Public API
# --------------------------------------------------------------------------- #
def load(name: str, default: Optional[dict] = None) -> dict:
    """The stored document, or `default` ({} when omitted).

    Mongo wins when it holds the document. When it does not but a local file
    does, that file is written up to Mongo first — the one-time migration that
    carries an existing deployment's configuration across without anyone
    re-entering it.
    """
    fallback = dict(default or {})
    col = _collection()
    if col is not None:
        try:
            doc = col.find_one({"_id": name})
            if doc is not None:
                return doc.get("data", fallback)
            # Mongo reachable but empty for this key: adopt any local file.
            local = _read_local(name)
            if local is not None:
                try:
                    col.update_one({"_id": name}, {"$set": {"data": local}},
                                   upsert=True)
                    print(f"[config_store] migrated {name} from local JSON "
                          f"into MongoDB.")
                except Exception as exc:
                    print(f"[config_store] migration of {name} failed ({exc}).")
                return local
            return fallback
        except Exception as exc:
            print(f"[config_store] Mongo read of {name} failed ({exc}); "
                  f"using local JSON.")

    local = _read_local(name)
    return fallback if local is None else local


def save(name: str, data: dict) -> None:
    """Persist a document. Writes to Mongo when reachable and ALWAYS mirrors
    to the local file, so neither store alone is a single point of failure and
    a Mongo-less setup behaves exactly as it did before this module existed."""
    with _lock:
        col = _collection()
        if col is not None:
            try:
                col.update_one({"_id": name}, {"$set": {"data": data}},
                               upsert=True)
            except Exception as exc:
                print(f"[config_store] Mongo write of {name} failed ({exc}); "
                      f"kept local JSON only.")
        _write_local(name, data)


def backend() -> str:
    """"MongoDB" or "Local JSON" — for surfacing where config actually lives,
    which is the difference between settings that survive a redeploy and
    settings that do not."""
    return "MongoDB" if _collection() is not None else "Local JSON"
