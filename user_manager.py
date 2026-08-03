"""
user_manager.py
Accounts for the FastAPI layer (api/) — admin + client logins, replacing
Phase 1's single hardcoded ADMIN_USERNAME/ADMIN_PASSWORD. Mirrors
db_manager.py's own pattern: MongoDB when reachable, local JSON file
fallback otherwise, so this never needs Mongo to be usable.

One admin account is bootstrapped automatically from .env
(ADMIN_USERNAME/ADMIN_PASSWORD) the first time this module runs, so
existing Phase 1 deployments keep working with zero config changes —
logging in as that account is identical to before, just checked against a
stored (hashed) record instead of a raw env-var compare.
"""
from __future__ import annotations

import json
import os
import threading
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

import config
import credentials_vault
from security import hash_password, verify_password

_USERS_PATH = os.path.join(config.LOCAL_DB_DIR, "users.json")
_lock = threading.Lock()

#: Never leaves this module in a record handed to an API layer. Stripping is
#: centralised here (rather than at each call site) so a new endpoint cannot
#: forget it — `list_users` previously returned raw `broker_tokens`.
_SENSITIVE_FIELDS = ("password_hash", "broker_tokens", "broker_credentials")


def _public(user: Optional[dict]) -> Optional[dict]:
    """A user record safe to hand to a router. Credentials are reachable only
    through the explicit accessors below, which decrypt for one use."""
    if user is None:
        return None
    return {k: v for k, v in user.items() if k not in _SENSITIVE_FIELDS}

# Shares the one process-wide MongoClient with DBManager rather than opening a
# second pool of its own (see mongo_client.py). The fallback contract is
# unchanged: if Mongo is unreachable, _db stays None and every accessor below
# transparently uses the local JSON store.
_client = None
_db = None
try:
    import mongo_client as _mongo
    _client = _mongo.get_client()
    if _client is None:
        raise RuntimeError("no Mongo client available")
    _db = _client[config.MONGO_DB_NAME]
    _db["users"].create_index("username", unique=True)
except Exception as exc:
    print(f"[user_manager] MongoDB unavailable ({exc}); using local JSON.")
    _client = None
    _db = None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# -- local JSON fallback ------------------------------------------------------ #
def _read_local() -> list[dict]:
    if not os.path.exists(_USERS_PATH):
        return []
    try:
        with open(_USERS_PATH, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return []


def _write_local(users: list[dict]) -> None:
    os.makedirs(os.path.dirname(_USERS_PATH), exist_ok=True)
    with open(_USERS_PATH, "w", encoding="utf-8") as fh:
        json.dump(users, fh, indent=2)


# -- CRUD ---------------------------------------------------------------------- #
def create_user(username: str, password: str, role: str = "client",
                display_name: str = "") -> dict:
    if get_user(username) is not None:
        raise ValueError(f"Username '{username}' already exists.")
    doc = {
        "user_id": str(uuid.uuid4()),
        "username": username,
        "password_hash": hash_password(password),
        "role": role,
        "status": "active",
        "display_name": display_name or username,
        "created_at": _now(),
        "broker_tokens": {},
        "broker_credentials": {},
    }
    with _lock:
        if _db is not None:
            _db["users"].insert_one(dict(doc))
        else:
            users = _read_local()
            users.append(doc)
            _write_local(users)
    return _public(doc)


def get_user(username: str) -> Optional[dict]:
    with _lock:
        if _db is not None:
            doc = _db["users"].find_one({"username": username}, {"_id": 0})
            return doc
        for u in _read_local():
            if u["username"] == username:
                return u
    return None


def get_user_by_id(user_id: str) -> Optional[dict]:
    with _lock:
        if _db is not None:
            return _db["users"].find_one({"user_id": user_id}, {"_id": 0})
        for u in _read_local():
            if u["user_id"] == user_id:
                return u
    return None


def list_users(role: Optional[str] = None) -> list[dict]:
    """Router-facing: every sensitive field is stripped (_SENSITIVE_FIELDS).
    Use the credential accessors below if you need an actual secret."""
    with _lock:
        if _db is not None:
            query = {"role": role} if role else {}
            raw = list(_db["users"].find(query, {"_id": 0}))
        else:
            raw = [u for u in _read_local()
                   if role is None or u["role"] == role]
    return [_public(u) for u in raw]


def _update(user_id: str, patch: dict[str, Any]) -> Optional[dict]:
    with _lock:
        if _db is not None:
            _db["users"].update_one({"user_id": user_id}, {"$set": patch})
            return _db["users"].find_one({"user_id": user_id}, {"_id": 0})
        users = _read_local()
        for u in users:
            if u["user_id"] == user_id:
                u.update(patch)
                _write_local(users)
                return u
    return None


def set_status(user_id: str, status: str) -> Optional[dict]:
    return _public(_update(user_id, {"status": status}))


def set_password(user_id: str, new_password: str) -> Optional[dict]:
    return _public(_update(user_id, {"password_hash": hash_password(new_password)}))


# -- per-client broker credentials (encrypted) --------------------------------- #
# Shape:  broker_credentials[broker] = {
#     api_key_enc, api_secret_enc, access_token_enc,
#     token_issued_at, updated_at }
# Nothing here is ever returned by an API; see credential_summary().

def _creds(user: dict, broker: str) -> dict:
    return dict((user.get("broker_credentials") or {}).get(broker) or {})


def set_broker_credentials(user_id: str, broker: str,
                           api_key: str, api_secret: str) -> Optional[dict]:
    """Store a client's own broker app credentials, encrypted. Raises
    credentials_vault.VaultUnavailable when no key is configured — callers
    surface that rather than writing plaintext."""
    user = get_user_by_id(user_id)
    if user is None:
        return None
    all_creds = dict(user.get("broker_credentials") or {})
    entry = _creds(user, broker)
    entry["api_key_enc"] = credentials_vault.encrypt(api_key)
    entry["api_secret_enc"] = credentials_vault.encrypt(api_secret)
    entry["updated_at"] = _now()
    # Replacing the app credentials invalidates any token minted by the old
    # ones — drop it so the client is told to reconnect instead of hitting a
    # confusing broker-side rejection on their first order.
    entry.pop("access_token_enc", None)
    entry.pop("token_issued_at", None)
    all_creds[broker] = entry
    return _public(_update(user_id, {"broker_credentials": all_creds}))


def get_broker_credentials(user_id: str, broker: str) -> tuple[str, str]:
    """(api_key, api_secret) in plaintext, for one server-side use. Returns
    ("", "") when unset or undecryptable."""
    user = get_user_by_id(user_id)
    if user is None:
        return "", ""
    entry = _creds(user, broker)
    return (credentials_vault.decrypt(entry.get("api_key_enc", "")),
            credentials_vault.decrypt(entry.get("api_secret_enc", "")))


def clear_broker_credentials(user_id: str, broker: str) -> Optional[dict]:
    user = get_user_by_id(user_id)
    if user is None:
        return None
    all_creds = dict(user.get("broker_credentials") or {})
    all_creds.pop(broker, None)
    tokens = dict(user.get("broker_tokens") or {})
    tokens.pop(broker, None)          # also drop any legacy plaintext token
    return _public(_update(user_id, {"broker_credentials": all_creds,
                                     "broker_tokens": tokens}))


def credential_summary(user_id: str, broker: str) -> dict:
    """The ONLY credential shape that may leave the server: a masked key and
    booleans. The secret has no read path anywhere in this module."""
    user = get_user_by_id(user_id)
    if user is None:
        return {"configured": False, "api_key_masked": "", "updated_at": "",
                "has_token": False}
    entry = _creds(user, broker)
    api_key = credentials_vault.decrypt(entry.get("api_key_enc", ""))
    has_secret = bool(entry.get("api_secret_enc"))
    return {
        "configured": bool(api_key and has_secret),
        "api_key_masked": credentials_vault.mask(api_key),
        "updated_at": entry.get("updated_at", ""),
        "has_token": bool(get_broker_token(user_id, broker)),
        "token_issued_at": entry.get("token_issued_at", ""),
    }


def set_broker_token(user_id: str, broker: str, token: str) -> Optional[dict]:
    """Store the daily access token, encrypted, alongside that broker's app
    credentials. The legacy plaintext `broker_tokens[broker]` is cleared on
    write so a record never carries both."""
    user = get_user_by_id(user_id)
    if user is None:
        return None
    all_creds = dict(user.get("broker_credentials") or {})
    entry = _creds(user, broker)
    patch: dict[str, Any] = {}
    try:
        entry["access_token_enc"] = credentials_vault.encrypt(token)
        entry["token_issued_at"] = _now()
        all_creds[broker] = entry
        patch["broker_credentials"] = all_creds
        tokens = dict(user.get("broker_tokens") or {})
        if tokens.pop(broker, None) is not None:
            patch["broker_tokens"] = tokens
    except credentials_vault.VaultUnavailable:
        # No key configured: fall back to the pre-vault behaviour rather than
        # dropping the token on the floor, so an un-migrated deployment keeps
        # working exactly as it did. The startup warning already flagged this.
        tokens = dict(user.get("broker_tokens") or {})
        tokens[broker] = token
        patch["broker_tokens"] = tokens
    return _public(_update(user_id, patch))


def get_broker_token(user_id: str, broker: str) -> Optional[str]:
    """The client's daily access token. Reads the encrypted store first and
    falls back to the legacy plaintext field, so clients connected before
    this change keep trading until their next reconnect."""
    user = get_user_by_id(user_id)
    if user is None:
        return None
    entry = _creds(user, broker)
    token = credentials_vault.decrypt(entry.get("access_token_enc", ""))
    if token:
        return token
    return (user.get("broker_tokens") or {}).get(broker) or None


def authenticate(username: str, password: str) -> Optional[dict]:
    user = get_user(username)
    if user is None or user.get("status") != "active":
        return None
    if not verify_password(password, user["password_hash"]):
        return None
    return user


def ensure_bootstrap_admin() -> None:
    """Create the one admin account from .env on first run, if no admin
    exists yet. Safe to call on every startup — idempotent."""
    if list_users(role="admin"):
        return
    admin_username = os.getenv("ADMIN_USERNAME", "admin")
    admin_password = os.getenv("ADMIN_PASSWORD", "admin123")
    create_user(admin_username, admin_password, role="admin",
               display_name="Admin")
    print(f"[user_manager] Bootstrapped admin account '{admin_username}' "
          "from .env.")
