"""
api/auth.py
JWT login for the React frontend, backed by user_manager.py's user store
(Mongo/local-JSON, same fallback pattern as db_manager.py). One admin
account is auto-bootstrapped from .env on first run — see
user_manager.ensure_bootstrap_admin() — so this replaced Phase 1's
hardcoded compare without changing how you log in.
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import Optional

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel

try:
    from dotenv import load_dotenv
    load_dotenv()   # same pattern as config.py — don't rely on import order
except Exception:
    pass

import user_manager

JWT_SECRET = os.getenv("JWT_SECRET", "dev-insecure-secret-change-me-in-.env")
JWT_ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 12 * 60

if JWT_SECRET == "dev-insecure-secret-change-me-in-.env":
    print("[api.auth] WARNING: JWT_SECRET not set in .env — using an insecure "
          "default. Set JWT_SECRET before exposing this server publicly.")

user_manager.ensure_bootstrap_admin()

_oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")

#: Shown when the ADMIN account reaches a client's server.
ADMIN_ON_CLIENT_SERVER = (
    "This is a client's server — the admin panel lives on the hub. Log in at "
    "your admin address (if one login page fronts several servers, the hub "
    "must be listed FIRST in VITE_API_BASE_URLS).")


def admin_refused_here(role: str) -> bool:
    """A fleet client node (NODE_MODE=worker) serves its ONE client and nobody
    else — never the admin. With a shared database the admin's password works
    on every server, so without this an admin panel pointed at a client's
    server logs in fine and quietly runs there: admin Start Bot on the
    client's machine, and broker panels asking for keys that (correctly) are
    not on it. The hub and any ordinary single server are unaffected."""
    if role != "admin":
        return False
    import hub_link
    return hub_link.is_worker()


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    role: str
    username: str
    user_id: str


class CurrentUser(BaseModel):
    user_id: str
    username: str
    role: str  # "admin" | "client"
    display_name: str = ""


def authenticate(username: str, password: str) -> Optional[CurrentUser]:
    user = user_manager.authenticate(username, password)
    if user is None:
        return None
    return CurrentUser(user_id=user["user_id"], username=user["username"],
                       role=user["role"], display_name=user.get("display_name", ""))


def create_access_token(user: CurrentUser) -> str:
    expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    payload = {"sub": user.username, "uid": user.user_id, "role": user.role,
               "exp": expire}
    # Session generation. Every password change bumps the stored counter
    # (user_manager.set_password), so tokens minted before it stop validating
    # in get_current_user — without this, changing a password would leave a
    # stolen session alive for its full 12-hour lifetime.
    record = user_manager.get_user_by_id(user.user_id)
    if record is not None:
        payload["tv"] = user_manager.token_version(record)
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def get_current_user(token: str = Depends(_oauth2_scheme)) -> CurrentUser:
    credentials_error = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except jwt.PyJWTError:
        raise credentials_error
    username, user_id, role = payload.get("sub"), payload.get("uid"), payload.get("role")
    if not username or not user_id or not role:
        raise credentials_error
    # Re-check status on every request (not baked into the token) so a
    # disabled client is locked out immediately, not just at their next login.
    user = user_manager.get_user_by_id(user_id)
    if user is None or user.get("status") != "active":
        raise credentials_error
    # A token issued before the last password change is dead. `tv` is absent
    # on tokens minted before this claim existed and defaults to 0, which
    # matches an untouched record — so shipping this does not log anyone out,
    # but the first password change after it invalidates every old session.
    if int(payload.get("tv", 0) or 0) != user_manager.token_version(user):
        raise credentials_error
    # An admin session already open on a client's server ends here too — 401,
    # so the UI drops it and returns to the login page instead of carrying on.
    if admin_refused_here(role):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, ADMIN_ON_CLIENT_SERVER,
                            headers={"WWW-Authenticate": "Bearer"})
    return CurrentUser(user_id=user_id, username=username, role=role,
                       display_name=user.get("display_name", ""))


def require_admin(user: CurrentUser = Depends(get_current_user)) -> CurrentUser:
    if user.role != "admin":
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Admin access required.")
    return user
