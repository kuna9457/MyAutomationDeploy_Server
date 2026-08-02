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
    return CurrentUser(user_id=user_id, username=username, role=role,
                       display_name=user.get("display_name", ""))


def require_admin(user: CurrentUser = Depends(get_current_user)) -> CurrentUser:
    if user.role != "admin":
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Admin access required.")
    return user
