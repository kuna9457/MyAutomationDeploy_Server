"""
api/main.py
FastAPI entrypoint. Run from the `backend/` directory:

    cd backend
    uvicorn api.main:app --reload --port 8000

Every router below imports the existing top-level modules (engine, strategy,
broker_api, db_manager, risk_manager, config, ...) unchanged — this file and
the rest of api/ are the only new backend code (see frontend_migration_plan.md).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Make backend/'s top-level modules (engine.py, config.py, ...) importable
# regardless of the process's working directory or how uvicorn was invoked.
_BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

# FIRST, before anything can open a socket. Prefers IPv4 for outbound
# connections so the address a broker's static-IP allowlist sees is the one it
# was written for — an IPv6 source can never match an IPv4 allowlist entry
# (Upstox UDAPI1154). Must precede the imports below: api.auth pulls in
# user_manager, which connects to Mongo at import time, and a pool built
# before this point would keep its own resolution.
import net_config  # noqa: E402
net_config.apply()

from fastapi import Depends, FastAPI, HTTPException  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.security import OAuth2PasswordRequestForm  # noqa: E402

from api.auth import (CurrentUser, TokenResponse, authenticate,  # noqa: E402
                      create_access_token, get_current_user)
from api.routers import (account, admin_users, backtest, bot,  # noqa: E402
                         broker, config_router, risk, trades)

app = FastAPI(title="Trading Bot API")

_origins = os.getenv("CORS_ORIGINS", "http://localhost:5173").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in _origins],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/auth/login", response_model=TokenResponse)
def login(form: OAuth2PasswordRequestForm = Depends()):
    user = authenticate(form.username, form.password)
    if user is None:
        raise HTTPException(401, "Invalid username or password.")
    token = create_access_token(user)
    return TokenResponse(access_token=token, role=user.role, username=user.username,
                         user_id=user.user_id)


@app.get("/auth/me")
def me(user: CurrentUser = Depends(get_current_user)):
    return user


@app.get("/health")
def health():
    return {"ok": True}


app.include_router(bot.router)
app.include_router(trades.router)
app.include_router(broker.router)
app.include_router(risk.router)
app.include_router(backtest.router)
app.include_router(config_router.router)
app.include_router(admin_users.router)
app.include_router(account.router)
