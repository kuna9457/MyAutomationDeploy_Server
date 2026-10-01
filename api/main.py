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

from api.auth import (ADMIN_ON_CLIENT_SERVER, CurrentUser,  # noqa: E402
                      TokenResponse, admin_refused_here, authenticate,
                      create_access_token, get_current_user)
from api.routers import (account, admin_users, advanced_backtest,  # noqa: E402
                         auditor, backtest, bot, broker, bulk_backtest,
                         chart, config_router, crudeoil_pipeline_api,
                         fleet, risk, trades)
import hub_link  # noqa: E402
from contextlib import asynccontextmanager  # noqa: E402


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    """Fleet startup. On a client NODE (NODE_MODE=worker) dial the hub; on the
    HUB resume a broadcast the admin left running across a restart. Both are
    no-ops on an ordinary single-server deployment, so nothing changes there.
    Resuming opens a market-data socket, so it runs off the event loop."""
    import threading
    if hub_link.start_if_worker():
        pass
    else:
        from fleet_broadcast import hub
        threading.Thread(target=hub.resume_if_active, daemon=True,
                         name="fleet-resume").start()
    yield


app = FastAPI(title="Trading Bot API", lifespan=_lifespan)

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
    if admin_refused_here(user.role):
        # 403, not 401: the password WAS right — this is just the wrong server.
        raise HTTPException(403, ADMIN_ON_CLIENT_SERVER)
    token = create_access_token(user)
    return TokenResponse(access_token=token, role=user.role, username=user.username,
                         user_id=user.user_id)


@app.get("/auth/me")
def me(user: CurrentUser = Depends(get_current_user)):
    return user


@app.get("/auth/home")
def auth_home(user: CurrentUser = Depends(get_current_user)):
    """Is THIS server the caller's home — where their bot is allowed to run?

    One client UI (e.g. algo.welthwest.com) can front several servers, and
    with a shared database a client's login succeeds on ALL of them. The UI
    logs in everywhere, asks each server this, and keeps the one that says yes.

    Answered from the shared server registry, not the live hub link, so a
    server that has not (yet) reached its hub still knows whom it serves:
      * client node: home only for the client it is bound to;
      * hub / single server: home for the admin, and for any client WITHOUT
        a server of their own (a client with one must use it — its static IP).
    """
    import fleet_registry
    if hub_link.is_worker():
        node_id = (os.getenv("NODE_ID", "") or "").strip()
        return {"home": user.role == "client"
                and user.username in fleet_registry.clients_of(node_id)}
    if user.role != "client":
        return {"home": True}
    return {"home": fleet_registry.node_for_client(user.username) is None}


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
# READ-ONLY LLM review of past trades. Imports nothing from the trading path and
# cannot act — see AI_AUDITOR_PLAN.md.
app.include_router(auditor.router)
# Trade replay: candles a trade was taken on, with the trade drawn over them.
app.include_router(chart.router)
# Combination search: which symbol x pattern actually works. Read-only.
app.include_router(advanced_backtest.router)
# MCX crude pipeline (crudeoil_pipeline/) — an ISOLATED, greenfield package.
# This adapter is the only thing that bridges it to the app; the package
# imports nothing from here, which is what keeps it independently testable
# (see crudeoil_pipeline/CLAUDE.md RULE #1 and tests/test_isolation.py).
app.include_router(crudeoil_pipeline_api.router)
# Multi-axis optimizer: separate funnel search (bulk_backtest/).
app.include_router(bulk_backtest.router)
# Fleet BROADCAST (hub only): its own runner and state, sharing nothing with the
# admin's own bot but the market-data socket. A client node never serves this —
# it only dials OUT to a hub — so the routes are not even mounted there.
if not hub_link.is_worker():
    app.include_router(fleet.router)
