"""
api/routers/fleet.py
The admin's BROADCAST surface, and the socket client nodes dial into.

Deliberately its own router and its own state (fleet_broadcast.hub): nothing
here is reachable from /bot, reads engine_registry or shares a Start/Stop with
the admin's own bot. Starting or stopping a broadcast does not touch it, and
the reverse.

    admin  ──HTTP──>  /fleet/nodes, /fleet/broadcast/*   (role=admin only)
    node   ──WS────>  /fleet/ws                          (node secret, not a JWT)

A node authenticates with the id + secret the admin generated for it, sent as
`Authorization: Bearer <node_id>:<secret>` in the handshake. That is only safe
over TLS, so the hub must be served over https/wss in production.
"""
from __future__ import annotations

import asyncio
import json
from typing import Optional

import fleet_registry
import user_manager
from api.auth import CurrentUser, require_admin
from config import Environment
from db_manager import DBManager
from fastapi import (APIRouter, Depends, HTTPException, WebSocket,
                     WebSocketDisconnect)
from fastapi.concurrency import run_in_threadpool
from fleet_broadcast import NodeConnection, hub
from fleet_protocol import BroadcastConfig
from fleet_reporting import client_pnl_report
from pydantic import BaseModel

router = APIRouter(prefix="/fleet", tags=["fleet"])

_db: Optional[DBManager] = None


def _trades_db() -> DBManager:
    """One long-lived DBManager, created on first use rather than at import so
    merely importing this router never opens a database connection."""
    global _db
    if _db is None:
        _db = DBManager()
    return _db

#: A node never has a reason to send more than a report; refuse anything large
#: rather than parse it.
_MAX_FRAME_BYTES = 256 * 1024


class NodeEnabled(BaseModel):
    enabled: bool


class BroadcastStart(BaseModel):
    """The same choices the Controls panel offers for the admin's own run."""
    environment: str = "Paper"
    mode: str = "Intraday"
    strategy_key: str = ""
    symbols: list[str] = []
    mcx_lots: dict[str, int] = {}
    risk_reward: float = 0.0
    min_score: float = 0.0
    square_off_time: str = ""
    square_off_enabled: bool = True
    exit_style: str = "strategy"
    trail_atr_mult: float = 0.0
    max_stop_pct: float = 0.0
    min_stop_pct: float = 0.0
    #: Live tells real client accounts to place real orders. The UI asks the
    #: admin to confirm; the API refuses without it so a stray call cannot.
    confirm_live: bool = False


# --------------------------------------------------------------------------- #
#  Nodes (admin)
# --------------------------------------------------------------------------- #
@router.get("/nodes")
def list_nodes(_: CurrentUser = Depends(require_admin)):
    return hub.status()["nodes"]


# There is deliberately no "create a server" here. One client, one server: a
# server is created WITH its client (POST /admin/users), so none can exist that
# trades nobody, and no client can end up with two.


@router.post("/nodes/{node_id}/enabled")
def set_node_enabled(node_id: str, req: NodeEnabled,
                     _: CurrentUser = Depends(require_admin)):
    if not fleet_registry.set_enabled(node_id, req.enabled):
        raise HTTPException(404, "Unknown node.")
    if not req.enabled:
        hub.disconnect(node_id)
    return {"ok": True}


@router.delete("/nodes/{node_id}")
def delete_node(node_id: str, _: CurrentUser = Depends(require_admin)):
    """Remove a server that belongs to NO client (one made before servers were
    tied to clients). A client's own server is managed from the Clients tab."""
    owner = fleet_registry.clients_of(node_id)
    if owner:
        raise HTTPException(
            400, f"This server belongs to client '{owner[0]}' — manage it from "
                 f"the Clients tab.")
    if not fleet_registry.delete_node(node_id):
        raise HTTPException(404, "Unknown node.")
    hub.disconnect(node_id)
    return {"ok": True}


# --------------------------------------------------------------------------- #
#  Broadcast (admin)
# --------------------------------------------------------------------------- #
@router.post("/broadcast/start")
def start_broadcast(req: BroadcastStart, _: CurrentUser = Depends(require_admin)):
    if req.environment == "Live" and not req.confirm_live:
        raise HTTPException(
            400, "Live broadcast needs confirm_live=true — it tells every "
                 "connected client account to place real orders.")
    cfg = BroadcastConfig.from_dict(req.model_dump(exclude={"confirm_live"}))
    try:
        hub.start(cfg)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    except RuntimeError as exc:
        raise HTTPException(409, str(exc))
    return {"ok": True, **hub.status()}


@router.post("/broadcast/stop")
def stop_broadcast(_: CurrentUser = Depends(require_admin)):
    """Ends the broadcast for EVERY node. Nodes stop their client bots; a
    position already open is left to its broker-side stop/target (use Flatten to
    close it)."""
    hub.stop()
    return {"ok": True, **hub.status()}


@router.get("/broadcast/status")
def broadcast_status(_: CurrentUser = Depends(require_admin)):
    return hub.status()


@router.post("/broadcast/flatten")
def flatten(node_id: Optional[str] = None, _: CurrentUser = Depends(require_admin)):
    """Ask one node (or every node) to close all open positions now."""
    return {"ok": True, "asked": hub.flatten(node_id)}


@router.post("/broadcast/resync")
def resync(node_id: Optional[str] = None, _: CurrentUser = Depends(require_admin)):
    """Ask one node (or every node) to start its clients on the running
    broadcast — e.g. after a client has connected their broker."""
    return {"ok": True, "asked": hub.resync(node_id)}


# --------------------------------------------------------------------------- #
#  How all the clients are doing (admin)
# --------------------------------------------------------------------------- #
@router.get("/pnl")
def fleet_pnl(environment: str = "Paper", _: CurrentUser = Depends(require_admin)):
    """Total / today's PnL, win rate and open trades for every client, plus a
    per-day series across all of them.

    Read from the shared trade database, scoped per client exactly as the
    per-client drill-down is (`/admin/clients/{username}/stats`) — so the two
    cannot disagree. It therefore shows a node's trades only when that node
    writes to THIS database; each server's row in /fleet/broadcast/status says
    when it does not (`db_problem`).
    """
    try:
        env = Environment(environment)
    except ValueError:
        raise HTTPException(400, "environment must be 'Paper' or 'Live'.")
    clients = user_manager.list_users(role="client")
    report = client_pnl_report(_trades_db().get_trades(env), clients,
                               hub.client_live, DBManager.today_key())
    return {"environment": env.value, **report}


# --------------------------------------------------------------------------- #
#  The node socket
# --------------------------------------------------------------------------- #
def _node_from_handshake(ws: WebSocket) -> Optional[dict]:
    header = ws.headers.get("authorization", "")
    if not header.lower().startswith("bearer "):
        return None
    node_id, _, secret = header[7:].strip().partition(":")
    if not node_id or not secret:
        return None
    return fleet_registry.authenticate(node_id, secret)


@router.websocket("/ws")
async def node_socket(ws: WebSocket):
    node = await run_in_threadpool(_node_from_handshake, ws)
    if node is None:
        # Closed before accept: the client sees a plain 403, and learns nothing
        # about whether the id exists.
        await ws.close(code=4401)
        return
    await ws.accept()
    conn = NodeConnection(node, ws, asyncio.get_running_loop())
    await run_in_threadpool(hub.on_connect, conn)
    try:
        while True:
            text = await ws.receive_text()
            if len(text) > _MAX_FRAME_BYTES:
                continue
            try:
                msg = json.loads(text)
            except ValueError:
                continue
            if isinstance(msg, dict):
                hub.on_message(conn, msg)
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        await run_in_threadpool(hub.on_disconnect, conn)
