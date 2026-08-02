"""
api/routers/trades.py
Trade Logs & Analytics tab, as an API: db_manager.py's existing
get_trades/analytics_summary/daily_pnl/export_excel, unchanged, just
returned as JSON (or a file download) instead of an st.dataframe.

Every read is scoped to the CALLER's own username (Phase 2) — a client only
ever sees their own trades, and admin's personal Trade Log tab shows only
the admin account's own trades. Cross-client rollups are a separate,
explicitly admin-only endpoint (api/routers/admin_users.py's
/admin/clients-overview) rather than something this router could leak by
omitting a filter.
"""
from __future__ import annotations

import os

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse

from api.auth import CurrentUser, get_current_user
from config import Environment
from db_manager import DBManager

router = APIRouter(prefix="/trades", tags=["trades"])

_db = DBManager()


def _env(environment: str) -> Environment:
    try:
        return Environment(environment)
    except ValueError:
        raise HTTPException(400, "environment must be 'Paper' or 'Live'.")


@router.get("")
def list_trades(environment: str = "Paper", user: CurrentUser = Depends(get_current_user)):
    df = _db.get_trades(_env(environment), user_id=user.username)
    return df.to_dict("records") if not df.empty else []


@router.get("/analytics")
def analytics(environment: str = "Paper", user: CurrentUser = Depends(get_current_user)):
    return _db.analytics_summary(_env(environment), user_id=user.username)


@router.get("/daily-pnl")
def daily_pnl(environment: str = "Paper", user: CurrentUser = Depends(get_current_user)):
    df = _db.daily_pnl(_env(environment), user_id=user.username)
    return df.to_dict("records") if not df.empty else []


@router.get("/export")
def export_excel(environment: str = "Paper", user: CurrentUser = Depends(get_current_user)):
    path = _db.export_excel(_env(environment), user_id=user.username)
    if not path or not os.path.exists(path):
        raise HTTPException(500, "Export failed.")
    return FileResponse(
        path, filename=os.path.basename(path),
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
