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
import config
from config import Environment
from db_manager import DBManager

router = APIRouter(prefix="/trades", tags=["trades"])

_db = DBManager()


def _env(environment: str) -> Environment:
    try:
        return Environment(environment)
    except ValueError:
        raise HTTPException(400, "environment must be 'Paper' or 'Live'.")


def _category(category: str = "") -> str:
    """Validate an optional category filter. "" / "All" means no filter —
    rejecting an unknown name rather than silently returning everything, since
    a typo that quietly widens the scope is the dangerous direction."""
    name = (category or "").strip()
    if not name or name.lower() == "all":
        return ""
    if name not in config.ALL_CATEGORIES:
        raise HTTPException(
            400, f"Unknown category {name!r}. Use one of "
                 f"{', '.join(config.ALL_CATEGORIES)}, or leave blank for all.")
    return name


@router.get("/categories")
def categories():
    """The asset classes trades are bucketed into. Crypto is listed before any
    crypto instrument exists, so the UI has a stable set to render."""
    return {"categories": list(config.ALL_CATEGORIES)}


@router.get("")
def list_trades(environment: str = "Paper", category: str = "",
                user: CurrentUser = Depends(get_current_user)):
    df = _db.get_trades(_env(environment), user_id=user.username,
                        category=_category(category) or None)
    return df.to_dict("records") if not df.empty else []


@router.get("/analytics")
def analytics(environment: str = "Paper", category: str = "",
              user: CurrentUser = Depends(get_current_user)):
    return _db.analytics_summary(_env(environment), user_id=user.username,
                                 category=_category(category) or None)


@router.get("/by-category")
def by_category(environment: str = "Paper",
                user: CurrentUser = Depends(get_current_user)):
    """One P&L row per asset class — the category-wise view. Every category
    appears, including ones with no trades yet (as zeroes), so an empty bucket
    reads as "nothing traded" rather than "something is missing"."""
    return _db.category_summary(_env(environment), user_id=user.username)


@router.get("/daily-pnl")
def daily_pnl(environment: str = "Paper", category: str = "",
              user: CurrentUser = Depends(get_current_user)):
    df = _db.daily_pnl(_env(environment), user_id=user.username,
                       category=_category(category) or None)
    return df.to_dict("records") if not df.empty else []


@router.get("/by-strategy")
def by_strategy(environment: str = "Paper", category: str = "",
                user: CurrentUser = Depends(get_current_user)):
    """All-time P&L per strategy, best first — which edge actually earns."""
    df = _db.strategy_summary(_env(environment), user_id=user.username,
                              category=_category(category) or None)
    return df.to_dict("records") if not df.empty else []


@router.get("/daily-strategy-pnl")
def daily_strategy_pnl(environment: str = "Paper", category: str = "",
                       user: CurrentUser = Depends(get_current_user)):
    """One row per trading day × strategy: on this day, this strategy traded
    this many symbols and made this much. Each day's rows sum to that day's
    /daily-pnl total."""
    df = _db.daily_strategy_pnl(_env(environment), user_id=user.username,
                                category=_category(category) or None)
    return df.to_dict("records") if not df.empty else []


@router.get("/export")
def export_excel(environment: str = "Paper", user: CurrentUser = Depends(get_current_user)):
    path = _db.export_excel(_env(environment), user_id=user.username)
    if not path or not os.path.exists(path):
        raise HTTPException(500, "Export failed.")
    return FileResponse(
        path, filename=os.path.basename(path),
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
