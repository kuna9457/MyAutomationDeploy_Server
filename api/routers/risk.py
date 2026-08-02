"""
api/routers/risk.py
LIVE-only risk guardrails — thin wrapper over risk_manager.py's
get_limits()/set_limits(), unchanged. Keyed per logged-in user (Phase 2):
each account reads/writes only its own limits file.
"""
from __future__ import annotations

from dataclasses import asdict

from fastapi import APIRouter, Depends

import risk_manager
from api.auth import CurrentUser, get_current_user
from api.schemas import RiskLimitsRequest

router = APIRouter(prefix="/risk", tags=["risk"])


@router.get("/limits")
def get_limits(user: CurrentUser = Depends(get_current_user)):
    return asdict(risk_manager.get_limits(user.username))


@router.put("/limits")
def set_limits(req: RiskLimitsRequest, user: CurrentUser = Depends(get_current_user)):
    updated = risk_manager.set_limits(user.username, **req.model_dump())
    return asdict(updated)
