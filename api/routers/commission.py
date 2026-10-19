"""
api/routers/commission.py
WelthWest's weekly profit share (commission.py): a client sees what they owe
and marks it paid; the admin sets each client's percent and tracks payments.

Mounted on EVERY server, client nodes included — a client's UI talks to their
own server, and the records live in the shared database. The admin routes are
admin-only, and an admin is already refused on a client's server.

Nothing here touches trading: no engine, strategy, broker or market-data import.
"""
from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

import commission
import commission_notify
from api.auth import CurrentUser, get_current_user, require_admin

router = APIRouter(prefix="/commission", tags=["commission"])


class MarkPaidRequest(BaseModel):
    #: The UPI reference / UTR from the client's payment app. Optional.
    reference: str = Field("", max_length=60)


class RateRequest(BaseModel):
    pct: float = Field(..., ge=0, le=commission.MAX_PCT)
    #: First week billed (any date in it). Blank = see commission.set_rate.
    effective_from: str = ""


class SettingsRequest(BaseModel):
    upi_id: str
    payee_name: str = ""
    environment: Literal["Live", "Paper"] = "Live"


class StatusRequest(BaseModel):
    status: Literal["verified", "due"]


def _run(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except KeyError:
        raise HTTPException(404, "No such client.")
    except ValueError as exc:
        raise HTTPException(400, str(exc))


# -- the client's own statement ------------------------------------------------ #
@router.get("/me")
def my_commission(user: CurrentUser = Depends(get_current_user)):
    return commission.client_view(user.username)


@router.post("/me/invoices/{week_start}/paid")
def mark_my_invoice_paid(week_start: str, req: MarkPaidRequest,
                         user: CurrentUser = Depends(get_current_user)):
    """Scoped to the caller's own username — a client can never mark, or even
    address, another client's invoice."""
    row = _run(commission.mark_paid, user.username, week_start, req.reference)
    # Confirmation to the client + heads-up to the admin, on a background
    # thread: the invoice is already saved, so mail can't fail this request.
    commission_notify.notify_paid(row)
    return row


# -- admin ----------------------------------------------------------------------- #
@router.get("/admin/overview")
def overview(_: CurrentUser = Depends(require_admin)):
    return commission.admin_overview()


@router.put("/admin/settings")
def put_settings(req: SettingsRequest, _: CurrentUser = Depends(require_admin)):
    return _run(commission.update_settings, req.upi_id, req.payee_name,
                req.environment)


@router.put("/admin/clients/{username}/rate")
def put_rate(username: str, req: RateRequest,
             _: CurrentUser = Depends(require_admin)):
    return _run(commission.set_rate, username, req.pct, req.effective_from)


@router.put("/admin/clients/{username}/invoices/{week_start}")
def put_invoice_status(username: str, week_start: str, req: StatusRequest,
                       _: CurrentUser = Depends(require_admin)):
    return _run(commission.admin_set_status, username, week_start, req.status)
