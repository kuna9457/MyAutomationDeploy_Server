"""
api/routers/auditor.py
The AI Auditor's HTTP surface. Admin-only, and READ-ONLY with respect to the
bot: nothing here can start, stop, size or place anything.

/preview is deliberately first-class — it returns the exact payload that WOULD
be sent, without calling any provider, so the operator can read what leaves the
machine before a key is ever used.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from ai_auditor import auditor, providers, store
from api.auth import CurrentUser, require_admin
from config import Environment
from db_manager import DBManager

router = APIRouter(prefix="/auditor", tags=["auditor"],
                   dependencies=[Depends(require_admin)])

# One long-lived manager, matching backtest.py / trades.py.
_db = DBManager()


class AuditRequest(BaseModel):
    #: "Paper" | "Live" — never mixed, because a paper fill is simulated and a
    #: live one is not (Immutable Rule #5 carried into reporting).
    environment: str = "Live"
    #: "YYYY-MM-DD"; empty means the whole history.
    start: str = ""
    end: str = ""
    provider: str = ""
    model: str = ""


def _env(name: str) -> Environment:
    try:
        return Environment(name)
    except ValueError:
        raise HTTPException(400, "environment must be 'Paper' or 'Live'.")


@router.get("/providers")
def list_providers():
    """Which back-ends are usable. A provider with no key is listed as
    unavailable rather than hidden, so the reason is visible."""
    return providers.available_providers()


@router.post("/preview")
def preview(req: AuditRequest, user: CurrentUser = Depends(require_admin)):
    """The exact audit pack that would be sent. No provider call, no cost."""
    try:
        return auditor.preview(_db, _env(req.environment), req.start, req.end,
                               user.username)
    except ValueError as exc:            # redaction tripped
        raise HTTPException(500, str(exc))


@router.post("/run")
def run(req: AuditRequest, user: CurrentUser = Depends(require_admin)):
    try:
        return auditor.run_audit(_db, _env(req.environment), req.start, req.end,
                                 req.provider, req.model, user.username)
    except providers.AuditProviderError as exc:
        # The provider's own message is written to be shown as-is.
        raise HTTPException(502, str(exc))
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(400, str(exc))


@router.get("/reports")
def reports():
    return store.list_reports()


@router.get("/reports/{report_id}")
def report(report_id: str):
    doc = store.get_report(report_id)
    if doc is None:
        raise HTTPException(404, "No such report.")
    return doc


@router.delete("/reports/{report_id}")
def delete(report_id: str):
    if not store.delete_report(report_id):
        raise HTTPException(404, "No such report.")
    return {"ok": True}
