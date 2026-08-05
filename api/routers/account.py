"""
api/routers/account.py
Everything an account owner does to their OWN login: change password, set the
address reset codes go to, and the two UNAUTHENTICATED forgot-password steps.

The reset endpoints are deliberately the only routes in the app outside
/auth/login that require no token — that is the whole point of a forgot-password
flow, and it is why they are written to give away nothing:

  * The request step returns the SAME response whether the account exists, is
    disabled, has no address on file, or is being throttled. Anything else
    turns it into a "does this person bank here" oracle.
  * The confirm step returns one message for every failure mode, so a wrong
    code and a wrong username are indistinguishable.

Nothing here touches trading. No engine, strategy, broker or market-data
import — a bug in this file cannot affect a running bot.
"""
from __future__ import annotations

import mailer
import password_reset
import user_manager
from api.auth import CurrentUser, create_access_token, get_current_user
from api.schemas import (ChangePasswordRequest, PasswordResetConfirm,
                         PasswordResetRequest, SetEmailRequest)
from fastapi import APIRouter, Depends, HTTPException
from security import verify_password

router = APIRouter(prefix="/account", tags=["account"])

#: Identical for every outcome of a reset request — see the module docstring.
_GENERIC_REQUEST_REPLY = {
    "ok": True,
    "message": ("If that account exists and has an email on file, a reset "
                "code has been sent. It expires in "
                f"{password_reset.OTP_TTL_MINUTES} minutes."),
}


# -- signed-in -------------------------------------------------------------- #
@router.get("/me")
def my_account(user: CurrentUser = Depends(get_current_user)):
    """The caller's own profile, including a MASKED email so they can confirm
    where a reset code would go without the full address being echoed back."""
    record = user_manager.get_user_by_id(user.user_id) or {}
    email = mailer.optional_recipient(record.get("email"))
    return {
        "username": user.username,
        "display_name": user.display_name,
        "role": user.role,
        "email_masked": mailer.mask(email),
        "has_email": bool(email),
        "reset_available": mailer.is_configured(),
    }


@router.put("/email")
def set_my_email(req: SetEmailRequest,
                 user: CurrentUser = Depends(get_current_user)):
    """Set (or clear, by sending "") where MY reset codes go."""
    email = mailer.normalise(req.email)
    if email and not mailer.looks_valid(email):
        raise HTTPException(400, "That doesn't look like a valid email address.")
    user_manager.set_email(user.user_id, email)
    return {"ok": True, "email_masked": mailer.mask(email), "has_email": bool(email)}


@router.post("/change-password")
def change_password(req: ChangePasswordRequest,
                    user: CurrentUser = Depends(get_current_user)):
    """Change my password, proving I know the current one.

    Requiring the current password is what stops a hijacked session from
    locking the real owner out. On success every OTHER session dies too
    (set_password bumps token_version), so this issues a fresh token for the
    caller — otherwise they would log themselves out mid-request.
    """
    record = user_manager.get_user_by_id(user.user_id)
    if record is None:
        raise HTTPException(404, "Account not found.")
    if not verify_password(req.current_password, record.get("password_hash", "")):
        raise HTTPException(400, "Your current password is incorrect.")
    try:
        new_password = password_reset.validate_password(req.new_password)
    except password_reset.ResetError as exc:
        raise HTTPException(400, str(exc))
    if verify_password(new_password, record.get("password_hash", "")):
        raise HTTPException(
            400, "That's the same as your current password — pick a new one.")

    user_manager.set_password(user.user_id, new_password)
    return {
        "ok": True,
        "message": "Password changed. Other devices have been signed out.",
        "access_token": create_access_token(user),
    }


# -- forgot password (NO auth) ---------------------------------------------- #
@router.get("/password-reset/available")
def reset_available():
    """Whether the login page should offer 'Forgot password?' at all. Says
    nothing about any account — only whether this server can send mail."""
    return {"available": mailer.is_configured()}


@router.post("/password-reset/request")
def request_reset(req: PasswordResetRequest):
    """Step 1: mail a one-time code to the address on file.

    Always answers with _GENERIC_REQUEST_REPLY. The only errors that surface
    are server-side ones (mail not configured / the send itself failing),
    because those are true regardless of which account was named and are
    actionable for the person reading them.
    """
    try:
        sent, detail = password_reset.request_code(req.username)
    except password_reset.ResetError as exc:
        raise HTTPException(503, str(exc))
    if not sent:
        # Logged for the operator, never returned: `detail` distinguishes
        # "no such account" from "no email on file", which the caller must
        # not be able to tell apart.
        print(f"[account] reset request for {req.username!r} not sent: {detail}")
    return _GENERIC_REQUEST_REPLY


@router.post("/password-reset/confirm")
def confirm_reset(req: PasswordResetConfirm):
    """Step 2: redeem the code and set the new password.

    No token is returned — the user is sent back to the login screen to sign
    in with the new password. That proves the change end-to-end and avoids
    handing a session to whoever happened to have the code.
    """
    try:
        username = password_reset.confirm_code(
            req.username, req.code, req.new_password)
    except password_reset.ResetError as exc:
        raise HTTPException(400, str(exc))
    return {"ok": True, "username": username,
            "message": "Password updated. Please sign in with your new password."}
