"""
password_reset.py
Forgot-password by emailed one-time code.

Flow, and why it is shaped this way:

    1. request(username)  -> a 6-digit code is mailed to the address on file
    2. confirm(username, code, new_password) -> password set, sessions killed

The account is identified by USERNAME, never by the email address alone. A
person may hold two logins on one mailbox (an admin and a client account), so
"reset whoever owns this address" would be ambiguous and could reset the
wrong one. The address is used only to DELIVER.

Security properties this file is responsible for:

  * The code is never stored. Only a PBKDF2 hash of it (security.py, the same
    primitive as passwords), so a database dump does not yield a live code.
  * Codes expire (OTP_TTL_MINUTES) and are single-use.
  * Wrong guesses are capped (MAX_ATTEMPTS). A 6-digit code is only 10^6
    wide, so without a cap it is trivially brute-forced within its lifetime.
  * Re-requests are throttled (RESEND_COOLDOWN_SECONDS), so this endpoint
    cannot be used to spam someone's inbox.
  * NOTHING in any response reveals whether an account, or an address for it,
    exists — the caller gets the same answer either way (see the router).

Nothing here touches trading: no engine, strategy, broker or config-store
import. It only reads and writes the user record.
"""
from __future__ import annotations

import secrets
from datetime import datetime, timedelta, timezone
from typing import Optional

import mailer
import user_manager
from security import hash_password, verify_password

#: Long enough to arrive and be typed, short enough that a leaked code is
#: usually already dead.
OTP_TTL_MINUTES = 10

#: 10^6 possibilities. Safe ONLY because of MAX_ATTEMPTS below — the cap, not
#: the length, is what stops a brute force here.
OTP_DIGITS = 6

#: Wrong guesses allowed before the code is destroyed and must be re-requested.
MAX_ATTEMPTS = 5

#: Minimum gap between two sends for one account, so this is not an email
#: cannon aimed at whoever's address is on file.
RESEND_COOLDOWN_SECONDS = 60

#: Matches the existing rule in the admin create-client route.
MIN_PASSWORD_LENGTH = 6


class ResetError(Exception):
    """A user-facing failure. The message is safe to show verbatim — it never
    distinguishes 'no such account' from 'no address on file'."""


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.isoformat()


def _parse(value: str) -> Optional[datetime]:
    try:
        dt = datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _generate_code() -> str:
    """Cryptographically random, zero-padded so every code is the same length
    (a variable-length code leaks a little and confuses users)."""
    return f"{secrets.randbelow(10 ** OTP_DIGITS):0{OTP_DIGITS}d}"


def validate_password(password: str) -> str:
    pwd = str(password or "")
    if len(pwd) < MIN_PASSWORD_LENGTH:
        raise ResetError(
            f"Password must be at least {MIN_PASSWORD_LENGTH} characters.")
    return pwd


def request_code(username: str) -> tuple[bool, str]:
    """Generate and mail a reset code for `username`.

    Returns (sent, detail). `sent` is for SERVER-SIDE logging only — the
    router must return an identical response to the caller either way, or
    this becomes an account-and-email enumeration oracle.

    Raises ResetError only for conditions that are NOT account-specific (the
    mail server being unconfigured or refusing), because those are safe to
    surface and actionable.
    """
    if not mailer.is_configured():
        raise ResetError(
            "Password reset by email isn't available right now. Please ask "
            "your admin to reset it for you.")

    user = user_manager.get_user(str(username or "").strip())
    if user is None:
        return False, "no such account"
    if user.get("status") != "active":
        return False, "account disabled"

    to_addr = mailer.optional_recipient(user.get("email"))
    if not to_addr:
        return False, "no email on file"

    existing = user_manager.get_reset_challenge(user["user_id"])
    sent_at = _parse(existing.get("sent_at", ""))
    if sent_at is not None:
        elapsed = (_now() - sent_at).total_seconds()
        if elapsed < RESEND_COOLDOWN_SECONDS:
            # Throttled, but still reported to the caller as a normal success
            # by the router — a "you're being throttled" response would confirm
            # the account exists.
            return False, "throttled"

    code = _generate_code()
    challenge = {
        "code_hash": hash_password(code),
        "expires_at": _iso(_now() + timedelta(minutes=OTP_TTL_MINUTES)),
        "attempts": 0,
        "sent_at": _iso(_now()),
    }
    ok, err = mailer.send_otp(to_addr, code, OTP_TTL_MINUTES,
                              user.get("display_name", ""))
    if not ok:
        # Do NOT arm the challenge for a mail that never left — otherwise the
        # cooldown would lock the user out of retrying a send that failed.
        raise ResetError(err)

    user_manager.set_reset_challenge(user["user_id"], challenge)
    return True, mailer.mask(to_addr)


def confirm_code(username: str, code: str, new_password: str) -> str:
    """Verify the code and set the new password. Returns the username on
    success; raises ResetError otherwise.

    Every failure below returns the SAME message. Distinguishing "no such
    account" from "wrong code" here would undo the enumeration protection the
    request step is careful to maintain.
    """
    invalid = ResetError("That code is invalid or has expired. "
                         "Request a new one and try again.")

    pwd = validate_password(new_password)
    user = user_manager.get_user(str(username or "").strip())
    if user is None or user.get("status") != "active":
        raise invalid

    challenge = user_manager.get_reset_challenge(user["user_id"])
    if not challenge or not challenge.get("code_hash"):
        raise invalid

    expires = _parse(challenge.get("expires_at", ""))
    if expires is None or _now() > expires:
        user_manager.set_reset_challenge(user["user_id"], None)
        raise invalid

    attempts = int(challenge.get("attempts", 0) or 0)
    if attempts >= MAX_ATTEMPTS:
        user_manager.set_reset_challenge(user["user_id"], None)
        raise invalid

    if not verify_password(str(code or "").strip(), challenge["code_hash"]):
        # Count the miss BEFORE returning, so repeated guesses actually burn
        # the budget rather than retrying against a fresh counter.
        challenge["attempts"] = attempts + 1
        if challenge["attempts"] >= MAX_ATTEMPTS:
            user_manager.set_reset_challenge(user["user_id"], None)
        else:
            user_manager.set_reset_challenge(user["user_id"], challenge)
        raise invalid

    # Correct. set_password clears the challenge and bumps token_version,
    # which logs out every session that existed under the old password —
    # including whoever prompted the reset.
    user_manager.set_password(user["user_id"], pwd)
    return user["username"]
