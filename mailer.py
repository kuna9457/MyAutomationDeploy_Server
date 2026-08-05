"""
mailer.py
Outbound email, for the password-reset OTP.

Stdlib `smtplib` only — no new dependency, matching the reasoning in
security.py and credentials_vault.py: requirements.txt is already heavy with
broker SDKs, and an email provider's SDK would add another native-build risk
for something one email a month needs.

Configuration (all from .env, nothing hardcoded):

    SMTP_HOST       e.g. smtp.gmail.com
    SMTP_PORT       587 for STARTTLS (default), 465 for implicit TLS
    SMTP_USER       the mailbox login
    SMTP_PASSWORD   an APP PASSWORD, never the account password — Gmail and
                    most providers require one when 2FA is on, and it can be
                    revoked without touching the account itself
    SMTP_FROM       the From: address (defaults to SMTP_USER)
    SMTP_FROM_NAME  display name (defaults to "Trading Bot")
    SMTP_SSL        "true" to connect with implicit TLS (port 465) instead of
                    STARTTLS

This module NEVER raises at import and never logs a message body, so an OTP
cannot end up in the server log. `is_configured()` lets the API tell a user
"password reset isn't available" instead of silently accepting a request that
could never deliver.
"""
from __future__ import annotations

import os
import smtplib
import ssl
from email.message import EmailMessage
from typing import Optional

try:
    from dotenv import load_dotenv
    load_dotenv()          # same pattern as config.py — don't rely on import order
except Exception:
    pass

_TIMEOUT_SECONDS = 20


def _env(key: str, default: str = "") -> str:
    return (os.getenv(key, default) or "").strip()


def _truthy(value: str) -> bool:
    return value.lower() in ("1", "true", "yes", "on")


def settings() -> dict:
    """Current SMTP settings. The password is NEVER included — callers that
    want to display configuration state get everything except the secret."""
    host = _env("SMTP_HOST")
    user = _env("SMTP_USER")
    try:
        port = int(_env("SMTP_PORT", "587") or "587")
    except ValueError:
        port = 587
    return {
        "host": host,
        "port": port,
        "user": user,
        "from_addr": _env("SMTP_FROM") or user,
        "from_name": _env("SMTP_FROM_NAME", "Trading Bot"),
        "use_ssl": _truthy(_env("SMTP_SSL")),
    }


def is_configured() -> bool:
    """True when enough is set to actually deliver mail. Checked BEFORE an OTP
    is generated, so we never burn a code on a send that cannot happen."""
    cfg = settings()
    return bool(cfg["host"] and cfg["from_addr"] and _env("SMTP_PASSWORD"))


def status_detail() -> str:
    """Operator-facing reason the mailer is unusable. Never names the
    password's value, only whether it is missing."""
    cfg = settings()
    missing = [name for name, ok in (
        ("SMTP_HOST", cfg["host"]),
        ("SMTP_USER or SMTP_FROM", cfg["from_addr"]),
        ("SMTP_PASSWORD", _env("SMTP_PASSWORD")),
    ) if not ok]
    if not missing:
        return ""
    return (f"Email is not configured — set {', '.join(missing)} in .env. "
            f"Use an app password, not your mailbox password.")


def send(to_addr: str, subject: str, body: str) -> tuple[bool, str]:
    """Send one plain-text email. Returns (ok, error_message).

    Never raises: a mail-server problem must surface as a handled error to the
    caller, not a 500 — and certainly not a traceback that quotes the body.
    """
    if not is_configured():
        return False, status_detail()
    cfg = settings()
    password = _env("SMTP_PASSWORD")

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = f"{cfg['from_name']} <{cfg['from_addr']}>"
    msg["To"] = to_addr
    msg.set_content(body)

    try:
        if cfg["use_ssl"]:
            ctx = ssl.create_default_context()
            with smtplib.SMTP_SSL(cfg["host"], cfg["port"], timeout=_TIMEOUT_SECONDS,
                                  context=ctx) as smtp:
                if cfg["user"]:
                    smtp.login(cfg["user"], password)
                smtp.send_message(msg)
        else:
            with smtplib.SMTP(cfg["host"], cfg["port"], timeout=_TIMEOUT_SECONDS) as smtp:
                smtp.ehlo()
                smtp.starttls(context=ssl.create_default_context())
                smtp.ehlo()
                if cfg["user"]:
                    smtp.login(cfg["user"], password)
                smtp.send_message(msg)
        return True, ""
    except Exception as exc:
        # The exception text can name the recipient and the server, which is
        # fine for an operator log — it never contains the body (the OTP).
        print(f"[mailer] send to {to_addr} failed: {exc}")
        return False, "Could not send the email. Try again in a moment."


def send_otp(to_addr: str, code: str, minutes_valid: int,
             display_name: str = "") -> tuple[bool, str]:
    """The password-reset email. Deliberately terse and states the expiry and
    the "ignore this" line, which is what makes an unrequested code
    actionable for the recipient rather than alarming."""
    who = f" {display_name}" if display_name else ""
    body = (
        f"Hello{who},\n\n"
        f"Your password reset code is:\n\n"
        f"    {code}\n\n"
        f"It expires in {minutes_valid} minutes and can be used once.\n\n"
        f"If you did not request a password reset, you can ignore this email — "
        f"your password has not been changed.\n"
    )
    return send(to_addr, "Your password reset code", body)


def mask(email: str) -> str:
    """A recognisable but non-disclosing echo: 'kunal@gmail.com' -> 'ku***@gmail.com'.
    Used only AFTER a code is verified, never in a pre-auth response."""
    if not email or "@" not in email:
        return ""
    local, _, domain = email.partition("@")
    head = local[:2] if len(local) > 2 else local[:1]
    return f"{head}***@{domain}"


def normalise(email: str) -> str:
    """Lower-cased and trimmed, so lookups are case-insensitive."""
    return (email or "").strip().lower()


def looks_valid(email: str) -> bool:
    """Shape check only — deliverability is proven by the code arriving, not
    by a regex. Rejects the obvious mistakes without pretending to validate
    RFC 5322."""
    e = normalise(email)
    if not e or e.count("@") != 1:
        return False
    local, domain = e.split("@")
    return bool(local) and "." in domain and not domain.startswith(".") \
        and not domain.endswith(".") and " " not in e


def optional_recipient(email: Optional[str]) -> str:
    return normalise(email or "")
