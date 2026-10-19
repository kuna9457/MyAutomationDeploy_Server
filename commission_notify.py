"""
commission_notify.py
Emails sent when a client marks a commission invoice paid: a confirmation to
the client, and a heads-up to the admin.

Sent on a background thread, so a slow or broken mail server can never delay
or fail the "mark as paid" request — the invoice is already saved by then.
Every failure is logged and swallowed. Uses mailer.py (the same SMTP settings
as the password-reset email); nothing is sent if SMTP isn't configured.

Recipients:
  * client — the email on their account (Account settings / recovery email).
  * admin  — COMMISSION_ADMIN_EMAIL in .env (comma-separated for several);
             if unset, the email on every admin account; if none, SMTP_FROM.
"""
from __future__ import annotations

import os
import threading
from datetime import datetime

import config
import mailer
import user_manager

#: Name used in the email text. COMMISSION_BRAND in .env overrides it.
DEFAULT_BRAND = "WelthWest Algo"


def _brand() -> str:
    return os.getenv("COMMISSION_BRAND", "").strip() or DEFAULT_BRAND


def _money(n: float) -> str:
    return f"Rs {float(n):,.2f}"


def _when(iso: str) -> str:
    """UTC ISO timestamp -> '19 Oct 2026, 03:42 PM IST'."""
    try:
        ts = datetime.fromisoformat(iso).astimezone(config.IST)
        return ts.strftime("%d %b %Y, %I:%M %p IST")
    except Exception:
        return iso or "-"


def _week(inv: dict) -> str:
    try:
        a = datetime.fromisoformat(inv["week_start"])
        b = datetime.fromisoformat(inv["week_end"])
        return f"{a:%d %b} - {b:%d %b %Y}"
    except Exception:
        return f"{inv.get('week_start', '')} - {inv.get('week_end', '')}"


def admin_recipients() -> list[str]:
    raw = os.getenv("COMMISSION_ADMIN_EMAIL", "")
    found = [mailer.normalise(e) for e in raw.split(",")]
    if not any(found):
        found = [mailer.normalise(u.get("email", ""))
                 for u in user_manager.list_users(role="admin")]
    if not any(found):
        found = [mailer.normalise(mailer.settings()["from_addr"])]
    out: list[str] = []
    for e in found:
        if e and mailer.looks_valid(e) and e not in out:
            out.append(e)
    return out


def _details(inv: dict) -> str:
    ref = inv.get("paid_ref") or "-"
    return (
        f"    Week:              {_week(inv)}\n"
        f"    Profit that week:  {_money(inv.get('profit', 0))} "
        f"({inv.get('trades', 0)} trades)\n"
        f"    Commission rate:   {inv.get('pct', 0):g}%\n"
        f"    Amount paid:       {_money(inv.get('amount', 0))}\n"
        f"    UPI reference:     {ref}\n"
        f"    Marked paid on:    {_when(inv.get('paid_at', ''))}\n"
    )


def client_message(inv: dict, display_name: str) -> tuple[str, str]:
    brand = _brand()
    subject = (f"Payment received: {inv.get('pct', 0):g}% commission to {brand} "
               f"({_money(inv.get('amount', 0))})")
    body = (
        f"Hello {display_name or inv['username']},\n\n"
        f"Thank you. Your payment of {inv.get('pct', 0):g}% commission to "
        f"{brand} has been recorded.\n\n"
        f"{_details(inv)}\n"
        f"We will match it against our bank statement; you don't need to do "
        f"anything else. You can see all your past payments on the Commission "
        f"tab of your dashboard.\n\n"
        f"- {brand}\n"
    )
    return subject, body


def admin_message(inv: dict, display_name: str) -> tuple[str, str]:
    who = f"{display_name} ({inv['username']})" if display_name else inv["username"]
    subject = (f"Commission paid: {who} - {_money(inv.get('amount', 0))} "
               f"for week of {inv.get('week_start', '')}")
    body = (
        f"{who} has marked a commission invoice as paid.\n\n"
        f"{_details(inv)}"
        f"    Book charged:      {inv.get('environment', '')}\n\n"
        f"Check the UPI reference against your bank statement, then press "
        f"'Received' on the admin Commission tab to verify it.\n"
    )
    return subject, body


def _send_all(inv: dict) -> None:
    try:
        user = user_manager.get_user(inv["username"]) or {}
        name = user.get("display_name") or ""
        client_email = mailer.normalise(user.get("email", ""))
        if client_email and mailer.looks_valid(client_email):
            ok, err = mailer.send(client_email, *client_message(inv, name))
            if not ok:
                print(f"[commission] client email for {inv['invoice_id']} failed: {err}")
        else:
            print(f"[commission] {inv['username']} has no email on file - "
                  f"no client confirmation sent.")
        subject, body = admin_message(inv, name)
        for addr in admin_recipients():
            ok, err = mailer.send(addr, subject, body)
            if not ok:
                print(f"[commission] admin email for {inv['invoice_id']} failed: {err}")
    except Exception as exc:                     # never let a mail problem escape
        print(f"[commission] payment emails for {inv.get('invoice_id')} failed: {exc}")


def notify_paid(inv: dict, background: bool = True) -> None:
    """Email the client and the admin that `inv` was marked paid."""
    if not mailer.is_configured():
        print("[commission] SMTP not configured - payment emails skipped.")
        return
    if background:
        threading.Thread(target=_send_all, args=(dict(inv),), daemon=True,
                         name="commission-mail").start()
    else:
        _send_all(dict(inv))
