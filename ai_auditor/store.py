"""
ai_auditor/store.py
Saved audit reports — same persisted-JSON pattern as presets and watchlists.

Reports are kept so two runs can be compared: if a later audit disagrees with an
earlier one, the stored provider, model, window and pack hash tell you whether
the BOT changed or only the MODEL did.
"""
from __future__ import annotations

import threading
from typing import Optional

import config_store

_KEY = "ai_audit_reports"
_lock = threading.Lock()

#: Reports are small but unbounded in principle; keep the newest N. Nothing here
#: is a trading record — the trades themselves live in the trade store.
MAX_REPORTS = 50


def _all() -> list[dict]:
    data = config_store.load(_KEY)
    if isinstance(data, dict):
        data = data.get("reports")
    return list(data) if isinstance(data, list) else []


def list_reports(include_body: bool = False) -> list[dict]:
    """Newest first. Without `include_body` only the headers are returned, so
    the history list stays small however long the reports are."""
    out = []
    for r in _all():
        if include_body:
            out.append(r)
            continue
        out.append({k: r.get(k) for k in
                    ("id", "created_at", "environment", "window", "provider",
                     "model", "verdict", "headline", "closed_trades",
                     "pack_bytes", "error")})
    return out


def get_report(report_id: str) -> Optional[dict]:
    for r in _all():
        if r.get("id") == report_id:
            return r
    return None


def save_report(report: dict) -> dict:
    with _lock:
        reports = _all()
        reports.insert(0, report)
        config_store.save(_KEY, {"reports": reports[:MAX_REPORTS]})
    return report


def delete_report(report_id: str) -> bool:
    with _lock:
        reports = _all()
        kept = [r for r in reports if r.get("id") != report_id]
        if len(kept) == len(reports):
            return False
        config_store.save(_KEY, {"reports": kept})
    return True
