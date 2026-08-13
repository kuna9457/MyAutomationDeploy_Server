"""
ai_auditor/auditor.py
Orchestration: build the pack -> check it -> call the model -> verify -> save.

The only place the pieces meet, and the only place that talks to the network.
"""
from __future__ import annotations

import hashlib
import json
import re
import threading
import uuid
from datetime import datetime

import config
from ai_auditor import audit_pack, prompts, providers, store
from config import Environment

#: One audit at a time per process. Two concurrent runs would double the cost
#: for no benefit — the pack is the same and the operator is one person.
_run_lock = threading.Lock()


def _numbers_in(text: str) -> set[str]:
    """Numeric tokens in a string, normalised (commas and signs stripped).

    The trailing "." strip matters: a figure that ends a sentence ("…was
    18400.") otherwise captures the full stop, which both mis-renders in the
    warning and makes a correctly-quoted number look invented.
    """
    out = set()
    for t in re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", text or ""):
        out.add(t.replace(",", "").lstrip("+-").rstrip("."))
    return out


def _verify_numbers(report: dict, pack: dict) -> list[str]:
    """Flag figures in the report that do not appear anywhere in the pack.

    The prompt forbids the model from computing anything; this is the check that
    the ban held. It is intentionally LENIENT — it only looks at the headline
    and evidence strings, tolerates rounding to whole numbers, and ignores small
    integers that are almost always sample counts or priorities. False alarms
    would train the operator to ignore the warning, which is worse than no
    warning at all.
    """
    pack_nums = _numbers_in(json.dumps(pack, default=str))
    # Rounded forms too: the pack holds 18400.0, a report may say 18400.
    for n in list(pack_nums):
        try:
            f = float(n)
        except ValueError:
            continue
        pack_nums.add(str(int(f)))
        pack_nums.add(f"{f:.1f}")
        pack_nums.add(f"{f:.2f}")

    suspicious: list[str] = []
    fields = [report.get("headline", "")]
    for key in ("what_is_working", "what_is_broken"):
        fields += [str(i.get("evidence", "")) for i in report.get(key, []) or []]
    for rec in report.get("recommendations", []) or []:
        fields.append(str(rec.get("evidence", "")))

    for text in fields:
        for n in _numbers_in(text):
            try:
                if abs(float(n)) < 100:      # sample sizes, priorities, hours
                    continue
            except ValueError:
                continue
            if n in pack_nums:
                continue
            if str(int(float(n))) in pack_nums or f"{float(n):.1f}" in pack_nums:
                continue
            suspicious.append(n)
    return sorted(set(suspicious))


def preview(db, environment: Environment, start: str = "", end: str = "",
            user_id: str = "admin") -> dict:
    """The exact pack that WOULD be sent — no network call, no cost.

    Exists so the payload can be read before any key is used. It runs the same
    redaction check the real path runs, so previewing genuinely proves what
    would leave the machine.
    """
    pack = audit_pack.build_pack(db, environment, start, end, user_id)
    audit_pack.assert_no_secrets(pack)
    return {"pack": pack, "bytes": audit_pack.pack_size_bytes(pack),
            "closed_trades": pack["meta"]["closed_trades"]}


def run_audit(db, environment: Environment, start: str = "", end: str = "",
              provider_name: str = "", model: str = "",
              user_id: str = "admin") -> dict:
    """Build, send, verify, save. Returns the stored report document."""
    if not _run_lock.acquire(blocking=False):
        raise RuntimeError("An audit is already running. Wait for it to finish.")
    try:
        pack = audit_pack.build_pack(db, environment, start, end, user_id)
        size = audit_pack.pack_size_bytes(pack)
        if size > audit_pack.MAX_PACK_BYTES:
            raise RuntimeError(
                f"The audit pack is {size:,} bytes, over the "
                f"{audit_pack.MAX_PACK_BYTES:,} limit. Narrow the date range.")
        if pack["meta"]["closed_trades"] == 0:
            raise RuntimeError(
                "No closed trades in that window — there is nothing to audit.")

        # LAST GATE BEFORE THE NETWORK. Never reorder this below the request.
        audit_pack.assert_no_secrets(pack)

        pack_json = json.dumps(pack, default=str, separators=(",", ":"))
        started = datetime.now().astimezone()

        # Try the preferred provider, then the other one. An audit is a manual,
        # occasional action — finishing on the second provider is worth more
        # than making the operator notice a failure and press the button again.
        # Every attempt's reason is kept so a fallback is explainable rather
        # than mysterious.
        chain = providers.provider_chain(provider_name, model)
        attempts: list[dict] = []
        res = None
        for prov in chain:
            try:
                res = prov.complete(prompts.SYSTEM_PROMPT,
                                    prompts.user_message(pack_json))
                attempts.append({"provider": prov.name, "model": res.model,
                                 "ok": True, "error": ""})
                break
            except providers.AuditProviderError as exc:
                attempts.append({"provider": prov.name,
                                 "model": getattr(prov, "model", ""),
                                 "ok": False, "error": str(exc)[:400]})
        if res is None:
            raise providers.AuditProviderError(
                "Every provider failed.\n" + "\n".join(
                    f"- {a['provider']}: {a['error']}" for a in attempts))

        report = res.report
        unverified = _verify_numbers(report, pack)

        doc = {
            "id": uuid.uuid4().hex[:12],
            "created_at": started.isoformat(timespec="seconds"),
            "environment": environment.value,
            "window": {"from": start or "all", "to": end or "all"},
            "provider": res.provider,
            "model": res.model,
            "closed_trades": pack["meta"]["closed_trades"],
            "pack_bytes": size,
            # Two packs with the same hash describe the same book, so a changed
            # verdict on an unchanged hash is the MODEL disagreeing with itself.
            "pack_hash": hashlib.sha256(pack_json.encode()).hexdigest()[:16],
            "usage": res.usage,
            # Every attempt, in order. When a fallback happened this says which
            # provider was tried first and exactly why it did not answer.
            "attempts": attempts,
            "fell_back": len(attempts) > 1,
            "verdict": report.get("verdict", ""),
            "headline": report.get("headline", ""),
            "report": report,
            "unverified_numbers": unverified,
            "error": "",
        }
        return store.save_report(doc)
    finally:
        _run_lock.release()
