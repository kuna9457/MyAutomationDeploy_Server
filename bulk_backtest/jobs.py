"""
bulk_backtest/jobs.py
Run a funnel in the background and report progress.

Same pattern as advanced_backtest/jobs.py but with PER-ROUND progress
reporting. The funnel is minutes of CPU, so an HTTP request that blocks dies
behind any proxy — it runs on its own thread and the client polls.

One funnel at a time: they are CPU-bound and already use a 5-thread pool
internally.
"""
from __future__ import annotations

import threading
import time
import traceback
import uuid
from datetime import datetime

import config_store
from bulk_backtest.funnel import FunnelSpec, run_funnel

_KEY = "bulk_backtest_jobs"
_lock = threading.Lock()
_jobs: dict[str, dict] = {}
MAX_SAVED = 20


def _persist(job: dict) -> None:
    try:
        with _lock:
            saved = config_store.load(_KEY)
            rows = saved.get("jobs", []) if isinstance(saved, dict) else []
            rows = [r for r in rows if r.get("id") != job["id"]]
            rows.insert(0, job)
            config_store.save(_KEY, {"jobs": rows[:MAX_SAVED]})
    except Exception as exc:
        print(f"[bulk_backtest] could not persist job: {exc}")


def start(spec: FunnelSpec) -> str:
    """Kick off a funnel search. Returns its id immediately."""
    with _lock:
        running = [j for j in _jobs.values() if j["status"] == "running"]
        if running:
            raise RuntimeError(
                "A bulk backtest is already running. Wait for it, or cancel it first.")

    job_id = uuid.uuid4().hex[:12]
    job = {
        "id": job_id,
        "status": "running",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "spec": {
            "symbols": spec.symbols,
            "strategy_keys": spec.strategy_keys,
            "start": spec.start,
            "end": spec.end,
            "capital": spec.capital,
            "mode": spec.mode,
            "rr_ladder": spec.rr_ladder,
            "folds": spec.folds,
            "baseline_rr_sweep": spec.baseline_rr_sweep,
        },
        "done": 0, "total": 0,
        "round": 0, "round_name": "starting…",
        "label": "starting…",
        "started": time.time(),
        "elapsed": 0.0,
        "results": None,
        "error": "",
    }
    _jobs[job_id] = job

    stop_flag = threading.Event()
    job["_stop"] = stop_flag

    def progress(round_info: dict) -> None:
        job["round"] = round_info.get("round", 0)
        job["round_name"] = round_info.get("name", "")
        job["done"] = round_info.get("done", 0)
        job["total"] = round_info.get("total", 0)
        survivors = round_info.get("survivors", "")
        noun = round_info.get("survivors_label", "survivors")
        job["label"] = (f"Round {job['round']}: {job['round_name']} "
                        f"({job['done']}/{job['total']}"
                        + (f", {survivors} {noun}" if survivors != "" else "")
                        + ")")
        job["elapsed"] = round(time.time() - job["started"], 1)

    def work() -> None:
        try:
            out = run_funnel(spec, progress=progress,
                             should_stop=stop_flag.is_set)
            job["results"] = out
            job["status"] = "cancelled" if out.get("cancelled") else "done"
        except Exception as exc:
            job["status"] = "error"
            job["error"] = f"{type(exc).__name__}: {exc}"[:300]
            traceback.print_exc()
        finally:
            job["elapsed"] = round(time.time() - job["started"], 1)
            job.pop("_stop", None)
            _persist({k: v for k, v in job.items() if not k.startswith("_")})

    threading.Thread(target=work, name=f"bulk-bt-{job_id}", daemon=True).start()
    return job_id


def get(job_id: str) -> dict | None:
    """Live job if we have it, else the persisted record."""
    job = _jobs.get(job_id)
    if job is not None:
        return {k: v for k, v in job.items() if not k.startswith("_")}
    try:
        saved = config_store.load(_KEY)
        for r in (saved.get("jobs", []) if isinstance(saved, dict) else []):
            if r.get("id") == job_id:
                return r
    except Exception:
        pass
    return None


def cancel(job_id: str) -> bool:
    job = _jobs.get(job_id)
    stop = job.get("_stop") if job else None
    if stop is None:
        return False
    stop.set()
    job["label"] = "cancelling…"
    return True


def recent() -> list[dict]:
    """Job headers, newest first — live ones plus what is on disk."""
    out = {}
    try:
        saved = config_store.load(_KEY)
        for r in (saved.get("jobs", []) if isinstance(saved, dict) else []):
            out[r["id"]] = r
    except Exception:
        pass
    for j in _jobs.values():
        out[j["id"]] = {k: v for k, v in j.items() if not k.startswith("_")}
    rows = sorted(out.values(), key=lambda r: r.get("created_at", ""),
                  reverse=True)
    return [{k: v for k, v in r.items() if k != "results"} for r in rows]
