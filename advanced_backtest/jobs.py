"""
advanced_backtest/jobs.py
Run a search in the background and report progress.

A search is minutes of CPU, not milliseconds. An HTTP request that blocks that
long dies behind any proxy and cannot be cancelled, so the search runs on its
own thread and the client polls.

One search at a time per process: they are CPU-bound and already use a 5-thread
pool internally, so a second concurrent search would halve both and finish
neither sooner.
"""
from __future__ import annotations

import threading
import time
import traceback
import uuid
from dataclasses import asdict
from datetime import datetime

import config_store
from advanced_backtest import ranking
from advanced_backtest.search import SearchSpec, run_search

_KEY = "advanced_backtest_jobs"
_lock = threading.Lock()

#: Live jobs, including finished ones until the process restarts. Results are
#: also persisted (see _persist) so a finished search survives a reload.
_jobs: dict[str, dict] = {}

#: Finished searches kept on disk. A search is minutes of work; losing one to a
#: page refresh would be its own small tragedy.
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
        print(f"[advanced_backtest] could not persist job: {exc}")


def start(spec: SearchSpec) -> str:
    """Kick off a search. Returns its id immediately."""
    with _lock:
        running = [j for j in _jobs.values() if j["status"] == "running"]
        if running:
            raise RuntimeError(
                "A search is already running. Wait for it, or cancel it first.")

    job_id = uuid.uuid4().hex[:12]
    job = {
        "id": job_id,
        "status": "running",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "spec": {**asdict(spec), "mode": spec.mode.value},
        "done": 0, "total": len(spec.symbols) + spec.verify_top,
        "label": "starting…", "started": time.time(),
        "elapsed": 0.0, "results": None, "error": "",
    }
    with _jobs_guard():
        _jobs[job_id] = job

    stop_flag = threading.Event()
    job["_stop"] = stop_flag

    def progress(done: int, total: int, label: str) -> None:
        job["done"] = done
        job["total"] = total
        job["label"] = label
        job["elapsed"] = round(time.time() - job["started"], 1)

    def work() -> None:
        try:
            out = run_search(spec, progress=progress,
                             should_stop=stop_flag.is_set)
            out["bucket"] = ranking.recommended_bucket(out["combos"])
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

    threading.Thread(target=work, name=f"advbt-{job_id}", daemon=True).start()
    return job_id


def _jobs_guard():
    return _lock


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
    # Headers only: a full result set is large and the list view never needs it.
    return [{k: v for k, v in r.items() if k != "results"} for r in rows]
