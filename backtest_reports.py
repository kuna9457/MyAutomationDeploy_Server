"""
backtest_reports.py
Named single-symbol backtest analyses — run a backtest, save its trades/metrics
under a name you choose, and reopen the exact same charts and insights later
without re-running the backtest.

Reports are bucketed by strategy category — Scalper / Intraday / Swing — so a
name only has to be unique within the category it was run under, and saving
always defaults to the category the backtest actually ran in.

Pure persistence, mirrors watchlists.py: {category: {name: report}} stored as
JSON under config.LOCAL_DB_DIR. No Streamlit/strategy/broker imports, so it
can't hamper anything else — the UI (app.py) is the only caller.
"""
from __future__ import annotations

import json
import os

import config

CATEGORIES = ["Scalper", "Intraday", "Swing"]
_DEFAULT_CATEGORY = "Intraday"


def _path() -> str:
    return os.path.join(config.LOCAL_DB_DIR, "backtest_reports.json")


def _empty() -> dict[str, dict]:
    return {c: {} for c in CATEGORIES}


def load_all() -> dict[str, dict[str, dict]]:
    """Every saved report as {category: {name: report}}. Never raises — a
    missing or corrupt file just reads as 'no reports' so the UI always
    renders.

    Transparently migrates the old flat {name: report} format (from before
    categories existed) into the new nested shape, bucketing each report by
    its own "mode" field."""
    path = _path()
    if not os.path.exists(path):
        return _empty()
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception:
        return _empty()
    if not isinstance(data, dict):
        return _empty()

    if set(data.keys()) <= set(CATEGORIES):
        out = _empty()
        for cat in CATEGORIES:
            bucket = data.get(cat, {})
            if isinstance(bucket, dict):
                out[cat] = bucket
        return out

    # Old flat {name: report} format — migrate in place.
    migrated = _empty()
    for name, report in data.items():
        if not isinstance(report, dict):
            continue
        cat = report.get("mode")
        if cat not in CATEGORIES:
            cat = _DEFAULT_CATEGORY
        migrated[cat][name] = report
    _write(migrated)
    return migrated


def names(category: str) -> list[str]:
    """Saved report names within `category`, alphabetical — ready for a
    selectbox."""
    return sorted(load_all().get(category, {}).keys())


def get(category: str, name: str) -> dict:
    """The report saved under `name` within `category` (empty dict if
    unknown)."""
    return load_all().get(category, {}).get(name, {})


def save(name: str, report: dict, category: str | None = None) -> bool:
    """Create or overwrite the report `name` inside `category`. If `category`
    is omitted, it's inferred from `report["mode"]` (falling back to
    Intraday). Returns False for a blank name or an empty report, so the UI
    can show a helpful message instead of writing junk."""
    name = str(name).strip()
    if not name or not report:
        return False
    if category not in CATEGORIES:
        category = report.get("mode") if report.get("mode") in CATEGORIES \
            else _DEFAULT_CATEGORY
    data = load_all()
    data.setdefault(category, {})[name] = report
    _write(data)
    return True


def delete(category: str, name: str) -> bool:
    """Remove a saved report from `category`. Returns True if it existed."""
    data = load_all()
    bucket = data.get(category, {})
    if name not in bucket:
        return False
    del bucket[name]
    _write(data)
    return True


def _write(data: dict) -> None:
    os.makedirs(config.LOCAL_DB_DIR, exist_ok=True)
    with open(_path(), "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, default=str)
