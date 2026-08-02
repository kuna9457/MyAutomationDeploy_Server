"""
watchlists.py
Named instrument watchlists — save a bucket of symbols you trade often under a
name, then load it in one click into the live bot's instrument picker AND the
bulk backtest bucket, instead of re-selecting the same stocks every time.

Pure persistence, nothing else: a watchlist is just {name: [symbol, ...]} stored
as JSON under the local data dir (config.LOCAL_DB_DIR, already gitignored). This
module imports NEITHER Streamlit NOR any strategy/broker/engine code, so it can't
hamper anything else — the UI (app.py) is the only caller.

Symbols are stored as plain strings. Validation against the live instrument
universe is the caller's job (an instrument can be added/removed from config
independently), so `get()` returns exactly what was saved and the UI filters it
to what currently exists before using it.
"""
from __future__ import annotations

import json
import os

import config


def _path() -> str:
    return os.path.join(config.LOCAL_DB_DIR, "watchlists.json")


def load_all() -> dict[str, list[str]]:
    """Every saved watchlist as {name: [symbols]}. Never raises — a missing or
    corrupt file just reads as 'no watchlists' so the UI always renders."""
    path = _path()
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception:
        return {}
    if not isinstance(data, dict):
        return {}
    out: dict[str, list[str]] = {}
    for name, syms in data.items():
        if isinstance(syms, list):
            out[str(name)] = [str(s) for s in syms]
    return out


def names() -> list[str]:
    """Watchlist names, alphabetical — ready for a selectbox."""
    return sorted(load_all().keys())


def get(name: str) -> list[str]:
    """The symbols saved under `name` (empty list if the name is unknown)."""
    return load_all().get(name, [])


def save(name: str, symbols: list[str]) -> bool:
    """Create or overwrite the watchlist `name` with `symbols` (order preserved,
    duplicates dropped). Returns False for a blank name or an empty symbol list,
    so the UI can show a helpful message instead of writing junk."""
    name = str(name).strip()
    clean = list(dict.fromkeys(s for s in symbols if s))   # dedupe, keep order
    if not name or not clean:
        return False
    data = load_all()
    data[name] = clean
    _write(data)
    return True


def delete(name: str) -> bool:
    """Remove a watchlist. Returns True if it existed."""
    data = load_all()
    if name not in data:
        return False
    del data[name]
    _write(data)
    return True


def _write(data: dict[str, list[str]]) -> None:
    os.makedirs(config.LOCAL_DB_DIR, exist_ok=True)
    with open(_path(), "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
