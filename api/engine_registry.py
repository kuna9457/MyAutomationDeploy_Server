"""
api/engine_registry.py
Replaces Streamlit's `st.session_state.engine` (one engine per browser
session) with a process-wide registry keyed by owner. Phase 1 has exactly one
owner ("admin"); Phase 2 adds one entry per logged-in client — the registry
shape already supports that, only the auth layer needs to grow.

TradingEngine itself (engine.py) is untouched: this module only holds
references to instances and starts/stops them, exactly like app.py's
sidebar buttons do today.
"""
from __future__ import annotations

import threading
from typing import Optional

from engine import TradingEngine

_lock = threading.Lock()
_engines: dict[str, TradingEngine] = {}


def get_engine(owner: str) -> Optional[TradingEngine]:
    with _lock:
        return _engines.get(owner)


def set_engine(owner: str, engine: TradingEngine) -> None:
    with _lock:
        _engines[owner] = engine


def stop_engine(owner: str) -> bool:
    with _lock:
        engine = _engines.get(owner)
    if engine is None:
        return False
    engine.stop()
    return True
