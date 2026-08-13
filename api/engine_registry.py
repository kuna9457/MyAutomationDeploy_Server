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
#: owner -> that owner's engines, in start order. A single-strategy run has
#: exactly one entry (every case that predates the strategy board); a run off
#: the board has one per enabled group, all sharing a CapitalLedger.
_engines: dict[str, list[TradingEngine]] = {}


def get_engine(owner: str) -> Optional[TradingEngine]:
    """The owner's PRIMARY engine — the first started.

    Every call site that predates multi-group trading keeps working through
    this: with one engine it is that engine, and with several it is a real,
    running one whose environment, mode and capital are shared by the whole
    board. Callers that must see the WHOLE board (status, stop, PnL) use
    get_engines() instead.
    """
    with _lock:
        engines = _engines.get(owner) or []
        return engines[0] if engines else None


def get_engines(owner: str) -> list[TradingEngine]:
    """Every engine this owner is running — one per strategy group."""
    with _lock:
        return list(_engines.get(owner) or [])


def set_engine(owner: str, engine: TradingEngine) -> None:
    """Replace the owner's engines with this single one — the ordinary
    single-strategy start. Any previous board is dropped, which is correct:
    starting a plain run replaces whatever was running before it."""
    with _lock:
        _engines[owner] = [engine]


def set_engines(owner: str, engines: list[TradingEngine]) -> None:
    """Replace the owner's engines with a whole strategy board."""
    with _lock:
        _engines[owner] = list(engines)


def stop_engine(owner: str) -> bool:
    """Stop the owner's engine, keeping the registry entry.

    The entry is deliberately NOT removed. /bot/status derives `started` from
    the engine's existence rather than state.running, so the dashboard keeps
    reporting the final snapshot (closed positions, realised PnL) after a stop.
    Popping here would blank that out.

    The cost is that a stopped engine's candle buffers stay reachable until the
    same owner starts a new one — bounded by user count, and those buffers are
    capped at 600 rows per instrument (data_feed.py), so it is small and does
    not grow. Releasing it belongs in TradingEngine.stop(), not here.

    Stops EVERY engine the owner is running, so one Stop Bot halts the whole
    strategy board rather than just its first group. Each stop is isolated:
    one group's broker or feed throwing must not leave the rest running.
    """
    with _lock:
        engines = list(_engines.get(owner) or [])
    if not engines:
        return False
    for engine in engines:
        try:
            engine.stop()
        except Exception as exc:                  # pragma: no cover
            print(f"[engine_registry] {owner}: a group failed to stop: {exc}")
    return True
