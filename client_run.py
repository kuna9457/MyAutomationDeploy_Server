"""
client_run.py
The ONE place that says what a client account trades.

    on the admin HUB (and any ordinary single server)
        -> admin_config / symbol_config, exactly as before this module existed.
    on a fleet NODE (NODE_MODE=worker)
        -> the broadcast the hub sent, held IN MEMORY by hub_link.

Why a node must not use admin_config
------------------------------------
The fleet shares ONE database, so `admin_config` and `symbol_config` on a node
are the ADMIN's own saved settings. The first version of the node wrote the
broadcast into them to reuse the client start path — which, on a shared
database, overwrote the admin's saved client defaults and (symbol_config
.replace_mode) DELETED the admin's per-stock settings for every symbol outside
the broadcast. Reading through here instead means a node never writes either.

Callers: api/routers/bot.py (client start, auto-start, platform signals) and
api/routers/config_router.py (/config/client-modes).
"""
from __future__ import annotations

from typing import Optional

import admin_config
import config
import symbol_config


def _link():
    import hub_link
    return hub_link.link() if hub_link.is_worker() else None


def active_mode() -> str:
    """THE mode clients trade right now, or "" when nothing is configured /
    no broadcast is running."""
    link = _link()
    if link is None:
        return admin_config.active_client_mode()
    run = link.current_run()
    return run.mode if run is not None else ""


def available_modes() -> list[str]:
    link = _link()
    if link is None:
        return admin_config.available_client_modes()
    mode = active_mode()
    return [mode] if mode else []


def mode_config(mode: str) -> admin_config.ModeConfig:
    """The run for `mode`, in the ModeConfig shape every caller already reads."""
    link = _link()
    if link is None:
        return admin_config.get_mode_config(mode)
    run = link.current_run()
    if run is None or run.mode != mode:
        return admin_config.ModeConfig()
    instruments = [config.INSTRUMENTS_BY_SYMBOL[s] for s in run.symbols
                   if s in config.INSTRUMENTS_BY_SYMBOL]
    return admin_config.ModeConfig(
        strategy_key=run.strategy_key,
        segments=sorted({i.segment.value for i in instruments}),
        symbols=[i.symbol for i in instruments],
        mcx_lots=dict(run.mcx_lots), risk_reward=run.risk_reward,
        min_score=run.min_score, square_off_time=run.square_off_time,
        square_off_enabled=run.square_off_enabled, exit_style=run.exit_style,
        trail_atr_mult=run.trail_atr_mult, max_stop_pct=run.max_stop_pct,
        min_stop_pct=run.min_stop_pct)


def rules_for(mode: str, symbols: Optional[list[str]] = None) -> dict:
    """Per-symbol rules for a CLIENT run. On a node these are the admin's
    settings as they were when the broadcast started (sent in the session
    frame) — the same ones the hub's runner is deciding with."""
    link = _link()
    if link is None:
        return symbol_config.rules_for(mode, symbols)
    wanted = set(symbols) if symbols is not None else None
    return {sym: r for sym, r in link.current_rules().items()
            if wanted is None or sym in wanted}
