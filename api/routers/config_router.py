"""
api/routers/config_router.py
Read-only lookups the sidebar needs before Start Bot: the instrument
universe, which strategies exist per mode, and watchlists. All of it is a
thin read over config.py / strategy.py / watchlists.py — no new logic.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends

import admin_config
import config
import strategy
import watchlists
from api.auth import get_current_user
from api.schemas import WatchlistSaveRequest
from config import Mode, Segment

router = APIRouter(prefix="/config", tags=["config"], dependencies=[Depends(get_current_user)])


@router.get("/instruments")
def list_instruments():
    return [
        {
            "symbol": i.symbol,
            "segment": i.segment.value,
            "lot_size": i.lot_size,
            "tick_size": i.tick_size,
            "reference_price": i.reference_price,
            "contract_multiplier": i.contract_multiplier,
        }
        for i in config.ALL_INSTRUMENTS
    ]


@router.get("/segments")
def list_segments():
    return [{"key": s.value, "label": "NSE Equity" if s == Segment.EQUITY
             else "MCX Commodity"} for s in Segment]


@router.get("/strategies")
def list_strategies(mode: str):
    try:
        mode_enum = Mode(mode)
    except ValueError:
        return []
    default_key = strategy.default_strategy(mode_enum).key
    out = []
    for s in strategy.strategies_for_mode(mode_enum):
        p = s.params
        out.append({
            "key": s.key,
            "name": s.name,
            "summary": s.summary,
            "is_default": s.key == default_key,
            "params": {
                "timeframe": p.timeframe,
                "risk_per_trade": p.risk_per_trade,
                "risk_reward": p.risk_reward,
                "atr_sl_mult": p.atr_sl_mult,
                "atr_period": p.atr_period,
                "allow_short": p.allow_short,
                "max_hold_minutes": p.max_hold_minutes,
            },
        })
    return out


@router.get("/client-modes")
def list_client_modes():
    """The phases a client may run, for their Start Bot picker.

    Deliberately does NOT name the strategy behind a phase: which edge admin
    runs is not a client's to see, and omitting it here (rather than hiding it
    in the UI) keeps it out of the response body too. What's left is what a
    client legitimately needs — which phase, how much it trades, and the
    risk/reward they're taking. The authoritative check still happens in
    /bot/start."""
    out = []
    for mode_name in admin_config.available_client_modes():
        mode_cfg = admin_config.get_mode_config(mode_name)
        bound = strategy.resolve_strategy(Mode(mode_name), mode_cfg.strategy_key)
        out.append({
            "key": mode_name,
            "label": admin_config.MODE_LABELS.get(mode_name, mode_name),
            "risk_reward": bound.params.risk_reward,
            "instrument_count": len(mode_cfg.symbols),
        })
    return out


@router.get("/watchlists")
def list_watchlists():
    return {name: watchlists.get(name) for name in watchlists.names()}


@router.post("/watchlists")
def save_watchlist(req: WatchlistSaveRequest):
    ok = watchlists.save(req.name, req.symbols)
    return {"ok": ok}


@router.delete("/watchlists/{name}")
def delete_watchlist(name: str):
    ok = watchlists.delete(name)
    return {"ok": ok}
