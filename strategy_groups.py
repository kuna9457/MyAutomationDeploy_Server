"""
strategy_groups.py
Which STRATEGY trades which STOCKS — the drag-and-drop board's storage.

A "group" is one strategy plus the instruments dropped onto it. Several groups
run at once, each as its own TradingEngine (one engine, one strategy — exactly
the shape that already works), coordinated by capital_ledger so they share one
wallet and never both open the same stock.

Design contract:

  * OPT-IN, and invisible when unused. `enabled_groups(mode)` returns [] when
    nothing is configured, and bot.py then takes the original single-strategy
    start path unchanged. Turning this feature on is the ONLY way to reach the
    multi-engine code at all.
  * A SYMBOL MAY APPEAR IN SEVERAL GROUPS. That is the point — the same stock
    can be watched by two strategies. Only one POSITION per symbol can exist
    at a time, and that is enforced at execution (capital_ledger's symbol
    lock + engine.holds), never here. Storage stays a plain description of
    intent.
  * ONE STRATEGY PER GROUP, and a strategy may appear only once per mode —
    two groups on the same strategy would be one group with a longer symbol
    list, and allowing it would make "which engine owns this signal"
    ambiguous for no gain.
  * Stored PER MODE, like admin_config and symbol_config: a strategy is bound
    to a mode (strategy.py's registry), so a 1-minute Scalper board and a
    15-minute Intraday board are different boards.

Imports only `config` and `config_store` — never engine/broker/strategy code.
"""
from __future__ import annotations

import threading
from dataclasses import asdict, dataclass, field

import config
import config_store

_KEY = "strategy_groups"
_lock = threading.Lock()


@dataclass
class StrategyGroup:
    """One strategy and the instruments dropped onto it."""
    strategy_key: str = ""
    symbols: list[str] = field(default_factory=list)
    #: Lots per MCX symbol in THIS group, mirroring the sidebar's own map.
    #: Absent symbols default to 1 (engine._mcx_fixed_size).
    mcx_lots: dict[str, int] = field(default_factory=dict)
    #: Risk:reward for this group. 0.0 = the strategy's own (config.RR_CHOICES).
    risk_reward: float = 0.0
    #: Signal-score threshold for this group. 0.0 = the strategy's own.
    min_score: float = 0.0
    #: Off keeps the group's stocks on the board without trading them — the
    #: way to park a strategy without losing the list you built.
    enabled: bool = True

    def is_runnable(self) -> bool:
        return bool(self.enabled and self.strategy_key and self.symbols)


def _from_disk() -> dict[str, list[StrategyGroup]]:
    """Everything on disk. Never raises: a missing or corrupt document reads as
    'no groups', which correctly falls back to single-strategy behaviour."""
    data = config_store.load(_KEY)
    if not isinstance(data, dict):
        return {}
    out: dict[str, list[StrategyGroup]] = {}
    for mode, raw_groups in data.items():
        if not isinstance(raw_groups, list):
            continue
        groups: list[StrategyGroup] = []
        for raw in raw_groups:
            if not isinstance(raw, dict):
                continue
            groups.append(StrategyGroup(
                **{k: v for k, v in raw.items()
                   if k in StrategyGroup.__dataclass_fields__}))
        if groups:
            out[str(mode)] = groups
    return out


def _to_disk(data: dict[str, list[StrategyGroup]]) -> None:
    config_store.save(_KEY, {mode: [asdict(g) for g in groups]
                             for mode, groups in data.items()})


def get_all(mode: str) -> list[StrategyGroup]:
    """Every group saved under this mode, in board order."""
    return _from_disk().get(str(mode), [])


def enabled_groups(mode: str) -> list[StrategyGroup]:
    """The groups that would actually run — enabled, with a strategy and at
    least one instrument. An empty list is the signal bot.py uses to take the
    original single-strategy path, so a half-built board never half-starts."""
    return [g for g in get_all(mode) if g.is_runnable()]


def validate(groups: list[StrategyGroup], mode: str) -> list[StrategyGroup]:
    """Clean a whole board, or raise ValueError with an HTTP-400-worthy
    message. Called by the route BEFORE anything is written, so what lands on
    disk is always startable."""
    seen: set[str] = set()
    clean: list[StrategyGroup] = []
    for g in groups:
        key = str(g.strategy_key or "").strip()
        if not key:
            raise ValueError("Every group needs a strategy.")
        if key in seen:
            raise ValueError(
                f"{key} appears twice — put all of its stocks in one group.")
        seen.add(key)

        symbols, unknown = [], []
        for s in g.symbols:
            (symbols if s in config.INSTRUMENTS_BY_SYMBOL else unknown).append(s)
        if unknown:
            raise ValueError(f"Unknown instrument(s): {', '.join(unknown)}.")
        # De-duplicate within a group while keeping drop order. Across groups
        # duplicates are legal and deliberate — see the module docstring.
        symbols = list(dict.fromkeys(symbols))

        rr = float(g.risk_reward or 0.0)
        if not config.is_valid_rr(rr):
            raise ValueError(
                f"risk_reward {rr:g} is not offered. Pick one of "
                f"{', '.join(config.rr_label(c) for c in config.RR_CHOICES)}, "
                f"or 0 to use the strategy's own.")
        score = float(g.min_score or 0.0)
        if not config.is_valid_min_score(score):
            raise ValueError(
                f"min_score {score:g} is out of range. Use "
                f"{config.MIN_SCORE_MIN:g}-{config.MIN_SCORE_MAX:g}, or 0 to "
                f"use the strategy's own.")

        clean.append(StrategyGroup(
            strategy_key=key, symbols=symbols,
            # Lots only for symbols still in the group, so removing a stock
            # cannot leave a stale lot count behind to reappear later.
            mcx_lots={s: int(n) for s, n in (g.mcx_lots or {}).items()
                      if s in symbols},
            risk_reward=rr, min_score=score, enabled=bool(g.enabled)))
    return clean


def replace_mode(mode: str, groups: list[StrategyGroup]) -> list[StrategyGroup]:
    """REPLACE this mode's whole board. The board is edited as one document in
    the UI (drag a stock from one strategy to another and two groups change at
    once), so a wholesale swap is the only write that cannot leave the two
    halves of a move inconsistent. Values must already have been validated."""
    mode = str(mode)
    with _lock:
        data = _from_disk()
        if groups:
            data[mode] = list(groups)
        else:
            data.pop(mode, None)
        _to_disk(data)
    return list(groups)


def symbols_for(mode: str) -> list[str]:
    """Every distinct instrument the board trades under this mode, in board
    order. Used for the shared symbol lock and for pre-flight checks."""
    seen: list[str] = []
    for g in enabled_groups(mode):
        for s in g.symbols:
            if s not in seen:
                seen.append(s)
    return seen
