"""
admin_config.py
The strategy/instrument selection admin sets centrally for clients to trade
(frontend_migration_plan.md §3/§5). Clients never pick a strategy, stock or
exchange segment — but they DO pick which trading mode to run, from the set
admin has enabled here.

Because a strategy is bound to a mode (`scalp_vwap_atr` only supports
Mode.SCALPER, `intraday_vwap_macd` only Mode.INTRADAY — see strategy.py's
registry), one flat strategy_key/symbols pair cannot serve two modes. So the
config is stored PER MODE: admin configures Intraday, configures Scalper, and
enables either or both for clients. A client's Start Bot names one of the
enabled modes and everything else is resolved from that mode's entry.

Same persisted-JSON pattern as risk_manager.py/watchlists.py. Admin's own
Start Bot is unaffected — this module is only consulted for role="client".
"""
from __future__ import annotations

import json
import os
import threading
from dataclasses import asdict, dataclass, field

import config

_PATH = os.path.join(config.LOCAL_DB_DIR, "admin_bot_config.json")
_lock = threading.Lock()

#: Modes a client is ever allowed to run. Swing is deliberately excluded — it
#: holds positions across days, which the client surface isn't built for.
#: Admin still trades every mode in config.Mode from their own sidebar.
CLIENT_SELECTABLE_MODES: tuple[str, ...] = ("Intraday", "Scalper")

MODE_LABELS: dict[str, str] = {
    "Intraday": "Intraday (15-minute)",
    "Scalper": "Scalping (1-minute)",
}


@dataclass
class ModeConfig:
    """What a client trades when they pick this one mode."""
    strategy_key: str = ""
    segments: list[str] = field(default_factory=lambda: ["NSE_EQUITY"])
    symbols: list[str] = field(default_factory=list)
    mcx_lots: dict[str, int] = field(default_factory=dict)


@dataclass
class BotConfig:
    #: mode name -> that mode's instruments/strategy. Only modes the admin has
    #: actually saved appear here.
    by_mode: dict[str, ModeConfig] = field(default_factory=dict)
    #: which of CLIENT_SELECTABLE_MODES the admin currently offers to clients.
    client_modes: list[str] = field(default_factory=list)


def _from_disk() -> BotConfig:
    try:
        with open(_PATH, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception:
        return BotConfig()

    # Pre-per-mode files stored one flat {mode, strategy_key, segments,
    # symbols, mcx_lots}. Fold it into that mode's entry and enable exactly
    # that mode, so an existing deployment's clients keep trading precisely
    # what they traded before this change.
    if "by_mode" not in data:
        legacy_mode = data.get("mode") or "Intraday"
        cfg = BotConfig()
        cfg.by_mode[legacy_mode] = ModeConfig(
            strategy_key=data.get("strategy_key", ""),
            segments=data.get("segments") or ["NSE_EQUITY"],
            symbols=data.get("symbols") or [],
            mcx_lots=data.get("mcx_lots") or {},
        )
        if legacy_mode in CLIENT_SELECTABLE_MODES:
            cfg.client_modes = [legacy_mode]
        return cfg

    by_mode = {
        name: ModeConfig(**{k: v for k, v in (entry or {}).items()
                            if k in ModeConfig.__dataclass_fields__})
        for name, entry in (data.get("by_mode") or {}).items()
    }
    client_modes = [m for m in (data.get("client_modes") or [])
                    if m in CLIENT_SELECTABLE_MODES]
    return BotConfig(by_mode=by_mode, client_modes=client_modes)


def _to_disk(cfg: BotConfig) -> None:
    os.makedirs(config.LOCAL_DB_DIR, exist_ok=True)
    with open(_PATH, "w", encoding="utf-8") as fh:
        json.dump(asdict(cfg), fh, indent=2)


def get_config() -> BotConfig:
    return _from_disk()


def get_mode_config(mode: str) -> ModeConfig:
    """This mode's saved config, or an empty one if admin never saved it."""
    return get_config().by_mode.get(mode) or ModeConfig()


def set_mode_config(mode: str, **kwargs) -> BotConfig:
    """Save one mode's strategy/instruments, leaving every other mode alone."""
    with _lock:
        cfg = _from_disk()
        current = asdict(cfg.by_mode.get(mode) or ModeConfig())
        current.update({k: v for k, v in kwargs.items()
                        if k in ModeConfig.__dataclass_fields__})
        cfg.by_mode[mode] = ModeConfig(**current)
        _to_disk(cfg)
        return cfg


def set_client_modes(modes: list[str]) -> BotConfig:
    """Replace the set of modes clients may choose from. Unknown or non-client
    modes (e.g. Swing) are dropped rather than rejected."""
    with _lock:
        cfg = _from_disk()
        cfg.client_modes = [m for m in dict.fromkeys(modes)
                            if m in CLIENT_SELECTABLE_MODES]
        _to_disk(cfg)
        return cfg


def is_set(mode: str) -> bool:
    """True once admin has picked at least one instrument for this mode — the
    signal a client's Start Bot uses to tell "not configured yet" apart from a
    legitimate empty-by-choice state."""
    return len(get_mode_config(mode).symbols) > 0


def available_client_modes() -> list[str]:
    """Modes a client can actually start right now: enabled by admin AND
    carrying instruments. A mode that's enabled but never configured is
    omitted rather than offered and then failing at Start Bot."""
    cfg = get_config()
    return [m for m in CLIENT_SELECTABLE_MODES
            if m in cfg.client_modes
            and (cfg.by_mode.get(m) or ModeConfig()).symbols]
