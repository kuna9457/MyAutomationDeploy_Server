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

import threading
from dataclasses import asdict, dataclass, field

import config_store

#: Storage key (was the filename admin_bot_config.json — kept identical so an
#: existing deployment's file is picked up and migrated, not orphaned).
_KEY = "admin_bot_config"
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
    #: Risk:reward override for this mode, as reward per 1 unit of risk (see
    #: config.RR_CHOICES). 0.0 = inherit whatever the chosen strategy declares,
    #: which is what every pre-existing saved config reads as — so adding this
    #: field changes nothing until an admin actually picks a value.
    #:
    #: Overrides the TARGET only. Position size is risk_budget / stop_distance
    #: and never consults RR, so this cannot widen a client's risk per trade.
    risk_reward: float = 0.0
    #: Signal-score threshold for this mode (config.MIN_SCORE_MIN..MAX).
    #: 0.0 = inherit whatever the chosen strategy declares, which is what every
    #: pre-existing saved config reads as — so adding this field changes
    #: nothing until an admin picks a value.
    #:
    #: Moves ENTRY SELECTIVITY only: higher trades less often with more
    #: agreement behind each entry, lower trades more often on weaker
    #: evidence. It never touches position size, the risk cap or the stop.
    min_score: float = 0.0
    #: End-of-session flat-out for this mode. "" = the segment's own default
    #: (config.DEFAULT_SQUARE_OFF — 15:09 equity, 23:15 MCX). "HH:MM" IST to
    #: override. Ignored entirely for Swing, which holds overnight by design.
    square_off_time: str = ""
    #: Master switch. ON by default: an intraday position left open past the
    #: close is either auto-squared by the broker at whatever the auction
    #: prints, or becomes an unfunded delivery. Turning this off means taking
    #: that on yourself.
    square_off_enabled: bool = True


@dataclass
class BotConfig:
    #: mode name -> that mode's instruments/strategy. Only modes the admin has
    #: actually saved appear here.
    by_mode: dict[str, ModeConfig] = field(default_factory=dict)
    #: which of CLIENT_SELECTABLE_MODES the admin currently offers to clients.
    client_modes: list[str] = field(default_factory=list)
    #: Start every eligible client's bot when ADMIN starts theirs, and publish
    #: the run admin just started as the config clients follow — so the setup
    #: never has to be saved for clients as a separate step.
    #:
    #: Only ever acts on a mode in CLIENT_SELECTABLE_MODES: an admin running
    #: Swing (or any mode clients cannot select) publishes nothing and starts
    #: nobody, so experimenting outside the client modes cannot overwrite what
    #: clients are trading.
    auto_start_clients: bool = True


def _from_disk() -> BotConfig:
    data = config_store.load(_KEY)
    if not data:
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
    # Absent in every file written before this field existed. Defaulting those
    # to False (rather than the dataclass default) keeps an existing
    # deployment's behaviour exactly as it was — nobody's clients start
    # trading because the server was upgraded. New installs get True.
    auto_start = bool(data.get("auto_start_clients", False))
    return BotConfig(by_mode=by_mode, client_modes=client_modes,
                     auto_start_clients=auto_start)


def _to_disk(cfg: BotConfig) -> None:
    config_store.save(_KEY, asdict(cfg))


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


def auto_start_clients() -> bool:
    """Whether admin's Start Bot also publishes its config to clients and
    starts them. See BotConfig.auto_start_clients."""
    return bool(get_config().auto_start_clients)


def set_auto_start_clients(enabled: bool) -> BotConfig:
    with _lock:
        cfg = _from_disk()
        cfg.auto_start_clients = bool(enabled)
        _to_disk(cfg)
        return cfg


def publish_run(mode: str, **kwargs) -> bool:
    """Make the run admin just started the one clients follow: save it as this
    mode's config AND make sure the mode is offered to clients.

    Returns False without writing anything when `mode` is not client-
    selectable — the guard that stops an admin's Swing (or any non-client)
    run from silently replacing what clients trade.

    This is exactly what "save as client default" did as a separate button;
    doing it here means the setup can never drift from the run that is
    actually happening.
    """
    if mode not in CLIENT_SELECTABLE_MODES:
        return False
    set_mode_config(mode, **kwargs)
    with _lock:
        cfg = _from_disk()
        # active_client_mode() reads the FIRST entry, so the published mode
        # goes to the front — starting Scalper after Intraday must move
        # clients onto Scalper, not leave them on the older one.
        cfg.client_modes = [mode] + [m for m in cfg.client_modes if m != mode]
        _to_disk(cfg)
    return True


def is_set(mode: str) -> bool:
    """True once admin has picked at least one instrument for this mode — the
    signal a client's Start Bot uses to tell "not configured yet" apart from a
    legitimate empty-by-choice state."""
    return len(get_mode_config(mode).symbols) > 0


def active_client_mode() -> str:
    """THE mode clients trade, or "" if admin hasn't configured one.

    Clients do not choose what to trade — admin does, and everything about
    the trade (mode, strategy, instruments, risk:reward, per-symbol windows)
    comes from admin's saved config. A client's account only decides HOW BIG
    the trade is for them and pushes it to their own broker.

    `client_modes` remains a list on disk for backward compatibility, but only
    the first configured entry is used; the admin UI offers a single choice.
    """
    modes = available_client_modes()
    return modes[0] if modes else ""


def available_client_modes() -> list[str]:
    """Modes a client can actually start right now: enabled by admin AND
    carrying instruments. A mode that's enabled but never configured is
    omitted rather than offered and then failing at Start Bot."""
    cfg = get_config()
    return [m for m in CLIENT_SELECTABLE_MODES
            if m in cfg.client_modes
            and (cfg.by_mode.get(m) or ModeConfig()).symbols]
