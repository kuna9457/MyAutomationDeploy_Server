"""
presets.py
Named snapshots of the ENTIRE Controls sidebar — environment, mode, broker,
strategy, segments, instruments, capital, MCX lots, and the per-symbol
settings (symbol_config.py) that go with that mode. Save a setup you like
under a name, pick it later, and the whole panel comes back exactly as it was.

Why this rather than reusing watchlists.py: a watchlist is a bucket of symbols
and nothing else. What makes a setup reproducible is the COMBINATION — the
same stocks under a different strategy, capital or per-stock window is a
different setup entirely, and re-picking each piece by hand is where mistakes
creep in.

Two deliberate design points:

  * A preset is INERT until loaded. Saving one never touches what the bot is
    doing, and loading one never starts or stops anything — it only repopulates
    the controls (and restores that mode's per-symbol settings). You still press
    Start Bot yourself.
  * Loading REPLACES the mode's per-symbol settings rather than merging them
    (symbol_config.replace_mode). A preset is a snapshot of a whole setup, so a
    stock customised after the preset was saved must not survive the load and
    silently alter what trades.

Same persisted-JSON pattern as watchlists.py / admin_config.py, under the
gitignored local data dir. Imports only config and symbol_config — no engine,
broker or strategy code, so it cannot affect a running bot.
"""
from __future__ import annotations

import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

import config
import config_store
import symbol_config

#: Storage key (was the filename control_presets.json).
_KEY = "control_presets"
_lock = threading.Lock()

#: Guard against a name that would make the file unwieldy or the UI unreadable.
MAX_NAME_LENGTH = 60


@dataclass
class ControlPreset:
    """One saved Controls setup. Field-for-field what the sidebar holds, plus
    the per-symbol settings snapshot for `mode`."""
    environment: str = "Paper"
    mode: str = "Intraday"
    #: Only meaningful when environment == "Live"; kept either way so switching
    #: back to Live restores the broker you had picked.
    broker: str = ""
    strategy_key: str = ""
    segments: list[str] = field(default_factory=lambda: ["NSE_EQUITY"])
    symbols: list[str] = field(default_factory=list)
    capital: float = 100_000.0
    mcx_lots: dict[str, int] = field(default_factory=dict)
    #: symbol -> that symbol's settings, as they were for `mode` when saved.
    #: Stored as plain dicts so the file stays readable and this module needs
    #: no special decoding; validated on the way in and out.
    symbol_configs: dict[str, dict] = field(default_factory=dict)
    #: UTC ISO timestamp, shown in the picker so several similar presets can
    #: be told apart.
    saved_at: str = ""


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


# --------------------------------------------------------------------------- #
#  Validation. Raises ValueError with a message fit for an HTTP 400.
# --------------------------------------------------------------------------- #
def clean_name(name: str) -> str:
    text = str(name or "").strip()
    if not text:
        raise ValueError("Give the preset a name.")
    if len(text) > MAX_NAME_LENGTH:
        raise ValueError(f"Preset name must be {MAX_NAME_LENGTH} characters or fewer.")
    return text


def validate(preset: ControlPreset) -> ControlPreset:
    """Return a cleaned copy, or raise ValueError.

    Unknown SYMBOLS are dropped rather than rejected: the tradable universe is
    regenerated periodically (MCX futures roll at expiry, index membership
    changes), and a preset saved last month must still load rather than fail
    outright because one instrument no longer exists. Unknown MODES and
    ENVIRONMENTS are rejected — those are a fixed enum, so a bad one is a real
    error, not drift.
    """
    try:
        mode = config.Mode(preset.mode).value
    except ValueError:
        raise ValueError(
            f"Invalid mode {preset.mode!r}. Use one of "
            f"{', '.join(m.value for m in config.Mode)}.")
    try:
        environment = config.Environment(preset.environment).value
    except ValueError:
        raise ValueError(
            f"Invalid environment {preset.environment!r}. Use Paper or Live.")

    valid_segments = {s.value for s in config.Segment}
    segments = [s for s in dict.fromkeys(preset.segments) if s in valid_segments]
    if not segments:
        segments = [config.Segment.EQUITY.value]

    symbols = [s for s in dict.fromkeys(preset.symbols)
               if s in config.INSTRUMENTS_BY_SYMBOL]

    capital = float(preset.capital or 0.0)
    if capital <= 0:
        raise ValueError("Capital must be greater than zero.")

    # Each per-symbol entry goes through symbol_config's own validation, so a
    # preset can never carry a window or RR the live editor would have refused.
    symbol_configs: dict[str, dict] = {}
    for sym, raw in (preset.symbol_configs or {}).items():
        if sym not in config.INSTRUMENTS_BY_SYMBOL:
            continue                       # same drop-unknown rule as symbols
        cfg = symbol_config.validate(symbol_config.SymbolConfig(
            **{k: v for k, v in (raw or {}).items()
               if k in symbol_config.SymbolConfig.__dataclass_fields__}))
        if not cfg.is_noop():
            symbol_configs[sym] = asdict(cfg)

    return ControlPreset(
        environment=environment,
        mode=mode,
        broker=str(preset.broker or ""),
        strategy_key=str(preset.strategy_key or ""),
        segments=segments,
        symbols=symbols,
        capital=capital,
        mcx_lots={str(k): int(v) for k, v in (preset.mcx_lots or {}).items()},
        symbol_configs=symbol_configs,
        saved_at=preset.saved_at or _now_iso(),
    )


# --------------------------------------------------------------------------- #
#  Persistence — {name: ControlPreset}
# --------------------------------------------------------------------------- #
def load_all() -> dict[str, ControlPreset]:
    """Every saved preset. Never raises — a missing or corrupt file reads as
    'no presets', so the sidebar always renders."""
    data = config_store.load(_KEY)
    if not isinstance(data, dict):
        return {}
    out: dict[str, ControlPreset] = {}
    for name, raw in data.items():
        if not isinstance(raw, dict):
            continue
        out[str(name)] = ControlPreset(
            **{k: v for k, v in raw.items()
               if k in ControlPreset.__dataclass_fields__})
    return out


def _write(data: dict[str, ControlPreset]) -> None:
    config_store.save(_KEY, {name: asdict(p) for name, p in data.items()})


def get(name: str) -> ControlPreset | None:
    return load_all().get(str(name))


def save(name: str, preset: ControlPreset) -> ControlPreset:
    """Create or overwrite a preset. `preset` must already have been
    validated. The save time is stamped here, not by the caller, so it always
    reflects the actual write."""
    name = clean_name(name)
    preset.saved_at = _now_iso()
    with _lock:
        data = load_all()
        data[name] = preset
        _write(data)
    return preset


def delete(name: str) -> bool:
    """Remove a preset. Returns True if it existed."""
    name = str(name)
    with _lock:
        data = load_all()
        if name not in data:
            return False
        data.pop(name)
        _write(data)
    return True


def apply(name: str) -> ControlPreset | None:
    """Load a preset: restore its per-symbol settings for its own mode, and
    return it so the caller can repopulate the rest of the controls.

    This is the ONLY part of loading with a server-side effect. It REPLACES
    that mode's settings (see symbol_config.replace_mode) so what you get is
    the setup you saved, not the saved one merged with whatever was customised
    since. Other modes are untouched, and no bot is started or stopped.
    """
    preset = get(name)
    if preset is None:
        return None
    configs = {
        sym: symbol_config.SymbolConfig(
            **{k: v for k, v in (raw or {}).items()
               if k in symbol_config.SymbolConfig.__dataclass_fields__})
        for sym, raw in (preset.symbol_configs or {}).items()
    }
    symbol_config.replace_mode(preset.mode, configs)
    return preset
