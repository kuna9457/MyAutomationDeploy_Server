"""
symbol_config.py
PER-SYMBOL overrides layered on top of a mode's strategy — the "⚙️ per stock"
settings in the sidebar. Three things can be tuned for one instrument without
touching the strategy that trades it:

  1. WHICH DAYS it may open new trades on (weekday filter).
  2. WHAT TIME WINDOW within the session it may open new trades in — the
     "limit losses by not trading the chop" control — plus an optional
     square-off when that window ends.
  3. ITS OWN RISK:REWARD, chosen from config.RR_CHOICES.

Design contract, and the whole reason this module is separate from strategy.py:

  * OPT-IN. A symbol with no saved entry, or one saved at every default, is a
    NO-OP — `rules_for()` never even returns an entry for it, so the engine's
    hot path is a dict miss and behaves exactly as it did before this existed.
  * ENTRY-SIDE ONLY. Nothing here can suppress the management of a position
    that is already open: stop-loss, target and time-exit are untouched. The
    window gates whether a NEW position may be opened (plus the explicit,
    default-off square_off_at_end).
  * RISK IS UNTOUCHABLE. risk_reward moves the TARGET only. Position size is
    risk_budget / stop_distance and never reads RR (Immutable Rule #1), so a
    per-symbol RR can never widen risk per trade. The value is validated
    against config.RR_CHOICES, so an arbitrary ratio cannot be stored.

Same persisted-JSON pattern as admin_config.py / watchlists.py, and stored PER
MODE for the same reason admin_config is: a 1-minute Scalper and a 15-minute
Intraday run on the same ticker want different windows. It imports only
`config`, never strategy/broker/engine code.
"""
from __future__ import annotations

import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, time
from typing import Optional

import config
import config_store

#: Storage key (was the filename symbol_configs.json — kept identical so an
#: existing local file is migrated rather than orphaned).
_KEY = "symbol_configs"
_lock = threading.Lock()

#: Weekday numbers as Python's datetime.weekday() reports them.
MONDAY, SUNDAY = 0, 6
WEEKDAY_LABELS: tuple[str, ...] = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")


# --------------------------------------------------------------------------- #
#  Stored shape
# --------------------------------------------------------------------------- #
@dataclass
class SymbolConfig:
    """One symbol's overrides, exactly as persisted. EVERY field's default is
    the "not configured" value, so an all-defaults instance is a no-op."""
    #: Weekdays new entries are allowed on, 0=Mon .. 6=Sun. EMPTY = no filter
    #: (every day the market is open), which is the pre-existing behaviour.
    trade_days: list[int] = field(default_factory=list)
    #: "HH:MM" IST. Empty = the segment's own open (config.market_hours_for_segment).
    start_time: str = ""
    #: "HH:MM" IST. Empty = the segment's own close.
    end_time: str = ""
    #: Reward per 1 unit of risk. 0.0 = inherit whatever the run already uses
    #: (the strategy's own RR, or admin's per-mode override).
    risk_reward: float = 0.0
    #: Close an OPEN position at market when the window above ends. Default
    #: False: without it this module only ever gates new entries, which is the
    #: conservative reading of "don't change how the bot behaves".
    square_off_at_end: bool = False

    def is_noop(self) -> bool:
        """True when this entry would change nothing — used to DELETE rather
        than store it, so 'reset to default' leaves no residue on disk."""
        return (not self.trade_days and not self.start_time and not self.end_time
                and not self.risk_reward and not self.square_off_at_end)


# --------------------------------------------------------------------------- #
#  Parsing / validation. Raises ValueError with a message fit for an HTTP 400.
# --------------------------------------------------------------------------- #
def parse_hhmm(value: str) -> Optional[time]:
    """"HH:MM" -> time, or None for an empty string (= 'use the segment's own')."""
    text = str(value or "").strip()
    if not text:
        return None
    try:
        hh, mm = text.split(":")
        parsed = time(int(hh), int(mm))
    except Exception:
        raise ValueError(f"{value!r} is not a valid time — use HH:MM (24-hour).")
    return parsed


def validate(cfg: SymbolConfig) -> SymbolConfig:
    """Return a cleaned copy, or raise ValueError. Called by the API route
    BEFORE anything is written, so the stored file is always well-formed and
    the engine can parse it without defensive branching."""
    days = sorted({int(d) for d in cfg.trade_days})
    bad = [d for d in days if d < MONDAY or d > SUNDAY]
    if bad:
        raise ValueError(
            f"Invalid weekday(s) {bad} — use 0 (Mon) through 6 (Sun).")

    start = parse_hhmm(cfg.start_time)
    end = parse_hhmm(cfg.end_time)
    # A window that ends before it starts would silently never open, so it is
    # rejected rather than saved. Sessions here never wrap past midnight —
    # even MCX closes at 23:30 (config.MCX_HOURS).
    if start is not None and end is not None and end <= start:
        raise ValueError(
            f"End time {cfg.end_time} must be after start time {cfg.start_time}.")

    rr = float(cfg.risk_reward or 0.0)
    if not config.is_valid_rr(rr):
        raise ValueError(
            f"risk_reward {rr:g} is not offered. Pick one of "
            f"{', '.join(config.rr_label(c) for c in config.RR_CHOICES)}, "
            f"or 0 to inherit.")

    return SymbolConfig(
        trade_days=days,
        start_time=start.strftime("%H:%M") if start else "",
        end_time=end.strftime("%H:%M") if end else "",
        risk_reward=rr,
        square_off_at_end=bool(cfg.square_off_at_end),
    )


# --------------------------------------------------------------------------- #
#  The resolved form the engine actually reads
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SymbolRules:
    """A validated SymbolConfig parsed once, at engine construction, into the
    types the trading loop compares against. Frozen and self-contained so the
    loop never re-parses strings or touches the filesystem per tick."""
    symbol: str
    trade_days: frozenset[int]
    start_t: Optional[time]
    end_t: Optional[time]
    risk_reward: float
    square_off_at_end: bool

    def day_allowed(self, when: datetime) -> bool:
        """No configured days = every day allowed (the unfiltered default)."""
        return not self.trade_days or when.weekday() in self.trade_days

    def window_open(self, now_t: time) -> bool:
        """Inside the configured intraday window. An unset bound means 'no
        limit on that side', so a start-only config runs to the session close
        and an end-only config runs from the session open."""
        if self.start_t is not None and now_t < self.start_t:
            return False
        if self.end_t is not None and now_t > self.end_t:
            return False
        return True

    def entry_block_reason(self, when: datetime) -> str:
        """Why a NEW entry is not allowed right now, or "" when it is. The
        string is what gets logged, so it names the actual configured limit."""
        if not self.day_allowed(when):
            allowed = ", ".join(WEEKDAY_LABELS[d] for d in sorted(self.trade_days))
            return (f"{WEEKDAY_LABELS[when.weekday()]} is not a configured "
                    f"trading day (allowed: {allowed})")
        if not self.window_open(when.time()):
            return (f"outside its trading window "
                    f"{self.start_t.strftime('%H:%M') if self.start_t else 'open'}"
                    f"–{self.end_t.strftime('%H:%M') if self.end_t else 'close'}")
        return ""

    def describe(self) -> str:
        """One-line summary for the start-up log, so the running bot's log
        states exactly which symbols deviate from the plain strategy."""
        bits = []
        if self.trade_days:
            bits.append("days " + "/".join(WEEKDAY_LABELS[d]
                                           for d in sorted(self.trade_days)))
        if self.start_t or self.end_t:
            bits.append(
                f"{self.start_t.strftime('%H:%M') if self.start_t else 'open'}"
                f"–{self.end_t.strftime('%H:%M') if self.end_t else 'close'}"
                + (" (square off at end)" if self.square_off_at_end else ""))
        if self.risk_reward > 0:
            bits.append(f"RR {config.rr_label(self.risk_reward)}")
        return f"{self.symbol}: " + ", ".join(bits)


def to_rules(symbol: str, cfg: SymbolConfig) -> Optional[SymbolRules]:
    """Resolve one stored config, or None when it would change nothing.

    Returning None for a no-op is the mechanism that keeps this feature
    invisible when unused: the engine keys a dict on the symbols that DO have
    rules, so an unconfigured symbol costs one dict miss and takes the
    original code path unchanged.
    """
    if cfg.is_noop():
        return None
    return SymbolRules(
        symbol=symbol,
        trade_days=frozenset(int(d) for d in cfg.trade_days),
        start_t=parse_hhmm(cfg.start_time),
        end_t=parse_hhmm(cfg.end_time),
        risk_reward=float(cfg.risk_reward or 0.0),
        square_off_at_end=bool(cfg.square_off_at_end),
    )


# --------------------------------------------------------------------------- #
#  Persistence — {mode: {symbol: SymbolConfig}}
# --------------------------------------------------------------------------- #
def _from_disk() -> dict[str, dict[str, SymbolConfig]]:
    """Everything on disk. Never raises: a missing or corrupt file reads as
    'nothing configured', which is precisely the safe default here."""
    data = config_store.load(_KEY)
    if not isinstance(data, dict):
        return {}
    out: dict[str, dict[str, SymbolConfig]] = {}
    for mode, entries in data.items():
        if not isinstance(entries, dict):
            continue
        by_symbol: dict[str, SymbolConfig] = {}
        for symbol, raw in entries.items():
            if not isinstance(raw, dict):
                continue
            by_symbol[str(symbol)] = SymbolConfig(
                **{k: v for k, v in raw.items()
                   if k in SymbolConfig.__dataclass_fields__})
        if by_symbol:
            out[str(mode)] = by_symbol
    return out


def _to_disk(data: dict[str, dict[str, SymbolConfig]]) -> None:
    config_store.save(_KEY, {
        mode: {sym: asdict(cfg) for sym, cfg in entries.items()}
        for mode, entries in data.items()})


def get_all(mode: str) -> dict[str, SymbolConfig]:
    """Every symbol configured for this mode, as stored."""
    return _from_disk().get(str(mode), {})


def get_symbol(mode: str, symbol: str) -> SymbolConfig:
    """One symbol's config, or an all-defaults (no-op) one if never saved."""
    return get_all(mode).get(str(symbol)) or SymbolConfig()


def set_symbol(mode: str, symbol: str, cfg: SymbolConfig) -> SymbolConfig:
    """Save one symbol's overrides for one mode, leaving every other symbol
    and every other mode untouched. A config that is entirely defaults is
    DELETED instead of written, so "reset to default" and "never configured"
    are the same state on disk — there is no half-configured third case for
    the engine to interpret.

    `cfg` must already have been through validate()."""
    mode, symbol = str(mode), str(symbol)
    with _lock:
        data = _from_disk()
        entries = data.get(mode, {})
        if cfg.is_noop():
            entries.pop(symbol, None)
        else:
            entries[symbol] = cfg
        if entries:
            data[mode] = entries
        else:
            data.pop(mode, None)
        _to_disk(data)
    return cfg


def delete_symbol(mode: str, symbol: str) -> bool:
    """Drop one symbol's overrides (back to plain strategy behaviour).
    Returns True if something was actually removed."""
    mode, symbol = str(mode), str(symbol)
    with _lock:
        data = _from_disk()
        entries = data.get(mode, {})
        if symbol not in entries:
            return False
        entries.pop(symbol)
        if entries:
            data[mode] = entries
        else:
            data.pop(mode, None)
        _to_disk(data)
    return True


def replace_mode(mode: str, configs: dict[str, SymbolConfig]) -> dict[str, SymbolConfig]:
    """REPLACE every per-symbol setting under one mode with `configs`.

    Unlike set_symbol (which edits one symbol and leaves the rest alone), this
    is a wholesale swap: symbols absent from `configs` have their settings
    REMOVED. That is what loading a saved preset needs — a preset restores the
    setup you saved, so a symbol you had customised afterwards must not linger
    on top of it and quietly change what trades.

    No-op entries are dropped rather than stored, keeping the same
    "not configured == absent" invariant the rest of this module relies on.
    Every value must already have been through validate().
    """
    mode = str(mode)
    clean = {str(sym): cfg for sym, cfg in configs.items() if not cfg.is_noop()}
    with _lock:
        data = _from_disk()
        if clean:
            data[mode] = clean
        else:
            data.pop(mode, None)
        _to_disk(data)
    return clean


def rules_for(mode: str, symbols: Optional[list[str]] = None
              ) -> dict[str, SymbolRules]:
    """The resolved rules the engine is constructed with: symbol -> SymbolRules,
    containing ONLY symbols whose config actually changes something.

    `symbols` narrows the result to the instruments this run trades; omit it
    for everything saved under the mode. A corrupt individual entry is skipped
    rather than raised — one bad row must not stop the bot from starting.
    """
    wanted = set(symbols) if symbols is not None else None
    out: dict[str, SymbolRules] = {}
    for symbol, cfg in get_all(mode).items():
        if wanted is not None and symbol not in wanted:
            continue
        try:
            rules = to_rules(symbol, cfg)
        except Exception:
            continue
        if rules is not None:
            out[symbol] = rules
    return out
