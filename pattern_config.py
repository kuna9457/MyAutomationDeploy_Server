"""
pattern_config.py
WHICH candlestick patterns are allowed to open a trade.

The candlestick engine recognises 49 patterns and treats them all as evidence.
Measured against a real book, some earn money and some only pay brokerage. This
is the switch that lets an operator trade the ones that work.

DESIGN CONTRACT — the important half:

  * OFF BY DEFAULT AND INERT. `enabled=False`, or an empty allow-list, means
    `filter_hits()` returns the hits it was handed, unchanged, and the strategy
    takes exactly the code path it did before this module existed. There is a
    test asserting an unconfigured system reproduces the identical backtest.
  * EVIDENCE-SIDE ONLY. Filtering happens between detect_patterns() and
    scoring. It can only ever REMOVE evidence, so it can make the bot trade
    less, never more, and never differently-sized: position size is
    risk_budget / stop_distance and never reads any of this.
  * PER STRATEGY, PER MODE. A pattern that works on 15-minute bars need not
    work on 1-minute ones, and the two candlestick strategies score
    differently, so each combination is configured on its own.
  * NAMES ARE MATCHED, NOT IDS. A name that no pattern emits simply never
    matches — which is why the UI shows which names have actually been seen in
    your own trades, and warns about ones that have not.

Same persisted-JSON pattern as symbol_config.py / strategy_groups.py. Imports
nothing from strategy, engine or broker code.
"""
from __future__ import annotations

import threading
import time
from dataclasses import asdict, dataclass, field

import config_store

_KEY = "pattern_config"
_lock = threading.Lock()

#: filter_hits() runs on EVERY bar for EVERY instrument — twice a second per
#: symbol on the Scalper — so it must never touch storage on the hot path. The
#: whole document is small, so it is cached wholesale and re-read at most once
#: per TTL. set_rules() clears the cache, making an edit from the dashboard take
#: effect on the next tick; the TTL exists only so a write from ANOTHER process
#: is eventually picked up too.
_CACHE_TTL_SECONDS = 5.0
_cache: tuple[float, dict] = (0.0, {})
_cache_lock = threading.Lock()


def _cached() -> dict[str, dict[str, "PatternRules"]]:
    global _cache
    now = time.monotonic()
    with _cache_lock:
        ts, data = _cache
        if now - ts < _CACHE_TTL_SECONDS:
            return data
    fresh = _from_disk()
    with _cache_lock:
        _cache = (now, fresh)
    return fresh


def _invalidate() -> None:
    global _cache
    with _cache_lock:
        _cache = (0.0, {})

#: Every pattern candlestick_engine can emit, for the picker. This is a UI
#: CONVENIENCE ONLY — the filter matches on the name a hit actually carries at
#: runtime, so a pattern added to the engine later still works here the moment
#: its name is added to an allow-list, with no change to this list.
PATTERN_CATALOGUE: tuple[str, ...] = (
    # singles
    "Doji", "Dragonfly Doji", "Gravestone Doji", "Long-legged Doji",
    "Marubozu Bullish", "Marubozu Bearish", "Hammer", "Hanging Man",
    "Inverted Hammer", "Shooting Star", "Spinning Top",
    # doubles
    "Bullish Engulfing", "Bearish Engulfing", "Bullish Harami",
    "Bearish Harami", "Bullish Harami Cross", "Bearish Harami Cross",
    "Tweezer Bottom", "Tweezer Top", "Piercing Pattern", "Dark Cloud Cover",
    "Matching Low", "Matching High", "Bullish Kicker", "Bearish Kicker",
    "On Neck", "In Neck", "Thrusting Pattern",
    # triples
    "Morning Star", "Evening Star", "Three White Soldiers", "Advance Block",
    "Deliberation", "Three Black Crows", "Identical Three Crows",
    "Three Inside Up", "Three Inside Down", "Three Outside Up",
    "Three Outside Down", "Abandoned Baby", "Tri-Star",
    "Upside Gap Two Crows", "Stick Sandwich", "Upside Tasuki Gap",
    "Downside Tasuki Gap",
    # fives
    "Rising Three Methods", "Falling Three Methods", "Mat Hold", "Breakaway",
)

#: Chart patterns (chart_pattern_engine). The engine appends " breakout" to the
#: detector's name when it builds the signal reason, so these are stored in the
#: SAME form the trade log and the combination search report — otherwise a name
#: chosen from a search result would not match the filter.
CHART_PATTERN_CATALOGUE: tuple[str, ...] = (
    "Double Bottom breakout", "Double Top breakout",
    "Head & Shoulders breakout", "Inverse Head & Shoulders breakout",
    "Ascending Triangle breakout", "Descending Triangle breakout",
    "Symmetrical Triangle breakout",
)

#: Context confluence factors (context_engine). NOT patterns — this strategy
#: has none. It scores a bar across several dimensions and trades when the
#: total clears a threshold, so what can be selected here is WHICH EVIDENCE
#: must be part of that total. See CONTEXT semantics below.
CONTEXT_FACTOR_CATALOGUE: tuple[str, ...] = (
    "uptrend", "downtrend", "vol+", "ATR↑", "at support", "at resistance",
)

#: How a strategy's filter behaves. Two genuinely different semantics, and
#: conflating them would make one of the controls lie:
#:
#:   "any-of"  — the signal names ONE thing (a candlestick pattern, a chart
#:               pattern). It may fire only if that thing is on the list.
#:   "require" — the signal is a SCORE assembled from several contributing
#:               factors. It may fire only if at least one CHOSEN factor is
#:               among the contributors. This narrows setups; it never changes
#:               the score or the threshold.
ANY_OF, REQUIRE = "any-of", "require"

STRATEGY_FILTERS: dict[str, dict] = {
    "candlestick_engine": {
        "kind": ANY_OF, "label": "Candlestick patterns",
        "catalogue": PATTERN_CATALOGUE,
        "help": "Only these patterns may open a trade.",
    },
    "candlestick_engine_v2": {
        "kind": ANY_OF, "label": "Candlestick patterns",
        "catalogue": PATTERN_CATALOGUE,
        "help": "Only these patterns may open a trade.",
    },
    "chart_pattern_engine": {
        "kind": ANY_OF, "label": "Chart patterns",
        "catalogue": CHART_PATTERN_CATALOGUE,
        "help": ("Only these breakouts may open a trade. Detectors are tried "
                 "in order, so excluding one lets a later detector fire on the "
                 "same bar instead of skipping the bar."),
    },
    "context_engine": {
        "kind": REQUIRE, "label": "Required context factors",
        "catalogue": CONTEXT_FACTOR_CATALOGUE,
        "help": ("This strategy scores confluence rather than naming a "
                 "pattern. Ticking factors requires at least one of them to be "
                 "among the evidence behind a signal — it does not change the "
                 "score or the confluence threshold."),
    },
}

#: Strategies whose signal can be filtered at all. Only these are offered in
#: the UI — a filter on a VWAP strategy would be a control that silently does
#: nothing.
FILTERABLE_STRATEGIES: tuple[str, ...] = tuple(STRATEGY_FILTERS)


def catalogue_for(strategy_key: str) -> tuple[str, ...]:
    """What may be ticked for this strategy. Empty for an unfilterable one."""
    return tuple(STRATEGY_FILTERS.get(str(strategy_key), {}).get("catalogue", ()))


def kind_for(strategy_key: str) -> str:
    return STRATEGY_FILTERS.get(str(strategy_key), {}).get("kind", ANY_OF)


@dataclass
class PatternRules:
    """One (strategy, mode) combination's allow-list."""
    #: Master switch. Off = no filtering whatsoever, whatever `allowed` holds —
    #: so a list can be built up and reviewed before it starts affecting trades.
    enabled: bool = False
    #: Only these patterns may contribute evidence. Empty while enabled is a
    #: deliberate no-op rather than "block everything": switching the feature on
    #: with an empty list would otherwise silently stop all trading.
    allowed: list[str] = field(default_factory=list)

    def is_active(self) -> bool:
        return bool(self.enabled and self.allowed)


def _from_disk() -> dict[str, dict[str, PatternRules]]:
    """{mode: {strategy_key: PatternRules}}. Never raises — a corrupt document
    reads as 'nothing configured', which is the safe default here."""
    data = config_store.load(_KEY)
    if not isinstance(data, dict):
        return {}
    out: dict[str, dict[str, PatternRules]] = {}
    for mode, by_strategy in data.items():
        if not isinstance(by_strategy, dict):
            continue
        entries: dict[str, PatternRules] = {}
        for key, raw in by_strategy.items():
            if not isinstance(raw, dict):
                continue
            entries[str(key)] = PatternRules(
                enabled=bool(raw.get("enabled", False)),
                allowed=[str(p) for p in (raw.get("allowed") or [])])
        if entries:
            out[str(mode)] = entries
    return out


def _to_disk(data: dict[str, dict[str, PatternRules]]) -> None:
    config_store.save(_KEY, {
        mode: {k: asdict(v) for k, v in entries.items()}
        for mode, entries in data.items()})


def _mode_key(mode) -> str:
    """Storage key for a mode, accepting either the enum or its value.

    NOT `str(mode)`: config.Mode is a str-Enum, and `str(Mode.INTRADAY)` is
    "Mode.INTRADAY", not "Intraday". Callers in the strategies pass
    `params.mode` (the enum) while the API passes the string, so both must
    land on the same key — getting this wrong makes the filter silently never
    match, which is the one failure nobody would notice.
    """
    return getattr(mode, "value", None) or str(mode)


def get_rules(strategy_key: str, mode) -> PatternRules:
    """This combination's rules, or an inert default. Served from the cache —
    see _cached()."""
    return (_cached().get(_mode_key(mode), {}).get(str(strategy_key))
            or PatternRules())


def get_all(mode) -> dict[str, PatternRules]:
    """Uncached: the dashboard must always see what is actually stored, never a
    value up to a TTL old that it is about to edit."""
    return _from_disk().get(_mode_key(mode), {})


def set_rules(strategy_key: str, mode, rules: PatternRules) -> PatternRules:
    """Save one combination. A fully-default entry is DELETED rather than
    stored, so 'never configured' and 'reset' are the same state on disk."""
    strategy_key, mode = str(strategy_key), _mode_key(mode)
    clean = PatternRules(
        enabled=bool(rules.enabled),
        # De-duplicated, order preserved, blanks dropped.
        allowed=[p for p in dict.fromkeys(
            str(x).strip() for x in (rules.allowed or [])) if p])
    with _lock:
        data = _from_disk()
        entries = data.get(mode, {})
        if not clean.enabled and not clean.allowed:
            entries.pop(strategy_key, None)
        else:
            entries[strategy_key] = clean
        if entries:
            data[mode] = entries
        else:
            data.pop(mode, None)
        _to_disk(data)
    # Takes effect on the very next tick rather than up to a TTL later.
    _invalidate()
    return clean


# --------------------------------------------------------------------------- #
#  The filter itself
# --------------------------------------------------------------------------- #
def allowed_set(strategy_key: str, mode, params=None):
    """The effective allow-list, or None when nothing is filtering.

    ONE resolution path for every strategy, most-specific first:

      1. `params.ignore_pattern_filter` — an explicit bypass, used by the
         combination search so its screen sees everything.
      2. `params.allowed_patterns` — a PER-RUN override (a backtest trying a
         set without touching the saved dashboard filter).
      3. The saved allow-list for this (strategy, mode).

    None means "no filtering" and every caller must treat it as such, so an
    unconfigured strategy takes exactly the path it did before this existed.
    Never raises: a storage problem degrades to no filtering rather than
    stopping a running bot from taking signals.
    """
    try:
        if getattr(params, "ignore_pattern_filter", False):
            return None
        override = tuple(getattr(params, "allowed_patterns", ()) or ())
        if override:
            return set(override)
        rules = get_rules(strategy_key, mode)
        if not rules.is_active():
            return None
        return set(rules.allowed)
    except Exception:
        return None


def filter_hits(hits, strategy_key: str, mode, params=None):
    """Drop pattern hits that are not on the effective allow-list.

    For the ANY-OF strategies that produce a LIST of pattern hits
    (candlestick). Returns `hits` UNCHANGED — the same list object — when
    nothing is filtering.
    """
    if not hits:
        return hits
    allowed = allowed_set(strategy_key, mode, params)
    if allowed is None:
        return hits
    return [h for h in hits if h.name in allowed]


def allows(name: str, strategy_key: str, mode, params=None) -> bool:
    """May a signal named `name` fire?

    For the ANY-OF strategies that produce ONE named signal rather than a list
    of hits (chart patterns). True whenever nothing is filtering.
    """
    allowed = allowed_set(strategy_key, mode, params)
    return allowed is None or name in allowed


def allows_factors(factors, strategy_key: str, mode, params=None) -> bool:
    """May a signal supported by `factors` fire?

    For the REQUIRE strategies (context confluence). The signal is a score, not
    a named pattern, so the rule is that at least ONE chosen factor must be
    among the evidence. True whenever nothing is filtering — and also true when
    the strategy emitted no factors at all, since refusing on missing evidence
    would silently disable a strategy whose notes are simply empty.
    """
    allowed = allowed_set(strategy_key, mode, params)
    if allowed is None:
        return True
    present = {str(f).strip() for f in (factors or []) if str(f).strip()}
    if not present:
        return True
    return bool(present & allowed)
