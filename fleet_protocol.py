"""
fleet_protocol.py
What crosses the wire between the admin HUB and a client NODE — and nothing else.

Both sides import this one module so they cannot disagree about a field name.
It knows nothing about sockets, engines or brokers: it only turns the runner's
in-process events (strategy_runner.TickEvent / SignalEvent) into JSON-safe dicts
and back, and owns the small set of rules that must be identical on both ends.

What is deliberately NOT here, and never goes over the link
-----------------------------------------------------------
  * a QUANTITY. Position size is `risk_budget / stop_distance` against each
    node's own capital (Immutable Rules #1 and #4); a signal carries the
    decision, never the size.
  * any broker key, secret or token, in either direction.

Frames
------
Hub -> node
  {"t":"session", "active":bool, "config":{...}, "feed_simulated":bool, "hub_ts":f}
  {"t":"poll", "seq":n, "hub_ts":f, "events":[ {"t":"begin"|"tick"|"exit"|"signal", ...} ]}
  {"t":"ping", "hub_ts":f}
Node -> hub
  {"t":"hello", "version":1, "name":str}
  {"t":"ack",   "signal_id":str, "entered":bool, "note":str}
  {"t":"report", ...}  (see hub_link.build_report)
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime
from typing import Any, Optional

import pandas as pd

import config
from data_feed import LiveQuote
from strategy import Signal
from strategy_runner import SignalEvent, TickEvent

PROTOCOL_VERSION = 1

#: A signal older than this on ARRIVAL is dropped, never entered late — a
#: stale entry is a worse fill than no entry. Exits are never dropped.
MAX_SIGNAL_AGE_SECONDS = 5.0


# --------------------------------------------------------------------------- #
#  The run a broadcast publishes
# --------------------------------------------------------------------------- #
@dataclass
class BroadcastConfig:
    """Everything a node needs to build the same run the hub is deciding.

    Field-for-field what admin_config.ModeConfig + the StartBotRequest admin
    fields carry, because a node turns this back into exactly that and lets the
    ordinary client start path do the rest.
    """
    environment: str = "Paper"
    mode: str = "Intraday"
    strategy_key: str = ""
    symbols: list[str] = field(default_factory=list)
    mcx_lots: dict[str, int] = field(default_factory=dict)
    risk_reward: float = 0.0
    min_score: float = 0.0
    square_off_time: str = ""
    square_off_enabled: bool = True
    exit_style: str = "strategy"
    trail_atr_mult: float = 0.0
    max_stop_pct: float = 0.0
    min_stop_pct: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "BroadcastConfig":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in (data or {}).items() if k in known})

    def validate(self) -> str:
        """"" when usable, else a short reason. Run by the hub before it starts
        AND by every node before it acts — a node never trusts the wire."""
        import admin_config
        if self.environment not in ("Paper", "Live"):
            return f"Unknown environment {self.environment!r}."
        if self.mode not in admin_config.CLIENT_SELECTABLE_MODES:
            return (f"Mode {self.mode!r} cannot be broadcast. Use one of "
                    f"{', '.join(admin_config.CLIENT_SELECTABLE_MODES)}.")
        known = [s for s in self.symbols if s in config.INSTRUMENTS_BY_SYMBOL]
        if not known:
            return "Select at least one known instrument."
        if not config.is_valid_exit_style(self.exit_style):
            return f"Unknown exit_style {self.exit_style!r}."
        if self.square_off_time and config.parse_clock(self.square_off_time) is None:
            return f"square_off_time {self.square_off_time!r} is not HH:MM."
        if not config.is_valid_min_score(self.min_score):
            return f"min_score {self.min_score:g} is out of range."
        if not config.is_valid_rr(self.risk_reward):
            return f"risk_reward {self.risk_reward:g} is not an offered RR."
        if (self.min_stop_pct and self.max_stop_pct
                and self.min_stop_pct > self.max_stop_pct):
            return "min_stop_pct cannot exceed max_stop_pct."
        return ""


# --------------------------------------------------------------------------- #
#  Event <-> dict
# --------------------------------------------------------------------------- #
def _iso(ts: Any) -> Optional[str]:
    if ts is None:
        return None
    try:
        return pd.Timestamp(ts).isoformat()
    except Exception:
        return None


def _parse_ts(text: Optional[str]) -> Any:
    return pd.Timestamp(text) if text else None


def tick_to_dict(ev: TickEvent) -> dict:
    q = ev.quote
    return {
        "t": "tick", "symbol": ev.instrument.symbol,
        "live_price": float(ev.live_price),
        "market_open": bool(ev.market_open),
        "bar_ts": _iso(ev.bar_ts),
        "now_dt": ev.now_dt.isoformat(),
        "atr": float(ev.atr or 0.0),
        "quote": None if q is None else {
            "ltp": float(q.ltp), "bid": float(q.bid), "ask": float(q.ask),
            "source": q.source, "age": float(q.age_seconds),
            "ts": q.ts.isoformat() if hasattr(q.ts, "isoformat") else None,
        },
    }


def tick_from_dict(d: dict) -> Optional[TickEvent]:
    inst = config.INSTRUMENTS_BY_SYMBOL.get(d.get("symbol", ""))
    if inst is None:
        return None
    qd = d.get("quote")
    quote = None
    if qd:
        quote = LiveQuote(
            ltp=float(qd["ltp"]),
            ts=datetime.fromisoformat(qd["ts"]) if qd.get("ts") else datetime.now(),
            # Re-anchor the age on THIS machine's monotonic clock: the quote
            # was already `age` seconds old when it left the hub.
            received_at=time.monotonic() - float(qd.get("age", 0.0)),
            bid=float(qd.get("bid", 0.0)), ask=float(qd.get("ask", 0.0)),
            source=qd.get("source", "ws"))
    return TickEvent(
        instrument=inst, quote=quote, live_price=float(d["live_price"]),
        market_open=bool(d["market_open"]), bar_ts=_parse_ts(d.get("bar_ts")),
        now_dt=datetime.fromisoformat(d["now_dt"]), atr=float(d.get("atr", 0.0)))


def signal_to_dict(ev: SignalEvent, signal_id: str) -> dict:
    s = ev.signal
    tick = None
    if ev.quote is not None:
        q = ev.quote
        tick = {"ltp": float(q.ltp), "bid": float(q.bid), "ask": float(q.ask),
                "source": q.source, "age": float(q.age_seconds),
                "ts": q.ts.isoformat() if hasattr(q.ts, "isoformat") else None}
    return {
        "t": "signal", "signal_id": signal_id, "symbol": ev.instrument.symbol,
        "side": s.side, "entry_price": float(s.entry_price),
        "stop_loss": float(s.stop_loss), "target": float(s.target),
        "reason": s.reason, "bar_ts": _iso(ev.bar_ts), "quote": tick,
    }


def signal_from_dict(d: dict) -> Optional[SignalEvent]:
    inst = config.INSTRUMENTS_BY_SYMBOL.get(d.get("symbol", ""))
    if inst is None:
        return None
    qd = d.get("quote")
    quote = None
    if qd:
        quote = LiveQuote(
            ltp=float(qd["ltp"]),
            ts=datetime.fromisoformat(qd["ts"]) if qd.get("ts") else datetime.now(),
            received_at=time.monotonic() - float(qd.get("age", 0.0)),
            bid=float(qd.get("bid", 0.0)), ask=float(qd.get("ask", 0.0)),
            source=qd.get("source", "ws"))
    sig = Signal(side=d["side"], entry_price=float(d["entry_price"]),
                 stop_loss=float(d["stop_loss"]), target=float(d["target"]),
                 reason=d.get("reason", ""))
    return SignalEvent(inst, sig, quote, _parse_ts(d.get("bar_ts")))


# --------------------------------------------------------------------------- #
#  Which database is this process writing trades to?
# --------------------------------------------------------------------------- #
def db_fingerprint() -> str:
    """A short, non-reversible id for the MongoDB this process writes to: the
    hosts plus the database name, nothing else.

    The hub reads every client's trades from ITS database. A node that writes to
    a different one — a forgotten MONGO_URI left at localhost, a second cluster —
    trades normally and its trades simply never appear on the hub. Nodes report
    this fingerprint so the hub can say so instead of showing a client that
    looks idle.

    Credentials are stripped BEFORE hashing (they differ per deployment and must
    never travel), and the URI is split by hand: pymongo's own parser resolves
    mongodb+srv:// over DNS, which this must never do on a timer.
    """
    uri = (config.MONGO_URI or "").strip()
    rest = uri.split("://", 1)[-1].rsplit("@", 1)[-1]
    hosts = rest.split("/", 1)[0].split("?", 1)[0]
    norm = ",".join(sorted(h.strip().lower() for h in hosts.split(",") if h.strip()))
    import hashlib
    return hashlib.sha256(f"{norm}/{config.MONGO_DB_NAME}".encode()).hexdigest()[:12]


def db_problem(node_backend: str, node_fingerprint: str,
               hub_fingerprint: str) -> str:
    """"" when a node's trades will reach the hub's database; otherwise why not,
    in words an admin can act on. Empty inputs (a node that has not reported
    yet, or an older build) are "unknown", not a problem."""
    if not node_backend and not node_fingerprint:
        return ""
    if node_backend and node_backend != "MongoDB":
        return ("This server is NOT writing to MongoDB — its trades are being "
                "saved on the server itself and will not appear here.")
    if node_fingerprint and node_fingerprint != hub_fingerprint:
        return ("This server uses a DIFFERENT database from the hub — its "
                "trades will not appear here. Check its MONGO_URI and "
                "MONGO_DB_NAME.")
    return ""


# --------------------------------------------------------------------------- #
#  Staleness, tolerant of clock skew
# --------------------------------------------------------------------------- #
class FrameAge:
    """How long a frame took to reach us, WITHOUT trusting the two clocks to
    agree.

    Each frame carries the hub's clock. `node_now - hub_ts` is clock skew plus
    network delay; the SMALLEST such value over recent frames is the skew plus
    the best-case delay, so anything above that minimum is genuine lateness.
    That is what a stale-signal check needs, and it needs no NTP: a node whose
    clock is a minute off would otherwise reject every signal or accept every
    stale one.
    """

    def __init__(self, window: int = 60):
        self._samples: deque[float] = deque(maxlen=window)

    def observe(self, hub_ts: float, node_now: Optional[float] = None) -> float:
        """Record a frame and return its estimated age in seconds."""
        now = time.time() if node_now is None else node_now
        d = now - float(hub_ts)
        self._samples.append(d)
        return max(0.0, d - min(self._samples))
