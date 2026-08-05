"""
strategy_runner.py
The DECIDING half of the bot, split out from the EXECUTING half.

One StrategyRunner owns a single market-data feed and evaluates the strategy
ONCE per bar, then broadcasts what it decided to every subscribed execution
account simultaneously. Each account then sizes the trade against its OWN
capital and risk limits and punches it through its OWN broker.

    StrategyRunner (one)                 accounts (many)
    ─────────────────────                ───────────────────────────
    admin token -> 1 WebSocket           own broker token -> orders
    enrich() + run_strategy()            position_size() against own capital
    SL/TP/time trigger                   own risk limits, trade book, PnL

Why this exists
---------------
Every user used to get a whole TradingEngine: its own WebSocket on the SAME
admin token, its own indicator computation, its own signal evaluation. Three
problems, all of which this fixes:

  * N sockets on one token — brokers cap concurrent streams, so it fails as
    clients are added, and it fails by silently degrading to REST polling.
  * Duplicated compute — the same indicators recomputed per user per tick.
  * Signals genuinely DIVERGED. Each engine kept its own re-entry cooldown and
    started at a different moment, so two clients on the same strategy did not
    take the same trades. That is the opposite of replication.

What this module does NOT do
----------------------------
It does not size positions, place orders, hold positions, book PnL or read
risk limits — all of that is per-account and stays exactly where it was, in
engine.py. Nothing here imports engine.py; accounts are duck-typed (see the
Account protocol below), which is also what keeps the import graph acyclic.

Strategy logic is untouched: this calls the same `run_strategy(sd, df,
session_open)` with the same arguments the engine always did. Only the caller
moved.
"""
from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor, wait
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional, Protocol

from config import Instrument, Mode, market_hours_for_segment, now_ist
from data_feed import LiveQuote, MarketDataFeed, SimulatedFeed, make_feed
from strategy import BoundStrategy, Signal, run_strategy

# A tick older than this during market hours means the stream has stalled; we
# stop calling it "live" in the UI rather than pricing PnL off stale data.
STALE_QUOTE_SECONDS = 30.0

# How long one account gets to handle a broadcast before the runner stops
# waiting on it. A hung broker HTTP call must never stall the other accounts
# or the next tick — the slow account simply misses that event.
ACCOUNT_TIMEOUT_SECONDS = 15.0

# Ceiling on concurrent account dispatch. Entries must land at the same
# moment across accounts, so they are submitted together rather than looped.
MAX_DISPATCH_WORKERS = 32


# --------------------------------------------------------------------------- #
#  What the runner broadcasts
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class TickEvent:
    """Per-instrument state for one poll. Frozen so every account sees the
    identical numbers — an account cannot mutate what the next one receives."""
    instrument: Instrument
    quote: Optional[LiveQuote]
    live_price: float
    market_open: bool
    bar_ts: Any
    now_dt: datetime


@dataclass(frozen=True)
class SignalEvent:
    """One entry decision, computed once and delivered to every account."""
    instrument: Instrument
    signal: Signal
    quote: Optional[LiveQuote]
    bar_ts: Any


class Account(Protocol):
    """What the runner needs from an execution account. Deliberately narrow —
    TradingEngine satisfies it, and nothing here depends on that class."""

    def begin_tick(self, now_dt: datetime, feed_status: str) -> None: ...
    def on_tick(self, ev: TickEvent) -> bool: ...
    def on_signal(self, ev: SignalEvent) -> bool: ...
    def on_exit(self, instrument: Instrument, price: float, reason: str) -> bool: ...
    def holds(self, symbol: str) -> bool: ...
    def push_log(self, msg: str) -> None: ...


# --------------------------------------------------------------------------- #
#  The runner's own reference position
# --------------------------------------------------------------------------- #
@dataclass
class _Reference:
    """The runner's view of a trade it opened, used ONLY to decide when the
    whole platform should exit. It is not a real position — no quantity, no
    PnL, no broker. Accounts hold the real ones."""
    side: str
    entry: float
    stop: float
    target: float
    opened_at: datetime


class StrategyRunner:
    def __init__(
        self,
        key: str,
        mode: Mode,
        strategy: BoundStrategy,
        params,
        instruments: list[Instrument],
        feed_token: str,
        poll_seconds: float,
        symbol_rules: Optional[dict] = None,
    ):
        self.key = key
        self.mode = mode
        self.strategy = strategy
        # RR-adjusted params (engine applies admin's per-mode override before
        # constructing us), so the target this runner quotes matches the one
        # accounts will book.
        self.params = params
        self.instruments = instruments
        self.feed_token = feed_token
        self.poll_seconds = poll_seconds
        self.symbol_rules = dict(symbol_rules or {})

        self.feed: Optional[MarketDataFeed] = None
        self._accounts: list[Account] = []
        self._accounts_lock = threading.Lock()
        self._pool: Optional[ThreadPoolExecutor] = None

        # Bar on which this runner last acted per symbol — the re-entry
        # cooldown. Now runner-level rather than per-account, which is exactly
        # what makes every account trade the SAME bars.
        self._last_action_bar: dict[str, Any] = {}
        self._window_skip_logged: dict[str, str] = {}
        self._refs: dict[str, _Reference] = {}
        # What this runner has DECIDED recently, independent of any account —
        # no quantity, because quantity is per-account. This is what lets a
        # client watch the platform trading before they have started their own
        # bot: the signals are real and already happening, they simply are not
        # punching them yet.
        self._recent_signals: list[dict] = []
        self._signals_lock = threading.Lock()
        # Rotates so the same account is not structurally last to be submitted
        # on every signal.
        self._dispatch_offset = 0

        self.running = False
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        # Several accounts can subscribe at once and all call ensure_started();
        # only the first may actually open the socket.
        self._start_lock = threading.Lock()

    # -- subscription -------------------------------------------------------- #
    def subscribe(self, account: Account) -> None:
        with self._accounts_lock:
            if account not in self._accounts:
                self._accounts.append(account)

    def unsubscribe(self, account: Account) -> int:
        """Remove an account; returns how many remain."""
        with self._accounts_lock:
            if account in self._accounts:
                self._accounts.remove(account)
            return len(self._accounts)

    def account_count(self) -> int:
        with self._accounts_lock:
            return len(self._accounts)

    def _snapshot_accounts(self) -> list[Account]:
        """A copy, rotated for fairness — dispatch must never hold the lock,
        since account handlers place orders and can block."""
        with self._accounts_lock:
            accounts = list(self._accounts)
        if len(accounts) > 1:
            self._dispatch_offset = (self._dispatch_offset + 1) % len(accounts)
            off = self._dispatch_offset
            accounts = accounts[off:] + accounts[:off]
        return accounts

    # -- lifecycle ----------------------------------------------------------- #
    def ensure_started(self) -> None:
        """Start the feed and loop if they aren't already. Idempotent and
        thread-safe: whichever account subscribes first opens the socket, the
        rest simply attach to it."""
        with self._start_lock:
            if not self.running:
                self.start()

    def start(self) -> None:
        if self.running:
            return
        self._stop.clear()
        self.feed = make_feed(prefer_real=bool(self.feed_token),
                              access_token=self.feed_token, mode=self.mode)
        try:
            self.feed.start(self.instruments)
        except Exception as exc:
            self._log(f"⚠️ Live feed unavailable ({exc}); using simulated feed.")
            self.feed = SimulatedFeed(mode=self.mode)
            self.feed.start(self.instruments)
        self._pool = ThreadPoolExecutor(
            max_workers=MAX_DISPATCH_WORKERS,
            thread_name_prefix=f"dispatch-{self.key[:16]}")
        self.running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self.running = False
        if self.feed:
            self.feed.stop()
        if self._pool is not None:
            self._pool.shutdown(wait=False)
            self._pool = None

    def feed_is_simulated(self) -> bool:
        return isinstance(self.feed, SimulatedFeed)

    def status(self) -> str:
        return self.feed.status() if self.feed else "🔴 Not started"

    # -- dispatch ------------------------------------------------------------ #
    def _broadcast(self, call, accounts: list[Account]) -> list[bool]:
        """Run `call(account)` across every account AT ONCE and collect the
        results. Failure is isolated per account: one broker raising, hanging
        or rejecting must never break the runner or the other accounts —
        before this split a crash took down only that user's own bot, so
        containing it here is mandatory, not defensive styling."""
        if not accounts:
            return []
        pool = self._pool
        if pool is None:                      # not started / already stopped
            return [self._guard(call, a) for a in accounts]
        futures = {pool.submit(self._guard, call, a): a for a in accounts}
        done, not_done = wait(futures, timeout=ACCOUNT_TIMEOUT_SECONDS)
        for fut in not_done:
            acct = futures[fut]
            self._safe_log(acct, "⚠️ timed out handling a bot event; skipped "
                                 "for this tick.")
        return [f.result() for f in done if not f.cancelled()]

    @staticmethod
    def _guard(call, account: Account) -> bool:
        try:
            return bool(call(account))
        except Exception as exc:
            try:
                account.push_log(f"⚠️ error handling bot event: {exc}")
            except Exception:
                pass
            return False

    @staticmethod
    def _safe_log(account: Account, msg: str) -> None:
        try:
            account.push_log(msg)
        except Exception:
            pass

    def _log(self, msg: str) -> None:
        for acct in self._snapshot_accounts():
            self._safe_log(acct, msg)

    # -- main loop ----------------------------------------------------------- #
    def _run_loop(self) -> None:
        while not self._stop.is_set():
            try:
                self.tick()
            except Exception as exc:          # never let one bad tick kill it
                self._log(f"⚠️ tick error: {exc}")
            self._stop.wait(self.poll_seconds)

    def tick(self) -> None:
        """One poll: refresh every instrument, let accounts manage what they
        hold, then decide entries once and broadcast them."""
        now_dt = now_ist()
        now_t = now_dt.time()
        accounts = self._snapshot_accounts()
        if not accounts:
            return
        feed_status = self.status()
        self._broadcast(lambda a: a.begin_tick(now_dt, feed_status), accounts)

        for inst in self.instruments:
            hours = market_hours_for_segment(inst.segment)
            market_open = hours.is_open(now_t)
            rules = self.symbol_rules.get(inst.symbol)

            quote = self.feed.get_quote(inst)
            df = self.feed.get_candles(inst, lookback=260)
            live_price, _src = self._live_price(quote, df)
            if live_price is None:
                continue
            bar_ts = df.index[-1] if not df.empty else None

            # 1) Accounts manage what they already hold — their own SL/TP/time
            # exit, unchanged. This runs BEFORE the runner's own reference
            # trigger below so single-account behaviour is exactly what it was
            # before this split.
            ev = TickEvent(inst, quote, live_price, market_open, bar_ts, now_dt)
            closed = self._broadcast(lambda a, e=ev: a.on_tick(e), accounts)
            if any(closed) and bar_ts is not None:
                self._last_action_bar[inst.symbol] = bar_ts

            # 2) The runner's own exit trigger: the unifier that gets every
            # account out at the SAME moment, whatever price each of them
            # happened to fill at. Accounts already out are a no-op.
            if self._reference_exit(inst, live_price, now_dt, accounts) and bar_ts:
                self._last_action_bar[inst.symbol] = bar_ts
                continue

            # 3) Entry gating — identical order to the original engine loop.
            if not market_open or df.empty:
                continue
            if rules is not None:
                blocked = rules.entry_block_reason(now_dt)
                if blocked:
                    self._log_window_skip(inst.symbol, blocked, accounts)
                    continue
                self._window_skip_logged.pop(inst.symbol, None)
            if not self._cooldown_elapsed(inst.symbol, df):
                continue

            # 4) THE single calculation. One enrich(), one strategy evaluation,
            # for every account on this platform.
            sig = run_strategy(self.strategy, df, session_open=hours.open_t)
            if sig is None:
                continue

            # 5) Record it platform-wide, then broadcast simultaneously. Each
            # account sizes it against its own capital and risk limits and
            # punches its own order.
            self._record_signal(inst, sig, now_dt)
            sev = SignalEvent(inst, sig, quote, bar_ts)
            entered = self._broadcast(lambda a, e=sev: a.on_signal(e), accounts)
            if any(entered):
                self._last_action_bar[inst.symbol] = bar_ts
                self._refs[inst.symbol] = _Reference(
                    side=sig.side, entry=sig.entry_price,
                    stop=sig.stop_loss,
                    target=self._effective_target(inst.symbol, sig),
                    opened_at=now_dt)

    # -- the platform signal log --------------------------------------------- #
    def _record_signal(self, inst: Instrument, sig: Signal,
                       now_dt: datetime) -> None:
        """Append what was decided, WITHOUT a quantity.

        Quantity is meaningless here: it is whatever each account's own
        capital and risk limits produce, so a platform-level log that quoted
        one would be wrong for everybody. This is the decision; the sizing
        belongs to the account rows."""
        row = {
            "time": now_dt.strftime("%H:%M:%S"),
            "symbol": inst.symbol,
            "segment": inst.segment.value,
            "side": sig.side,
            "entry": round(sig.entry_price, 2),
            "stop": round(sig.stop_loss, 2),
            "target": round(self._effective_target(inst.symbol, sig), 2),
            "rr": round(self._rr_for(inst.symbol), 2),
            "reason": sig.reason,
        }
        with self._signals_lock:
            self._recent_signals.insert(0, row)
            self._recent_signals = self._recent_signals[:50]

    def recent_signals(self) -> list[dict]:
        with self._signals_lock:
            return list(self._recent_signals)

    # -- the runner's reference exit ----------------------------------------- #
    def _rr_for(self, symbol: str) -> float:
        """The symbol's own RR if it has one, else the run's. Mirrors
        TradingEngine._rr_for so the runner's exit trigger and the accounts'
        stored targets agree."""
        rules = self.symbol_rules.get(symbol)
        if rules is not None and rules.risk_reward > 0:
            return rules.risk_reward
        return self.params.risk_reward

    def _effective_target(self, symbol: str, sig: Signal) -> float:
        """The target accounts will actually book."""
        rr = self._rr_for(symbol)
        dist = abs(sig.entry_price - sig.stop_loss)
        return (sig.entry_price + rr * dist if sig.side == "BUY"
                else sig.entry_price - rr * dist)

    def _reference_exit(self, inst: Instrument, live_price: float,
                        now_dt: datetime, accounts: list[Account]) -> bool:
        """Fire a platform-wide exit when the reference levels are hit.

        The reference is anchored to the SIGNAL's prices, not to any one
        account's fill, which is the point: it gets everybody out together
        instead of each account drifting on its own fill. Accounts that are
        already flat ignore it, and each still books its own real exit price
        and PnL.
        """
        ref = self._refs.get(inst.symbol)
        if ref is None:
            return False
        reason = ""
        if ref.side == "BUY":
            if live_price <= ref.stop:
                reason = "STOP-LOSS"
            elif live_price >= ref.target:
                reason = "TARGET"
        else:
            if live_price >= ref.stop:
                reason = "STOP-LOSS"
            elif live_price <= ref.target:
                reason = "TARGET"
        if not reason and self.params.max_hold_minutes > 0:
            held = (now_dt - ref.opened_at).total_seconds() / 60.0
            if held >= self.params.max_hold_minutes:
                reason = f"TIME-EXIT ({self.params.max_hold_minutes}m)"
        if not reason:
            # Nobody holds it any more (all stopped out on their own levels) —
            # drop the reference so a later signal can re-arm it.
            if not any(a.holds(inst.symbol) for a in accounts):
                self._refs.pop(inst.symbol, None)
            return False

        self._refs.pop(inst.symbol, None)
        closed = self._broadcast(
            lambda a: a.on_exit(inst, live_price, reason), accounts)
        return any(closed)

    # -- gates (moved verbatim from the engine) ------------------------------ #
    def _cooldown_elapsed(self, symbol: str, df) -> bool:
        """True if enough NEW bars have closed since this runner last acted on
        the symbol. Runner-level now, so every account is held to the same
        bars — two accounts can no longer disagree about whether a setup is
        still fresh."""
        last = self._last_action_bar.get(symbol)
        if last is None:
            return True
        need = max(self.params.reentry_cooldown_bars, 0)
        if need <= 0:
            return True
        try:
            fresh = int((df.index > last).sum())
        except Exception:
            return True
        return fresh >= need

    def _log_window_skip(self, symbol: str, reason: str,
                         accounts: list[Account]) -> None:
        """Log a day/window entry block ONCE per reason, not once per poll."""
        if self._window_skip_logged.get(symbol) == reason:
            return
        self._window_skip_logged[symbol] = reason
        for acct in accounts:
            self._safe_log(
                acct,
                f"⏸️ {symbol}: new entries paused — {reason} (custom settings). "
                f"Open positions are still managed normally.")

    @staticmethod
    def _live_price(quote: Optional[LiveQuote], df) -> tuple[Optional[float], str]:
        """Resolve the price used to mark positions and fire stops.

        Order matters: a fresh WebSocket tick wins over everything. Only when
        no usable tick exists do we read a candle close, and a stale tick is
        the last resort — never preferred over fresher candle data.
        """
        if (quote is not None and quote.ltp > 0
                and quote.age_seconds <= STALE_QUOTE_SECONDS):
            return float(quote.ltp), quote.source
        if df is not None and not df.empty:
            return float(df["close"].iloc[-1]), "candle"
        if quote is not None and quote.ltp > 0:
            return float(quote.ltp), f"{quote.source}:stale"
        return None, "none"


# --------------------------------------------------------------------------- #
#  Registry — who shares a runner with whom
# --------------------------------------------------------------------------- #
#: When replication is ON, accounts trading the same mode/strategy/instruments
#: share ONE runner: one socket, one calculation, simultaneous entries. When
#: OFF, every account gets a private runner, which reproduces the original
#: one-engine-per-user behaviour exactly — the switch is only about SHARING,
#: never about how anything is decided or sized. That makes it a genuine
#: rollback: flip the env var and the platform is back to its previous shape
#: without a redeploy.
def replication_enabled() -> bool:
    import os
    return (os.getenv("REPLICATION_ENABLED", "true") or "").strip().lower() \
        not in ("0", "false", "no", "off")


_registry_lock = threading.Lock()
_runners: dict[str, StrategyRunner] = {}


def runner_key(mode: Mode, strategy_key: str, instruments: list[Instrument],
               risk_reward: float, feed_token: str,
               min_score: float = 0.0) -> str:
    """Everything that must match for two accounts to share a decision.

    risk_reward is in the key because it moves the TARGET, and the runner's
    exit trigger quotes that target. min_score is in it because it decides
    WHETHER a signal fires at all — two accounts on different thresholds are
    not running the same decision and must never share one runner, or the
    stricter account would silently receive the looser one's entries.
    """
    syms = ",".join(sorted(i.symbol for i in instruments))
    return (f"{mode.value}|{strategy_key}|{risk_reward:g}|{min_score:g}"
            f"|{syms}|{hash(feed_token)}")


def acquire(key: str, factory) -> StrategyRunner:
    """The shared runner for `key`, created via `factory` on first use.

    Returned UNSTARTED — the caller subscribes first and then calls
    ensure_started(), so the runner never takes a tick with nobody listening.
    A runner that has been stopped is removed from the registry by release(),
    so anything found here is live or about to be.

    With replication off the caller passes a key unique to itself, so this
    hands back a private runner and the platform behaves exactly as it did
    when every user had their own engine.
    """
    with _registry_lock:
        runner = _runners.get(key)
        if runner is None:
            runner = factory()
            _runners[key] = runner
        return runner


def release(key: str, account: Account) -> None:
    """Unsubscribe an account and stop the runner once the last one leaves —
    no point holding a socket open for nobody."""
    with _registry_lock:
        runner = _runners.get(key)
        if runner is None:
            return
        remaining = runner.unsubscribe(account)
        if remaining == 0:
            runner.stop()
            _runners.pop(key, None)


def get(key: str) -> Optional[StrategyRunner]:
    """The live runner for `key`, or None if nobody is running it.

    This is how an account that has NOT started yet finds the decisions being
    made on its behalf: the key is derived from admin's saved config, so a
    client can watch the platform trade before punching anything themselves.
    """
    with _registry_lock:
        return _runners.get(key)


def active() -> dict[str, int]:
    """{runner key: subscriber count} — for diagnostics."""
    with _registry_lock:
        return {k: r.account_count() for k, r in _runners.items()}
