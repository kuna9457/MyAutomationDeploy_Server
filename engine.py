"""
engine.py
The orchestrator. Ties the four decoupled layers together into one running bot:

    data_feed  --candles-->  strategy  --signal-->  broker  --fill-->  db_manager

It runs in a background thread so the Streamlit UI stays responsive, exposes a
thread-safe snapshot of state (connection status, live signals, open positions,
PnL) for the dashboard, and honours per-segment market hours — critically, MCX
commodities keep trading into the night while equity stops at 15:30.

Live pricing is WEBSOCKET-DRIVEN: open positions are marked, and stops/targets
evaluated, against feed.get_quote() — the live tick — not against a candle close.
Candles decide *whether to enter*; ticks decide *what things are worth now*.

Start/Stop from the UI simply flip the thread on and off.
"""
from __future__ import annotations

import math
import threading
from datetime import datetime
from typing import Optional

import time as _time

import config
import risk_manager
from broker_api import BaseBroker, fetch_upstox_margin, make_broker
from config import (Broker, Environment, Instrument, Mode, Segment,
                    market_hours_for_segment, now_ist, params_for_mode)
from data_feed import LiveQuote, MarketDataFeed, SimulatedFeed, make_feed
from db_manager import DBManager
from strategy import position_size, resolve_strategy, run_strategy

# TTL for the cached broker-reported available-funds check (engine._broker_available_
# funds). Account-wide and read-only, but re-fetched on every tick would hammer the
# broker API at the Scalper's 0.5s poll rate — a real balance doesn't move that fast.
BROKER_FUNDS_TTL_SECONDS = 20.0

# TTL for the cached per-symbol broker position (engine._broker_position), which
# drives the LIVE dashboard's "real" PnL/avg-price. Short — this is meant to feel
# live — but still short-circuits the Scalper's 0.5s loop from re-fetching every
# open symbol's position on every single tick.
BROKER_POSITION_TTL_SECONDS = 5.0

# A tick older than this during market hours means the stream has stalled; we
# stop calling it "live" in the UI rather than pricing PnL off stale data.
STALE_QUOTE_SECONDS = 30.0


class BotState:
    """Thread-safe container the UI polls each rerun."""

    def __init__(self):
        self.lock = threading.Lock()
        self.running = False
        self.feed_status = "🔴 Not started"
        self.broker_name = "-"
        self.db_backend = "-"
        self.last_signals: list[dict] = []      # recently detected signals
        self.open_positions: dict[str, dict] = {}   # symbol -> trade doc + live px
        self.live_quotes: dict[str, dict] = {}      # symbol -> WS tick snapshot
        self.day_pnl = 0.0
        self.realized_pnl = 0.0
        self.unrealized_pnl = 0.0
        # The trading day the live counters belong to. When the calendar day rolls
        # over (pre-market), realized_pnl is reset to that new day's total (≈0) so
        # the dashboard shows a fresh day — while every past day stays on disk.
        self.trading_day = ""
        # Day-wise history rebuilt from storage (list of dict rows). Survives
        # restarts: it's read back from the persisted trades, never held only here.
        self.daily_pnl: list[dict] = []
        self.log: list[str] = []
        # LIVE-only risk-guardrail status (risk_manager.LiveRiskLimits applied),
        # refreshed every tick so the dashboard can show it. Empty dict in Paper.
        self.risk_status: dict = {}

    def push_log(self, msg: str) -> None:
        stamp = now_ist().strftime("%H:%M:%S")
        with self.lock:
            self.log.insert(0, f"[{stamp}] {msg}")
            self.log = self.log[:200]

    def snapshot(self) -> dict:
        with self.lock:
            return {
                "running": self.running,
                "feed_status": self.feed_status,
                "broker_name": self.broker_name,
                "db_backend": self.db_backend,
                "last_signals": list(self.last_signals),
                "open_positions": dict(self.open_positions),
                "live_quotes": dict(self.live_quotes),
                "day_pnl": self.day_pnl,
                "realized_pnl": self.realized_pnl,
                "unrealized_pnl": self.unrealized_pnl,
                "trading_day": self.trading_day,
                "daily_pnl": list(self.daily_pnl),
                "log": list(self.log),
                "risk_status": dict(self.risk_status),
            }


class TradingEngine:
    def __init__(
        self,
        environment: Environment,
        mode: Mode,
        broker_choice: Broker,
        instruments: list[Instrument],
        total_capital: float,
        poll_seconds: Optional[float] = None,
        strategy_key: str = "",
        mcx_lots: Optional[dict[str, int]] = None,
        user_id: str = "admin",
        broker_access_token: str = "",
        broker_api_key: str = "",
    ):
        self.environment = environment
        self.mode = mode
        self.broker_choice = broker_choice
        self.instruments = instruments
        self.total_capital = total_capital
        # Multi-tenancy (frontend_migration_plan.md §3, Phase 2): whose trades/
        # risk-limits/positions this instance owns. "admin" (the default)
        # reproduces Phase 1 exactly — one engine, one shared trade history.
        # A client's engine passes their own user_id, so THEIR trades get
        # tagged separately and THEIR risk limits are read/written
        # independently of every other account, including the admin's.
        self.user_id = user_id
        # A client's own broker access token (Upstox/Zerodha), obtained via
        # their own OAuth login, not the shared .env credentials — see
        # broker_api.make_broker. Empty string = fall back to the global
        # config token, i.e. today's admin-only behavior.
        self.broker_access_token = broker_access_token
        # Only meaningful for Zerodha, whose tokens are bound to the api_key
        # that minted them (see broker_api.ZerodhaBroker). Empty = shared .env
        # app, i.e. exactly today's behaviour.
        self.broker_api_key = broker_api_key
        # Per-symbol FIXED lot count for MCX commodities, chosen by the user in the
        # sidebar BEFORE the bot starts (symbol -> number of lots/contracts). This is
        # the whole reason commodity sizing is separate: unlike equity — which is
        # risk-sized (qty = risk_budget / stop_distance, capped by the 20%-of-account
        # rule) — a commodity signal trades exactly the lots configured here, and the
        # SL/TP/margin all follow from that fixed quantity. A symbol absent from the
        # map defaults to 1 lot. See `_mcx_fixed_size`. Equity NEVER consults this.
        self.mcx_lots = {str(k): int(v) for k, v in (mcx_lots or {}).items()}
        # The strategy owns its parameters — risk, RR, stop multiple and filters
        # all travel with it, so switching strategy switches the whole contract.
        self.strategy = resolve_strategy(mode, strategy_key)
        self.params = self.strategy.params
        # The Scalper trades 1-minute bars with a 7-minute time exit, so a 3s loop
        # would be a meaningful share of the whole trade. Poll sub-second there.
        self.poll_seconds = poll_seconds or (
            0.5 if mode == Mode.SCALPER else 3.0)

        self.state = BotState()
        self.db = DBManager()
        self.broker: Optional[BaseBroker] = None
        self.feed: Optional[MarketDataFeed] = None
        # Last candle on which we entered or exited each symbol. Guards against
        # re-trading the same bar (see reentry_cooldown_bars).
        self._last_action_bar: dict[str, object] = {}
        # Cache of real MCX margins keyed by (symbol, quantity, side), so we hit the
        # broker's margin API at most once per distinct size per session instead of
        # on every tick. Margin drifts intraday but not enough to matter for a
        # funding check; the cache is dropped on restart, which re-fetches fresh.
        self._mcx_margin_cache: dict[tuple, float] = {}
        # Cache of broker.available_funds() (account-wide, real ₹ free balance),
        # refreshed at most every BROKER_FUNDS_TTL_SECONDS — see that constant.
        self._funds_cache: tuple[float, Optional[float]] = (0.0, None)
        # Cache of broker.get_position() per symbol (real ₹ avg price/PnL),
        # refreshed at most every BROKER_POSITION_TTL_SECONDS — see that constant.
        self._broker_position_cache: dict[str, tuple[float, Optional[dict]]] = {}
        # Cache of broker.get_all_positions() (the full broker-truth position
        # book, for the dashboard's Broker Positions panel) — same TTL.
        self._all_positions_cache: tuple[float, list] = (0.0, [])

        # LIVE-only risk-guardrail bookkeeping (risk_manager.py). Trades-today is
        # rebuilt from storage on start (see _rehydrate), not just counted
        # in-memory, so a restart mid-day doesn't reset a max-trades kill switch.
        self._trades_today = 0
        self._live_halt_logged = False

        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()

    # -- lifecycle ---------------------------------------------------------- #
    def start(self) -> None:
        if self.state.running:
            return
        self._stop.clear()

        # Broker: real if creds exist for the chosen env, else Simulated.
        # A non-empty broker_access_token (a client's own OAuth token) takes
        # priority over the shared .env credentials — see make_broker.
        self.broker = make_broker(self.environment, self.broker_choice,
                                  access_token=self.broker_access_token,
                                  api_key=self.broker_api_key)

        # Feed: real Upstox data if a token is present, else simulated.
        # Market data is READ-ONLY, so we prefer the LIVE token (which is the one
        # that stays valid) for real candles even in Paper mode — that gives true
        # paper trading = real market data + simulated execution. Fall back to the
        # sandbox token, then to the simulator.
        token = config.UPSTOX_LIVE_ACCESS_TOKEN or config.UPSTOX_SANDBOX_TOKEN
        self.feed = make_feed(prefer_real=bool(token), access_token=token,
                              mode=self.mode)
        self._start_feed_resilient()

        # SAFETY: never place real orders against simulated prices.
        #
        # make_broker returns a REAL broker whenever the caller has a valid
        # token, but make_feed falls back to SimulatedFeed whenever the admin
        # market-data token is missing or expired. In Live that combination
        # would size and fire genuine orders off invented candles, so refuse
        # to run instead. Paper is unaffected: simulated fills against
        # simulated prices is a coherent (and intended) mode.
        if (self.environment == Environment.LIVE
                and isinstance(self.feed, SimulatedFeed)):
            self.feed.stop()
            raise RuntimeError(
                "Live trading blocked: real market data is unavailable, so the "
                "bot would be trading on simulated prices. Refresh the Upstox "
                "market-data token and try again.")

        with self.state.lock:
            self.state.running = True
            self.state.broker_name = self.broker.name
            self.state.db_backend = self.db.backend
            self.state.feed_status = self.feed.status()

        # Rebuild live state from disk BEFORE the loop starts, so a restart picks up
        # exactly where it left off: today's PnL, the day-wise history, and any
        # still-open positions all come back instead of resetting to zero.
        self._rehydrate()

        self.state.push_log(
            f"Bot started — {self.environment.value} / {self.mode.value} / "
            f"strategy: {self.strategy.name} / {self.broker.name} / "
            f"{len(self.instruments)} instruments / {self.params.timeframe} / "
            f"risk {self.params.risk_per_trade:.1%} per trade / "
            f"RR 1:{self.params.risk_reward:g} / "
            f"SL {self.params.atr_sl_mult:g}×ATR({self.params.atr_period})")
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def _start_feed_resilient(self) -> None:
        """Start the chosen feed; if a real feed fails (no data / auth / network),
        fall back to the SimulatedFeed so Start Bot never throws in the UI."""
        try:
            self.feed.start(self.instruments)
        except Exception as exc:
            self.state.push_log(
                f"⚠️ Live feed unavailable ({exc}); using simulated feed.")
            self.feed = SimulatedFeed(mode=self.mode)
            self.feed.start(self.instruments)

    def stop(self) -> None:
        self._stop.set()
        if self.feed:
            self.feed.stop()
        with self.state.lock:
            self.state.running = False
            self.state.feed_status = "🔴 Stopped"
        self.state.push_log("Bot stopped.")

    # -- persistence / rehydration ------------------------------------------ #
    def _rehydrate(self) -> None:
        """Rebuild live state from the trade store on (re)start.

        Three things are restored so a restart resumes rather than resets:
          * OPEN positions are put back under management, with their entry time and
            committed margin reconstructed. Without this a restart would leave them
            orphaned as OPEN in the DB forever, and the engine — seeing no position
            — could open a SECOND one on the same symbol.
          * realized_pnl is set to TODAY's booked PnL, read back from disk, so the
            dashboard shows the real day figure instead of ₹0.
          * the day-wise history table is loaded for the dashboard.
        """
        self.state.trading_day = self.db.today_key()
        open_positions: dict[str, dict] = {}
        try:
            for trade in self.db.get_open_trades(self.environment, user_id=self.user_id):
                sym = trade.get("ticker")
                if not sym:
                    continue
                inst = config.INSTRUMENTS_BY_SYMBOL.get(sym)
                mult = int(trade.get("contract_multiplier", 1) or 1)
                # Local-only fields (leading underscore = never persisted) the live
                # loop needs, rebuilt so a reloaded position behaves like a fresh
                # one. _live_price seeds from entry until the first tick updates it.
                trade["_live_price"] = trade["entry_price"]
                trade["_entry_dt"] = self._reconstruct_entry_dt(trade)
                notional = trade["entry_price"] * trade["quantity"] * max(mult, 1)
                if inst is not None and inst.segment == Segment.MCX:
                    # Restore the broker's REAL margin (same source as entry) so
                    # available-capital accounting survives a restart intact.
                    trade["_margin"], _ = self._mcx_margin(
                        inst, trade["quantity"], trade.get("side", "BUY"))
                else:
                    lev = (config.max_leverage_for(inst.segment, self.params)
                           if inst else self.params.max_leverage)
                    # Best-effort: mirror _tick's LIVE equity-leverage override so
                    # a restart's committed-margin accounting matches what a fresh
                    # entry would book right now. If the risk panel's
                    # intraday_leverage has changed since this trade was actually
                    # opened, this is an approximation, same as the rest of
                    # rehydration — it only affects capital bookkeeping, never the
                    # position's real SL/TP/quantity.
                    if (self.environment == Environment.LIVE and inst is not None
                            and inst.segment == Segment.EQUITY
                            and self.mode in (Mode.INTRADAY, Mode.SCALPER)):
                        rl = risk_manager.get_limits(self.user_id)
                        lev = min(self.params.max_leverage,
                                 max(rl.intraday_leverage, 1.0))
                    trade["_margin"] = notional / lev if lev > 0 else notional
                open_positions[sym] = trade
        except Exception as exc:                       # never block start on this
            self.state.push_log(f"⚠️ Could not reload open positions ({exc}).")

        today_real = self.db.today_realized(self.environment, user_id=self.user_id)
        self._trades_today = self.db.trades_opened_today(self.environment, user_id=self.user_id)
        self._live_halt_logged = False
        with self.state.lock:
            self.state.open_positions = open_positions
            self.state.realized_pnl = today_real
        self._refresh_daily()
        self._recompute_unrealized()
        if open_positions or today_real:
            self.state.push_log(
                f"Restored {len(open_positions)} open position(s) and today's "
                f"realized PnL ₹{today_real:,.2f} from storage.")

    @staticmethod
    def _reconstruct_entry_dt(trade: dict) -> datetime:
        """Local-clock entry time that PRESERVES elapsed duration across a restart,
        so the Scalper's time-exit still measures real minutes held rather than
        minutes since the reload (the stored timestamp is UTC)."""
        try:
            entry_utc = datetime.fromisoformat(str(trade["timestamp"]))
            return now_ist() - (datetime.utcnow() - entry_utc)
        except Exception:
            return now_ist()

    def _refresh_daily(self) -> None:
        """Reload the day-wise history snapshot from storage into state."""
        try:
            df = self.db.daily_pnl(self.environment, user_id=self.user_id)
            rows = df.to_dict("records") if not df.empty else []
        except Exception as exc:
            self.state.push_log(f"⚠️ Daily PnL refresh failed ({exc}).")
            rows = []
        with self.state.lock:
            self.state.daily_pnl = rows

    def _rollover_if_new_day(self) -> None:
        """At the first tick of a new trading day (00:00 UTC / 05:30 IST, before the
        market opens) reset the live day-PnL counter to that day's total (≈0). Past
        days are untouched on disk — only the LIVE counter rolls; the history keeps
        every day."""
        today = self.db.today_key()
        if today == self.state.trading_day:
            return
        today_real = self.db.today_realized(self.environment, user_id=self.user_id)
        self._trades_today = self.db.trades_opened_today(self.environment, user_id=self.user_id)
        self._live_halt_logged = False
        with self.state.lock:
            self.state.trading_day = today
            self.state.realized_pnl = today_real
        self._refresh_daily()
        self.state.push_log(f"📆 New trading day {today} — daily PnL reset.")

    # -- main loop ---------------------------------------------------------- #
    def _run_loop(self) -> None:
        while not self._stop.is_set():
            try:
                self._tick()
            except Exception as exc:  # never let one bad tick kill the bot
                self.state.push_log(f"⚠️ tick error: {exc}")
            self._stop.wait(self.poll_seconds)

    def _tick(self) -> None:
        now_t = now_ist().time()
        # Reset the live day-PnL counter when the calendar day turns over.
        self._rollover_if_new_day()
        with self.state.lock:
            self.state.feed_status = self.feed.status()

        # LIVE-only risk guardrails (risk_manager.py). Computed ONCE per tick —
        # never in Paper/backtest, and never touching strategy/signal logic,
        # only whether/how big a NEW entry may be. Read fresh every tick (not
        # cached at start()) so editing the sidebar's risk panel takes effect
        # on THIS running bot immediately.
        live_limits, live_halt_reason = self._live_risk_check()

        for inst in self.instruments:
            hours = market_hours_for_segment(inst.segment)
            market_open = hours.is_open(now_t)

            quote = self.feed.get_quote(inst)
            self._publish_quote(inst, quote, market_open)

            df = self.feed.get_candles(inst, lookback=260)
            live_price, price_src = self._live_price(quote, df)
            if live_price is None:
                continue

            bar_ts = df.index[-1] if not df.empty else None

            # 1) manage an open position (exit on SL/TP/time) regardless of signals
            if inst.symbol in self.state.open_positions:
                if self._manage_open(inst, live_price) and bar_ts is not None:
                    # Record the exit bar so the cooldown starts from here.
                    self._last_action_bar[inst.symbol] = bar_ts
                continue

            # 2) only look for new entries while that segment's market is open
            if not market_open or df.empty:
                continue

            # LIVE risk kill-switch: existing open positions above are still
            # managed to their SL/TP/time-exit either way — this only blocks
            # OPENING new risk once today's loss/trade-count ceiling is hit.
            if live_halt_reason:
                continue

            # 3) never act twice on the same candle. The strategy re-evaluates the
            # same bar every poll, so without this an exit is immediately followed
            # by re-entry into the identical setup — a loss loop. It also stops us
            # trading a feed whose bars have stopped advancing.
            if not self._cooldown_elapsed(inst.symbol, df):
                continue

            # The segment's own open time drives the "skip the first 15 minutes"
            # filter — equity skips to 09:30, MCX to 09:15.
            sig = run_strategy(self.strategy, df, session_open=hours.open_t)
            if sig is None:
                continue

            lev = config.max_leverage_for(inst.segment, self.params)
            # LIVE-only: equity Intraday/Scalper sizes against the REAL MIS
            # leverage configured in the risk panel instead of the pinned 1x
            # (config.SEGMENT_MAX_LEVERAGE) used everywhere else — Paper/backtest
            # and Swing (overnight = delivery = 1x, non-negotiable) are untouched.
            # The strategy's own ceiling (self.params.max_leverage) still bounds
            # it either way, so this can only move the effective cap DOWN from 15x,
            # never past the mode's own limit.
            if (live_limits is not None and inst.segment == Segment.EQUITY
                    and self.mode in (Mode.INTRADAY, Mode.SCALPER)):
                lev = min(self.params.max_leverage,
                         max(live_limits.intraday_leverage, 1.0))
            available = self._available_capital()
            # Two completely separate sizing paths — kept apart on purpose so the
            # commodity logic never leaks into equity (and vice-versa):
            #   * MCX commodity  -> FIXED lots the user configured in the sidebar.
            #                       No risk %, no 20% cap; the only limit is whether
            #                       the account can fund the margin (checked inside).
            #   * NSE equity/etc -> risk-based sizing (qty = risk_budget / stop
            #                       distance), bounded by leverage and the 20%-of-
            #                       account cap. Sized against AVAILABLE capital so an
            #                       unfundable trade floors to qty=0 and is skipped.
            mcx_skip: Optional[str] = None
            if inst.segment == Segment.MCX:
                qty, risk_amt, mcx_skip = self._mcx_fixed_size(
                    inst, sig, available, lev)
            else:
                qty, risk_amt = position_size(
                    available, sig, self.params, inst.lot_size,
                    inst.contract_multiplier, lev,
                    account_capital=self.total_capital)

            # LIVE-only: hard clip on the SIZE of a single order, independent of
            # (and always at least as strict as) the risk/notional maths above.
            live_qty_clip = ""
            if (live_limits is not None and live_limits.max_qty_per_trade > 0
                    and qty > live_limits.max_qty_per_trade):
                live_qty_clip = (f" (clipped from {qty} by the live max-qty/"
                                 f"trade cap of {live_limits.max_qty_per_trade})")
                qty = live_limits.max_qty_per_trade

            # Capital this trade would deploy = notional ÷ effective leverage —
            # the SAME margin figure booked as `_margin` on entry (see _enter).
            # Shown so the dashboard's signal row states what the trade ties up;
            # qty=0 (skipped) correctly reads ₹0 deployed.
            mult = max(inst.contract_multiplier, 1)
            notional = sig.entry_price * qty * mult
            deployed = notional / lev if lev > 0 else notional

            # LIVE-only: independent real-broker-funds gate. Every ₹ ceiling
            # above (capital_allocated, risk sizing, notional cap) is OUR
            # bookkeeping of what should be free — this instead asks the
            # broker directly what actually IS free right now, catching drift
            # from anything outside the bot (manual trades, other margin use
            # in the same account). Skips the trade rather than trusting our
            # own accounting alone. Brokers that don't implement
            # available_funds() (returns None) are silently skipped — this is
            # an ADDITIONAL check, never the only one.
            live_funds_block = ""
            if live_limits is not None and qty > 0:
                broker_avail = self._broker_available_funds()
                if broker_avail is not None and deployed > broker_avail:
                    live_funds_block = (
                        f"{inst.symbol}: signal skipped — broker reports only "
                        f"₹{broker_avail:,.0f} available but this trade needs "
                        f"₹{deployed:,.0f} (live funds check).")
                    qty = 0

            sig_row = {
                "time": now_ist().strftime("%H:%M:%S"),
                "symbol": inst.symbol, "segment": inst.segment.value,
                "side": sig.side, "entry": round(sig.entry_price, 2),
                "stop": round(sig.stop_loss, 2), "target": round(sig.target, 2),
                "rr": round(sig.risk_reward, 2), "qty": qty,
                "deployed": round(deployed, 2), "reason": sig.reason,
            }
            with self.state.lock:
                self.state.last_signals.insert(0, sig_row)
                self.state.last_signals = self.state.last_signals[:50]

            if qty <= 0:
                # The live broker-funds gate has its own precise reason —
                # report that verbatim rather than the generic maths below,
                # which would otherwise describe a trade that funds check
                # already rejected for an unrelated reason.
                if live_funds_block:
                    self.state.push_log(live_funds_block)
                    continue
                # MCX fixed-lot sizing carries its OWN skip reason (zero lots
                # configured, or the account can't fund the margin for the lots
                # chosen). Report that verbatim instead of the equity risk maths.
                if mcx_skip:
                    self.state.push_log(mcx_skip)
                    continue
                # Report BOTH bounds — qty=0 can mean "stop too wide for the risk
                # budget" or "position too expensive to fund from available cash",
                # and the fix differs.
                mult = max(inst.contract_multiplier, 1)
                pct_budget = available * self.params.risk_per_trade
                cash_cap = self.params.risk_per_trade_cash
                risk_budget = (min(cash_cap, pct_budget)
                               if cash_cap and cash_cap > 0 else pct_budget)
                by_risk = risk_budget / max(
                    abs(sig.entry_price - sig.stop_loss) * mult, 1e-9)
                by_notional = ((available * lev)
                               / max(sig.entry_price * mult, 1e-9))
                cap_pct = getattr(self.params, "max_capital_per_trade_pct", 0.0)
                cap_msg = ""
                if cap_pct and cap_pct > 0:
                    by_capital = ((self.total_capital * cap_pct)
                                  / max(sig.entry_price * mult, 1e-9))
                    cap_msg = (f" {cap_pct:.0%}-of-account capital cap allows "
                               f"{by_capital:.2f};")
                self.state.push_log(
                    f"{inst.symbol}: signal skipped (qty=0). Available capital "
                    f"₹{available:,.0f}; risk budget ₹{risk_budget:,.0f} allows "
                    f"{by_risk:.2f} units; {lev:g}x notional cap allows "
                    f"{by_notional:.2f};{cap_msg} lot size is {inst.lot_size}.")
                continue

            if self._enter(inst, sig, qty, risk_amt, quote, live_qty_clip, lev):
                self._last_action_bar[inst.symbol] = bar_ts

    # -- commodity (MCX) fixed-lot sizing ----------------------------------- #
    def _mcx_fixed_size(
        self, inst: Instrument, sig, available: float, lev: float
    ) -> tuple[int, float, Optional[str]]:
        """Size a commodity trade at the FIXED number of lots the user configured,
        completely bypassing the risk-% / leverage / 20%-of-account maths equity
        uses. Returns (quantity, risk_amount, skip_reason).

        Contract: on a valid trade skip_reason is None; when the trade cannot be
        taken it is a human-readable string and quantity is 0 (the caller logs it
        and moves on). The two ways it is skipped:
          * lots configured as 0 for this symbol, or
          * the account's AVAILABLE capital can't post the margin the chosen lots
            require. There is deliberately NO percentage cap here (commodities may
            deploy the whole account per the user's instruction) — the margin the
            account can actually fund is the only ceiling.

        Quantity is the LOT COUNT (not lots × lot_size). The broker's margin/order
        APIs treat MCX quantity as lots (quantity=1 => one contract), and every
        downstream consumer — margin, notional, PnL — multiplies by
        contract_multiplier, which is the per-lot point value. So one lot of
        CRUDEOILM is quantity=1 needing ~₹24.75k, never ₹247k. risk_amount is
        lots × price-distance × contract_multiplier. SL/TP price levels are the
        strategy's — this method only decides HOW MANY, never WHERE."""
        lots = self.mcx_lots.get(inst.symbol, 1)
        if lots <= 0:
            return 0, 0.0, (f"{inst.symbol}: signal skipped — 0 lots configured "
                            f"for this commodity (set lots in MCX settings).")
        qty = lots                           # quantity is a lot count
        mult = max(inst.contract_multiplier, 1)
        per_unit = abs(sig.entry_price - sig.stop_loss)
        risk_amt = lots * per_unit * mult
        # REAL margin the broker blocks for these lots (SPAN + exposure + peak),
        # fetched live and cached — the same figure booked as `_margin` on entry, so
        # the fund check and later bookkeeping agree exactly.
        margin, _src = self._mcx_margin(inst, qty, sig.side)
        if margin > available:
            return 0, 0.0, (
                f"{inst.symbol}: signal skipped — {lots} lot(s) need "
                f"₹{margin:,.0f} margin but only ₹{available:,.0f} is available. "
                f"Lower the lots or free capital.")
        return qty, risk_amt, None

    def _mcx_margin(
        self, inst: Instrument, qty: int, side: str = "BUY"
    ) -> tuple[float, str]:
        """Real (₹) margin to hold `qty` of an MCX instrument, with its source.

        PAPER mode uses the hardcoded per-lot table (config.MCX_MARGIN_PER_LOT) as
        the source of truth — the paper bot should not depend on a live token, and
        the table holds the user's real broker-side per-lot figures. LIVE mode
        prefers the broker's own margin API and only falls back to the table.

        Order of preference:
          PAPER:  "hardcoded" table  ->  "notional" (last resort)
          LIVE:   "broker" API  ->  "upstox" direct call  ->  "hardcoded"  ->  "notional"

        `qty` is the LOT COUNT, so margin = per_lot × qty directly (and the live
        Upstox API, which also counts lots, gets the right quantity).
        Cached per (symbol, qty, side) so the API is hit at most once per size."""
        key = (inst.symbol, int(qty), side)
        if key in self._mcx_margin_cache:
            return self._mcx_margin_cache[key], "cache"

        margin, src = 0.0, ""
        # LIVE-only: consult the broker's real margin API first. In PAPER we skip
        # straight to the hardcoded table below.
        if self.environment == Environment.LIVE:
            if self.broker is not None:
                m = self.broker.required_margin(inst, qty, side)
                if m and m > 0:
                    margin, src = float(m), "broker"
            if margin <= 0:
                token = config.UPSTOX_LIVE_ACCESS_TOKEN or config.UPSTOX_SANDBOX_TOKEN
                m = fetch_upstox_margin(token, inst.instrument_key, qty, side, "D")
                if m and m > 0:
                    margin, src = float(m), "upstox"
        if margin <= 0:
            per_lot = config.mcx_margin_per_lot(inst.symbol)
            if per_lot > 0:
                margin, src = per_lot * qty, "hardcoded"   # qty is the lot count
        if margin <= 0:                      # nothing else worked — rough estimate
            lev = config.max_leverage_for(inst.segment, self.params)
            notional = inst.reference_price * qty * max(inst.contract_multiplier, 1)
            margin = notional / lev if lev > 0 else notional
            src = "notional"

        self._mcx_margin_cache[key] = margin
        return margin, src

    # -- re-entry guard ----------------------------------------------------- #
    def _cooldown_elapsed(self, symbol: str, df) -> bool:
        """True if enough NEW bars have closed since we last traded this symbol.

        Counts bars strictly after the last action, so acting on the same bar
        twice is impossible regardless of how fast the poll loop spins.
        """
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

    # -- capital tracking --------------------------------------------------- #
    def _available_capital(self) -> float:
        """The capital ceiling minus the margin already committed to open
        positions.

        Each open trade records `_margin` at entry = its notional ÷ effective
        leverage — the real cash the broker ties up. For equity (pinned to 1×,
        or the LIVE risk panel's real MIS leverage — see _tick) that is the
        notional ÷ that leverage. For MCX (~15×) it is the futures margin,
        which is what keeps a ₹1.4cr GOLD contract fundable at all. Sizing off
        this figure stops concurrent positions from collectively
        over-committing the account.

        The ceiling itself is `total_capital` (the sidebar figure) — EXCEPT in
        LIVE, where the risk panel's `capital_allocated` (if set) caps it
        further. This is what makes "only use the capital I've allocated
        today, not my whole broker balance" strict: sizing never sees more
        than min(total_capital, capital_allocated).
        """
        with self.state.lock:
            committed = sum(float(t.get("_margin", 0.0))
                            for t in self.state.open_positions.values())
        ceiling = self.total_capital
        if self.environment == Environment.LIVE:
            limits = risk_manager.get_limits(self.user_id)
            if limits.capital_allocated and limits.capital_allocated > 0:
                ceiling = min(ceiling, limits.capital_allocated)
        return max(ceiling - committed, 0.0)

    def _live_risk_check(self) -> tuple[Optional["risk_manager.LiveRiskLimits"],
                                        Optional[str]]:
        """LIVE-only: resolve the current risk-panel limits and whether new
        entries should be halted for the rest of today (daily-loss kill switch
        or max-trades-per-day). Returns (None, None) in Paper — this function
        never touches Paper/backtest behaviour. Also publishes the resolved
        status to state.risk_status for the dashboard, and logs a halt exactly
        ONCE per day (not every tick) via self._live_halt_logged."""
        if self.environment != Environment.LIVE:
            return None, None
        limits = risk_manager.get_limits(self.user_id)
        base_capital = (limits.capital_allocated
                        if limits.capital_allocated and limits.capital_allocated > 0
                        else self.total_capital)
        loss_cap = risk_manager.daily_loss_limit(limits, base_capital)
        realized = self.state.realized_pnl
        halt_reason: Optional[str] = None
        if loss_cap > 0 and realized <= -loss_cap:
            halt_reason = (f"daily loss limit hit (realized ₹{realized:,.2f} "
                           f"<= -₹{loss_cap:,.2f})")
        elif (limits.max_trades_per_day > 0
              and self._trades_today >= limits.max_trades_per_day):
            halt_reason = (f"max trades/day reached ({self._trades_today}/"
                           f"{limits.max_trades_per_day})")

        if halt_reason and not self._live_halt_logged:
            self.state.push_log(
                f"🛑 LIVE new-entry halt: {halt_reason}. Existing open "
                "positions are still managed to their SL/TP/time-exit.")
            self._live_halt_logged = True
        elif not halt_reason:
            self._live_halt_logged = False

        with self.state.lock:
            self.state.risk_status = {
                "capital_allocated": base_capital,
                "trades_today": self._trades_today,
                "max_trades_per_day": limits.max_trades_per_day,
                "max_qty_per_trade": limits.max_qty_per_trade,
                "intraday_leverage": limits.intraday_leverage,
                "daily_loss_limit": loss_cap,
                "realized_pnl": realized,
                "halted": bool(halt_reason),
                "halt_reason": halt_reason or "",
            }
        return limits, halt_reason

    def _broker_available_funds(self) -> Optional[float]:
        """Cached wrapper over broker.available_funds() — see
        BROKER_FUNDS_TTL_SECONDS for why this isn't fetched on every tick."""
        now = _time.time()
        ts, val = self._funds_cache
        if now - ts < BROKER_FUNDS_TTL_SECONDS:
            return val
        try:
            val = self.broker.available_funds() if self.broker else None
        except Exception:
            val = None
        self._funds_cache = (now, val)
        return val

    # -- live pricing (WebSocket is the source of truth) --------------------- #
    @staticmethod
    def _live_price(quote: Optional[LiveQuote],
                    df) -> tuple[Optional[float], str]:
        """Resolve the price used to mark positions and fire stops.

        Order matters: a fresh WebSocket tick wins over everything. Only when no
        usable tick exists do we read a candle close, and a stale tick is the last
        resort — never preferred over fresher candle data.
        """
        if (quote is not None and quote.ltp > 0
                and quote.age_seconds <= STALE_QUOTE_SECONDS):
            return float(quote.ltp), quote.source
        if df is not None and not df.empty:
            return float(df["close"].iloc[-1]), "candle"
        if quote is not None and quote.ltp > 0:
            return float(quote.ltp), f"{quote.source}:stale"
        return None, "none"

    def _publish_quote(self, inst: Instrument, quote: Optional[LiveQuote],
                       market_open: bool) -> None:
        """Mirror the live tick into UI state so the dashboard can prove the
        WebSocket is alive per instrument, not just globally."""
        row = {
            "Symbol": inst.symbol,
            "Segment": "MCX" if inst.segment == Segment.MCX else "NSE",
            "Market": "🟢 OPEN" if market_open else "⚪ CLOSED",
            "LTP": round(quote.ltp, 2) if quote else None,
            "Bid": round(quote.bid, 2) if quote and quote.bid else None,
            "Ask": round(quote.ask, 2) if quote and quote.ask else None,
            "Source": quote.source.upper() if quote else "—",
            "Tick age (s)": round(quote.age_seconds, 1) if quote else None,
        }
        with self.state.lock:
            self.state.live_quotes[inst.symbol] = row

    def _entry_limit_price(self, inst: Instrument, sig,
                           quote: Optional[LiveQuote]) -> Optional[float]:
        """A limit price one tick inside the spread (scalping.md §4).

        Requires a genuine two-sided quote from the WebSocket depth — without one
        the spread is unknown, so we return None and let the caller send a market
        order rather than invent a price.
        """
        if not self.params.use_limit_entry:
            return None
        if quote is None or quote.bid <= 0 or quote.ask <= 0 or quote.ask < quote.bid:
            return None
        tick = inst.tick_size or 0.05
        offset = self.params.limit_offset_ticks * tick
        if sig.side == "BUY":
            px = min(quote.bid + offset, quote.ask)   # improve the bid, never cross
        else:
            px = max(quote.ask - offset, quote.bid)   # improve the ask, never cross
        return round(round(px / tick) * tick, 2)

    # -- order handling ----------------------------------------------------- #
    def _enter(self, inst: Instrument, sig, qty: int, risk_amt: float,
               quote: Optional[LiveQuote] = None, qty_clip_note: str = "",
               lev: Optional[float] = None) -> bool:
        """Returns True if a position was opened.

        `lev` is the EFFECTIVE leverage the caller sized this trade against
        (may be the LIVE risk panel's real intraday_leverage override for
        equity — see _tick) and MUST be reused for the `_margin` booked below,
        never recomputed from config.max_leverage_for alone: recomputing would
        silently disagree with what was actually sized, corrupting
        _available_capital's committed-margin accounting."""
        limit_px = self._entry_limit_price(inst, sig, quote)
        if limit_px is not None:
            res = self.broker.place_limit_order(inst, sig.side, qty, limit_px,
                                                sig.entry_price)
            how = f"LIMIT {limit_px:.2f} (bid {quote.bid:.2f} / ask {quote.ask:.2f})"
        else:
            res = self.broker.place_market_order(inst, sig.side, qty,
                                                 sig.entry_price)
            how = "MARKET"
        if not res.ok:
            self.state.push_log(f"{inst.symbol}: order rejected — {res.message}")
            return False

        # Re-anchor the risk levels to the price we ACTUALLY filled at. The signal
        # sized its stop off the candle close, but a limit can fill elsewhere, and
        # the RR contract (1:1 scalp, 1:2 intraday, 1:3 swing) is defined from the
        # real entry. Distance is preserved; only the anchor moves.
        fill = res.filled_price or sig.entry_price
        risk_dist = abs(sig.entry_price - sig.stop_loss)
        # Snap the stop to the instrument's tick grid — an off-tick stop price is
        # not a placeable order. Round AWAY from entry (down for a long, up for a
        # short) so rounding never tightens the stop below the intended risk, then
        # recompute the target from the rounded distance so the RR contract (1:1
        # scalp / 1:2 intraday / 1:3 swing) stays exact.
        tick = inst.tick_size or 0.05
        if sig.side == "BUY":
            stop = round(math.floor((fill - risk_dist) / tick) * tick, 2)
        else:
            stop = round(math.ceil((fill + risk_dist) / tick) * tick, 2)
        risk_dist = abs(fill - stop)
        if risk_dist <= 0:                      # degenerate: tick wider than stop
            self.state.push_log(
                f"{inst.symbol}: order rejected — stop rounds onto entry "
                f"(tick {tick}).")
            self.broker.square_off(inst, sig.side, qty, fill)  # unwind the fill
            return False
        if sig.side == "BUY":
            target = round(fill + self.params.risk_reward * risk_dist, 2)
        else:
            target = round(fill - self.params.risk_reward * risk_dist, 2)

        # LIVE-only: mirror this trade's SL/TP as a REAL protective order at
        # the broker (e.g. a Kite GTT OCO), so it's still protected if this
        # bot goes offline — not just watched by our own polling loop. A
        # failure here does NOT unwind the entry: the position is already
        # correctly opened and the engine's own SL/TP/time-exit still manages
        # it exactly as before this existed; it just won't have the extra
        # broker-side safety net for this one trade.
        gtt_id = ""
        if self.environment == Environment.LIVE:
            ok, gtt_id, gtt_msg = self.broker.place_oco_exit(
                inst, sig.side, qty, stop, target, fill)
            if ok:
                self.state.push_log(
                    f"{inst.symbol}: broker-side SL/TP protection placed "
                    f"(GTT {gtt_id}).")
            elif gtt_msg and gtt_msg != "not supported by this broker":
                self.state.push_log(
                    f"⚠️ {inst.symbol}: broker-side SL/TP protection failed "
                    f"({gtt_msg}) — still protected by the bot's own polling.")

        trade = self.db.new_trade(
            mode=self.mode.value, environment=self.environment.value,
            user_id=self.user_id,
            broker=self.broker.name, ticker=inst.symbol, side=sig.side,
            entry_price=fill, stop_loss=stop, target=target, quantity=qty,
            risk_amount=risk_amt, segment=inst.segment.value,
            contract_multiplier=inst.contract_multiplier,
            strategy=self.strategy.key, entry_reason=sig.reason,
            broker_gtt_id=gtt_id,
        )
        self.db.insert_trade(trade, self.environment)
        # Local-only bookkeeping (leading underscore = never persisted): set after
        # insert so the stored document stays exactly the Section-5 schema.
        trade["_live_price"] = fill
        trade["_entry_dt"] = now_ist()
        # Capital this position ties up, deducted from available capital until it
        # closes (see _available_capital). EQUITY is 1×, so it commits its full
        # notional. MCX posts the broker's REAL futures margin (SPAN + exposure +
        # peak) fetched live — not notional ÷ leverage, which is nowhere near right.
        notional = fill * qty * max(inst.contract_multiplier, 1)
        if inst.segment == Segment.MCX:
            trade["_margin"], _msrc = self._mcx_margin(inst, qty, sig.side)
        else:
            eff_lev = lev if lev is not None else config.max_leverage_for(
                inst.segment, self.params)
            trade["_margin"] = notional / eff_lev if eff_lev > 0 else notional
        with self.state.lock:
            self.state.open_positions[inst.symbol] = trade
        self._trades_today += 1
        self.state.push_log(
            f"ENTER {inst.symbol} {sig.side} qty={qty} @ {fill:.2f} via {how} | "
            f"SL={stop:.2f} TP={target:.2f} | RR 1:{self.params.risk_reward:g} "
            f"| risk ₹{risk_amt:,.0f} ({risk_amt / self.total_capital:.2%}) "
            f"| notional ₹{notional:,.0f} "
            f"({notional / self.total_capital:.2f}x capital)"
            f"{qty_clip_note} | {sig.reason}")
        return True

    def _manage_open(self, inst: Instrument, live_price: float) -> bool:
        """Returns True if the position was closed on this tick."""
        # .get() not [] — a manual close on the UI thread may have removed this
        # position between the caller's membership check and here.
        trade = self.state.open_positions.get(inst.symbol)
        if trade is None:
            return False
        trade["_live_price"] = live_price
        exit_price, reason = None, ""

        # Direction-aware by necessity: a SHORT's stop sits ABOVE its entry and its
        # target BELOW. Long-only comparisons would stop out every short instantly.
        if trade["side"] == "BUY":
            if live_price <= trade["stop_loss"]:
                exit_price, reason = trade["stop_loss"], "STOP-LOSS"
            elif live_price >= trade["target"]:
                exit_price, reason = trade["target"], "TARGET"
        else:
            if live_price >= trade["stop_loss"]:
                exit_price, reason = trade["stop_loss"], "STOP-LOSS"
            elif live_price <= trade["target"]:
                exit_price, reason = trade["target"], "TARGET"

        # Time exit (scalping.md §4): a scalp that hasn't resolved has lost its
        # edge — close it at market rather than let it drift.
        if exit_price is None and self.params.max_hold_minutes > 0:
            entered = trade.get("_entry_dt")
            if entered is not None:
                held_min = (now_ist() - entered).total_seconds() / 60.0
                if held_min >= self.params.max_hold_minutes:
                    exit_price = live_price
                    reason = f"TIME-EXIT ({self.params.max_hold_minutes}m)"

        if exit_price is None:
            self._recompute_unrealized()
            return False

        # Claim the position (remove it under the lock) BEFORE any broker/DB work,
        # so a concurrent manual close can't square the same position off twice.
        with self.state.lock:
            claimed = self.state.open_positions.pop(inst.symbol, None)
        if claimed is None:
            return True                    # already closed (e.g. manual button)
        self._finalize_close(inst, claimed, exit_price, reason)
        return True

    def close_position(self, symbol: str) -> bool:
        """Manually close ONE open position now, at the latest live price. Invoked
        from the UI's per-trade ❌ button. Returns False if the symbol isn't open
        (already closed, or never was).

        Thread-safe with the trading loop: the position is claimed (popped under the
        lock) before any order is sent, so the loop's own SL/TP/time exit cannot
        also close it — whichever thread pops first owns the close."""
        with self.state.lock:
            trade = self.state.open_positions.pop(symbol, None)
        if trade is None:
            return False
        inst = config.INSTRUMENTS_BY_SYMBOL.get(symbol)
        if inst is None:
            # Open positions come from configured instruments, so this is virtually
            # impossible — but never drop a claimed trade on the floor: restore it.
            with self.state.lock:
                self.state.open_positions[symbol] = trade
            self.state.push_log(
                f"⚠️ Manual close failed: {symbol} not in instrument list.")
            return False
        exit_price = float(trade.get("_live_price", trade["entry_price"]))
        self._finalize_close(inst, trade, exit_price, "MANUAL")
        return True

    # -- broker-truth positions (LIVE only) ---------------------------------- #
    def broker_positions(self) -> list[dict]:
        """EVERY non-flat position the broker reports right now — the ground
        truth, independent of this bot's own open_positions tracking (which
        can drift: a position closed manually in the broker's own app, or by
        anything outside this bot, still shows here even though our DB never
        heard about it — see close_broker_position). Empty in Paper, or if
        the broker doesn't support it. Cached briefly like _broker_position."""
        if self.environment != Environment.LIVE or self.broker is None:
            return []
        now = _time.time()
        ts, val = self._all_positions_cache
        if now - ts < BROKER_POSITION_TTL_SECONDS:
            return val
        try:
            val = self.broker.get_all_positions()
        except Exception:
            val = []
        self._all_positions_cache = (now, val)
        return val

    def close_broker_position(self, symbol: str, quantity: int,
                              side: str) -> tuple[bool, str]:
        """Square off a REAL broker position directly, sourced from the
        broker's own book (broker_positions()) rather than our internal
        open_positions — this is what lets the UI close ANY real position,
        including ones this bot never opened or has since lost track of.

        If `symbol` also matches a bot-tracked OPEN trade, that trade is
        closed the NORMAL way (_finalize_close: cancels its GTT, updates the
        DB, books PnL) so the two views can't drift apart from this action.
        If it doesn't match anything tracked, this places a plain square-off
        order and logs it — there's no DB trade to reconcile, since the bot
        never opened it.

        `side`/`quantity` MUST come from the broker's own reported position
        (broker_positions()), not guessed — squaring off the wrong side or
        size would open a NEW position instead of closing this one."""
        inst = config.INSTRUMENTS_BY_SYMBOL.get(symbol)
        if inst is None:
            return False, f"{symbol}: not a configured instrument."

        with self.state.lock:
            tracked = self.state.open_positions.pop(symbol, None)
        if tracked is not None:
            exit_price = float(tracked.get("_live_price", tracked["entry_price"]))
            pnl = self._finalize_close(inst, tracked, exit_price, "MANUAL")
            self._all_positions_cache = (0.0, [])   # force a fresh read next time
            return True, f"Closed tracked position ({pnl:+,.2f} PnL)."

        # Not tracked by this bot at all — square off directly; nothing in our
        # own DB describes it, so there is nothing to update there.
        res = self.broker.square_off(inst, side, quantity, 0.0)
        self._all_positions_cache = (0.0, [])       # force a fresh read next time
        if res.ok:
            self.state.push_log(
                f"{symbol}: closed an UNTRACKED broker position ({side} "
                f"{quantity}) directly (order {res.order_id}).")
            return True, "Closed untracked broker position."
        self.state.push_log(
            f"{symbol}: closing untracked broker position FAILED — "
            f"{res.message}")
        return False, res.message

    def _finalize_close(self, inst: Instrument, trade: dict,
                        exit_price: float, reason: str) -> float:
        """Square off, persist the close, and refresh PnL for an ALREADY-CLAIMED
        position (one already removed from open_positions under the lock). Shared by
        the auto-exit loop and the manual close button so both book exits
        identically. Returns the realized PnL.

        LIVE only, two extra steps around the actual square-off:
          1. Cancel the broker-side protective order (GTT) FIRST — it's now
             redundant, and leaving it armed risks it firing later on a
             position we're about to close some other way.
          2. Check the broker's OWN position for this symbol, with a forced
             (uncached) read, right before deciding to send the square-off
             order. If the broker already shows it flat, that protective
             order fired first — sending our own square-off too would be a
             genuine DOUBLE EXIT (selling/buying shares we no longer hold /
             hold twice over), so it's skipped rather than risked.
        """
        if self.environment == Environment.LIVE:
            gtt_id = trade.get("broker_gtt_id")
            if gtt_id and self.broker.cancel_oco_exit(gtt_id):
                self.state.push_log(
                    f"{inst.symbol}: cancelled broker-side SL/TP protection "
                    f"(GTT {gtt_id}).")

        already_flat = False
        if self.environment == Environment.LIVE:
            pos = self._broker_position(inst, force=True)
            already_flat = pos is not None and int(pos.get("quantity", -1)) == 0

        if already_flat:
            self.state.push_log(
                f"{inst.symbol}: broker already shows this position flat — "
                "its own protective order likely fired first. Skipping a "
                "duplicate square-off order.")
            order_id = "already-flat"
        else:
            res = self.broker.square_off(inst, trade["side"], trade["quantity"],
                                         exit_price)
            order_id = res.order_id

        closed = self.db.close_trade(trade["trade_id"], exit_price,
                                     self.environment, exit_reason=reason)
        pnl = closed["realized_pnl"] if closed else 0.0
        # Re-derive today's realized PnL from storage rather than incrementing, so
        # the live figure always agrees with the persisted day-wise record (and a
        # restart mid-day rebuilds to the identical number). Then refresh history.
        today_real = self.db.today_realized(self.environment, user_id=self.user_id)
        with self.state.lock:
            self.state.realized_pnl = today_real
        self._refresh_daily()
        self.state.push_log(
            f"EXIT {inst.symbol} {trade['side']} @ {exit_price:.2f} [{reason}] "
            f"PnL ₹{pnl:,.2f} (order {order_id})")
        self._recompute_unrealized()
        return pnl

    def reset_portfolio(self) -> dict:
        """Full reset: permanently delete this environment's entire trade history
        AND clear the live in-memory state, returning the bot to a fresh start.
        IRREVERSIBLE. Only this environment is touched (paper vs live stay separate).

        Any positions currently open in memory are simply dropped — for paper/sim
        that's harmless, but a LIVE user should square real positions off first
        (the UI warns when open positions exist). Storage is wiped via the DB layer;
        the counters and history are then reset so the dashboard shows a clean
        slate immediately."""
        stats = self.db.reset_environment(self.environment, user_id=self.user_id)
        self._trades_today = 0
        self._live_halt_logged = False
        with self.state.lock:
            self.state.open_positions = {}
            self.state.last_signals = []
            self.state.realized_pnl = 0.0
            self.state.unrealized_pnl = 0.0
            self.state.day_pnl = 0.0
            self.state.daily_pnl = []
            self.state.trading_day = self.db.today_key()
            self.state.risk_status = {}
        self.state.push_log(
            f"♻️ Portfolio reset ({self.environment.value}) — removed "
            f"{stats['trades_removed']} trade(s). Starting fresh.")
        return stats

    def _recompute_unrealized(self) -> None:
        """Unrealized PnL for the dashboard. LIVE, per-symbol: prefers the
        broker's OWN reported position (real average price + real PnL,
        including any slippage/partial-fill the broker actually saw) over our
        internal entry_price-vs-tick calculation. Falls back to the internal
        calc for Paper, MCX (not supported), or whenever the broker call
        doesn't return a live position. Network calls happen OUTSIDE the lock
        (see _broker_position) so a slow broker response never blocks the UI
        thread reading state.snapshot()."""
        with self.state.lock:
            positions = list(self.state.open_positions.items())
        unreal = 0.0
        for sym, t in positions:
            broker_pos = None
            if self.environment == Environment.LIVE:
                inst = config.INSTRUMENTS_BY_SYMBOL.get(sym)
                if inst is not None:
                    broker_pos = self._broker_position(inst)
            if broker_pos is not None and broker_pos.get("quantity", 0) != 0:
                # Mutates the SAME dict object referenced by state.open_positions
                # (this is a shallow copy of the outer dict only), so the
                # dashboard sees these fields without a separate write-back.
                t["_broker_avg_price"] = broker_pos["average_price"]
                t["_broker_pnl"] = broker_pos["pnl"]
                unreal += broker_pos["pnl"]
            else:
                unreal += _position_pnl(t, t.get("_live_price", t["entry_price"]))
        with self.state.lock:
            self.state.unrealized_pnl = round(unreal, 2)
            self.state.day_pnl = round(self.state.realized_pnl + unreal, 2)

    def _broker_position(self, inst: Instrument, force: bool = False) -> Optional[dict]:
        """Cached wrapper over broker.get_position() — TTL'd like
        _broker_available_funds, but much shorter (this drives the dashboard's
        "live" PnL, which should feel live). `force=True` bypasses the cache
        entirely for the one moment it matters most: the instant BEFORE a
        software-triggered close decides whether to square off (see
        _finalize_close) — a stale "still open" read there risks a duplicate
        order against a position the broker's own GTT already flattened."""
        if force:
            try:
                val = self.broker.get_position(inst) if self.broker else None
            except Exception:
                val = None
            self._broker_position_cache[inst.symbol] = (_time.time(), val)
            return val
        now = _time.time()
        ts, val = self._broker_position_cache.get(inst.symbol, (0.0, None))
        if now - ts < BROKER_POSITION_TTL_SECONDS:
            return val
        try:
            val = self.broker.get_position(inst) if self.broker else None
        except Exception:
            val = None
        self._broker_position_cache[inst.symbol] = (now, val)
        return val


def _position_pnl(trade: dict, price: float) -> float:
    """Mark-to-market PnL in rupees. Mirrors DBManager._pnl exactly — same sign
    convention, same multiplier — so unrealized and realized never disagree."""
    direction = 1 if trade["side"] == "BUY" else -1
    mult = int(trade.get("contract_multiplier", 1) or 1)
    return (price - trade["entry_price"]) * trade["quantity"] * direction * mult
