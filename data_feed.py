"""
data_feed.py
Market-data layer. Streams candles AND live quotes to the engine.

    MarketDataFeed (interface)
      ├── SimulatedFeed       -> random-walk candles. Runs with zero credentials
      │                          so the bot always starts.
      ├── UpstoxWebSocketFeed -> real Upstox Market Data WebSocket V3. DEFAULT.
      └── UpstoxRestFeed      -> polled candles; opt-in fallback only.

WEBSOCKET-FIRST CONTRACT
------------------------
Live state — the current price, the bid/ask, the forming candle, and therefore
the live PnL — comes from WebSocket ticks and nothing else. REST is used for
exactly one job: seeding and back-correcting COMPLETED historical candles, which
is what makes indicators (VWAP, ATR, 200 EMA) valid the moment the bot starts.
REST never touches the forming bar and never fabricates a live price.

Consumers read two things:
    get_candles(instrument) -> DataFrame   (history + the live forming bar)
    get_quote(instrument)   -> LiveQuote   (the live tick: price, bid/ask, age)

Every quote carries a `source` ("ws" / "rest" / "sim") and an `age_seconds`, so
a stalled or degraded stream is visible rather than silently priced as live.
MCX instruments flow through the exact same path as equity — the only difference
is market hours (config).
"""
from __future__ import annotations

import random
import threading
import time as _time
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd

from config import Instrument, Mode, Segment


CANDLE_COLUMNS = ["open", "high", "low", "close", "volume"]

# How many candle minutes each mode trades on. The Scalper works the 1-minute
# tape, which is why ticks (not polled candles) have to drive it.
TF_MINUTES = {Mode.SCALPER: 1, Mode.INTRADAY: 15, Mode.SWING: 24 * 60}


@dataclass
class LiveQuote:
    """One instrument's current live state — THE single source of truth for price
    and PnL. `source` records where it came from so the UI can never silently
    present a polled REST candle as if it were a live tick:

        "ws"   -> a real WebSocket tick (the only genuinely live source)
        "rest" -> last REST candle close, used while the socket is down
        "sim"  -> simulated feed
    """
    ltp: float
    ts: datetime                  # exchange last-trade-time, IST wall-clock
    received_at: float            # time.monotonic() when we accepted it
    bid: float = 0.0
    ask: float = 0.0
    source: str = "ws"

    @property
    def age_seconds(self) -> float:
        """Seconds since this quote arrived. A climbing age during market hours
        means the stream has stalled."""
        return max(0.0, _time.monotonic() - self.received_at)

    @property
    def is_live(self) -> bool:
        return self.source == "ws"

    @property
    def spread(self) -> float:
        return (self.ask - self.bid) if (self.bid > 0 and self.ask > 0) else 0.0


class MarketDataFeed:
    """Interface. status() feeds the Live Dashboard 'connection status' widget."""

    def start(self, instruments: list[Instrument]) -> None:
        raise NotImplementedError

    def stop(self) -> None:
        raise NotImplementedError

    def get_candles(self, instrument: Instrument, lookback: int = 250) -> pd.DataFrame:
        raise NotImplementedError

    def status(self) -> str:
        raise NotImplementedError

    def get_quote(self, instrument: Instrument) -> Optional[LiveQuote]:
        """Latest live quote, or None if nothing has arrived yet. The engine
        prices open positions from this — never from a candle close — so live PnL
        is WebSocket-driven end to end."""
        return None


# --------------------------------------------------------------------------- #
#  Simulated feed
# --------------------------------------------------------------------------- #
class SimulatedFeed(MarketDataFeed):
    """
    Builds a seed history per instrument, then appends a fresh candle every
    `tick_seconds`. Prices follow a gentle random walk with occasional trending
    bursts so the strategies actually fire from time to time.
    """

    def __init__(self, tick_seconds: float = 2.0, seed_bars: int = 260,
                 mode: Mode = Mode.INTRADAY):
        self.tick_seconds = tick_seconds
        self.seed_bars = seed_bars
        # Bar width must match the mode, or a Scalper fallback would be handed
        # 15-minute candles and quietly trade the wrong timeframe.
        self.tf_min = TF_MINUTES.get(mode, 15)
        self._data: dict[str, pd.DataFrame] = {}
        self._quotes: dict[str, LiveQuote] = {}
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._running = False

    # -- history generation ------------------------------------------------- #
    def _seed(self, inst: Instrument) -> pd.DataFrame:
        price = inst.reference_price
        rows = []
        drift = 0.0
        for _ in range(self.seed_bars):
            if random.random() < 0.05:                  # occasional regime change
                drift = random.uniform(-0.0008, 0.0012)
            ret = random.gauss(drift, 0.004)
            new = max(price * (1 + ret), inst.tick_size)
            o, c = price, new
            hi = max(o, c) * (1 + abs(random.gauss(0, 0.0015)))
            lo = min(o, c) * (1 - abs(random.gauss(0, 0.0015)))
            vol = int(abs(random.gauss(50_000, 20_000))) + 1000
            rows.append([o, hi, lo, c, vol])
            price = new
        idx = pd.date_range(end=datetime.now(), periods=self.seed_bars,
                            freq=f"{self.tf_min}min")
        return pd.DataFrame(rows, columns=CANDLE_COLUMNS, index=idx)

    def _append_tick(self, inst: Instrument) -> None:
        df = self._data[inst.symbol]
        last_close = float(df["close"].iloc[-1])
        drift = random.choice([0.0, 0.0, 0.001, -0.001])
        ret = random.gauss(drift, 0.004)
        new = max(last_close * (1 + ret), inst.tick_size)
        o, c = last_close, new
        hi = max(o, c) * (1 + abs(random.gauss(0, 0.0015)))
        lo = min(o, c) * (1 - abs(random.gauss(0, 0.0015)))
        vol = int(abs(random.gauss(60_000, 25_000))) + 1000
        row = pd.DataFrame([[o, hi, lo, c, vol]], columns=CANDLE_COLUMNS,
                           index=[df.index[-1] + timedelta(minutes=self.tf_min)])
        with self._lock:
            self._data[inst.symbol] = pd.concat([df, row]).tail(600)
            self._quotes[inst.symbol] = LiveQuote(
                ltp=c, ts=row.index[-1], received_at=_time.monotonic(),
                bid=lo, ask=hi, source="sim")

    # -- lifecycle ---------------------------------------------------------- #
    def start(self, instruments: list[Instrument]) -> None:
        with self._lock:
            for inst in instruments:
                if inst.symbol not in self._data:
                    self._data[inst.symbol] = self._seed(inst)
        self._instruments = instruments
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while self._running:
            for inst in self._instruments:
                self._append_tick(inst)
            _time.sleep(self.tick_seconds)

    def stop(self) -> None:
        self._running = False

    def get_candles(self, instrument: Instrument, lookback: int = 250) -> pd.DataFrame:
        with self._lock:
            df = self._data.get(instrument.symbol)
            return df.tail(lookback).copy() if df is not None else pd.DataFrame(
                columns=CANDLE_COLUMNS)

    def get_quote(self, instrument: Instrument) -> Optional[LiveQuote]:
        with self._lock:
            return self._quotes.get(instrument.symbol)

    def status(self) -> str:
        return "🟢 Simulated feed streaming" if self._running else "🔴 Stopped"


# --------------------------------------------------------------------------- #
#  Shared REST candle fetch — used to SEED history for both the REST feed and the
#  WebSocket feed (so 200 EMA / MACD are valid the instant the bot starts).
# --------------------------------------------------------------------------- #
def _fetch_rest_candles(hist_api, inst: Instrument, mode: Mode,
                        history_days: int) -> pd.DataFrame:
    """Pull real candles for one instrument and return an ascending OHLCV df
    indexed in IST wall-clock time. Raises on hard API errors (Swing path)."""
    today = datetime.now().strftime("%Y-%m-%d")
    frm = (datetime.now() - timedelta(days=history_days)).strftime("%Y-%m-%d")
    if mode == Mode.SWING:
        resp = hist_api.get_historical_candle_data1(
            inst.instrument_key, "day", today, frm, api_version="v2")
        candles = resp.data.candles
    else:
        candles = []
        try:
            r1 = hist_api.get_historical_candle_data1(
                inst.instrument_key, "1minute", today, frm, api_version="v2")
            candles += r1.data.candles
        except Exception:
            pass
        try:
            r2 = hist_api.get_intra_day_candle_data(
                inst.instrument_key, "1minute", api_version="v2")
            candles += r2.data.candles
        except Exception:
            pass
    if not candles:
        return pd.DataFrame(columns=CANDLE_COLUMNS)

    df = pd.DataFrame(candles, columns=[
        "ts", "open", "high", "low", "close", "volume", "oi"][:len(candles[0])])
    # Upstox stamps candles in IST (+05:30). Convert to IST wall-clock, drop tz.
    df["ts"] = (pd.to_datetime(df["ts"], utc=True)
                .dt.tz_convert("Asia/Kolkata").dt.tz_localize(None))
    df = (df.set_index("ts").sort_index()
          [["open", "high", "low", "close", "volume"]].astype(float))
    df = df[~df.index.duplicated(keep="last")]
    if mode == Mode.INTRADAY:
        df = df.resample("15min").agg(
            {"open": "first", "high": "max", "low": "min",
             "close": "last", "volume": "sum"}).dropna()
    # SCALPER keeps the raw 1-minute bars Upstox already returns — no resample.
    return df.tail(600)


# --------------------------------------------------------------------------- #
#  Real Upstox WebSocket V3 feed — TICK-BY-TICK live streaming (the real-time
#  path the project asked for). Design is HYBRID:
#    1. Seed real candle history once via REST  -> indicators valid immediately.
#    2. Stream live ticks over WebSocket V3     -> aggregate into the current
#       forming candle in real time (O/H/L/C from ltp, volume from vtt delta).
#    3. Periodically re-seed completed candles via REST to correct any drift.
#  If the WebSocket can't connect (e.g. expired token — WS needs a valid token,
#  unlike the read-only history endpoints), it degrades to REST polling so the
#  bot keeps running; status() shows which mode is live.
# --------------------------------------------------------------------------- #
class UpstoxWebSocketFeed(MarketDataFeed):
    def __init__(self, access_token: str, mode: Mode,
                 reseed_seconds: float = 120.0, history_days: int = 0):
        self.access_token = access_token
        self.mode = mode
        self.tf_min = TF_MINUTES.get(mode, 15)
        self.reseed_seconds = reseed_seconds
        # Seed depth per mode. Swing needs 200+ DAILY bars for the 200 EMA; the
        # Scalper needs only a couple of sessions of 1-minute bars (asking for
        # years of minutes would be a huge request the API would refuse).
        default_days = {Mode.SWING: 1100, Mode.INTRADAY: 5, Mode.SCALPER: 3}
        self.history_days = history_days or default_days.get(mode, 5)

        self._hist_api = None
        self._streamer = None
        self._key_to_symbol: dict[str, str] = {}
        self._seed: dict[str, pd.DataFrame] = {}        # completed candles
        self._forming: dict[str, dict] = {}            # symbol -> live candle
        self._quotes: dict[str, LiveQuote] = {}        # symbol -> live tick state
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._running = False
        self._ws_connected = False
        self._tick_count = 0
        self._degraded = False   # True => WebSocket down, using REST poll fallback

    # -- helpers ------------------------------------------------------------ #
    def _bucket(self, ts: pd.Timestamp) -> pd.Timestamp:
        """Floor a tick timestamp to the start of its candle bucket (IST-naive)."""
        ts = pd.Timestamp(ts)
        if self.mode == Mode.SWING:
            return ts.normalize()
        minute = (ts.minute // self.tf_min) * self.tf_min
        return ts.replace(minute=minute, second=0, microsecond=0, nanosecond=0)

    @staticmethod
    def _to_ist(ltt_ms) -> pd.Timestamp:
        """Convert Upstox last-trade-time (epoch ms, UTC) to IST wall-clock naive."""
        try:
            return (pd.Timestamp(int(ltt_ms), unit="ms", tz="UTC")
                    .tz_convert("Asia/Kolkata").tz_localize(None))
        except Exception:
            return pd.Timestamp(datetime.now())

    def _seed_history(self, instruments: list[Instrument]) -> int:
        ok = 0
        for inst in instruments:
            try:
                df = _fetch_rest_candles(self._hist_api, inst, self.mode,
                                         self.history_days)
                if not df.empty:
                    with self._lock:
                        self._seed[inst.symbol] = df
                    ok += 1
            except Exception as exc:
                print(f"[UpstoxWebSocketFeed] seed {inst.symbol} failed: {exc}")
        return ok

    # -- lifecycle ---------------------------------------------------------- #
    def start(self, instruments: list[Instrument]) -> None:
        import upstox_client  # type: ignore
        self._instruments = instruments
        self._key_to_symbol = {i.instrument_key: i.symbol for i in instruments}

        cfg = upstox_client.Configuration()
        cfg.access_token = self.access_token
        self._hist_api = upstox_client.HistoryApi(upstox_client.ApiClient(cfg))

        # 1) Seed history so indicators are valid immediately.
        if self._seed_history(instruments) == 0:
            raise RuntimeError("UpstoxWebSocketFeed: no seed data returned")

        # 2) Try to open the live WebSocket tick stream.
        try:
            ws_cfg = upstox_client.Configuration()
            ws_cfg.access_token = self.access_token
            self._streamer = upstox_client.MarketDataStreamerV3(
                upstox_client.ApiClient(ws_cfg),
                instrumentKeys=[i.instrument_key for i in instruments],
                mode="full",
            )
            self._streamer.on("open", self._on_open)
            self._streamer.on("message", self._on_message)
            self._streamer.on("error", self._on_error)
            self._streamer.on("close", self._on_close)
            self._streamer.connect()          # non-blocking (runs its own thread)
        except Exception as exc:
            print(f"[UpstoxWebSocketFeed] WebSocket connect failed ({exc}); "
                  "degrading to REST polling.")
            self._degraded = True

        self._running = True
        self._thread = threading.Thread(target=self._maintain_loop, daemon=True)
        self._thread.start()

    def _maintain_loop(self) -> None:
        """Periodically re-seed COMPLETED candles (corrects tick drift) and, in
        degraded mode, act as a plain REST poller so data still flows. Uses a
        short base sleep so it reacts quickly when the WebSocket drops."""
        last_reseed = 0.0
        while self._running:
            _time.sleep(5.0)
            if not self._running:
                break
            interval = 20.0 if self._degraded else self.reseed_seconds
            if _time.monotonic() - last_reseed < interval:
                continue
            last_reseed = _time.monotonic()
            for inst in self._instruments:
                try:
                    df = _fetch_rest_candles(self._hist_api, inst, self.mode,
                                             self.history_days)
                    if df.empty:
                        continue
                    with self._lock:
                        # The WebSocket owns the live bar. Drop any REST row at or
                        # after the forming bucket so a lagging poll can never
                        # overwrite fresher tick data — REST corrects history only.
                        f = self._forming.get(inst.symbol)
                        if f is not None:
                            df = df[df.index < f["ts"]]
                        if df.empty:
                            continue
                        base = self._seed.get(inst.symbol)
                        merged = (df if base is None else
                                  pd.concat([base, df]))
                        merged = merged[~merged.index.duplicated(keep="last")]
                        self._seed[inst.symbol] = merged.sort_index().tail(600)
                        # While the socket is down there are no ticks, so publish a
                        # REST-derived quote — tagged "rest" so the UI and the
                        # engine can tell it apart from a real tick.
                        if self._degraded:
                            self._quotes[inst.symbol] = LiveQuote(
                                ltp=float(merged["close"].iloc[-1]),
                                ts=merged.index[-1].to_pydatetime(),
                                received_at=_time.monotonic(), source="rest")
                except Exception:
                    pass

    # -- WebSocket event handlers ------------------------------------------ #
    def _on_open(self) -> None:
        self._ws_connected = True
        self._degraded = False
        print("[UpstoxWebSocketFeed] WebSocket open — streaming live ticks.")

    def _on_error(self, err=None) -> None:
        print(f"[UpstoxWebSocketFeed] WebSocket error: {err}")

    def _on_close(self, *args) -> None:
        self._ws_connected = False
        # Fall back to REST polling until it reconnects.
        self._degraded = True

    @staticmethod
    def _best_bid_ask(market_ff: dict) -> tuple[float, float]:
        """Best bid/ask from level-1 of the depth ladder. Protobuf path:
        fullFeed.marketFF.marketLevel.bidAskQuote[0] -> {bidP, bidQ, askP, askQ}.
        MessageToDict omits zero-valued fields, hence the .get defaults."""
        quotes = (market_ff.get("marketLevel", {}) or {}).get("bidAskQuote") or []
        if not quotes:
            return 0.0, 0.0
        top = quotes[0] or {}
        try:
            return float(top.get("bidP", 0.0) or 0.0), float(top.get("askP", 0.0) or 0.0)
        except (TypeError, ValueError):
            return 0.0, 0.0

    def _on_message(self, message) -> None:
        if not isinstance(message, dict):
            return
        feeds = message.get("feeds", {})
        for key, feed in feeds.items():
            sym = self._key_to_symbol.get(key)
            if not sym:
                continue
            ff = (feed.get("fullFeed", {}) or {}).get("marketFF", {}) or {}
            ltpc = ff.get("ltpc") or feed.get("ltpc") or {}
            ltp = ltpc.get("ltp")
            if ltp is None:
                continue
            ts = (self._to_ist(ltpc["ltt"]) if ltpc.get("ltt")
                  else pd.Timestamp(datetime.now()))
            vtt = ff.get("vtt")
            bid, ask = self._best_bid_ask(ff)
            try:
                self._apply_tick(sym, float(ltp), ts,
                                 float(vtt) if vtt is not None else None,
                                 bid, ask)
            except Exception as exc:
                print(f"[UpstoxWebSocketFeed] tick apply error {sym}: {exc}")

    def _apply_tick(self, sym: str, ltp: float, ts: pd.Timestamp,
                    vtt: float | None, bid: float = 0.0,
                    ask: float = 0.0) -> None:
        b = self._bucket(ts)
        with self._lock:
            self._tick_count += 1
            # Not every tick carries the depth ladder — a trade-only update has an
            # ltpc and no marketLevel. Carry the last known bid/ask forward instead
            # of zeroing it, or the spread appears unknown at random and limit
            # entries silently degrade to market orders.
            prev = self._quotes.get(sym)
            if prev is not None:
                bid = bid or prev.bid
                ask = ask or prev.ask
            # The quote is the live source of truth for price/PnL — publish it
            # before any candle bookkeeping so a bad bucket can't withhold it.
            self._quotes[sym] = LiveQuote(
                ltp=ltp, ts=pd.Timestamp(ts), received_at=_time.monotonic(),
                bid=bid, ask=ask, source="ws")

            f = self._forming.get(sym)
            if f is None or b > f["ts"]:
                if f is not None:
                    self._commit_forming(sym, f)      # close previous candle
                f = {"ts": b, "open": ltp, "high": ltp, "low": ltp,
                     "close": ltp, "volume": 0.0,
                     "vtt_start": vtt if vtt is not None else 0.0}
                self._forming[sym] = f
            else:
                f["high"] = max(f["high"], ltp)
                f["low"] = min(f["low"], ltp)
                f["close"] = ltp
            if vtt is not None and vtt >= f["vtt_start"]:
                f["volume"] = vtt - f["vtt_start"]

    def _commit_forming(self, sym: str, f: dict) -> None:
        row = pd.DataFrame(
            [[f["open"], f["high"], f["low"], f["close"], f["volume"]]],
            columns=CANDLE_COLUMNS, index=[f["ts"]])
        base = self._seed.get(sym)
        merged = row if base is None else pd.concat([base, row])
        merged = merged[~merged.index.duplicated(keep="last")]
        self._seed[sym] = merged.sort_index().tail(600)

    # -- consumption -------------------------------------------------------- #
    def get_candles(self, instrument: Instrument, lookback: int = 250) -> pd.DataFrame:
        sym = instrument.symbol
        with self._lock:
            base = self._seed.get(sym)
            base = base.copy() if base is not None else pd.DataFrame(
                columns=CANDLE_COLUMNS)
            f = self._forming.get(sym)
            if f is not None:
                row = pd.DataFrame(
                    [[f["open"], f["high"], f["low"], f["close"], f["volume"]]],
                    columns=CANDLE_COLUMNS, index=[f["ts"]])
                base = base[base.index != f["ts"]]
                base = pd.concat([base, row]).sort_index()
        return base.tail(lookback).copy()

    def get_quote(self, instrument: Instrument) -> Optional[LiveQuote]:
        with self._lock:
            return self._quotes.get(instrument.symbol)

    @property
    def tick_count(self) -> int:
        return self._tick_count

    def stop(self) -> None:
        self._running = False
        try:
            if self._streamer:
                self._streamer.disconnect()
        except Exception:
            pass

    def status(self) -> str:
        if not self._running:
            return "🔴 Disconnected"
        if self._ws_connected:
            return f"🟢 Upstox WebSocket V3 (live ticks: {self._tick_count})"
        if self._degraded:
            return "🟡 Upstox REST fallback (WebSocket down — refresh token?)"
        # connect() is non-blocking, so there is a brief window after start()
        # where the socket is simply still opening. Saying "down — refresh token?"
        # here would send you chasing a token problem that doesn't exist.
        return "🟡 Connecting to Upstox WebSocket…"


# --------------------------------------------------------------------------- #
#  Real Upstox REST feed — seeds full indicator history, then polls.
#  This is the DEFAULT real feed: unlike a pure tick stream it has 100s of bars
#  the instant it starts, so strategies (200 EMA, MACD…) work immediately, and
#  it keeps working whether the market is open or closed (serving the latest
#  available candles). It uses only READ-ONLY history endpoints — it never
#  places orders — so it is safe to drive with a live token in Paper mode.
# --------------------------------------------------------------------------- #
class UpstoxRestFeed(MarketDataFeed):
    def __init__(self, access_token: str, mode: Mode,
                 refresh_seconds: float = 0.0, history_days: int = 0):
        self.access_token = access_token
        self.mode = mode
        # Swing needs 200+ daily bars for the 200 EMA (~3 years); the Scalper
        # needs a couple of sessions of minutes and a fast refresh.
        default_refresh = {Mode.SWING: 300.0, Mode.INTRADAY: 30.0,
                           Mode.SCALPER: 10.0}
        default_days = {Mode.SWING: 1100, Mode.INTRADAY: 10, Mode.SCALPER: 3}
        self.refresh_seconds = refresh_seconds or default_refresh.get(mode, 30.0)
        self.history_days = history_days or default_days.get(mode, 10)
        self._hist_api = None
        self._data: dict[str, pd.DataFrame] = {}
        self._quotes: dict[str, LiveQuote] = {}
        self._instruments: list[Instrument] = []
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._running = False
        self._ok = False

    def _fetch(self, inst: Instrument) -> pd.DataFrame:
        """Pull real candles for one instrument (delegates to the shared helper)."""
        return _fetch_rest_candles(self._hist_api, inst, self.mode,
                                   self.history_days)

    def _store(self, inst: Instrument, df: pd.DataFrame) -> None:
        """Cache candles and publish the derived quote. Tagged "rest", never
        "ws" — this feed polls, so its price is by definition not a live tick."""
        with self._lock:
            self._data[inst.symbol] = df
            self._quotes[inst.symbol] = LiveQuote(
                ltp=float(df["close"].iloc[-1]),
                ts=df.index[-1].to_pydatetime(),
                received_at=_time.monotonic(), source="rest")

    def start(self, instruments: list[Instrument]) -> None:
        import upstox_client  # type: ignore
        cfg = upstox_client.Configuration()
        cfg.access_token = self.access_token
        self._hist_api = upstox_client.HistoryApi(upstox_client.ApiClient(cfg))
        self._instruments = instruments

        ok = 0
        for inst in instruments:
            try:
                df = self._fetch(inst)
                if not df.empty:
                    self._store(inst, df)
                    ok += 1
            except Exception as exc:
                print(f"[UpstoxRestFeed] {inst.symbol} fetch failed: {exc}")
        if ok == 0:
            # Nothing loaded — signal the caller to fall back to simulation.
            raise RuntimeError("UpstoxRestFeed: no instruments returned data")

        self._ok = True
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while self._running:
            for inst in self._instruments:
                if not self._running:
                    break
                try:
                    df = self._fetch(inst)
                    if not df.empty:
                        self._store(inst, df)
                except Exception:
                    pass
            _time.sleep(self.refresh_seconds)

    def stop(self) -> None:
        self._running = False

    def get_candles(self, instrument: Instrument, lookback: int = 250) -> pd.DataFrame:
        with self._lock:
            df = self._data.get(instrument.symbol)
            return df.tail(lookback).copy() if df is not None else pd.DataFrame(
                columns=CANDLE_COLUMNS)

    def get_quote(self, instrument: Instrument) -> Optional[LiveQuote]:
        with self._lock:
            return self._quotes.get(instrument.symbol)

    def status(self) -> str:
        return ("🟡 Upstox live data (REST poll — not tick-by-tick)"
                if self._running else "🔴 Disconnected")


# --------------------------------------------------------------------------- #
#  Factory
# --------------------------------------------------------------------------- #
def make_feed(prefer_real: bool, access_token: str = "",
              mode: Mode = Mode.INTRADAY,
              use_websocket: bool = True) -> MarketDataFeed:
    """
    Pick the market-data feed:
      • WebSocket (default) — real-time tick-by-tick via Upstox Streamer V3, seeded
        with REST history so indicators work immediately. Degrades to REST polling
        internally if the socket can't connect (e.g. expired token).
      • REST (use_websocket=False) — poll-based real candles, no live socket.
      • Simulated — when no token / SDK, so the bot always runs.
    Both real feeds raise from .start() if no data returns, so the engine's
    resilient starter falls back to Simulated.
    """
    if prefer_real and access_token:
        import importlib.util
        if importlib.util.find_spec("upstox_client") is not None:
            if use_websocket:
                return UpstoxWebSocketFeed(access_token, mode)
            return UpstoxRestFeed(access_token, mode)
        print("[make_feed] upstox_client not installed — using Simulated feed. "
              "Run 'pip install upstox-python-sdk' for the live feed.")
    return SimulatedFeed(mode=mode)
