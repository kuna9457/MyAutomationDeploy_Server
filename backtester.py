"""
backtester.py
Vectorized backtesting engine.

Given historical daily/intraday candles for one instrument, it walks the same
strategy signals used live and computes the Section-3 metrics:
Total Return, Max Drawdown, Sharpe, Calmar, Win Rate, plus an equity curve for
the UI chart.

Historical data source, in priority order:
  1. real Upstox candles (needs an instrument key + token) — the good path
  2. yfinance (if installed and a mappable ticker is given)
  3. synthetic random-walk series, seeded per ticker (always available)

Timeframe follows the mode: Swing = daily, Intraday = 15m, Scalper = 1m.
Longs and shorts are both simulated; the Scalper is two-sided.
"""
from __future__ import annotations

import glob
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd

import config
from config import Mode, Segment, params_for_mode
from strategy import enrich, position_size, resolve_strategy


TRADING_DAYS = 252

# Bulk backtests hit the SAME broker/data-source repeatedly (once per ticker).
# 5 concurrent workers is comfortably inside Upstox's rate limits while cutting
# wall-clock time roughly 5x versus the old sequential loop.
BULK_MAX_WORKERS = 5

# --------------------------------------------------------------------------- #
#  Local disk cache for fetched history — a backtest re-run over the same
#  ticker/interval/date-range (very common while iterating on a strategy) reads
#  straight from disk instead of re-hitting Upstox/yfinance. Only REAL data
#  (upstox/yfinance) is cached; synthetic is cheap to regenerate and caching it
#  would risk masking a data-source outage as "no cache miss".
# --------------------------------------------------------------------------- #
_CACHE_DIR = os.path.join(config.LOCAL_DB_DIR, "hist_cache")


def _cache_stem(ticker: str, interval: str, start: str, end: str) -> str:
    safe = f"{ticker}_{interval}_{start}_{end}".replace("/", "-").replace(":", "-")
    return os.path.join(_CACHE_DIR, safe)


# --------------------------------------------------------------------------- #
#  SUPERSET cache — one file per (ticker, interval) holding the widest range
#  ever fetched, sliced in memory per request.
#
#  The range-keyed cache above only hits on an EXACT date match, so moving the
#  start date by a day re-downloaded everything. Real evidence from a live cache
#  directory: ADANIENT_15m_2026-01-01_2026-08-16 sitting beside
#  ADANIENT_15m_2026-07-01_2026-08-09 — the same bars, downloaded twice.
#
#  That is fine for the odd manual backtest and useless for a combination
#  search, which re-runs the same symbols over and over. With a superset a
#  20-symbol search downloads once, ever, per timeframe; narrowing the window
#  then costs nothing.
#
#  The old files stay readable (see _load_cached_history) so nothing already
#  downloaded is wasted.
# --------------------------------------------------------------------------- #
def _superset_path(ticker: str, interval: str, source: str) -> str:
    safe = f"{ticker}_{interval}__{source}".replace("/", "-").replace(":", "-")
    return os.path.join(_CACHE_DIR, f"super_{safe}.parquet")


def _slice_window(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """Rows within [start, end] INCLUSIVE of the end date.

    `end` is a date, and an intraday frame carries times, so a naive
    `df.loc[:end]` would silently drop the whole of the final day. +1 day and a
    strict upper bound is the correct reading of "up to and including this
    date".
    """
    out = df
    if start:
        out = out[out.index >= pd.Timestamp(start)]
    if end:
        out = out[out.index < pd.Timestamp(end) + pd.Timedelta(days=1)]
    return out


def _load_superset(ticker: str, interval: str, start: str,
                   end: str) -> tuple[pd.DataFrame, str] | None:
    """The requested window from a superset file, or None if no file covers it.

    "Covers" is judged on the file's own first/last bar, not on the range it
    was requested with — a symbol simply has no bars before it listed, and
    demanding otherwise would make the cache permanently miss.
    """
    for path in glob.glob(_superset_path(ticker, interval, "*")):
        source = os.path.basename(path).rsplit("__", 1)[-1][:-len(".parquet")]
        try:
            df = pd.read_parquet(path)
        except Exception as exc:
            print(f"[backtester] superset cache unreadable for {ticker} ({exc}).")
            continue
        if df.empty:
            continue
        want_start = pd.Timestamp(start) if start else df.index[0]
        want_end = (pd.Timestamp(end) + pd.Timedelta(days=1) if end
                    else df.index[-1])
        # A one-day tolerance at the start: the requested date may fall on a
        # weekend or holiday, when no bar can exist however complete the file.
        if df.index[0] > want_start + pd.Timedelta(days=1):
            continue
        if df.index[-1] < want_end - pd.Timedelta(days=1):
            continue
        window = _slice_window(df, start, end)
        if window.empty:
            continue
        return window, source
    return None


def _merge_into_superset(ticker: str, interval: str, df: pd.DataFrame,
                         source: str) -> None:
    """Union freshly-fetched bars into the superset for this (ticker, interval).

    Never raises — caching is a speed optimisation, and a write failure must
    not break the backtest that produced the data.
    """
    if df is None or df.empty:
        return
    try:
        os.makedirs(_CACHE_DIR, exist_ok=True)
        path = _superset_path(ticker, interval, source)
        merged = df
        if os.path.exists(path):
            try:
                merged = pd.concat([pd.read_parquet(path), df])
            except Exception:
                merged = df
        # Fresher rows win on overlap: a re-fetch corrects a partial last bar.
        merged = merged[~merged.index.duplicated(keep="last")].sort_index()
        merged.to_parquet(path)
    except Exception as exc:
        print(f"[backtester] superset cache write failed for {ticker} ({exc}).")


def _load_cached_history(ticker: str, interval: str, start: str,
                         end: str) -> tuple[pd.DataFrame, str] | None:
    matches = glob.glob(_cache_stem(ticker, interval, start, end) + "__*.parquet")
    if not matches:
        return None
    path = matches[0]
    source = os.path.basename(path).rsplit("__", 1)[-1][:-len(".parquet")]
    try:
        return pd.read_parquet(path), source
    except Exception as exc:
        print(f"[backtester] cache read failed for {ticker} ({exc}); refetching.")
        return None


def _save_cached_history(ticker: str, interval: str, start: str, end: str,
                         df: pd.DataFrame, source: str) -> None:
    try:
        os.makedirs(_CACHE_DIR, exist_ok=True)
        path = _cache_stem(ticker, interval, start, end) + f"__{source}.parquet"
        df.to_parquet(path)
    except Exception as exc:
        # Caching is a pure speed optimisation — never let a write failure
        # (e.g. missing pyarrow) break the backtest itself.
        print(f"[backtester] cache write failed for {ticker} ({exc}); continuing.")

# Bars per year, per timeframe — used to annualise Sharpe and the Calmar CAGR.
# Equity: ~6.25h/day => 25 fifteen-minute bars, 375 one-minute bars.
BARS_PER_YEAR = {"1d": TRADING_DAYS, "15m": TRADING_DAYS * 25,
                 "1m": TRADING_DAYS * 375}


@dataclass
class BacktestResult:
    metrics: dict
    equity_curve: pd.Series
    trades: pd.DataFrame


# --------------------------------------------------------------------------- #
#  Data acquisition
# --------------------------------------------------------------------------- #
def _yf_symbol(ticker: str) -> str:
    """Map our symbols to yfinance tickers where a sensible mapping exists."""
    mapping = {
        "GOLD": "GC=F", "CRUDEOIL": "CL=F", "NATURALGAS": "NG=F", "SILVER": "SI=F",
        # MCX mini/micro contracts track the SAME underlying spot price as their
        # full-size sibling — only the contract size (and therefore margin) differs,
        # not the price series — so they map to the identical Yahoo futures ticker.
        "GOLDM": "GC=F", "CRUDEOILM": "CL=F", "NATGASMINI": "NG=F",
        "SILVERM": "SI=F", "SILVERMIC": "SI=F",
    }
    if ticker in mapping:
        return mapping[ticker]
    return ticker if ticker.endswith(".NS") else f"{ticker}.NS"


# Upstox serves at most ~1 month of 1-minute history per request; a wider window
# throws ApiException. Daily has no such limit. So 1-minute ranges are fetched in
# sub-month chunks and stitched — WITHOUT this, any intraday/scalper backtest
# longer than a month silently fell through to synthetic data (the bug that made
# backtest trade prices not match the real instrument).
_MINUTE_CHUNK_DAYS = 25


def _fetch_upstox_candles_raw(hist_api, instrument_key: str, up_interval: str,
                              start: str, end: str) -> list:
    """Raw Upstox candle lists over [start, end]. 'day' is one call (spans years);
    '1minute' is walked backwards in <=_MINUTE_CHUNK_DAYS windows and concatenated,
    because Upstox caps a single 1-minute request at roughly one month. Overlaps
    are harmless — the caller de-duplicates by timestamp."""
    if up_interval == "day":
        resp = hist_api.get_historical_candle_data1(
            instrument_key, up_interval, end, start, api_version="v2")
        return resp.data.candles or []

    start_dt = datetime.strptime(start, "%Y-%m-%d").date()
    end_dt = datetime.strptime(end, "%Y-%m-%d").date()
    candles: list = []
    ok_chunks = errors = 0
    cur = end_dt
    while cur >= start_dt:
        chunk_from = max(start_dt, cur - timedelta(days=_MINUTE_CHUNK_DAYS))
        try:
            resp = hist_api.get_historical_candle_data1(
                instrument_key, up_interval, str(cur), str(chunk_from),
                api_version="v2")
            candles += (resp.data.candles or [])
            ok_chunks += 1
        except Exception as exc:
            # One bad month must not sink the whole fetch — a partial real series
            # is still real. Only a total wipe-out falls back to synthetic.
            errors += 1
            print(f"[backtester] 1-min chunk {chunk_from}->{cur} failed: {exc}")
        cur = chunk_from - timedelta(days=1)
    if errors:
        print(f"[backtester] 1-min fetch: {ok_chunks} chunk(s) OK, "
              f"{errors} failed.")
    return candles


def _fetch_upstox_hist(
    instrument_key: str, start: str, end: str, interval: str, token: str
) -> pd.DataFrame:
    """Real historical candles from Upstox for one instrument over [start, end].
    Daily for swing; 1-minute resampled to 15m for intraday. Indexed IST-naive."""
    import upstox_client  # type: ignore
    cfg = upstox_client.Configuration()
    cfg.access_token = token
    hist = upstox_client.HistoryApi(upstox_client.ApiClient(cfg))
    up_interval = "day" if interval == "1d" else "1minute"
    candles = _fetch_upstox_candles_raw(hist, instrument_key, up_interval,
                                        start, end)
    if not candles:
        return pd.DataFrame()
    df = pd.DataFrame(candles, columns=[
        "ts", "open", "high", "low", "close", "volume", "oi"][:len(candles[0])])
    df["ts"] = (pd.to_datetime(df["ts"], utc=True)
                .dt.tz_convert("Asia/Kolkata").dt.tz_localize(None))
    df = (df.set_index("ts").sort_index()
          [["open", "high", "low", "close", "volume"]].astype(float))
    df = df[~df.index.duplicated(keep="last")]
    # Only 15m needs building; "1m" is already what Upstox returned, and
    # resampling it to 15min would silently backtest the wrong timeframe.
    if interval == "15m":
        df = df.resample("15min").agg(
            {"open": "first", "high": "max", "low": "min",
             "close": "last", "volume": "sum"}).dropna()
    return df


def fetch_history(
    ticker: str, start: str, end: str, interval: str = "1d",
    instrument_key: str = "", token: str = "",
) -> tuple[pd.DataFrame, str]:
    """Return (candles, source). `source` is one of "upstox", "yfinance" or
    "synthetic" so callers can tell REAL market data from a synthetic random walk
    and warn the user instead of presenting fake trade prices as genuine."""
    # 0) Local disk cache — a re-run over the same ticker/interval/range (very
    #    common while iterating on a strategy, or across bulk-backtest tickers
    #    re-run later) skips the network entirely.
    # 0a) SUPERSET cache first — covers any window inside what was ever
    #     fetched, so changing the dates no longer re-downloads.
    superset = _load_superset(ticker, interval, start, end)
    if superset is not None:
        df, source = superset
        print(f"[backtester] {ticker}: {len(df)} candles from superset cache "
              f"({source}).")
        return df, source

    # 0b) The original exact-range cache. Kept so files downloaded before the
    #     superset existed are still used rather than re-fetched; a hit here is
    #     also promoted into the superset so it is reusable next time.
    cached = _load_cached_history(ticker, interval, start, end)
    if cached is not None:
        df, source = cached
        print(f"[backtester] {ticker}: {len(df)} candles from local cache "
              f"({source}).")
        _merge_into_superset(ticker, interval, df, source)
        return df, source

    # 1) REAL Upstox historical data — the good path. Ticker-specific & real, so
    #    every instrument gives genuinely different results.
    if instrument_key and token:
        try:
            df = _fetch_upstox_hist(instrument_key, start, end, interval, token)
            if len(df) > 30:
                print(f"[backtester] {ticker}: {len(df)} real Upstox candles.")
                _save_cached_history(ticker, interval, start, end, df, "upstox")
                _merge_into_superset(ticker, interval, df, "upstox")
                return df, "upstox"
            print(f"[backtester] {ticker}: Upstox returned too few candles "
                  f"({len(df)}); trying next source.")
        except Exception as exc:
            print(f"[backtester] {ticker}: Upstox history failed ({exc}); "
                  "trying next source.")

    # 2) yfinance, if installed
    try:
        import yfinance as yf  # type: ignore
        df = yf.download(_yf_symbol(ticker), start=start, end=end,
                         interval=interval, progress=False, auto_adjust=True,
                         timeout=20)
        if df is not None and not df.empty:
            df = df.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
            df.index = pd.to_datetime(df.index)
            df = df.dropna()
            _save_cached_history(ticker, interval, start, end, df, "yfinance")
            _merge_into_superset(ticker, interval, df, "yfinance")
            return df, "yfinance"
    except Exception as exc:
        print(f"[backtester] yfinance unavailable ({exc}); using synthetic data.")

    # 3) Synthetic — seeded PER TICKER so different symbols give different series
    #    (a fixed seed made every backtest identical — the bug being fixed here).
    #    Not cached: it's cheap to regenerate and caching it would risk masking
    #    a real data-source outage as a harmless cache hit.
    return synthetic_history(start, end, interval, seed_key=ticker), "synthetic"


def synthetic_history(start: str, end: str, interval: str = "1d",
                      seed_key: str = "") -> pd.DataFrame:
    freq = {"1d": "1D", "15m": "15min", "1m": "1min"}.get(interval, "15min")
    idx = pd.date_range(start=start, end=end, freq=freq)
    if len(idx) < 50:
        idx = pd.date_range(end=config.now_ist(), periods=400, freq=freq)
    # Real intraday history (e.g. yfinance) only spans ~60 days, so a multi-year
    # 15m range would balloon to 100k+ bars and stall the backtest for no realism.
    # Cap to the most recent slice, mirroring what a real intraday feed would give.
    MAX_INTRADAY_BARS = 6000
    if freq != "1D" and len(idx) > MAX_INTRADAY_BARS:
        idx = idx[-MAX_INTRADAY_BARS:]
    n = len(idx)
    # Derive the seed from the ticker so each symbol has its own price path — with a
    # fixed seed, every ticker produced the exact same numbers.
    seed = (abs(hash(seed_key)) % (2**32)) if seed_key else 42
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0004, 0.012, n)
    # inject a few trends so signals trigger
    for _ in range(max(3, n // 120)):
        s = rng.integers(0, n - 20)
        rets[s:s + 15] += rng.normal(0.003, 0.001)
    price = 100 * np.exp(np.cumsum(rets))
    close = pd.Series(price, index=idx)
    open_ = close.shift(1).fillna(close.iloc[0])
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.004, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.004, n)))
    vol = np.abs(rng.normal(1_000_000, 300_000, n)).astype(int) + 1000
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": vol},
        index=idx,
    )


# --------------------------------------------------------------------------- #
#  Entry filters — BACKTEST ONLY
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class TradeFilters:
    """Restrict WHEN and WHICH WAY new entries may open, for research.

    Three deliberate properties:

      * ENTRIES ONLY. An open position is managed to its stop/target/time-exit
        on every bar regardless — a filter must never strand a position by
        switching off the bars that would have closed it.
      * EMPTY MEANS UNRESTRICTED. A default TradeFilters() changes nothing, and
        run_backtest(filters=None) takes the identical code path it did before
        this existed.
      * BACKTEST ONLY. engine.py never reads this. The live equivalent is
        symbol_config.py's per-symbol days/hours, which is a different feature
        with a different scope (per symbol, and it can also square off).

    `days` are Python weekday numbers (Monday = 0). `hours` are IST hours 0-23,
    where hour H covers H:00-H:59 — so [9, 10] means 09:00-10:59.
    """
    days: frozenset[int] = frozenset()
    hours: frozenset[int] = frozenset()
    side: str = "BOTH"                      # "BOTH" | "BUY" | "SELL"

    @property
    def active(self) -> bool:
        return bool(self.days or self.hours or self.side != "BOTH")

    def allows_bar(self, ts) -> bool:
        """May a NEW entry open on this bar's timestamp?"""
        if self.days:
            try:
                if ts.weekday() not in self.days:
                    return False
            except AttributeError:          # non-datetime index; no day to test
                return True
        if self.hours:
            try:
                if ts.hour not in self.hours:
                    return False
            except AttributeError:
                return True
        return True

    def allows_side(self, side: str) -> bool:
        return self.side == "BOTH" or self.side == side

    def describe(self) -> str:
        bits = []
        if self.days:
            names = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
            bits.append("/".join(names[d] for d in sorted(self.days)))
        if self.hours:
            bits.append(", ".join(f"{h}:00" for h in sorted(self.hours)))
        if self.side != "BOTH":
            bits.append(f"{self.side} only")
        return " · ".join(bits)


def parse_filters(days=None, hours=None, side: str = "BOTH"
                  ) -> Optional[TradeFilters]:
    """Build a TradeFilters, or None when nothing is actually restricted.

    Returning None for the unrestricted case keeps the "no filters means the
    original code path" property honest at the call site rather than relying on
    every branch below to check `.active`.
    """
    day_set = frozenset(int(d) for d in (days or []) if 0 <= int(d) <= 6)
    hour_set = frozenset(int(h) for h in (hours or []) if 0 <= int(h) <= 23)
    side = (side or "BOTH").upper()
    if side not in ("BOTH", "BUY", "SELL"):
        raise ValueError(f"side must be BOTH, BUY or SELL — got {side!r}.")
    f = TradeFilters(days=day_set, hours=hour_set, side=side)
    return f if f.active else None


# --------------------------------------------------------------------------- #
#  Core simulation
# --------------------------------------------------------------------------- #
def run_backtest(
    ticker: str,
    start: str,
    end: str,
    initial_capital: float,
    mode: Mode,
    lot_size: int = 1,
    strategy_key: str = "",
    risk_reward: float = 0.0,
    min_score: float = 0.0,
    filters: Optional["TradeFilters"] = None,
    patterns: Optional[list[str]] = None,
) -> BacktestResult:
    # Same resolution the engine uses, so a backtest measures exactly the
    # strategy the bot would trade — parameters included. That has to include
    # the RR override (engine.TradingEngine applies the identical replace), or
    # a backtest would silently model a different target than the live bot.
    sd = resolve_strategy(mode, strategy_key)
    params = sd.params
    if risk_reward and risk_reward > 0:
        params = replace(params, risk_reward=float(risk_reward))
    # ...and the signal-score threshold, for the same reason: it decides
    # WHETHER a setup is taken, so a backtest run at a different threshold
    # than the live bot is measuring a different strategy entirely. This is
    # the control that makes "change the score and test the market" honest.
    if min_score and min_score > 0:
        params = replace(params, cs_min_score=float(min_score))
    # ...and the candlestick pattern allow-list, for the same reason again: it
    # decides WHICH setups are even eligible. Passing it per-run lets a
    # backtest try a pattern set WITHOUT editing the saved dashboard filter the
    # live bot is trading on. Empty/None leaves params untouched, so the run
    # falls through to whatever is saved — which is what makes an un-overridden
    # backtest measure the same strategy the bot is actually running.
    if patterns:
        params = replace(params, allowed_patterns=tuple(patterns))
    # NOTE: this local `params` is what actually reaches the strategy — both
    # enrich() and sd.fn() below are called with it explicitly rather than
    # through run_strategy(), so the overrides take effect without rebinding
    # `sd`. (The live engine DOES rebind, because its runner goes through
    # run_strategy(sd, ...) and would otherwise read the strategy's own.)
    interval = {Mode.SWING: "1d", Mode.INTRADAY: "15m", Mode.SCALPER: "1m"}[mode]
    # Resolve the Upstox instrument key + live token so we backtest on REAL data.
    inst = config.INSTRUMENTS_BY_SYMBOL.get(ticker)
    instrument_key = inst.instrument_key if inst else ""
    contract_multiplier = inst.contract_multiplier if inst else 1
    session_open = (config.market_hours_for_segment(inst.segment).open_t
                    if inst else None)
    # Same notional cap the live engine applies, so backtest quantities are ones
    # the account could actually have funded.
    max_leverage = (config.max_leverage_for(inst.segment, params) if inst
                    else params.max_leverage)
    # BACKTEST-ONLY sizing: a stock backtest deploys the FULL capital field per
    # position — the 20%-per-trade cap (params.max_capital_per_trade_pct) is
    # DROPPED here and ONLY here. That cap is a LIVE risk control: it forces
    # diversification across ~5 names so no single position dominates the real
    # account. A single-instrument backtest is measuring the strategy's edge on
    # ALL the capital, so the cap would artificially shrink every position (and
    # the returns) instead. The notional/leverage cap still stands (equity = 1x =
    # exactly "all the capital"), and risk-% sizing is unchanged. Live/paper
    # (engine.py) are untouched and keep the 20% cap.
    size_params = replace(params, max_capital_per_trade_pct=0.0)
    # MCX commodities are NOT risk-sized like equity. The live engine trades a
    # FIXED number of lots (engine._mcx_fixed_size) and only checks that the
    # account can fund the margin — no risk-%, no 20%-of-account cap. The backtest
    # must mirror that, otherwise position_size floors a commodity to 0 lots (one
    # CRUDEOIL lot's ~₹7.5L notional busts the leverage/capital caps) and the
    # backtest silently takes NO trades — the "no result" bug for MCX symbols.
    is_mcx = inst is not None and inst.segment == Segment.MCX
    mcx_lots_per_trade = 1                       # same default as the live engine
    mcx_margin_per_lot = config.mcx_margin_per_lot(ticker) if is_mcx else 0.0
    token = config.UPSTOX_LIVE_ACCESS_TOKEN or config.UPSTOX_SANDBOX_TOKEN
    data, source = fetch_history(ticker, start, end, interval,
                                 instrument_key=instrument_key, token=token)
    if data.empty:
        empty = pd.Series(dtype=float)
        return BacktestResult(_metrics(empty, pd.DataFrame(), initial_capital,
                                       interval, source), empty, pd.DataFrame())
    # Enrich ONCE up front. Indicators are causal (each row uses only past/current
    # data), so a value at row i is identical whether computed on the full series
    # or on data[:i+1]. This lets the walk-forward loop read pre-computed columns
    # instead of re-enriching a growing window every bar (which was O(n^2) and made
    # intraday backtests hang). We call the mode's signal fn directly on the slice.
    data = enrich(data, params)

    def signal_fn(w):
        # `w` is already enriched, so call the strategy fn directly — going via
        # run_strategy would re-enrich a growing window every bar (O(n^2)).
        return sd.fn(w, params, session_open)

    capital = initial_capital
    equity = []
    trades = []
    position = None  # dict: side, entry, stop, target, qty
    # Bar index of the last entry/exit. The live engine refuses to re-trade a bar
    # it has already acted on (reentry_cooldown_bars); the backtest must model the
    # same guard or it measures a bot that doesn't exist.
    last_action_i = None
    if mode == Mode.SWING:
        warmup = params.ema_trend + 2
    elif mode == Mode.SCALPER:
        warmup = max(params.atr_median_window, params.context_bars,
                     params.ema_fast, params.vol_avg_period) + 2
    else:
        warmup = params.macd_slow + 2

    # The signal fns read only the last two bars (indicators are pre-computed), but
    # keep a >= warmup-sized tail so their internal length guard still passes.
    tail = warmup + 6
    for i in range(warmup, len(data)):
        window = data.iloc[max(0, i - tail + 1): i + 1]
        bar = data.iloc[i]

        # manage an open position first
        if position is not None:
            is_long = position["side"] == "BUY"
            # Direction-aware: a short is stopped out by a HIGH above its stop and
            # targeted by a LOW below its target — the mirror of a long.
            if is_long:
                hit_sl = bar["low"] <= position["stop"]
                hit_tp = bar["high"] >= position["target"]
            else:
                hit_sl = bar["high"] >= position["stop"]
                hit_tp = bar["low"] <= position["target"]
            exit_price, exit_reason = None, ""
            if hit_sl:
                exit_price, exit_reason = position["stop"], "STOP-LOSS"  # worst case when both hit
            elif hit_tp:
                exit_price, exit_reason = position["target"], "TARGET"
            # Time exit (Scalper): bars_held is exact because bars are fixed-width.
            if exit_price is None and params.max_hold_minutes > 0:
                held_bars = i - position["bar"]
                if held_bars >= params.max_hold_minutes:   # 1 bar == 1 minute
                    exit_price = float(bar["close"])
                    exit_reason = f"TIME-EXIT ({params.max_hold_minutes}m)"
            if exit_price is not None:
                direction = 1 if is_long else -1
                pnl = ((exit_price - position["entry"]) * position["qty"]
                       * direction * contract_multiplier)
                capital += pnl
                trades.append({
                    "entry_time": position["time"], "exit_time": bar.name,
                    "side": position["side"],
                    "entry": position["entry"], "exit": exit_price,
                    "qty": position["qty"], "pnl": pnl,
                    "rr": params.risk_reward, "win": pnl > 0,
                    # WHY the trade was taken and WHY it closed — mirrors the live
                    # log so a backtest row explains itself, not just its numbers.
                    "entry_reason": position["reason"], "exit_reason": exit_reason,
                })
                position = None
                last_action_i = i

        # Look for a new entry only when flat AND off cooldown. Without the
        # cooldown an exit would be followed by re-entry into the identical setup
        # on the very same bar.
        cooling = (last_action_i is not None
                   and (i - last_action_i) < params.reentry_cooldown_bars)
        # `window` is already enriched, so we call the signal fn directly
        # (generate_signal would re-enrich = O(n^2)).
        # Entry filters gate ENTRIES ONLY — the position block above has
        # already run, so a bar excluded here can still close a position.
        bar_allowed = filters is None or filters.allows_bar(bar.name)
        if position is None and not cooling and bar_allowed:
            sig = signal_fn(window)
            if sig is not None and filters is not None \
                    and not filters.allows_side(sig.side):
                sig = None
            if sig is not None:
                if is_mcx:
                    # Fixed-lot commodity sizing (mirrors engine._mcx_fixed_size):
                    # trade a set number of lots as long as the account can fund the
                    # per-lot margin. qty is a LOT COUNT; PnL below multiplies by
                    # contract_multiplier (the per-lot point value).
                    margin_needed = mcx_margin_per_lot * mcx_lots_per_trade
                    if 0 < margin_needed <= capital:
                        qty = mcx_lots_per_trade
                        risk_amt = (mcx_lots_per_trade
                                    * abs(sig.entry_price - sig.stop_loss)
                                    * contract_multiplier)
                    else:
                        qty, risk_amt = 0, 0.0
                else:
                    # size_params drops the 20% per-trade cap for the backtest
                    # (see its definition above); every other risk input is params'.
                    qty, risk_amt = position_size(capital, sig, size_params,
                                                  lot_size, contract_multiplier,
                                                  max_leverage)
                if qty > 0:
                    position = {
                        "side": sig.side,
                        "entry": sig.entry_price, "stop": sig.stop_loss,
                        "target": sig.target, "qty": qty, "time": bar.name,
                        "bar": i, "reason": sig.reason,
                    }
                    last_action_i = i

        equity.append(capital)

    equity_curve = pd.Series(equity, index=data.index[warmup:])
    trades_df = pd.DataFrame(trades)
    metrics = _metrics(equity_curve, trades_df, initial_capital, interval, source)
    return BacktestResult(metrics, equity_curve, trades_df)


# --------------------------------------------------------------------------- #
#  Trade analytics — the cuts of a trade log worth plotting
# --------------------------------------------------------------------------- #
WEEKDAY_NAMES = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")

#: Below this many trades a bucket's win rate is noise, so insights refuse to
#: name it. The improvement doc uses the same rule of thumb ("don't trust any
#: symbol bucket with fewer than ~30 trades") — 8 is the floor for saying
#: anything at all about a weekday or hour, which have far fewer buckets.
MIN_BUCKET_TRADES = 8


def _bucket(df: pd.DataFrame, key) -> list[dict]:
    """[{key, pnl, trades, wins, win_rate}] grouped by `key`, key order kept."""
    out = []
    for k, g in df.groupby(key, sort=True):
        wins = int(g["win"].sum())
        n = int(len(g))
        out.append({
            "key": k,
            "pnl": round(float(g["pnl"].sum()), 2),
            "trades": n,
            "wins": wins,
            "win_rate": round(100.0 * wins / n, 2) if n else 0.0,
        })
    return out


def trade_analytics(trades: pd.DataFrame, top_setups: int = 10) -> dict:
    """Cuts of the trade log for the analytics charts.

    Everything is derived from the trade rows the simulation already records —
    no second simulation, so these numbers cannot disagree with the metrics
    beside them. Bucketed by ENTRY time throughout: a trade belongs to the hour
    and the day it was TAKEN, which is the decision being judged. Bucketing by
    exit would attribute a Monday decision to Tuesday whenever a position was
    held across the boundary.

    Returns empty lists (never raises) when there are no trades, so a filtered
    run that took none still renders.
    """
    empty = {"by_weekday": [], "by_hour": [], "by_setup": [],
             "by_side": [], "insights": [], "total_trades": 0}
    if trades is None or trades.empty or "pnl" not in trades.columns:
        return empty

    t = trades.copy()
    t["pnl"] = pd.to_numeric(t["pnl"], errors="coerce").fillna(0.0)
    if "win" not in t.columns:
        t["win"] = t["pnl"] > 0
    t["win"] = t["win"].astype(bool)

    entry = pd.to_datetime(t["entry_time"], errors="coerce")
    t = t[entry.notna()].copy()
    if t.empty:
        return empty
    entry = entry[entry.notna()]
    t["_weekday"] = entry.dt.weekday.values
    t["_hour"] = entry.dt.hour.values

    by_weekday = [{**b, "label": WEEKDAY_NAMES[int(b["key"])]}
                  for b in _bucket(t, "_weekday")]
    by_hour = [{**b, "label": f"{int(b['key']):02d}:00"}
               for b in _bucket(t, "_hour")]

    # A "setup" is the entry reason the strategy logged. Candlestick strategies
    # put the pattern names there, which is exactly the cut worth ranking; a
    # strategy with one fixed reason simply yields one row.
    if "entry_reason" in t.columns:
        t["_setup"] = (t["entry_reason"].astype(str)
                       # Drop the parenthetical detail ("(evidence 4.2, ...)")
                       # so the same pattern set groups together instead of
                       # splitting into one bucket per evidence value.
                       .str.split("(").str[0].str.strip().replace("", "unnamed"))
    else:
        t["_setup"] = "unnamed"
    setups = sorted(_bucket(t, "_setup"), key=lambda b: b["pnl"], reverse=True)
    by_setup = [{**b, "label": str(b["key"])} for b in setups[:top_setups]]

    by_side = [{**b, "label": str(b["key"])} for b in _bucket(t, "side")]

    return {
        "by_weekday": by_weekday,
        "by_hour": by_hour,
        "by_setup": by_setup,
        "by_side": by_side,
        "total_trades": int(len(t)),
        "insights": _insights(by_weekday, by_hour, by_setup, by_side, int(len(t))),
    }


def _insights(by_weekday, by_hour, by_setup, by_side, total) -> list[str]:
    """Plain-language findings from the buckets above.

    Every claim names the sample it rests on, and a bucket under
    MIN_BUCKET_TRADES is never held up as a finding — a 100% win rate on three
    trades is the single easiest way to talk yourself into a bad change.
    """
    out: list[str] = []
    solid = lambda rows: [r for r in rows if r["trades"] >= MIN_BUCKET_TRADES]

    wk = solid(by_weekday)
    if wk:
        best = max(wk, key=lambda r: r["pnl"])
        worst = min(wk, key=lambda r: r["pnl"])
        if best["label"] != worst["label"]:
            out.append(
                f"{best['label']} is the best day (₹{best['pnl']:,.0f} over "
                f"{best['trades']} trades); {worst['label']} is the worst "
                f"(₹{worst['pnl']:,.0f} over {worst['trades']}).")
        if worst["pnl"] < 0:
            out.append(
                f"Dropping {worst['label']} would have removed "
                f"₹{abs(worst['pnl']):,.0f} of losses — test it before trusting it.")

    hr = solid(by_hour)
    if hr:
        best = max(hr, key=lambda r: r["pnl"])
        worst = min(hr, key=lambda r: r["pnl"])
        out.append(
            f"{best['label']} is the strongest hour (₹{best['pnl']:,.0f}, "
            f"{best['win_rate']:.0f}% win over {best['trades']} trades)"
            + (f"; {worst['label']} is the weakest (₹{worst['pnl']:,.0f})."
               if worst["label"] != best["label"] else "."))
        losing = [r for r in hr if r["pnl"] < 0]
        if losing:
            out.append(
                "Loss-making hours: "
                + ", ".join(f"{r['label']} (₹{r['pnl']:,.0f})" for r in losing)
                + ".")

    if by_setup:
        top = by_setup[0]
        out.append(
            f"Best setup is {top['label']} (₹{top['pnl']:,.0f} over "
            f"{top['trades']} trades, {top['win_rate']:.0f}% win).")
        bad = [s for s in by_setup
               if s["pnl"] < 0 and s["trades"] >= MIN_BUCKET_TRADES]
        if bad:
            out.append(
                f"{len(bad)} setup(s) lost money on a meaningful sample, worst "
                f"{min(bad, key=lambda s: s['pnl'])['label']}.")

    sides = {r["label"]: r for r in by_side}
    buy, sell = sides.get("BUY"), sides.get("SELL")
    if buy and sell and min(buy["trades"], sell["trades"]) >= MIN_BUCKET_TRADES:
        better, worse = ((buy, sell) if buy["pnl"] >= sell["pnl"]
                         else (sell, buy))
        out.append(
            f"{better['label']} outperformed {worse['label']}: "
            f"₹{better['pnl']:,.0f} at {better['win_rate']:.0f}% win vs "
            f"₹{worse['pnl']:,.0f} at {worse['win_rate']:.0f}%.")
    elif buy and not sell:
        out.append("Long-only in this run — no short trades to compare.")
    elif sell and not buy:
        out.append("Short-only in this run — no long trades to compare.")

    if total < 30:
        out.append(
            f"Only {total} trades — too few to act on. Treat everything above "
            "as a hint, not a finding.")
    return out


# --------------------------------------------------------------------------- #
#  RR sweep — the SAME symbol and window at a range of risk:reward ratios
# --------------------------------------------------------------------------- #

#: Ceiling on how many RRs one sweep may run. Each step is a full simulation,
#: so an unbounded (start, step, end) typo — say step 0.001 — would otherwise
#: queue thousands of runs and hang the request. Raising this only costs time,
#: never correctness.
RR_SWEEP_MAX_STEPS = 40

#: Floor on the step. Below this the runs are indistinguishable anyway: a stop
#: is tick-rounded, so a 0.01 change in RR frequently produces the identical
#: target price and therefore a byte-identical run.
RR_SWEEP_MIN_STEP = 0.05


def rr_sweep_values(start: float, step: float, end: float) -> list[float]:
    """The RR ladder a sweep will run, inclusive of both ends.

    Built with integer arithmetic rather than repeated addition: accumulating
    0.1 in binary float lands on 1.0000000000000007 by the eighth step, which
    would both mis-label the row and defeat the cache key.

    Raises ValueError with an HTTP-400-worthy message on a nonsensical range.
    """
    start, step, end = float(start), float(step), float(end)
    if start <= 0:
        raise ValueError("Start RR must be greater than 0.")
    if end < start:
        raise ValueError(f"End RR ({end:g}) must be at or above start ({start:g}).")
    if step < RR_SWEEP_MIN_STEP:
        raise ValueError(
            f"Step must be at least {RR_SWEEP_MIN_STEP:g} — smaller steps often "
            "produce the identical target once the stop is tick-rounded.")
    steps = int(round((end - start) / step)) + 1
    if steps > RR_SWEEP_MAX_STEPS:
        raise ValueError(
            f"That range needs {steps} runs; the limit is {RR_SWEEP_MAX_STEPS}. "
            "Use a bigger step or a narrower range.")
    return [round(start + i * step, 4) for i in range(steps)]


def run_rr_sweep(
    ticker: str,
    start_date: str,
    end_date: str,
    initial_capital: float,
    mode: Mode,
    rr_start: float,
    rr_step: float,
    rr_end: float,
    lot_size: int = 1,
    strategy_key: str = "",
    min_score: float = 0.0,
    patterns: Optional[list[str]] = None,
) -> list[dict]:
    """Run the same backtest once per RR and return one summary row each.

    Purely a LOOP over the existing run_backtest — no simulation logic is
    duplicated or altered, so a sweep row and a single run at the same RR are
    the same number by construction. RR is already a first-class override on
    run_backtest (it does `replace(params, risk_reward=...)`, exactly as the
    live engine does), which is what makes this honest rather than an
    approximation.

    History is fetched once and served from the parquet cache thereafter, so N
    runs cost roughly one download plus N simulations.

    A failing RR yields a row with an `error` rather than aborting the sweep —
    one bad step must not throw away the rows already computed.

    NOTE ON VALIDATION: RR here is deliberately NOT checked against
    config.RR_CHOICES. Those choices bound what may be armed on LIVE money;
    this is research on history, and the whole point is to discover whether a
    ratio outside the offered set is better. Immutable Rule #1 is untouched
    either way — RR moves the TARGET only, and position size is
    risk_budget / stop_distance, which never reads it.
    """
    rows: list[dict] = []
    for rr in rr_sweep_values(rr_start, rr_step, rr_end):
        try:
            res = run_backtest(ticker, start_date, end_date, initial_capital,
                               mode, lot_size=lot_size,
                               strategy_key=strategy_key,
                               risk_reward=rr, min_score=min_score,
                               patterns=patterns)
            m = res.metrics
            rows.append({
                "risk_reward": rr,
                "trades": m.get("Total Trades", 0),
                "return_pct": m.get("Total Return %", 0.0),
                "win_rate": m.get("Win Rate %", 0.0),
                "max_drawdown": m.get("Max Drawdown %", 0.0),
                "sharpe": m.get("Sharpe Ratio", 0.0),
                "calmar": m.get("Calmar Ratio", 0.0),
                "final_equity": m.get("Final Equity", 0.0),
                "data_source": m.get("Data Source", ""),
                "error": "",
            })
        except Exception as exc:
            rows.append({"risk_reward": rr, "trades": 0, "return_pct": 0.0,
                         "win_rate": 0.0, "max_drawdown": 0.0, "sharpe": 0.0,
                         "calmar": 0.0, "final_equity": 0.0, "data_source": "",
                         "error": f"{type(exc).__name__}: {exc}"[:200]})
    return rows


# --------------------------------------------------------------------------- #
#  Bulk simulation — same strategy/params across a bucket of instruments
# --------------------------------------------------------------------------- #
def run_bulk_backtest(
    tickers: list[str],
    start: str,
    end: str,
    initial_capital: float,
    mode: Mode,
    strategy_key: str = "",
    progress_cb=None,
    max_workers: int = BULK_MAX_WORKERS,
    risk_reward: float = 0.0,
    min_score: float = 0.0,
    filters: Optional["TradeFilters"] = None,
    patterns: Optional[list[str]] = None,
) -> dict[str, BacktestResult]:
    """Run the SAME strategy with the SAME parameters over every ticker in the
    bucket and return {ticker: BacktestResult}. Each instrument is simulated
    independently on its own real data, starting from the identical capital, so
    their equity curves are directly comparable in a single chart.

    Tickers are backtested concurrently, up to `max_workers` at once (default 5
    — comfortably inside broker rate limits while cutting wall-clock time
    roughly 5x versus a sequential loop). Each ticker's `run_backtest` call is
    independent (its own data fetch, its own local variables), so this is safe.

    `progress_cb(done, total, ticker)` — optional; called from the calling
    thread as each symbol finishes (not from a worker thread), so it's safe to
    use with UI frameworks like Streamlit that are picky about thread origin.
    Lot size is taken per-instrument from config, the same way the
    single-ticker path resolves it, so quantities stay realistic.
    """
    total = len(tickers)
    interval = {Mode.SWING: "1d", Mode.INTRADAY: "15m", Mode.SCALPER: "1m"}[mode]

    def _run_one(ticker: str) -> tuple[str, BacktestResult]:
        inst = config.INSTRUMENTS_BY_SYMBOL.get(ticker)
        lot_size = inst.lot_size if inst else 1
        try:
            return ticker, run_backtest(
                ticker, start, end, initial_capital, mode,
                lot_size=lot_size, strategy_key=strategy_key,
                risk_reward=risk_reward, min_score=min_score,
                filters=filters, patterns=patterns)
        except Exception as exc:
            # One bad symbol must not sink the whole bucket — record an empty
            # result so the UI can show it failed rather than aborting the run.
            print(f"[backtester] bulk: {ticker} failed ({exc}).")
            empty = pd.Series(dtype=float)
            return ticker, BacktestResult(
                _metrics(empty, pd.DataFrame(), initial_capital, interval,
                         "error"),
                empty, pd.DataFrame())

    results: dict[str, BacktestResult] = {}
    done = 0
    done_lock = threading.Lock()
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_run_one, ticker) for ticker in tickers]
        for future in as_completed(futures):
            ticker, result = future.result()
            results[ticker] = result
            with done_lock:
                done += 1
                done_snapshot = done
            if progress_cb is not None:
                progress_cb(done_snapshot, total, ticker)

    # as_completed finishes in whatever order threads happen to land in;
    # reorder back to the caller's ticker order so downstream consumers see
    # deterministic, reproducible output.
    return {ticker: results[ticker] for ticker in tickers if ticker in results}


def bulk_summary_frame(results: dict[str, BacktestResult]) -> pd.DataFrame:
    """Flatten bulk results into one comparison table, best return first."""
    rows = []
    for ticker, res in results.items():
        m = res.metrics
        rows.append({
            "Ticker": ticker,
            "Total Return %": m["Total Return %"],
            "Max Drawdown %": m["Max Drawdown %"],
            "Sharpe": m["Sharpe"],
            "Calmar": m["Calmar"],
            "Win Rate %": m["Win Rate %"],
            "Trades": m["Total Trades"],
            "Final Equity": m["Final Equity"],
            "Data Source": m.get("Data Source", "synthetic"),
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("Total Return %", ascending=False).reset_index(drop=True)
    return df


# --------------------------------------------------------------------------- #
#  Metrics
# --------------------------------------------------------------------------- #
def _metrics(equity: pd.Series, trades: pd.DataFrame,
             initial_capital: float, interval: str,
             source: str = "synthetic") -> dict:
    if equity.empty:
        return {"Total Return %": 0.0, "Max Drawdown %": 0.0, "Sharpe": 0.0,
                "Calmar": 0.0, "Win Rate %": 0.0, "Total Trades": 0,
                "Final Equity": initial_capital, "Data Source": source}

    total_return = (equity.iloc[-1] / initial_capital - 1) * 100

    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max
    max_dd = drawdown.min() * 100  # negative

    rets = equity.pct_change().dropna()
    periods_per_year = BARS_PER_YEAR.get(interval, TRADING_DAYS * 25)
    if rets.std() > 0:
        sharpe = (rets.mean() / rets.std()) * np.sqrt(periods_per_year)
    else:
        sharpe = 0.0

    years = max(len(equity) / periods_per_year, 1e-9)
    cagr = (equity.iloc[-1] / initial_capital) ** (1 / years) - 1
    calmar = (cagr / abs(max_dd / 100)) if max_dd != 0 else 0.0

    win_rate = (100 * trades["win"].mean()) if not trades.empty else 0.0

    return {
        "Total Return %": round(total_return, 2),
        "Max Drawdown %": round(max_dd, 2),
        "Sharpe": round(float(sharpe), 2),
        "Calmar": round(float(calmar), 2),
        "Win Rate %": round(float(win_rate), 2),
        "Total Trades": int(len(trades)),
        "Final Equity": round(float(equity.iloc[-1]), 2),
        "Data Source": source,
    }
