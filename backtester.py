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

import alpaca_data
import config
import exit_manager
from config import Mode, Segment, params_for_mode
from cost_model import INTRADAY_EQUITY, MCX_COMMODITY, US_EQUITY
from strategy import (clamp_signal_risk, enrich, position_size,
                      resolve_strategy)


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
# The session length the two intraday constants above already encode (375 min
# = 25 bars * 15 min = 6.25h). Any OTHER intraday bar size (a 5m Intraday run,
# say) derives its own bars/year from this instead of silently reusing the 15m
# constant via a bare dict .get() fallback — that would under-annualise Sharpe
# by the size of the mismatch (3x too few periods/year for a 5m run) without
# ever raising an error.
EQUITY_SESSION_MINUTES = 6.25 * 60
#: US regular session is 09:30-16:00 = 6.5h, slightly longer than NSE's 6.25h.
US_SESSION_MINUTES = 6.5 * 60


def _bars_per_year(interval: str) -> float:
    """Bars/year for Sharpe/Calmar annualisation, generalised beyond the two
    hardcoded intraday constants above so an alternate Intraday timeframe
    (interval == "5m", "10m", ...) still annualises correctly."""
    if interval in BARS_PER_YEAR:
        return BARS_PER_YEAR[interval]
    if interval.endswith("m"):
        try:
            tf = int(interval[:-1])
            if tf > 0:
                return TRADING_DAYS * (EQUITY_SESSION_MINUTES / tf)
        except ValueError:
            pass
    return TRADING_DAYS * 25


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
    # A US ticker IS its Yahoo symbol — appending ".NS" (the old unconditional
    # behaviour) asked Yahoo for a non-existent NSE listing and got nothing, so
    # every US backtest silently fell through to synthetic data.
    inst = config.INSTRUMENTS_BY_SYMBOL.get(ticker)
    if inst is not None and inst.segment == Segment.US_EQUITY:
        # Yahoo writes class shares with a dash (BRK-B), Alpaca with a dot.
        return ticker.replace(".", "-")
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


def _interval_minutes(interval: str) -> Optional[int]:
    """"5m" -> 5, "15m" -> 15, "1m"/"1d" -> None (nothing to resample to — 1m
    IS Upstox's native grain and 1d is fetched as native daily candles)."""
    if interval in ("1d", "1m") or not interval.endswith("m"):
        return None
    try:
        n = int(interval[:-1])
        return n if n > 0 else None
    except ValueError:
        return None


def _fetch_upstox_hist(
    instrument_key: str, start: str, end: str, interval: str, token: str
) -> pd.DataFrame:
    """Real historical candles from Upstox for one instrument over [start, end].
    Daily for swing; 1-minute resampled to the requested intraday bar size
    (15m by default, or any other "Xm" — e.g. a 5m Intraday run) otherwise.
    Indexed IST-naive."""
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
    # Only a coarser-than-1m interval needs building; "1m" is already what
    # Upstox returned, and resampling it to itself would silently backtest the
    # wrong timeframe. Generalised beyond 15m so any "Xm" Intraday override
    # (5m, 10m, ...) gets built off the identical raw 1-minute candles.
    tf = _interval_minutes(interval)
    if tf:
        df = df.resample(f"{tf}min").agg(
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

    # 0c) ALPACA — the good path for US equities, and only for them. Tried
    #     ahead of Upstox because an Upstox instrument_key is never valid for a
    #     US ticker, and ahead of yfinance because Alpaca serves years of
    #     intraday history where yfinance caps out at 60 days. Silently skipped
    #     when unconfigured, so a US backtest still works on yfinance daily
    #     bars with no credentials at all.
    _inst = config.INSTRUMENTS_BY_SYMBOL.get(ticker)
    if _inst is not None and _inst.segment == Segment.US_EQUITY:
        if alpaca_data.is_configured():
            df = alpaca_data.fetch_bars(ticker, start, end, interval)
            if len(df) > 30:
                print(f"[backtester] {ticker}: {len(df)} Alpaca bars "
                      f"({alpaca_data.data_feed_name()} feed).")
                _save_cached_history(ticker, interval, start, end, df, "alpaca")
                _merge_into_superset(ticker, interval, df, "alpaca")
                return df, "alpaca"
            print(f"[backtester] {ticker}: Alpaca returned {len(df)} bars; "
                  "falling back to yfinance.")
        elif interval == "1d":
            print(f"[backtester] {ticker}: ALPACA_API_KEY/SECRET not set — "
                  "using yfinance daily bars (deep history, no key needed).")
        else:
            # The trap this guards: yfinance serves only ~60 days of intraday
            # history, so an out-of-range US intraday request falls all the way
            # through to the SYNTHETIC random walk and returns a full set of
            # plausible-looking trades that mean nothing. The source is still
            # reported as "synthetic" downstream, but by then you have already
            # read the numbers.
            # ASCII only: this goes to a console that may be cp1252 (Windows),
            # where an emoji raises UnicodeEncodeError and kills the backtest.
            print(f"[backtester] WARNING {ticker}: US INTRADAY without Alpaca keys. "
                  f"yfinance serves only ~60 days at {interval}; anything older "
                  "cannot be fetched and will fall back to SYNTHETIC data. "
                  "Set ALPACA_API_KEY/ALPACA_API_SECRET for real intraday "
                  "history, or use Swing (daily) which needs no key.")

    # 1) REAL Upstox historical data — the good path. Ticker-specific & real, so
    #    every instrument gives genuinely different results. Skipped for US
    #    symbols: an Upstox key is an "NSE_EQ|INE..." string, so passing "AAPL"
    #    just earns a UDAPI1021 "invalid format" round trip before failing over.
    _is_us = _inst is not None and _inst.segment == Segment.US_EQUITY
    if instrument_key and token and not _is_us:
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
            # yfinance >= 0.2.5x returns MultiIndex columns — ("Close", "AAPL")
            # — even for a single ticker. Left alone, the rename+select below
            # yields a frame whose "close" is itself a one-column FRAME, and
            # the first strategy call dies on "truth value of a Series is
            # ambiguous". Flatten to the field name first.
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
            # An intraday yfinance frame is tz-aware UTC while a daily one is
            # naive. Everything downstream assumes naive exchange-local time,
            # so convert rather than carry a mixed convention.
            idx = pd.to_datetime(df.index)
            if getattr(idx, "tz", None) is not None:
                tzname = ("America/New_York"
                          if _inst is not None
                          and _inst.segment == Segment.US_EQUITY else "Asia/Kolkata")
                idx = idx.tz_convert(tzname).tz_localize(None)
            df.index = idx
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
    if interval == "1d":
        freq = "1D"
    elif interval == "1m":
        freq = "1min"
    else:
        # Any "Xm" Intraday override (15m default, or a 5m/10m run) — falls
        # back to 15min only for a genuinely malformed interval string.
        tf = _interval_minutes(interval)
        freq = f"{tf}min" if tf else "15min"
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
def interval_for(mode: Mode, timeframe_minutes: int = 0) -> str:
    """The bar size a run uses. ONE definition, shared.

    timeframe_minutes ONLY overrides Intraday's bar size (0 = its default, 15).
    Swing (daily, by design) and Scalper (native 1m, by design) are not this
    knob's business — an override passed under either mode is silently ignored
    rather than raising, matching how risk_reward=0 / min_score=0 already mean
    "don't override" everywhere else here.

    Public because api/routers/backtest.py's chart endpoint has to fetch the
    SAME candles the simulation walked: a chart drawn on 15m bars for a run
    that happened on 5m ones would put every marker on the wrong candle.
    """
    if mode == Mode.INTRADAY and timeframe_minutes and timeframe_minutes > 0:
        return f"{int(timeframe_minutes)}m"
    return {Mode.SWING: "1d", Mode.INTRADAY: "15m", Mode.SCALPER: "1m"}[mode]


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
    ignore_saved_patterns: bool = False,
    timeframe_minutes: int = 0,
    include_costs: bool = False,
    max_stop_pct: float = 0.0,
    min_stop_pct: float = 0.0,
    exit_style: str = "strategy",
    trail_atr_mult: float = 0.0,
    partial_exit_fraction: float = -1.0,
    runner_rr_mult: float = -1.0,
    hold_overnight: bool = False,
    ignore_entry_cutoff: bool = False,
) -> BacktestResult:
    """Simulate one strategy over one instrument.

    `hold_overnight` — LET INTRADAY POSITIONS LIVE PAST THE SESSION.

    Off by default, so every existing caller measures exactly what it measured
    before. With it on, the end-of-session flat-out is removed and a position
    closes on ONE of its own rules: stop, target, trail, or the strategy's own
    max_hold_minutes.

    `ignore_entry_cutoff` — DROP THE LATE-ENTRY GATE (params.
    entry_cutoff_before_close, 11:59 for Candlestick Intraday). A separate flag
    from hold_overnight on purpose: it changes how many trades are TAKEN, where
    hold_overnight changes how they are CLOSED. Setting both at once moves two
    variables and the result cannot be attributed to either.

    THIS IS A DIFFERENT BOT, NOT A LOOSER BACKTEST. Three consequences follow,
    and reading the result without them will mislead:

      1. IT IS NO LONGER INTRADAY. A position held past 15:30 in the NSE cash
         segment is DELIVERY. That means full cash funding (equity is already
         pinned to 1x by config.SEGMENT_MAX_LEVERAGE, so sizing does not
         change) and it means SHORTS ARE NOT POSSIBLE — you cannot take
         delivery of a short. Candlestick Intraday sets allow_short=True, so a
         run with this flag on will contain short trades that the cash segment
         could not have held overnight. Judge the long side separately, or
         re-run with a long-only params set, before believing the total.

      2. COSTS ARE STILL PRICED AS INTRADAY. bulk_backtester picks
         cost_model.INTRADAY_EQUITY for every non-MCX symbol and never looks at
         the mode. Delivery pays STT of 0.1% on BOTH legs against intraday's
         0.025% sell-only, plus 0.015% stamp against 0.003% — roughly 28 bps of
         notional against 10. A net figure from this flag is therefore
         OPTIMISTIC by that difference. Pass an explicit delivery CostModel to
         bulk_backtester.run_with_costs to price it honestly.

      3. GAPS FILL AT THE STOP. exit_manager.step_bar hands back price=st.stop
         on a stop hit, never the bar's open. Within a session that is
         accurate (15m bars do not gap); across sessions it is not, and every
         overnight gap through the stop is credited a fill the market never
         offered. This biases the result in the flattering direction and the
         bias grows with holding period.

    None of the above is a reason not to run it — the question "does this
    signal have edge when it is given room?" is worth a real answer. They are
    the reasons the answer is a starting point rather than a P&L forecast.
    """
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
    # ...or ignore the saved filter altogether. The combination search's screen
    # needs the full pattern set; without this it could only rediscover
    # whatever was already allowed on the dashboard.
    elif ignore_saved_patterns:
        params = replace(params, ignore_pattern_filter=True)
    # ...and the EXIT STYLE, for the same reason as all of the above: what
    # happens after the first target is part of the strategy, and it is the
    # one part that could not be measured at all until exit_manager existed.
    # "strategy" (the default) changes nothing, so every existing caller keeps
    # the result it had. The other styles let one run be compared against
    # another with a single variable moved:
    #     "fixed"         — the baseline: fixed ATR stop/target, no management
    #     "trail_full"    — Approach 1: trail the whole position from entry
    #     "partial_trail" — Approach 2: book part at 1R, trail the runner
    # None of them touch the entry, the initial stop/target or the sizing.
    params = config.apply_exit_style(
        params, exit_style, trail_atr_mult=trail_atr_mult,
        partial_exit_fraction=partial_exit_fraction,
        runner_rr_mult=runner_rr_mult)
    # ...and the stop-distance band, for the same reason as everything above:
    # it decides WHERE the stop and target sit, so a run at a different band
    # is measuring a different strategy. 0 leaves the ATR stop entirely alone.
    if max_stop_pct and max_stop_pct > 0:
        params = replace(params, max_stop_pct=float(max_stop_pct))
    if min_stop_pct and min_stop_pct > 0:
        params = replace(params, min_stop_pct=float(min_stop_pct))
    # NOTE: this local `params` is what actually reaches the strategy — both
    # enrich() and sd.fn() below are called with it explicitly rather than
    # through run_strategy(), so the overrides take effect without rebinding
    # `sd`. (The live engine DOES rebind, because its runner goes through
    # run_strategy(sd, ...) and would otherwise read the strategy's own.)
    interval = interval_for(mode, timeframe_minutes)
    # Resolve the Upstox instrument key + live token so we backtest on REAL data.
    inst = config.INSTRUMENTS_BY_SYMBOL.get(ticker)
    instrument_key = inst.instrument_key if inst else ""
    contract_multiplier = inst.contract_multiplier if inst else 1
    session_open = (config.market_hours_for_segment(inst.segment).open_t
                    if inst else None)
    # End-of-session flat-out — the backtest's counterpart to
    # strategy_runner._square_off_due/_do_square_off. Live, an INTRADAY/SCALPER
    # position is force-closed at this wall-clock time regardless of P&L
    # (15:09 equity, 23:15 MCX — config.DEFAULT_SQUARE_OFF); a backtest that
    # skips this lets a position run past it and grants trades runway the live
    # bot never gives them, which systematically flatters slow-moving symbols
    # and high-RR targets that would have been cut off. None for Swing (holds
    # overnight by design) and when the instrument is unknown.
    square_off_cutoff = (config.square_off_time_for(inst.segment, mode)
                         if inst else None)
    # ...UNLESS this run is explicitly asking the other question: what would
    # this signal have earned if the position were simply left alone until its
    # stop, its target or its trail closed it? See the hold_overnight docstring
    # above for what that measures and what it does NOT.
    if hold_overnight:
        square_off_cutoff = None
    # Latest bar on which a NEW position may open. None for every strategy
    # that sets no cutoff, which is all of them bar Candlestick Intraday and
    # CRUDEOIL — so this is inert unless asked for. Enforced HERE as well as
    # in the live runner because a cutoff that only one of them honours would
    # make the backtest describe a bot that does not exist.
    #
    # A SEPARATE KNOB FROM hold_overnight, deliberately, even though the two
    # are related: config.entry_cutoff_for derives the cutoff from the flat-out
    # ("a trade opened near the close has no runway"), so removing the flat-out
    # does weaken its justification. But the cutoff was measured to be worth
    # +0.96pp on its own and it halves the position count, and this book's
    # dominant failure mode is turnover — so whether it still pays once
    # positions can run is an EMPIRICAL question, not a corollary.
    #
    # Tying the two together would move two variables per run and make that
    # question unanswerable. They are independent flags so the 2x2 can be read.
    entry_cutoff = (config.entry_cutoff_for(inst.segment, mode, params)
                    if inst and not ignore_entry_cutoff else None)
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
    # COSTS (include_costs=True). OFF by default so every existing caller keeps
    # the byte-identical gross result it had before this existed — bulk_backtester
    # and advanced_backtest both deduct their OWN costs from the gross `pnl`
    # column below, and switching this on by default would double-charge them.
    # `pnl` therefore STAYS GROSS whatever this flag says; costs are carried in
    # the separate `cost` / `net_pnl` columns, and only the CAPITAL the
    # simulation compounds (and so the equity curve and every metric derived
    # from it) is reduced by them.
    is_us = inst is not None and inst.segment == Segment.US_EQUITY
    if not include_costs:
        cost_model = None
    elif is_mcx:
        cost_model = MCX_COMMODITY
    elif is_us:
        # USD, and a completely different fee structure — no STT, no stamp, no
        # GST. Figures from a US run are therefore in DOLLARS; do not add them
        # to a rupee book (config.SEGMENT_CURRENCY).
        cost_model = US_EQUITY
    else:
        cost_model = INTRADAY_EQUITY
    # The SAME exit state-machine the live engine runs (engine._manage_open).
    # Everything after the first target — the partial, the break-even move,
    # the runner target, the ATR chandelier — is decided by exit_manager here
    # too, which is the whole point: before this, the backtest exited on the
    # fixed entry stop/target only, so no trailing or scaling result it
    # produced described the bot that would actually trade.
    #
    # Inert at the defaults. With partial_exit_fraction=0, trail_remainder=
    # False and trail_from_entry=False, step_bar exits on exactly the stop /
    # target the old inline check used, so an unconfigured strategy's
    # backtest is unchanged.
    #
    # Built by the SAME function engine._exit_params calls, so the two cannot
    # drift apart. The only argument the engine passes and this does not is
    # the per-symbol trail override: that is a LIVE admin control, and a
    # backtest measures the strategy rather than one admin's symbol settings.
    exit_params = exit_manager.params_from_strategy(
        params, (inst.tick_size if inst else 0.05), contract_multiplier)
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
        # The stop-distance clamp still has to run, and through the SAME
        # function run_strategy uses, or a capped stop would be a live-only
        # behaviour the backtest never modelled.
        return clamp_signal_risk(sd.fn(w, params, session_open), params)

    capital = initial_capital
    equity = []
    trades = []
    position = None  # dict: side, entry, stop, target, qty, exit_state

    def _leg(pos: dict, qty: int, exit_price: float, exit_reason: str,
             ts) -> float:
        """Book ONE exit leg — a partial or the final close — as its own trade
        row, and return the change in capital.

        A trade with a scale-out produces TWO rows sharing an entry, exactly
        as the live engine writes a child trade document for the partial. Both
        carry `position_id` so the legs of one trade can be re-grouped, and
        `leg` numbers them; summing `qty` over a position_id returns the entry
        quantity, which is the quantity-conservation check.

        COSTS: each leg pays a FULL round trip on its own quantity. That is
        deliberately a shade conservative — a two-leg trade is really three
        orders (one entry, two exits), not four, so a flat per-order brokerage
        is over-charged once. Being wrong in the direction that makes scaling
        out look WORSE is the correct way to be wrong here: the point of
        testing the partial at all is that its extra exit is a real cost, and
        an optimistic model would be the one thing that could make a bad
        scale-out look good.
        """
        nonlocal trades
        qty = int(qty)
        if qty <= 0:
            return 0.0
        direction = 1 if pos["side"] == "BUY" else -1
        pnl = ((exit_price - pos["entry"]) * qty * direction
               * contract_multiplier)
        cost = (cost_model.round_trip_cost(pos["entry"], exit_price, qty,
                                           contract_multiplier)
                if cost_model is not None else 0.0)
        pos["leg"] = pos.get("leg", 0) + 1
        if pos["leg"] > 1 and cost_model is not None:
            # A scaled-out position is ONE entry and TWO exits — three orders,
            # not four. Every turnover-based head already sums correctly across
            # the legs; only the flat per-order brokerage (and its GST) would
            # be counted twice, so it comes off here. See
            # CostModel.duplicate_entry_charge.
            cost = max(cost - cost_model.duplicate_entry_charge(), 0.0)
        row = {
            "entry_time": pos["time"], "exit_time": ts,
            "side": pos["side"],
            "entry": pos["entry"], "exit": exit_price,
            "qty": qty, "pnl": pnl,
            # `win` follows the money that reached the account, so a
            # cost-aware run cannot report a win rate the P&L denies.
            "rr": params.risk_reward, "win": (pnl - cost) > 0,
            # WHY the trade was taken and WHY it closed — mirrors the live
            # log so a backtest row explains itself, not just its numbers.
            "entry_reason": pos["reason"], "exit_reason": exit_reason,
            # Which trade this leg belongs to, and its order within it. A
            # single-leg trade (every strategy that does not scale out) is
            # position_id=n, leg=1 — additive columns, nothing else reads them.
            "position_id": pos["id"], "leg": pos["leg"],
            # |entry - initial stop|, i.e. 1R for this trade. Written out
            # rather than left to be reconstructed from entry/exit/rr, which
            # only works for rows that exited exactly on the stop or the
            # target — a trailed or squared-off row cannot be reversed, and
            # the MFE study (analysis/mfe_study.py) needs 1R for every trade.
            "risk_dist": round(pos["exit_state"].risk_dist, 4),
        }
        # Present ONLY on a cost-aware run — their absence is what tells
        # _metrics (and any caller) that `pnl` is the whole story.
        if cost_model is not None:
            row["cost"] = round(cost, 2)
            row["net_pnl"] = pnl - cost
        trades.append(row)
        # Deduct from the COMPOUNDING capital, not from `pnl`: a cost paid on
        # trade n really does shrink the account that sizes trade n+1, which
        # post-hoc subtraction cannot reproduce.
        return pnl - cost
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

    # SESSION-ANCHORED strategies (Commodity.md) are the exception: they do not
    # read the last two bars, they rebuild a whole session — an anchored VWAP
    # from the US open, an opening range, and a resampled higher timeframe —
    # from the window they are handed. A 34-bar tail would silently truncate the
    # session and make the anchored VWAP disagree with the live bot's.
    #
    # Gated on the strategy DECLARING an anchor or a higher timeframe, so every
    # existing strategy keeps the exact warmup and tail it had before this
    # existed and its backtests stay bit-identical.
    if getattr(params, "orb_minutes", 0) or getattr(params, "htf_minutes", 0):
        # Derived from the ACTUAL resolved interval (honours timeframe_minutes)
        # rather than re-hardcoding 15 for Intraday, so a session-anchored /
        # HTF-bias strategy run at an overridden bar size still sizes its
        # session and HTF look-back correctly.
        bar_min = (_interval_minutes(interval)
                  or {Mode.SWING: 24 * 60, Mode.INTRADAY: 15, Mode.SCALPER: 1}[mode])
        # One MCX session is 09:00-23:30 IST; two of them covers the anchor even
        # when the window opens mid-session, plus the ATR median look-back.
        session_bars = int((14.5 * 60) / bar_min) + 1
        htf_bars_needed = (int(params.htf_minutes / bar_min) * (params.htf_bars + 2)
                           if params.htf_minutes else 0)
        warmup = max(warmup, params.atr_median_window + params.atr_period + 2)
        tail = max(tail, session_bars * 2, htf_bars_needed,
                   params.atr_median_window + params.atr_period + 6)
    for i in range(warmup, len(data)):
        window = data.iloc[max(0, i - tail + 1): i + 1]
        bar = data.iloc[i]
        # True from the first bar whose wall-clock time is at/after the
        # session's square-off cutoff. Resets naturally on the next day's
        # first bar since it's read straight off the bar's own timestamp —
        # no day-rollover bookkeeping needed, unlike the live poller.
        past_cutoff = (square_off_cutoff is not None
                       and bar.name.time() >= square_off_cutoff)

        # manage an open position first
        if position is not None:
            st = position["exit_state"]
            # ATR from the PREVIOUS bar, never this one. Live, the trail reads
            # an ATR computed off completed candles while it manages the price
            # inside the forming bar (strategy_runner._trail_atr); reading
            # bar i's own ATR here would let the trail see the very high/low
            # it is about to be measured against — look-ahead, and the exact
            # kind that flatters a trailing system.
            prev_atr = float(data["atr"].iloc[i - 1]) if i > 0 else 0.0
            if not np.isfinite(prev_atr) or prev_atr <= 0:
                prev_atr = 0.0
            exit_price, exit_reason = None, ""
            # ONE decision function, shared with the live engine. step_bar
            # assumes the ADVERSE intrabar path (the stop is tested at the
            # bar's worst price before any favourable move is credited), so
            # OHLC's hidden path cannot flatter a trail or a scale-out.
            for act in exit_manager.step_bar(
                    st, float(bar["open"]), float(bar["high"]),
                    float(bar["low"]), float(bar["close"]),
                    prev_atr, exit_params):
                if act.kind == exit_manager.ActionType.PARTIAL:
                    # A partial is a REAL exit leg with its OWN row, mirroring
                    # the child trade document the live engine writes, so
                    # `quantity` reads correctly everywhere and the leg is
                    # charged its own costs (see _leg below).
                    capital += _leg(position, act.qty, float(act.price),
                                    act.reason or "PARTIAL-TARGET", bar.name)
                    position["qty"] -= int(act.qty)
                elif act.kind == exit_manager.ActionType.MOVE_STOP:
                    position["stop"] = float(act.new_stop)
                    position["target"] = float(st.target)
                else:
                    exit_price, exit_reason = float(act.price), act.reason
            # Time exit (Scalper): bars_held is exact because bars are fixed-width.
            if exit_price is None and params.max_hold_minutes > 0:
                held_bars = i - position["bar"]
                if held_bars >= params.max_hold_minutes:   # 1 bar == 1 minute
                    exit_price = float(bar["close"])
                    exit_reason = f"TIME-EXIT ({params.max_hold_minutes}m)"
            # End-of-session square-off. Checked LAST, same order as the live
            # runner (own SL/TP/time-exit first, square-off after) — it only
            # fires when nothing else already closed the position this bar.
            # Unconditional on P&L: a time stop, not a decision.
            if exit_price is None and past_cutoff:
                exit_price = float(bar["close"])
                exit_reason = f"SQUARE-OFF ({square_off_cutoff.strftime('%H:%M')})"
            if exit_price is not None:
                capital += _leg(position, position["qty"], exit_price,
                                exit_reason, bar.name)
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
        # `not past_cutoff` mirrors the live runner: past the square-off time
        # nothing new may be opened, whatever the setup looks like.
        bar_allowed = ((filters is None or filters.allows_bar(bar.name))
                       and not past_cutoff
                       and (entry_cutoff is None
                            or bar.name.time() < entry_cutoff))
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
                        "id": len(trades) + 1, "leg": 0,
                        # Seeded from the SIGNAL's own levels, so risk_dist is
                        # the distance the position was sized against and the
                        # runner target the exit manager derives from it is
                        # the same one the live engine would derive.
                        "exit_state": exit_manager.ExitState(
                            side=sig.side, entry=float(sig.entry_price),
                            qty=int(qty), stop=float(sig.stop_loss),
                            target=float(sig.target),
                            risk_dist=abs(float(sig.entry_price)
                                          - float(sig.stop_loss))),
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
    timeframe_minutes: int = 0,
    exit_style: str = "strategy",
    trail_atr_mult: float = 0.0,
    partial_exit_fraction: float = -1.0,
    runner_rr_mult: float = -1.0,
    max_stop_pct: float = 0.0,
    min_stop_pct: float = 0.0,
    #: Remove the end-of-session flat-out so positions run to their own
    #: stop/target/trail across sessions. Forwarded verbatim to run_backtest,
    #: where the caveats are documented. False = unchanged behaviour.
    hold_overnight: bool = False,
    ignore_entry_cutoff: bool = False,
    #: Deduct costs from the compounding capital, exactly as run_backtest
    #: does. FALSE by default so no existing caller silently flips from gross
    #: to net; the API routes pass True, which is what makes a bulk ranking
    #: comparable with the single-symbol tab beside it.
    include_costs: bool = False,
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
                               patterns=patterns,
                               timeframe_minutes=timeframe_minutes,
                               exit_style=exit_style,
                               trail_atr_mult=trail_atr_mult,
                               partial_exit_fraction=partial_exit_fraction,
                               runner_rr_mult=runner_rr_mult,
                               max_stop_pct=max_stop_pct,
                               min_stop_pct=min_stop_pct,
                               hold_overnight=hold_overnight,
                               ignore_entry_cutoff=ignore_entry_cutoff,
                               include_costs=include_costs)
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
    timeframe_minutes: int = 0,
    exit_style: str = "strategy",
    trail_atr_mult: float = 0.0,
    partial_exit_fraction: float = -1.0,
    runner_rr_mult: float = -1.0,
    max_stop_pct: float = 0.0,
    min_stop_pct: float = 0.0,
    #: Remove the end-of-session flat-out so positions run to their own
    #: stop/target/trail across sessions. Forwarded verbatim to run_backtest,
    #: where the caveats are documented. False = unchanged behaviour.
    hold_overnight: bool = False,
    ignore_entry_cutoff: bool = False,
    #: Deduct costs from the compounding capital, exactly as run_backtest
    #: does. FALSE by default so no existing caller silently flips from gross
    #: to net; the API routes pass True, which is what makes a bulk ranking
    #: comparable with the single-symbol tab beside it.
    include_costs: bool = False,
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
    if mode == Mode.INTRADAY and timeframe_minutes and timeframe_minutes > 0:
        interval = f"{int(timeframe_minutes)}m"
    else:
        interval = {Mode.SWING: "1d", Mode.INTRADAY: "15m", Mode.SCALPER: "1m"}[mode]

    def _run_one(ticker: str) -> tuple[str, BacktestResult]:
        inst = config.INSTRUMENTS_BY_SYMBOL.get(ticker)
        lot_size = inst.lot_size if inst else 1
        try:
            return ticker, run_backtest(
                ticker, start, end, initial_capital, mode,
                lot_size=lot_size, strategy_key=strategy_key,
                risk_reward=risk_reward, min_score=min_score,
                filters=filters, patterns=patterns,
                timeframe_minutes=timeframe_minutes,
                exit_style=exit_style, trail_atr_mult=trail_atr_mult,
                partial_exit_fraction=partial_exit_fraction,
                runner_rr_mult=runner_rr_mult,
                max_stop_pct=max_stop_pct, min_stop_pct=min_stop_pct,
                hold_overnight=hold_overnight,
                ignore_entry_cutoff=ignore_entry_cutoff,
                include_costs=include_costs)
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
    periods_per_year = _bars_per_year(interval)
    if rets.std() > 0:
        sharpe = (rets.mean() / rets.std()) * np.sqrt(periods_per_year)
    else:
        sharpe = 0.0

    years = max(len(equity) / periods_per_year, 1e-9)
    cagr = (equity.iloc[-1] / initial_capital) ** (1 / years) - 1
    calmar = (cagr / abs(max_dd / 100)) if max_dd != 0 else 0.0

    win_rate = (100 * trades["win"].mean()) if not trades.empty else 0.0

    out = {
        "Total Return %": round(total_return, 2),
        "Max Drawdown %": round(max_dd, 2),
        "Sharpe": round(float(sharpe), 2),
        "Calmar": round(float(calmar), 2),
        "Win Rate %": round(float(win_rate), 2),
        "Total Trades": int(len(trades)),
        "Final Equity": round(float(equity.iloc[-1]), 2),
        "Data Source": source,
    }
    # Cost-aware run (run_backtest(include_costs=True)). Everything above is
    # already NET, because costs were taken out of the compounding capital the
    # equity curve is built from. What is added here is the gross counterpart,
    # so the two are readable side by side and the size of the drag is explicit
    # rather than inferred.
    if not trades.empty and "cost" in trades.columns:
        total_cost = float(trades["cost"].sum())
        gross_pnl = float(trades["pnl"].sum())
        n = len(trades)
        out["Costs Applied"] = True
        out["Total Costs"] = round(total_cost, 2)
        out["Cost per Trade"] = round(total_cost / n, 2) if n else 0.0
        out["Gross P&L"] = round(gross_pnl, 2)
        out["Gross Return %"] = round(100.0 * gross_pnl / initial_capital, 2)
        out["Gross Win Rate %"] = round(100.0 * float((trades["pnl"] > 0).mean()), 2)
    else:
        out["Costs Applied"] = False
    # A scaled-out trade closes in TWO legs and therefore writes TWO rows (the
    # live engine writes two documents for the same reason), so "Total Trades"
    # above counts EXIT LEGS. That is the right denominator for cost-per-trade
    # and for win rate — a partial really is a separately-won or separately-
    # lost booking — but it is not the number of positions the strategy took,
    # so both are reported. They are equal for every strategy that does not
    # scale out, which is all of them at their defaults.
    if not trades.empty and "position_id" in trades.columns:
        out["Total Positions"] = int(trades["position_id"].nunique())
        out["Exit Legs"] = int(len(trades))
    return out
