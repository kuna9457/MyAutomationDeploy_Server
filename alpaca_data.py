"""
alpaca_data.py
US equity market data from Alpaca — historical bars for the backtester.

WHY REST AND NOT THE SDK
------------------------
This talks to Alpaca's Market Data v2 HTTP API with `requests`, which the
project already depends on, instead of pulling in `alpaca-py`. Two reasons, and
both are about not making you configure things:

  * Nothing new to install. `pip install` stays exactly as it is, and a server
    that can already run this bot can already reach Alpaca.
  * The SDK's model classes churn between majors. The bars endpoint has been
    stable for years; a 40-line client against it will outlive several SDK
    upgrades.

ZERO-CONFIG BEHAVIOUR
---------------------
Two environment variables and nothing else:

    ALPACA_API_KEY / ALPACA_API_SECRET

Alpaca's own tooling names them APCA_API_KEY_ID / APCA_API_SECRET_KEY, so both
spellings are accepted — paste whichever your dashboard hands you. With NO keys
at all the module simply reports `is_configured() == False` and the backtester
falls through to yfinance, which needs no credentials and is enough for daily
bars. So US backtesting works before you sign up for anything, and works better
once you do.

    ALPACA_DATA_FEED   optional; "iex" (default, free) or "sip" (paid, full
                       market). IEX is a single venue — roughly 2-3% of
                       consolidated volume — so its bars are thinner and its
                       volume figures are NOT comparable to SIP or to a chart
                       on any retail site. Fine for shape and for pattern
                       geometry; do not read anything into IEX volume levels.

TIME ZONE — THE PART THAT MATTERS
---------------------------------
Alpaca returns RFC-3339 UTC timestamps. This module converts them to NAIVE
America/New_York wall-clock time before returning, because that is the
convention config.now_for_segment(US_EQUITY) establishes and everything
downstream assumes: with bars in exchange-local time a US session sits inside
one calendar day, so MarketHours.is_open(), the square-off comparison and
strategy.vwap()'s per-day reset all work unmodified. Bars in UTC or IST would
break all three. See the US_TZ note in config.py.

Only REGULAR-HOURS bars are returned (09:30-16:00 ET). Alpaca will happily
serve pre- and post-market prints; those trade on thin books at prices the
strategy's ATR was never calibrated against, and the live bot could not act on
them anyway.
"""
from __future__ import annotations

import os
from datetime import datetime, time as dtime
from typing import Optional
from zoneinfo import ZoneInfo

import pandas as pd
import requests

_BARS_URL = "https://data.alpaca.markets/v2/stocks/bars"
_QUOTES_URL = "https://data.alpaca.markets/v2/stocks/quotes/latest"
_TRADES_URL = "https://data.alpaca.markets/v2/stocks/trades/latest"
# TRADING api (assets, and later orders) — this one is NOT shared between the
# two environments. Paper keys authenticate ONLY against paper-api, live keys
# ONLY against api. The three data URLs above are common to both, which is why
# market data works with either set of keys and this one does not.
#
# Paper is the default because it is where everyone starts and it is the safe
# wrong-guess: a paper key against the live host fails cleanly with 401, while
# a live key against paper would silently address the wrong ACCOUNT.
_PAPER_TRADING_BASE = "https://paper-api.alpaca.markets"
_LIVE_TRADING_BASE = "https://api.alpaca.markets"

US_TZ = ZoneInfo("America/New_York")

#: Regular session. Bars outside it are dropped — see the module docstring.
_SESSION_OPEN, _SESSION_CLOSE = dtime(9, 30), dtime(16, 0)

#: Alpaca caps a single bars response; the API pages with `next_page_token`.
_PAGE_LIMIT = 10_000

#: Guard against a pathological loop if the API ever returns a repeating token.
_MAX_PAGES = 200

_TIMEOUT = 30

#: Our interval strings -> Alpaca timeframes. Anything not listed is rejected
#: rather than guessed, so a typo fails loudly instead of silently fetching the
#: wrong bar size and producing a plausible, wrong backtest.
_TIMEFRAME = {
    "1m": "1Min", "2m": "2Min", "3m": "3Min", "5m": "5Min",
    "10m": "10Min", "15m": "15Min", "30m": "30Min",
    "1h": "1Hour", "1d": "1Day",
}


def _key() -> str:
    return (os.getenv("ALPACA_API_KEY") or os.getenv("APCA_API_KEY_ID") or "").strip()


def _secret() -> str:
    return (os.getenv("ALPACA_API_SECRET")
            or os.getenv("APCA_API_SECRET_KEY") or "").strip()


def is_paper() -> bool:
    """True unless ALPACA_LIVE is explicitly turned on. Defaults to paper so a
    misconfiguration can never point order-shaped calls at real money."""
    return (os.getenv("ALPACA_LIVE") or "").strip().lower() not in (
        "1", "true", "yes", "on")


def trading_base() -> str:
    """Base URL for the TRADING api, matching whichever key set is in use."""
    return _PAPER_TRADING_BASE if is_paper() else _LIVE_TRADING_BASE


def data_feed_name() -> str:
    """"iex" (free) or "sip" (paid). Defaults to the one everybody has."""
    return (os.getenv("ALPACA_DATA_FEED") or "iex").strip().lower()


def is_configured() -> bool:
    """True when both credentials are present. Callers treat False as 'skip me'
    and fall through to their next data source — never as an error."""
    return bool(_key() and _secret())


def _headers() -> dict:
    return {"APCA-API-KEY-ID": _key(), "APCA-API-SECRET-KEY": _secret()}


def supported_interval(interval: str) -> bool:
    return interval in _TIMEFRAME


def fetch_bars(symbol: str, start: str, end: str,
               interval: str = "1d") -> pd.DataFrame:
    """OHLCV bars for one US symbol, indexed by NAIVE New York local time.

    `start`/`end` are plain YYYY-MM-DD dates; `end` is INCLUSIVE, matching the
    backtester's own convention (a date, not an instant, so excluding it would
    silently drop the final session).

    Returns an EMPTY frame — never raises — when the symbol has no data, the
    interval is unsupported, or credentials are missing. The backtester treats
    an empty frame as "try the next source", which is what makes Alpaca an
    optional accelerator rather than a hard dependency.
    """
    if not is_configured() or not supported_interval(interval):
        return pd.DataFrame()

    params = {
        "symbols": symbol,
        "timeframe": _TIMEFRAME[interval],
        "start": start,
        # +1 day because Alpaca's `end` is an exclusive instant while ours is an
        # inclusive date. Without this every backtest loses its last session.
        "end": (pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        "limit": _PAGE_LIMIT,
        "adjustment": "all",   # split- AND dividend-adjusted, so a split does
                               # not read as a 50% overnight crash the way the
                               # unadjusted Upstox history does.
        "feed": data_feed_name(),
        "sort": "asc",
    }
    rows: list[dict] = []
    token: Optional[str] = None
    try:
        for _ in range(_MAX_PAGES):
            if token:
                params["page_token"] = token
            r = requests.get(_BARS_URL, params=params, headers=_headers(),
                             timeout=_TIMEOUT)
            if r.status_code != 200:
                print(f"[alpaca] {symbol}: HTTP {r.status_code} — "
                      f"{r.text[:160]}")
                return pd.DataFrame()
            body = r.json()
            rows.extend(body.get("bars", {}).get(symbol, []) or [])
            token = body.get("next_page_token")
            if not token:
                break
    except Exception as exc:
        print(f"[alpaca] {symbol}: bars request failed ({exc}).")
        return pd.DataFrame()

    if not rows:
        return pd.DataFrame()
    return _to_frame(rows, interval)


def _to_frame(rows: list[dict], interval: str) -> pd.DataFrame:
    """Alpaca bar dicts -> the OHLCV frame the rest of the system expects."""
    df = pd.DataFrame(rows)
    # t=timestamp o/h/l/c/v = open/high/low/close/volume
    df = df.rename(columns={"t": "ts", "o": "open", "h": "high", "l": "low",
                            "c": "close", "v": "volume"})
    # UTC -> New York -> drop the tz, so the index is naive exchange-local time
    # exactly like the IST frames the Indian path produces.
    df["ts"] = (pd.to_datetime(df["ts"], utc=True)
                  .dt.tz_convert(US_TZ).dt.tz_localize(None))
    df = df.set_index("ts").sort_index()
    df = df[["open", "high", "low", "close", "volume"]].astype(float)

    # Regular hours only. A daily bar is stamped 00:00, so the session filter
    # must not be applied to it — it would drop every row.
    if interval != "1d":
        t = df.index.time
        df = df[(t >= _SESSION_OPEN) & (t < _SESSION_CLOSE)]
    return df[~df.index.duplicated(keep="last")]


def list_tradable_symbols(limit: int = 0) -> list[str]:
    """Every active, tradable US equity Alpaca knows about.

    Used by tools/refresh_us.py to build the instrument list, so the universe
    is discovered rather than hand-maintained. Returns [] on any failure —
    the caller falls back to the checked-in list.
    """
    if not is_configured():
        return []
    # Try the environment we think we are in, then the other one. Which host a
    # key belongs to is not knowable from the key itself, so asking is cheaper
    # than making the user tell us — and this is the only place it matters.
    bases = [trading_base()]
    other = _LIVE_TRADING_BASE if is_paper() else _PAPER_TRADING_BASE
    bases.append(other)
    r = None
    try:
        for base in bases:
            r = requests.get(f"{base}/v2/assets",
                             params={"status": "active",
                                     "asset_class": "us_equity"},
                             headers=_headers(), timeout=_TIMEOUT)
            if r.status_code == 200:
                break
            print(f"[alpaca] assets via {base}: HTTP {r.status_code}")
        if r is None or r.status_code != 200:
            print("[alpaca] assets: neither paper nor live host accepted these "
                  "keys — check ALPACA_API_KEY / ALPACA_API_SECRET.")
            return []
        out = [a["symbol"] for a in r.json()
               if a.get("tradable") and a.get("exchange") in ("NYSE", "NASDAQ",
                                                              "ARCA", "AMEX")]
        out.sort()
        return out[:limit] if limit else out
    except Exception as exc:
        print(f"[alpaca] assets request failed ({exc}).")
        return []


# --------------------------------------------------------------------------- #
#  Live quotes (used by the AlpacaRestFeed, not by the backtester)
# --------------------------------------------------------------------------- #
def fetch_latest_quotes(symbols: list[str]) -> dict[str, dict]:
    """{symbol: {"bid", "ask", "last", "ts"}} for many symbols in ONE request.

    Both endpoints are bulk, which matters: polling 100 names one at a time
    would burn the rate limit for no reason. `last` falls back to the mid when
    no trade has printed yet (common right at the open, and common all day on
    the free IEX feed for thin names).

    Returns {} on any failure — the feed treats a missing quote as "no tick
    yet", which it already handles.
    """
    if not is_configured() or not symbols:
        return {}
    joined = ",".join(sorted(set(symbols)))
    params = {"symbols": joined, "feed": data_feed_name()}
    out: dict[str, dict] = {}
    try:
        r = requests.get(_QUOTES_URL, params=params, headers=_headers(),
                         timeout=_TIMEOUT)
        if r.status_code == 200:
            for sym, q in (r.json().get("quotes") or {}).items():
                bid, ask = float(q.get("bp") or 0.0), float(q.get("ap") or 0.0)
                mid = (bid + ask) / 2.0 if bid > 0 and ask > 0 else (bid or ask)
                out[sym] = {"bid": bid, "ask": ask, "last": mid,
                            "ts": _parse_ts(q.get("t"))}
        else:
            print(f"[alpaca] quotes: HTTP {r.status_code} — {r.text[:140]}")
    except Exception as exc:
        print(f"[alpaca] quotes request failed ({exc}).")
        return {}

    # Overlay the real last-traded price where one exists; a mid is a decent
    # stand-in but it is not a print, and PnL should follow prints.
    try:
        r = requests.get(_TRADES_URL, params=params, headers=_headers(),
                         timeout=_TIMEOUT)
        if r.status_code == 200:
            for sym, t in (r.json().get("trades") or {}).items():
                px = float(t.get("p") or 0.0)
                if px > 0:
                    row = out.setdefault(sym, {"bid": 0.0, "ask": 0.0})
                    row["last"] = px
                    row["ts"] = _parse_ts(t.get("t")) or row.get("ts")
    except Exception:
        pass          # quotes alone are enough to price a position
    return out


def _parse_ts(value) -> Optional[datetime]:
    """Alpaca RFC-3339 UTC -> naive New York local, matching fetch_bars()."""
    if not value:
        return None
    try:
        ts = pd.Timestamp(value)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        return ts.tz_convert(US_TZ).tz_localize(None).to_pydatetime()
    except Exception:
        return None
