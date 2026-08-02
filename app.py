"""
app.py
Streamlit front-end. Sidebar for global controls, three tabs for the main views.

Run:  streamlit run app.py

The engine runs in a background thread, so the UI is a thin control + display
layer. It reads a thread-safe snapshot each rerun and never touches broker or
strategy logic directly (keeping the separation from Immutable Rule #3).
"""
from __future__ import annotations

import re
from datetime import date, timedelta

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import config
import risk_manager
from config import (Broker, Environment, Mode, Segment, ALL_INSTRUMENTS,
                    instruments_for_segment)
from db_manager import DBManager
from engine import TradingEngine
import backtester
import backtest_reports
import broker_api
import kite_auth
import strategy
import upstox_auth
import watchlists

st.set_page_config(page_title="Algo Trading Bot", page_icon="📈", layout="wide")

# --------------------------------------------------------------------------- #
#  Session state
# --------------------------------------------------------------------------- #
if "engine" not in st.session_state:
    st.session_state.engine = None
if "db" not in st.session_state:
    st.session_state.db = DBManager()

db: DBManager = st.session_state.db


# --------------------------------------------------------------------------- #
#  Upstox token refresh (OAuth). Live tokens expire daily (~03:30 IST), so the
#  UI lets you re-auth in-place: log in at Upstox, come back, token is saved to
#  .env and activated live (no restart). Works two ways —
#    • Auto-capture: if your Upstox app's redirect URI points at THIS Streamlit
#      URL, login redirects back here with ?code=... and we exchange it instantly.
#    • Manual paste: land on any registered redirect URI, copy the URL, paste it.
# --------------------------------------------------------------------------- #
def refresh_token_from_code(code: str) -> tuple[bool, str]:
    api_key, api_secret, redirect_uri = upstox_auth.get_credentials()
    if not (api_key and api_secret):
        return False, "UPSTOX_LIVE_API_KEY / UPSTOX_LIVE_SECRET missing in .env."
    res = upstox_auth.exchange_code(code, api_key, api_secret, redirect_uri)
    if not res["ok"]:
        return False, res["error"]
    try:
        upstox_auth.save_token(res["token"])
    except Exception:
        return False, ("Token fetched but .env couldn't be written "
                       "(python-dotenv missing). Copy it manually.")
    config.reload_tokens()   # activate the new token in this running process
    who = res.get("user_name") or res.get("email") or "user"
    return True, f"✅ Token refreshed for {who}. Saved to .env and active now."


# --------------------------------------------------------------------------- #
#  Zerodha (Kite Connect) token refresh. Kite sessions expire daily (~06:00
#  IST), so — like Upstox — this is a normal part of each trading day, not a
#  one-time setup. Same two-path pattern: auto-capture if the registered
#  Redirect URL points back at this app, or manual paste otherwise.
# --------------------------------------------------------------------------- #
def refresh_zerodha_token(request_token: str) -> tuple[bool, str]:
    api_key, api_secret = kite_auth.get_credentials()
    if not (api_key and api_secret):
        return False, "ZERODHA_API_KEY / ZERODHA_API_SECRET missing in .env."
    res = kite_auth.exchange_request_token(request_token, api_key, api_secret)
    if not res["ok"]:
        return False, res["error"]
    try:
        kite_auth.save_token(res["token"])
    except Exception:
        return False, ("Token fetched but .env couldn't be written "
                       "(python-dotenv missing). Copy it manually.")
    config.reload_tokens()
    who = res.get("user_name") or res.get("email") or "user"
    return True, f"✅ Zerodha token refreshed for {who}. Saved to .env and active now."


# Auto-capture: handle a redirect that landed back on this app with ?code=...
# (Upstox) or ?request_token=... (Zerodha).
_code = st.query_params.get("code")
if _code and st.session_state.get("_last_code") != _code:
    st.session_state["_last_code"] = _code
    ok, msg = refresh_token_from_code(_code)
    st.session_state["_token_msg"] = ("success" if ok else "error", msg)
    try:
        del st.query_params["code"]        # codes are single-use; drop from URL
    except Exception:
        st.query_params.clear()

_req_tok = st.query_params.get("request_token")
if _req_tok and st.session_state.get("_last_req_tok") != _req_tok:
    st.session_state["_last_req_tok"] = _req_tok
    ok, msg = refresh_zerodha_token(_req_tok)
    st.session_state["_zerodha_token_msg"] = ("success" if ok else "error", msg)
    try:
        del st.query_params["request_token"]   # single-use; drop from URL
    except Exception:
        st.query_params.clear()


# --------------------------------------------------------------------------- #
#  Sidebar — global controls
# --------------------------------------------------------------------------- #
st.sidebar.title("⚙️ Controls")

env_label = st.sidebar.radio(
    "Environment", ["Paper Trading (Sandbox)", "Live Trading"], index=0)
environment = Environment.LIVE if env_label.startswith("Live") else Environment.PAPER

MODE_LABELS = {
    "Intraday (15m)": Mode.INTRADAY,
    "Swing (Daily)": Mode.SWING,
    "⚡ Aggressive Scalper (1m)": Mode.SCALPER,
}
mode_label = st.sidebar.radio("Trading Mode", list(MODE_LABELS), index=0)
mode = MODE_LABELS[mode_label]

# --- Strategy picker -------------------------------------------------------- #
# The mode fixes the timeframe; the strategy is chosen separately, so one mode can
# host several. The list is built from the registry, so a newly registered
# strategy appears here with no change to this file.
_choices = strategy.strategies_for_mode(mode)
_phase_labels = [f"Phase {i + 1}" for i in range(len(_choices))]
_default_key = strategy.default_strategy(mode).key
_default_ix = next((i for i, s in enumerate(_choices) if s.key == _default_key), 0)
_picked_label = st.sidebar.selectbox("Strategy", _phase_labels, index=_default_ix,
                                     key="live_strategy")
selected_strategy = _choices[_phase_labels.index(_picked_label)]

_p = selected_strategy.params
st.sidebar.caption(f"_{_picked_label} parameters_")
st.sidebar.caption(
    f"**{_p.timeframe}** · risk **{_p.risk_per_trade:.0%}**/trade · "
    f"RR **1:{_p.risk_reward:g}** · SL **{_p.atr_sl_mult:g}×ATR({_p.atr_period})**"
    + (f" · {'long+short' if _p.allow_short else 'long only'}")
    + (f" · exit after {_p.max_hold_minutes}m" if _p.max_hold_minutes else "")
)

broker_choice = Broker.SIMULATED
if environment == Environment.LIVE:
    broker_name = st.sidebar.selectbox(
        "Live Broker", ["Upstox", "Dhan", "Zerodha", "Kotak Neo"])
    broker_choice = {"Upstox": Broker.UPSTOX, "Dhan": Broker.DHAN,
                     "Zerodha": Broker.ZERODHA,
                     "Kotak Neo": Broker.KOTAK}[broker_name]
    st.sidebar.warning("⚠️ Live mode places REAL orders when broker "
                       "credentials are present in .env.")

st.sidebar.markdown("### Universe")

# Human-readable segment labels <-> the Segment enum, used both by the Segments
# picker and by watchlist loading (a watchlist auto-enables the segments its
# instruments belong to).
_SEG_LABELS = {Segment.EQUITY: "NSE Equity", Segment.MCX: "MCX Commodity"}
_ALL_SEG_LABELS = ["NSE Equity", "MCX Commodity"]
_WL_NONE = "— None —"


def _apply_live_watchlist() -> None:
    """Callback: load the picked watchlist into the sidebar's Segments +
    Instruments pickers. Runs BEFORE the rerun, which is the only point Streamlit
    allows a widget's value to be set programmatically."""
    name = st.session_state.get("live_watchlist_pick", _WL_NONE)
    if not name or name == _WL_NONE:
        return
    syms = [s for s in watchlists.get(name) if s in config.INSTRUMENTS_BY_SYMBOL]
    # Turn on whatever segments these instruments need, so none get filtered out.
    needed = {_SEG_LABELS.get(config.INSTRUMENTS_BY_SYMBOL[s].segment) for s in syms}
    needed.discard(None)
    cur = set(st.session_state.get("live_segments", []))
    st.session_state["live_segments"] = [s for s in _ALL_SEG_LABELS
                                         if s in (cur | needed)]
    st.session_state["live_instruments"] = syms


def _delete_live_watchlist() -> None:
    """Callback: delete the currently loaded watchlist and reset the picker. Must
    run as a callback (before the rerun) — Streamlit forbids setting a widget's
    key value once the widget has been instantiated in the same run."""
    pick = st.session_state.get("live_watchlist_pick", _WL_NONE)
    if pick and pick != _WL_NONE and watchlists.delete(pick):
        st.session_state["live_watchlist_pick"] = _WL_NONE
        st.session_state["_wl_flash"] = f"Deleted '{pick}'."


# One-time init of the keyed pickers (see _apply_live_watchlist for why they are
# keyed rather than using `default=`).
if "live_segments" not in st.session_state:
    st.session_state["live_segments"] = list(_ALL_SEG_LABELS)

segments = st.sidebar.multiselect(
    "Segments", _ALL_SEG_LABELS, key="live_segments")

universe: list = []
if "NSE Equity" in segments:
    universe += instruments_for_segment(Segment.EQUITY)
if "MCX Commodity" in segments:
    universe += instruments_for_segment(Segment.MCX)

# The equity universe is the full Nifty 100 (~114 names). Selecting ALL of them by
# default would subscribe 100+ live WebSocket feeds the moment the bot starts, so
# the default is capped to a manageable slice — every instrument is still
# selectable, and a saved Watchlist loads an exact bucket in one click.
_DEFAULT_UNIVERSE_CAP = 15
_universe_syms = [i.symbol for i in universe]
if "live_instruments" not in st.session_state:
    st.session_state["live_instruments"] = _universe_syms[:_DEFAULT_UNIVERSE_CAP]
# Drop any selected symbol whose segment is currently unticked, BEFORE the widget
# is created — a keyed multiselect raises if its stored value isn't in the options.
st.session_state["live_instruments"] = [
    s for s in st.session_state["live_instruments"] if s in _universe_syms]

symbols = st.sidebar.multiselect(
    "Instruments", _universe_syms, key="live_instruments",
    help=f"{len(_universe_syms)} instruments available (Nifty 100 + MCX). "
         "Add or remove as you like, or load a saved Watchlist below.")
selected = [i for i in universe if i.symbol in symbols]

# --- Watchlists: save the current bucket under a name, reload it in one click - #
with st.sidebar.expander("📋 Watchlists", expanded=False):
    _wl_names = watchlists.names()
    st.selectbox(
        "Load a watchlist", [_WL_NONE] + _wl_names,
        key="live_watchlist_pick", on_change=_apply_live_watchlist,
        help="Loads its instruments into the Instruments picker above and "
             "enables the segments they need. The same watchlists appear in the "
             "Backtesting tab's bulk test.")
    st.caption(f"**{len(symbols)}** instrument(s) selected right now.")
    _wl_new = st.text_input("Save current selection as", key="wl_new_name",
                            placeholder="e.g. My Momentum Stocks")
    _cs, _cd = st.columns(2)
    if _cs.button("💾 Save", use_container_width=True, key="wl_save"):
        if watchlists.save(_wl_new, symbols):
            st.success(f"Saved '{_wl_new.strip()}' ({len(symbols)} symbols).")
            st.rerun()
        else:
            st.error("Enter a name and select at least one instrument first.")
    _pick = st.session_state.get("live_watchlist_pick", _WL_NONE)
    _cd.button("🗑️ Delete", use_container_width=True, key="wl_delete",
               on_click=_delete_live_watchlist,
               disabled=(_pick == _WL_NONE or _pick not in _wl_names))
    _flash = st.session_state.pop("_wl_flash", "")
    if _flash:
        st.success(_flash)

capital = st.sidebar.number_input(
    "Total Capital (₹)", min_value=10_000.0, value=float(config.TOTAL_CAPITAL),
    step=10_000.0)

# --------------------------------------------------------------------------- #
#  LIVE-only risk guardrails (risk_manager.py). Dynamic on purpose: these are
#  meant to change day to day as real-money risk appetite changes, so they are
#  read/written independently of engine start() — editing them here updates a
#  RUNNING bot on its very next tick, no restart needed. Paper/backtest never
#  see this panel and are completely unaffected by it.
# --------------------------------------------------------------------------- #
if environment == Environment.LIVE:
    _rl = risk_manager.get_limits()
    with st.sidebar.expander("🛡️ Live Risk Controls", expanded=True):
        st.caption(
            "Extra restrictions on top of the strategy's own risk/RR rules — "
            "these only ever make Live trading STRICTER, never looser. Saved "
            "immediately and picked up by a running bot on its next tick.")
        _cap_alloc = st.number_input(
            "Capital allocated for trading today (₹)", min_value=0.0,
            value=float(_rl.capital_allocated), step=5_000.0,
            help="The bot will size and cap trades against min(Total Capital "
                 "above, this figure) — never your whole broker balance. "
                 "0 = no extra cap; Total Capital above is used as-is.")
        _c1, _c2 = st.columns(2)
        _loss_cash = _c1.number_input(
            "Max daily loss (₹)", min_value=0.0,
            value=float(_rl.max_daily_loss_cash), step=500.0,
            help="Kill switch: once today's REALIZED loss reaches this, no "
                 "new trades open for the rest of the day. Open positions "
                 "still exit normally on SL/TP/time. 0 = disabled.")
        _loss_pct = _c2.number_input(
            "...or max daily loss (%)", min_value=0.0, max_value=100.0,
            value=float(_rl.max_daily_loss_pct), step=0.5,
            help="Same kill switch as % of the allocated capital above. "
                 "Whichever of the two (₹ / %) is set AND stricter wins.")
        _c3, _c4 = st.columns(2)
        _max_trades = _c3.number_input(
            "Max trades / day", min_value=0, value=int(_rl.max_trades_per_day),
            step=1, help="0 = unlimited.")
        _max_qty = _c4.number_input(
            "Max qty / trade", min_value=0, value=int(_rl.max_qty_per_trade),
            step=1, help="Hard cap on shares/lots in any single order. "
                        "0 = unlimited (risk sizing still bounds it).")
        _lev = st.number_input(
            "Intraday equity leverage (x)", min_value=1.0, max_value=50.0,
            value=float(_rl.intraday_leverage), step=0.5,
            help="Real MIS leverage your broker actually offers for equity "
                 "Intraday/Scalper — set this to match your broker's product, "
                 "not a wish. Swing is never leveraged (overnight = delivery). "
                 "The bot still checks the broker's REAL available funds "
                 "before every trade regardless of this number.")
        if st.button("💾 Save Risk Limits", use_container_width=True,
                     type="primary"):
            risk_manager.set_limits(
                capital_allocated=float(_cap_alloc),
                max_daily_loss_cash=float(_loss_cash),
                max_daily_loss_pct=float(_loss_pct),
                max_trades_per_day=int(_max_trades),
                max_qty_per_trade=int(_max_qty),
                intraday_leverage=float(_lev),
            )
            st.success("Saved — a running bot picks this up on its next tick.")

        _eng_for_status = st.session_state.get("engine")
        if (_eng_for_status and _eng_for_status.state.running
                and _eng_for_status.environment == Environment.LIVE):
            rs = _eng_for_status.state.snapshot().get("risk_status", {})
            if rs:
                st.markdown("---")
                st.caption("**Today, live:**")
                st.caption(
                    f"Trades: {rs.get('trades_today', 0)}"
                    + (f" / {rs.get('max_trades_per_day')}"
                       if rs.get('max_trades_per_day') else "")
                    + f"  ·  Realized PnL: ₹{rs.get('realized_pnl', 0):,.2f}"
                    + (f"  ·  Loss limit: ₹{rs.get('daily_loss_limit'):,.2f}"
                       if rs.get('daily_loss_limit') else ""))
                if rs.get("halted"):
                    st.error(f"🛑 New entries halted: {rs.get('halt_reason')}")

# --------------------------------------------------------------------------- #
#  MCX commodity settings — FIXED lots per symbol (separate from equity).
#
#  Commodities are NOT risk-sized like equity. Here the user pre-selects how many
#  lots to trade per symbol when a signal fires; the strategy still decides the
#  SL/TP price levels, and the margin follows automatically from the lots. The
#  estimate below uses each contract's reference price and effective leverage so
#  the user sees the funding requirement before starting the bot.
# --------------------------------------------------------------------------- #
mcx_selected = [i for i in selected if i.segment == Segment.MCX]
mcx_lots: dict[str, int] = {}


@st.cache_data(ttl=300, show_spinner=False)
def _mcx_margin_preview(instrument_key: str, symbol: str, lots: int,
                        ref_price: float, mult: int,
                        token: str, is_paper: bool) -> tuple[float, str]:
    """Margin (₹) for the sidebar preview, cached for 5 min so the auto-refreshing
    UI doesn't hammer the margin API. Mirrors engine._mcx_margin exactly:
      PAPER: hardcoded per-lot table (config.MCX_MARGIN_PER_LOT) → notional
      LIVE:  live Upstox fetch → hardcoded table → notional÷leverage
    `lots` is the LOT COUNT (Upstox counts MCX quantity in lots), so margin is
    per_lot × lots. `token`/`is_paper` are in the cache key so switching either
    re-computes."""
    if lots <= 0:
        return 0.0, "none"
    # PAPER never calls the live API — it uses the hardcoded broker-side figures.
    if not is_paper:
        m = broker_api.fetch_upstox_margin(token, instrument_key, lots, "BUY", "D")
        if m and m > 0:
            return float(m), "live"
    per_lot = config.mcx_margin_per_lot(symbol)
    if per_lot > 0:
        return per_lot * lots, "hardcoded"
    lev = config.SEGMENT_MAX_LEVERAGE.get(Segment.MCX, 1.0)
    notional = ref_price * lots * mult
    return (notional / lev if lev > 0 else notional), "notional"


if mcx_selected:
    _tok = config.UPSTOX_LIVE_ACCESS_TOKEN or config.UPSTOX_SANDBOX_TOKEN
    _is_paper = environment == Environment.PAPER
    _SRC_LABEL = {"live": "live from broker", "hardcoded": "fixed broker-side rate",
                  "notional": "rough estimate", "none": ""}
    with st.sidebar.expander("🛢️ MCX Commodity Settings", expanded=True):
        st.caption(
            "Commodities trade a **fixed number of lots** you set here — not "
            "risk-based sizing. The strategy still sets SL/TP; margin is a "
            "**fixed per-lot** broker-side figure (₹/lot × lots) in Paper mode, "
            "and the broker's real SPAN+exposure margin in Live mode. "
            "No % cap applies — the only limit is the margin your capital can fund.")
        for inst in mcx_selected:
            lots = st.number_input(
                f"{inst.symbol} — lots per trade", min_value=0, max_value=1000,
                value=1, step=1, key=f"mcx_lots_{inst.symbol}",
                help=f"Lot size {inst.lot_size} · quoted per contract. 0 = don't "
                     f"trade this commodity.")
            mcx_lots[inst.symbol] = int(lots)
            margin, src = _mcx_margin_preview(
                inst.instrument_key, inst.symbol, int(lots),
                inst.reference_price, max(inst.contract_multiplier, 1), _tok,
                _is_paper)
            if int(lots) <= 0:
                st.caption("_0 lots — this commodity won't be traded._")
            else:
                st.caption(f"≈ **₹{margin:,.0f}** margin for {int(lots)} lot(s) "
                           f"· _{_SRC_LABEL.get(src, src)}_")

st.sidebar.markdown("---")
col_a, col_b = st.sidebar.columns(2)
start_clicked = col_a.button("▶️ Start Bot", use_container_width=True,
                             type="primary")
stop_clicked = col_b.button("⏹️ Stop Bot", use_container_width=True)

if start_clicked:
    if not selected:
        st.sidebar.error("Select at least one instrument.")
    else:
        if st.session_state.engine and st.session_state.engine.state.running:
            st.session_state.engine.stop()
        eng = TradingEngine(environment, mode, broker_choice, selected, capital,
                            strategy_key=selected_strategy.key, mcx_lots=mcx_lots)
        eng.start()
        st.session_state.engine = eng
        st.sidebar.success(f"Bot started — {selected_strategy.name}.")

if stop_clicked and st.session_state.engine:
    st.session_state.engine.stop()
    st.sidebar.info("Bot stopped.")

# --------------------------------------------------------------------------- #
#  Sidebar — Broker Status lights. One read-only auth check per broker so you
#  can see, at a glance, which credentials are actually live RIGHT NOW —
#  independent of whether the bot is running. Cached briefly so opening the
#  sidebar doesn't hammer every broker's API on every rerun; the Refresh
#  button forces an immediate re-check (e.g. right after saving a new token).
# --------------------------------------------------------------------------- #
@st.cache_data(ttl=30, show_spinner=False)
def _broker_status_cached(_nonce: int) -> dict:
    return broker_api.check_broker_status()


st.sidebar.markdown("---")
with st.sidebar.expander("🚦 Broker Status", expanded=True):
    if st.button("🔄 Refresh now", use_container_width=True, key="broker_status_refresh"):
        _broker_status_cached.clear()
        st.session_state["_broker_status_nonce"] = (
            st.session_state.get("_broker_status_nonce", 0) + 1)
    _status = _broker_status_cached(st.session_state.get("_broker_status_nonce", 0))
    for _bname, _info in _status.items():
        if _info["ok"] is True:
            _dot = "🟢"
        elif _info["ok"] is False:
            _dot = "🔴"
        else:
            _dot = "⚪"
        _tag = " ← selected" if (environment == Environment.LIVE
                                 and _bname == broker_name) else ""
        st.markdown(f"{_dot} **{_bname}**{_tag}")
        st.caption(_info["detail"])
    st.caption("Read-only checks (profile/funds), cached ~30s. This proves "
               "the CREDENTIAL is live — it does not confirm the currently "
               "RUNNING bot is using it (check the dashboard's Broker field "
               "for that; if it ever says 'Simulated' while you picked a "
               "real broker, the connection failed and silently fell back).")

# --------------------------------------------------------------------------- #
#  Sidebar — Upstox token refresh panel
# --------------------------------------------------------------------------- #
st.sidebar.markdown("---")
_pending_msg = st.session_state.get("_token_msg")
with st.sidebar.expander("🔑 Upstox Token", expanded=bool(_pending_msg)):
    if _pending_msg:
        kind, text = st.session_state.pop("_token_msg")
        (st.success if kind == "success" else st.error)(text)

    _tok = config.UPSTOX_LIVE_ACCESS_TOKEN
    if _tok:
        st.caption(f"Current live token: `{_tok[:8]}…`  ({len(_tok)} chars)")
    else:
        st.caption("No live token set.")

    api_key, api_secret, redirect_uri = upstox_auth.get_credentials()
    if not (api_key and api_secret):
        st.warning("Add **UPSTOX_LIVE_API_KEY** and **UPSTOX_LIVE_SECRET** to "
                   ".env to refresh the token from here.")
    else:
        login_url = upstox_auth.build_login_url(api_key, redirect_uri)
        st.link_button("1) Log in at Upstox ↗", login_url,
                       use_container_width=True)
        st.caption(f"Redirect URI: `{redirect_uri}`  — must EXACTLY match the "
                   "one registered in your Upstox app.")
        pasted = st.text_input(
            "2) Paste the redirected URL (or just the code)",
            key="tok_paste", placeholder="https://127.0.0.1:5000/?code=...")
        if st.button("3) Exchange & Save Token", use_container_width=True,
                     type="primary"):
            code = upstox_auth.extract_code(pasted)
            if not code:
                st.error("Couldn't find an authorization code in that input.")
            else:
                ok, msg = refresh_token_from_code(code)
                st.session_state["_token_msg"] = (
                    "success" if ok else "error", msg)
                st.rerun()

    if st.button("Check token validity", use_container_width=True):
        r = upstox_auth.check_token(config.UPSTOX_LIVE_ACCESS_TOKEN)
        if r["ok"]:
            st.success(f"Valid ✓ — {r['user_name']}")
        else:
            st.error(r["error"])

# --------------------------------------------------------------------------- #
#  Sidebar — Zerodha (Kite Connect) token refresh panel. Same daily-refresh
#  need as Upstox (Kite sessions expire ~06:00 IST every trading day).
# --------------------------------------------------------------------------- #
_pending_zmsg = st.session_state.get("_zerodha_token_msg")
with st.sidebar.expander("🔑 Zerodha Token", expanded=bool(_pending_zmsg)):
    if _pending_zmsg:
        kind, text = st.session_state.pop("_zerodha_token_msg")
        (st.success if kind == "success" else st.error)(text)

    _ztok = config.ZERODHA_ACCESS_TOKEN
    if _ztok:
        st.caption(f"Current token: `{_ztok[:8]}…`  ({len(_ztok)} chars)")
    else:
        st.caption("No Zerodha token set.")

    z_api_key, z_api_secret = kite_auth.get_credentials()
    if not (z_api_key and z_api_secret):
        st.warning("Add **ZERODHA_API_KEY** and **ZERODHA_API_SECRET** to "
                   ".env to refresh the token from here (from your Kite "
                   "Connect developer app).")
    else:
        z_login_url = kite_auth.build_login_url(z_api_key)
        st.link_button("1) Log in at Zerodha ↗", z_login_url,
                       use_container_width=True)
        st.caption("Redirect URL is whatever you registered in your Kite "
                   "Connect app — land there, then copy the URL back here.")
        z_pasted = st.text_input(
            "2) Paste the redirected URL (or just the request_token)",
            key="ztok_paste", placeholder="...?request_token=...")
        if st.button("3) Exchange & Save Token", use_container_width=True,
                     type="primary", key="ztok_exchange"):
            rtok = kite_auth.extract_request_token(z_pasted)
            if not rtok:
                st.error("Couldn't find a request_token in that input.")
            else:
                ok, msg = refresh_zerodha_token(rtok)
                st.session_state["_zerodha_token_msg"] = (
                    "success" if ok else "error", msg)
                st.rerun()

    if st.button("Check token validity", use_container_width=True,
                 key="ztok_check"):
        r = kite_auth.check_token(config.ZERODHA_ACCESS_TOKEN, z_api_key)
        if r["ok"]:
            st.success(f"Valid ✓ — {r['user_name']}")
        else:
            st.error(r["error"])

# --------------------------------------------------------------------------- #
#  Sidebar — Dhan status. Dhan's access token is generated manually from the
#  Dhan web console (no OAuth redirect flow like Upstox/Zerodha) and pasted
#  into .env directly, so this panel only validates it — it can't refresh it.
# --------------------------------------------------------------------------- #
with st.sidebar.expander("🔑 Dhan Token", expanded=False):
    if config.has_dhan():
        st.caption(f"DHAN_CLIENT_ID set · token "
                   f"`{config.DHAN_ACCESS_TOKEN[:8]}…` "
                   f"({len(config.DHAN_ACCESS_TOKEN)} chars)")
    else:
        st.caption("DHAN_CLIENT_ID / DHAN_ACCESS_TOKEN not set in .env.")
    st.caption("Generate/renew the token from the Dhan web console "
               "(dhan.co → DhanHQ Trading APIs) and paste it into .env as "
               "DHAN_ACCESS_TOKEN, then reload.")
    if st.button("Check token validity", use_container_width=True,
                 key="dhan_check"):
        b = broker_api.DhanBroker()
        if b.connect():
            funds = b.available_funds()
            st.success("Valid ✓" + (f" — available ₹{funds:,.2f}"
                                    if funds is not None else ""))
        else:
            st.error("Connect failed — token missing/expired or fund-limits "
                     "call was rejected. Check the console log for details.")

engine = st.session_state.engine


# --------------------------------------------------------------------------- #
#  Header
# --------------------------------------------------------------------------- #
st.title("📈 Automated Trading Bot")
st.caption(f"Environment: **{environment.value}**  |  Mode: **{mode.value}**  "
           f"|  Strategy: **{selected_strategy.name}**  "
           f"|  MCX commodities trade until 23:30 IST")

_live_eng = st.session_state.engine
if (_live_eng and _live_eng.state.running
        and _live_eng.strategy.key != selected_strategy.key):
    # The sidebar shows what WOULD start; the engine keeps trading what it was
    # started with. Saying so prevents "I switched strategy" surprises.
    st.warning(
        f"⚠️ The bot is still running **{_live_eng.strategy.name}**. Your "
        f"selection (**{selected_strategy.name}**) takes effect when you press "
        f"**Start Bot** again.")

tab_dash, tab_pos, tab_act, tab_logs, tab_bt = st.tabs(
    ["🖥️ Live Dashboard", "📌 Holdings", "📝 Activity Log",
     "📒 Trade Log & Analytics", "🧪 Backtesting Engine"])


# --------------------------------------------------------------------------- #
#  Live views
#
#  Auto-refresh is done with st.fragment(run_every=...), which reruns ONLY that
#  fragment on a timer. Crucially it does NOT reload the page, so st.session_state
#  (and the running engine handle) survive — a full-page meta-refresh would start
#  a new session, drop the engine, and make the bot look like it "stopped" after a
#  few seconds while leaking the old background thread. The engine is re-read from
#  session_state on every fragment rerun so it always reflects the latest state.
# --------------------------------------------------------------------------- #
_eng0 = st.session_state.engine
_running = bool(_eng0 and _eng0.state.running)
# The Scalper works 1-minute bars with a 7-minute time exit, so the UI has to keep
# up with it; the slower modes don't need a 1s cadence.
_refresh = (1 if (_eng0 and _eng0.mode == Mode.SCALPER) else 2) if _running else None


def _need_engine() -> bool:
    if st.session_state.engine is None:
        st.info("Bot not started. Configure the sidebar and press **Start Bot**. "
                "It runs in Paper/Simulation mode out of the box — no broker "
                "tokens required.")
        return False
    return True


def _position_row(sym: str, t: dict) -> dict:
    """Mark-to-market display values for one open position.

    LIVE positions carry `_broker_avg_price` / `_broker_pnl` when the engine
    could fetch them from the broker (see engine._recompute_unrealized) —
    those are preferred over our own entry_price/tick-based calc, since they
    reflect the broker's OWN fill/PnL rather than our assumption. `Entry`/`R`
    still use OUR recorded entry_price/stop/target regardless, since those are
    what the bot is actually managing the exit against.
    """
    lp = t.get("_live_price", t["entry_price"])
    direction = 1 if t["side"] == "BUY" else -1
    mult = int(t.get("contract_multiplier", 1) or 1)
    broker_pnl = t.get("_broker_pnl")
    upnl = (broker_pnl if broker_pnl is not None else
           (lp - t["entry_price"]) * t["quantity"] * direction * mult)
    risk = abs(t["entry_price"] - t["stop_loss"])
    # How far price has travelled from entry toward the target, as a fraction
    # of the risk taken. +1.0R == at target, -1.0R == at stop.
    r_mult = ((lp - t["entry_price"]) * direction / risk) if risk else 0.0
    return {
        "Symbol": sym,
        "Side": "🟢 LONG" if t["side"] == "BUY" else "🔴 SHORT",
        "Qty": t["quantity"],
        "Entry": round(t["entry_price"], 2), "LTP": round(lp, 2),
        "SL": round(t["stop_loss"], 2), "Target": round(t["target"], 2),
        "R": round(r_mult, 2),
        "Unreal PnL (₹)": round(upnl, 2),
        "PnL src": "broker" if broker_pnl is not None else "internal",
        "Avg (broker)": (round(t["_broker_avg_price"], 2)
                        if t.get("_broker_avg_price") is not None else None),
    }


# Column widths shared by the open-positions header and each trade row, so the
# ❌ Close button lines up under its own column.
_POS_COLS = [1.4, 1.1, 0.7, 1.0, 1.0, 1.0, 1.0, 0.7, 1.2, 1.2, 1.1]
_POS_HEADERS = ["Symbol", "Side", "Qty", "Entry", "LTP", "SL", "Target", "R",
                "Unreal PnL (₹)", "Avg (Broker)", "Action"]


def _render_open_positions(eng, snap: dict) -> None:
    """One row per open trade, each with its own ❌ Close button for a manual,
    at-market exit. Kept as individual rows (not a single dataframe) precisely so
    every trade can carry its own button."""
    positions = snap["open_positions"]
    if not positions:
        st.write("No open positions.")
        return
    head = st.columns(_POS_COLS)
    for col, label in zip(head, _POS_HEADERS):
        col.markdown(f"**{label}**")
    for sym, t in positions.items():
        r = _position_row(sym, t)
        row = st.columns(_POS_COLS)
        row[0].write(r["Symbol"])
        row[1].write(r["Side"])
        row[2].write(str(r["Qty"]))
        row[3].write(f"{r['Entry']:.2f}")
        row[4].write(f"{r['LTP']:.2f}")
        row[5].write(f"{r['SL']:.2f}")
        row[6].write(f"{r['Target']:.2f}")
        row[7].write(f"{r['R']:+.2f}")
        pnl_label = f"{r['Unreal PnL (₹)']:,.2f}"
        row[8].write(f"{pnl_label} 🏦" if r["PnL src"] == "broker" else pnl_label)
        row[9].write(f"{r['Avg (broker)']:.2f}" if r["Avg (broker)"] is not None
                    else "—")
        # Stable per-symbol key so Streamlit keeps the buttons distinct across reruns.
        if row[10].button("❌ Close", key=f"close_{sym}",
                          help=f"Close {sym} now at the latest price"):
            if eng.close_position(sym):
                st.toast(f"Closed {sym} at market.")
            else:
                st.toast(f"{sym} was already closed.")
            st.rerun()
    st.caption("**R** = progress in units of risk: +1.00 is the target, "
               "−1.00 is the stop. LTP updates from WebSocket ticks. 🏦 next "
               "to Unreal PnL means that figure came straight from the "
               "broker's own position (Live only) rather than our internal "
               "calc — **Avg (Broker)** is the broker's real average fill "
               "price for comparison. **❌ Close** squares the position off "
               "immediately at the latest price — the exit is logged with "
               "reason `MANUAL`.")


def _pnl_header(snap: dict) -> None:
    """The number the user actually watches — driven by WebSocket ticks. This is
    TODAY's PnL: it resets each morning but every past day is kept on disk (see the
    Day-wise PnL table), so a restart rebuilds today's figure instead of losing it."""
    total, real, unreal = (snap["day_pnl"], snap["realized_pnl"],
                           snap["unrealized_pnl"])
    day = snap.get("trading_day") or ""
    c1, c2, c3 = st.columns([2, 1, 1])
    c1.metric(f"💰 Today's PnL (₹){f' · {day}' if day else ''}", f"{total:,.2f}",
              delta=f"{unreal:+,.2f} open", delta_color="normal")
    c2.metric("Realized today (₹)", f"{real:,.2f}")
    c3.metric("Open positions", len(snap["open_positions"]))


def _daily_pnl_frame(snap: dict) -> pd.DataFrame:
    """Day-wise history from the engine snapshot (rebuilt from storage)."""
    return pd.DataFrame(snap.get("daily_pnl") or [])


@st.fragment(run_every=_refresh)
def render_dashboard() -> None:
    if not _need_engine():
        return
    snap = st.session_state.engine.state.snapshot()

    _pnl_header(snap)
    eng = st.session_state.engine
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Feed", snap["feed_status"])
    c2.metric("Strategy", eng.strategy.name)
    c3.metric("Broker", snap["broker_name"])
    c4.metric("Storage", snap["db_backend"])

    st.subheader("📶 Live Market Data (WebSocket)")
    st.caption("Proof the stream is alive. **Source** must read `WS` and "
               "**Tick age** must stay low while the market is open — `REST` "
               "means the socket dropped and it fell back to polling.")
    if snap["live_quotes"]:
        st.dataframe(pd.DataFrame(list(snap["live_quotes"].values())),
                     use_container_width=True, hide_index=True)
    else:
        st.write("Waiting for the first tick…")

    st.subheader("📡 Detected Signals")
    if snap["last_signals"]:
        sig_df = pd.DataFrame(snap["last_signals"]).rename(
            columns={"deployed": "Capital Deployed (₹)"})
        st.dataframe(sig_df, use_container_width=True, hide_index=True)
    else:
        st.write("No signals yet — waiting for strategy conditions to align.")

    st.subheader("📅 Day-wise PnL")
    st.caption("One row per trading day, newest first. Persisted to local storage, "
               "so every past day survives a restart — today's live figure above "
               "resets each morning; the days below never disappear.")
    daily = _daily_pnl_frame(snap)
    if daily.empty:
        st.write("No trading days recorded yet.")
    else:
        st.dataframe(daily, use_container_width=True, hide_index=True)

    st.caption(f"🔄 Live — refreshing every {_refresh}s" if snap["running"]
               else "⏸️ Bot stopped.")


def _render_broker_positions(eng) -> None:
    """EVERY non-flat position the broker reports right now — ground truth,
    independent of this bot's own tracking (see engine.close_broker_position
    for why the two can drift: a position closed outside the bot, or one this
    bot never opened, still shows up here). Each row's ❌ Close sends a REAL
    square-off order for that exact broker-reported side/quantity."""
    positions = eng.broker_positions()
    if not positions:
        st.write("No broker-reported open positions.")
        return
    cols = [1.4, 1.1, 0.8, 1.2, 1.2, 1.3, 1.2]
    headers = ["Symbol", "Side", "Qty", "Avg (Broker)", "LTP", "PnL (₹)", "Action"]
    head = st.columns(cols)
    for col, label in zip(head, headers):
        col.markdown(f"**{label}**")
    tracked_symbols = set(eng.state.snapshot()["open_positions"].keys())
    for p in positions:
        sym = p["symbol"]
        row = st.columns(cols)
        row[0].write(sym + ("" if sym in tracked_symbols else " 🆕"))
        row[1].write("🟢 LONG" if p["side"] == "BUY" else "🔴 SHORT")
        row[2].write(str(p["quantity"]))
        row[3].write(f"{p['average_price']:.2f}")
        row[4].write(f"{p['last_price']:.2f}")
        row[5].write(f"{p['pnl']:,.2f}")
        if row[6].button("❌ Close", key=f"close_broker_{sym}",
                         help=f"Square off {sym} at the broker right now"):
            ok, msg = eng.close_broker_position(sym, p["quantity"], p["side"])
            st.toast(msg if ok else f"Failed: {msg}")
            st.rerun()
    st.caption("This table comes straight from the broker's own position "
               "book, refreshed every few seconds — it will show positions "
               "this bot never opened or has lost track of (marked 🆕), not "
               "just the ones in **Currently Open** below. Closing here sends "
               "a REAL order sized to match the broker's own reported "
               "quantity, and also updates this bot's own record if the "
               "symbol matches a tracked trade.")


@st.fragment(run_every=_refresh)
def render_holdings() -> None:
    if not _need_engine():
        return
    eng = st.session_state.engine
    snap = eng.state.snapshot()
    _pnl_header(snap)
    if eng.environment == Environment.LIVE:
        st.subheader("🏦 Broker Positions (ground truth)")
        _render_broker_positions(eng)
        st.markdown("---")
    st.subheader("📌 Currently Open (bot-tracked)")
    _render_open_positions(eng, snap)
    st.caption(f"🔄 Live — refreshing every {_refresh}s" if snap["running"]
               else "⏸️ Bot stopped.")


@st.fragment(run_every=_refresh)
def render_activity() -> None:
    if not _need_engine():
        return
    snap = st.session_state.engine.state.snapshot()
    st.subheader("📝 Activity Log")
    st.caption("Newest first. Entries, exits, rejections and feed problems.")
    n = st.slider("Lines to show", 20, 200, 60, step=20, key="log_lines")
    st.code("\n".join(snap["log"][:n]) or "—", language="text")
    st.caption(f"🔄 Live — refreshing every {_refresh}s" if snap["running"]
               else "⏸️ Bot stopped.")


with tab_dash:
    render_dashboard()

with tab_pos:
    render_holdings()

with tab_act:
    render_activity()


# --------------------------------------------------------------------------- #
#  Tab 2 — Trade Logs & Analytics
# --------------------------------------------------------------------------- #
with tab_logs:
    view_env = st.radio("Show trades from", ["Paper", "Live"], horizontal=True)
    env_sel = Environment.PAPER if view_env == "Paper" else Environment.LIVE

    summary = db.analytics_summary(env_sel)
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Total Trades", summary["total_trades"])
    s2.metric("Win Rate %", summary["win_rate"])
    s3.metric("Total PnL (₹)", summary["total_pnl"])
    s4.metric("Avg PnL (₹)", summary["avg_pnl"])

    st.subheader("📅 Day-wise PnL")
    st.caption("Rebuilt from stored trades — the permanent day-by-day record. "
               "Visible even when the bot is stopped, because it reads from disk.")
    daily = db.daily_pnl(env_sel)
    if daily.empty:
        st.write("No trading days recorded yet.")
    else:
        st.dataframe(daily, use_container_width=True, hide_index=True)

    trades = db.get_trades(env_sel)
    st.subheader(f"{view_env} Trade History  ·  collection: "
                 f"`{'paper_trades' if env_sel==Environment.PAPER else 'live_trades'}`")
    if trades.empty:
        st.write("No trades recorded yet.")
    else:
        st.dataframe(trades, use_container_width=True, hide_index=True)

    if st.button("⬇️ Export High-Level Analysis to Excel"):
        path = db.export_excel(env_sel)
        st.success(f"Exported to: {path}")
        try:
            with open(path, "rb") as fh:
                st.download_button("Download workbook", fh.read(),
                                   file_name=path.split("\\")[-1])
        except Exception:
            pass

    # -- Danger zone: full portfolio reset --------------------------------- #
    with st.expander("🧨 Danger Zone — Reset Portfolio"):
        st.warning(
            f"This permanently deletes **all {view_env} trades** recorded till "
            "date (every day, closed and open) and the running Excel log, then "
            "starts fresh from zero. **This cannot be undone.** Only the "
            f"**{view_env}** book is affected — the other environment is untouched.")
        st.caption("Tip: export to Excel first if you want a copy before wiping.")

        eng = st.session_state.engine
        eng_here = eng is not None and eng.environment == env_sel
        if eng_here and eng.state.snapshot()["open_positions"]:
            st.error("⚠️ There are OPEN positions in this environment. For live "
                     "trading, close them first — resetting only forgets them "
                     "here, it does not square off real broker positions.")

        confirm = st.checkbox(
            f"I understand — permanently delete all {view_env} data.",
            key="reset_confirm")
        if st.button("🗑️ Reset Portfolio Now", type="primary", disabled=not confirm,
                     key="reset_btn"):
            # Use the engine when it owns THIS environment so the live dashboard is
            # cleared too; otherwise just wipe storage.
            if eng_here:
                stats = eng.reset_portfolio()
            else:
                stats = db.reset_environment(env_sel)
            st.session_state.reset_confirm = False
            st.success(f"Portfolio reset — removed {stats['trades_removed']} "
                       f"{view_env} trade(s). Starting fresh.")
            st.rerun()


# --------------------------------------------------------------------------- #
#  Single-symbol backtest analysis — day/hour/setup/side breakdowns.
#  Diverging blue/red pair = PnL sign, fixed BUY=blue/SELL=orange for side
#  identity (both from the validated palette, unchanged).
# --------------------------------------------------------------------------- #
_PNL_PROFIT, _PNL_LOSS = "#2a78d6", "#e34948"
_SIDE_COLOR = {"BUY": "#2a78d6", "SELL": "#eb6834"}
_GRIDLINE, _AXIS_LINE = "#e1e0d9", "#c3c2b7"
_NUM_RE = re.compile(r"[-+]?\d+\.?\d*")


def _reason_bucket(reason: str) -> str:
    """Collapse a reason string's live numbers (ATR value, coil level, volume
    multiple, bar count) into a stable category, so trades from the same setup
    group together instead of each getting a one-off bucket."""
    cleaned = _NUM_RE.sub("", str(reason))
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip(" .:-x")
    return cleaned or str(reason)


def _build_trade_analysis(trades: pd.DataFrame) -> dict:
    """Turn a raw trades DataFrame into the aggregates the analysis charts
    need. Kept separate from rendering so a saved report's trades (plain
    dicts, string timestamps) can be re-plotted later the same way."""
    t = trades.copy()
    t["entry_time"] = pd.to_datetime(t["entry_time"])
    t["_day"] = t["entry_time"].dt.day_name()
    t["_hour"] = t["entry_time"].dt.hour

    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
                 "Saturday", "Sunday"]
    day_pnl = t.groupby("_day")["pnl"].sum().reindex(
        [d for d in day_order if d in t["_day"].unique()])

    # Swing (one daily bar) has a single entry hour, so the chart is not
    # meaningful there — only build it when trades actually span >1 hour.
    hour_pnl = (t.groupby("_hour")["pnl"].sum().sort_index()
                if t["_hour"].nunique() > 1 else None)

    setup = t["entry_reason"].apply(_reason_bucket)
    reason_pnl = t.groupby(setup)["pnl"].sum().sort_values(ascending=False).head(10)

    side_stats = t.groupby("side").agg(
        trades=("pnl", "size"), win_rate=("win", "mean"),
        total_pnl=("pnl", "sum")).reset_index()
    side_stats["win_rate"] *= 100

    return {"day_pnl": day_pnl, "hour_pnl": hour_pnl,
            "reason_pnl": reason_pnl, "side_stats": side_stats}


def _pnl_bar_figure(series: pd.Series, title: str, x_title: str,
                    tickangle: int = 0) -> go.Figure:
    """A PnL-by-category bar chart on the diverging blue/red pair — blue for
    net-profitable buckets, red for net-loss ones — each its own legend
    entry, so the sign is never color-alone."""
    labels = [str(i) for i in series.index]
    pos_y = [v if v >= 0 else None for v in series.values]
    neg_y = [v if v < 0 else None for v in series.values]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=labels, y=pos_y, name="Profit",
                        marker_color=_PNL_PROFIT,
                        hovertemplate="%{x}<br>₹%{y:,.2f}<extra></extra>"))
    fig.add_trace(go.Bar(x=labels, y=neg_y, name="Loss",
                        marker_color=_PNL_LOSS,
                        hovertemplate="%{x}<br>₹%{y:,.2f}<extra></extra>"))
    fig.update_layout(
        title=title, height=360, showlegend=True, bargap=0.35,
        xaxis_title=x_title, yaxis_title="Total PnL (₹)",
        xaxis=dict(showgrid=False, linecolor=_AXIS_LINE, tickangle=tickangle),
        yaxis=dict(gridcolor=_GRIDLINE, zerolinecolor=_AXIS_LINE),
        margin=dict(t=48, b=40),
    )
    return fig


def _side_figure(side_stats: pd.DataFrame) -> go.Figure:
    """Win % by side — BUY/SELL identity is fixed to the same two hues every
    time (never re-cycled), and the x-axis labels already name each bar, so
    no legend box is needed."""
    colors = [_SIDE_COLOR.get(s, "#898781") for s in side_stats["side"]]
    fig = go.Figure(go.Bar(
        x=side_stats["side"], y=side_stats["win_rate"], marker_color=colors,
        text=[f"{n} trades" for n in side_stats["trades"]],
        textposition="outside",
        hovertemplate="%{x}<br>Win rate %{y:.1f}%<extra></extra>"))
    fig.update_layout(
        title="Win % by Side (BUY vs SELL)", height=360, showlegend=False,
        yaxis_title="Win Rate (%)",
        xaxis=dict(showgrid=False, linecolor=_AXIS_LINE),
        yaxis=dict(gridcolor=_GRIDLINE, zerolinecolor=_AXIS_LINE, range=[0, 100]),
        margin=dict(t=48, b=40),
    )
    return fig


def _trade_insights(analysis: dict) -> list[str]:
    """Plain-English read of the aggregates — which conditions worked, which
    didn't, so a person tweaking the bot has a starting point without having
    to eyeball four charts themselves."""
    bullets = []
    day_pnl = analysis["day_pnl"]
    if not day_pnl.empty:
        best_day, worst_day = day_pnl.idxmax(), day_pnl.idxmin()
        bullets.append(
            f"**{best_day}** was the best day to trade (₹{day_pnl[best_day]:,.2f} "
            f"total), while **{worst_day}** was the weakest "
            f"(₹{day_pnl[worst_day]:,.2f}).")
    hour_pnl = analysis["hour_pnl"]
    if hour_pnl is not None and not hour_pnl.empty:
        best_hr, worst_hr = hour_pnl.idxmax(), hour_pnl.idxmin()
        bullets.append(
            f"The **{best_hr}:00** hour was the most profitable entry window "
            f"(₹{hour_pnl[best_hr]:,.2f}); **{worst_hr}:00** was the weakest "
            f"(₹{hour_pnl[worst_hr]:,.2f}) — consider filtering entries around it.")
    reason_pnl = analysis["reason_pnl"]
    if not reason_pnl.empty:
        top_setup = reason_pnl.index[0]
        bullets.append(
            f"**\"{top_setup}\"** was the most profitable setup, contributing "
            f"₹{reason_pnl.iloc[0]:,.2f} across this run.")
    side_stats = analysis["side_stats"].set_index("side")
    if "BUY" in side_stats.index and "SELL" in side_stats.index:
        buy_wr, sell_wr = side_stats.loc["BUY", "win_rate"], side_stats.loc["SELL", "win_rate"]
        leader = "SELL" if sell_wr > buy_wr else "BUY"
        bullets.append(
            f"**{leader}** trades had the higher win rate (BUY {buy_wr:.1f}% vs "
            f"SELL {sell_wr:.1f}%) across {int(side_stats['trades'].sum())} total "
            f"trades — {int(side_stats.loc['BUY', 'trades'])} long / "
            f"{int(side_stats.loc['SELL', 'trades'])} short.")
    return bullets


def _render_trade_analysis(trades: pd.DataFrame, key_prefix: str) -> None:
    """Render the four analysis charts (PnL by day, PnL by hour, PnL by
    setup, win% by side) plus the plain-English insights list. `key_prefix`
    keeps Streamlit element keys unique between a fresh run and a re-opened
    saved report shown on the same page."""
    analysis = _build_trade_analysis(trades)

    c1, c2 = st.columns(2)
    if not analysis["day_pnl"].empty:
        c1.plotly_chart(
            _pnl_bar_figure(analysis["day_pnl"], "Total PnL by Day of Week", "Day"),
            use_container_width=True, key=f"{key_prefix}_day")
    if analysis["hour_pnl"] is not None:
        c2.plotly_chart(
            _pnl_bar_figure(analysis["hour_pnl"], "Total PnL by Hour of Day", "Hour"),
            use_container_width=True, key=f"{key_prefix}_hour")
    else:
        c2.caption("Hour-of-day breakdown needs Intraday/Scalper trades "
                   "spanning more than one entry hour.")

    c3, c4 = st.columns(2)
    if not analysis["reason_pnl"].empty:
        c3.plotly_chart(
            _pnl_bar_figure(analysis["reason_pnl"], "Top Setups by Total PnL",
                           "Setup", tickangle=-20),
            use_container_width=True, key=f"{key_prefix}_reason")
    if not analysis["side_stats"].empty:
        c4.plotly_chart(_side_figure(analysis["side_stats"]),
                        use_container_width=True, key=f"{key_prefix}_side")

    bullets = _trade_insights(analysis)
    if bullets:
        st.markdown("**Key Insights**")
        for b in bullets:
            st.markdown(f"- {b}")


# --------------------------------------------------------------------------- #
#  Tab 3 — Backtesting Engine
# --------------------------------------------------------------------------- #
with tab_bt:
    bc1, bc2, bc3 = st.columns(3)
    ticker = bc1.selectbox("Ticker", [i.symbol for i in ALL_INSTRUMENTS])
    BT_MODES = {"Swing": Mode.SWING, "Intraday": Mode.INTRADAY,
                "Scalper": Mode.SCALPER}
    bt_mode = bc2.selectbox("Mode", list(BT_MODES))
    init_cap = bc3.number_input("Initial Capital (₹)", value=100_000.0,
                                step=10_000.0)

    _bt_choices = strategy.strategies_for_mode(BT_MODES[bt_mode])
    _bt_name = st.selectbox("Strategy", [s.name for s in _bt_choices],
                            key="bt_strategy")
    bt_strategy = next(s for s in _bt_choices if s.name == _bt_name)
    st.caption(f"_{bt_strategy.summary}_")
    # 1-minute history is only available for a short window, so a year-long
    # Scalper range would return nothing usable and silently fall back to
    # synthetic data. Default to a range each mode can actually be tested over.
    _default_span = {"Swing": 365, "Intraday": 30, "Scalper": 5}[bt_mode]
    d1, d2 = st.columns(2)
    start_d = d1.date_input("Start",
                            value=date.today() - timedelta(days=_default_span))
    end_d = d2.date_input("End", value=date.today())
    if bt_mode == "Scalper":
        st.caption("⚡ Scalper backtests run on **1-minute** candles. Upstox "
                   "serves only a short window of minute history — keep the "
                   "range to a few days or it will fall back to synthetic data.")

    if st.button("🚀 Run Backtest", type="primary"):
        inst = config.INSTRUMENTS_BY_SYMBOL[ticker]
        with st.spinner(f"Backtesting {bt_strategy.name} on {ticker}..."):
            result = backtester.run_backtest(
                ticker, str(start_d), str(end_d), init_cap,
                BT_MODES[bt_mode], lot_size=inst.lot_size,
                strategy_key=bt_strategy.key,
            )
        # Stashed in session_state, not a local var — a later rerun (typing in
        # the "save analysis" box, clicking Save) would otherwise re-evaluate
        # this button as False and the whole result would vanish.
        st.session_state["bt_last_result"] = result
        st.session_state["bt_last_meta"] = {
            "ticker": ticker, "mode": bt_mode, "strategy": bt_strategy.name,
            "start": str(start_d), "end": str(end_d),
        }

    if "bt_last_result" in st.session_state:
        result = st.session_state["bt_last_result"]
        meta = st.session_state["bt_last_meta"]
        m = result.metrics
        # Make the data source explicit. A silent fall-back to synthetic
        # random-walk prices is exactly what made backtest trade prices not
        # match the real instrument — surface it loudly instead of hiding it.
        _src = m.get("Data Source", "synthetic")
        if _src == "upstox":
            st.success(f"📈 Real Upstox historical candles for {meta['ticker']}.")
        elif _src == "yfinance":
            st.info(f"📊 Real yfinance historical candles for {meta['ticker']}.")
        else:
            st.warning(
                "⚠️ **Synthetic data** — these trades are on a simulated "
                "random walk, NOT real prices for this instrument, so the "
                "entry/exit prices will not match the market. Upstox history "
                "was unavailable for this range (intraday/scalper minute data "
                "is limited; try a shorter range or check the access token).")
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Total Return %", m["Total Return %"])
        k2.metric("Max Drawdown %", m["Max Drawdown %"])
        k3.metric("Sharpe", m["Sharpe"])
        k4.metric("Calmar", m["Calmar"])
        k5.metric("Win Rate %", m["Win Rate %"])
        st.caption(f"Trades: {m['Total Trades']}  ·  Final Equity: "
                   f"₹{m['Final Equity']:.2f}")

        if not result.equity_curve.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=result.equity_curve.index, y=result.equity_curve.values,
                mode="lines", name="Equity", line=dict(color="#2E86DE")))
            fig.update_layout(title="Equity Curve", height=420,
                              xaxis_title="Time", yaxis_title="Equity (₹)")
            st.plotly_chart(fig, use_container_width=True, key="bt_equity_chart")

        if not result.trades.empty:
            st.subheader("Backtest Trades")
            st.dataframe(result.trades, use_container_width=True, hide_index=True)

            st.markdown("---")
            st.subheader("📊 Trade Analysis")
            _render_trade_analysis(result.trades, key_prefix="bt_live")

            st.markdown("---")
            sc1, sc2 = st.columns([2, 1])
            _report_name = sc1.text_input(
                "Save this backtest analysis as", key="bt_report_new_name",
                placeholder=f"e.g. {meta['ticker']} {meta['mode']} {date.today()}")
            st.caption(f"Will be saved under the **{meta['mode']}** category "
                       "(matches the mode this backtest ran in).")
            if sc2.button("💾 Save analysis", key="bt_report_save",
                         use_container_width=True):
                _payload = {
                    "ticker": meta["ticker"], "mode": meta["mode"],
                    "strategy": meta["strategy"], "start": meta["start"],
                    "end": meta["end"], "metrics": m,
                    "trades": result.trades.assign(
                        entry_time=result.trades["entry_time"].astype(str),
                        exit_time=result.trades["exit_time"].astype(str),
                    ).to_dict("records"),
                }
                if backtest_reports.save(_report_name, _payload,
                                         category=meta["mode"]):
                    st.success(f"Saved analysis as '{_report_name.strip()}' "
                              f"under {meta['mode']}.")
                else:
                    st.error("Enter a name to save this analysis under.")
        else:
            st.info("No trades were generated in this window. Try a wider date "
                    "range or the other mode.")

    # ----------------------------------------------------------------------- #
    #  Saved single-symbol analyses — reopen a named report's charts and
    #  insights later without re-running the backtest.
    # ----------------------------------------------------------------------- #
    st.markdown("---")
    st.subheader("📂 Saved Backtest Analyses")
    _bt_categories = list(BT_MODES)  # ["Swing", "Intraday", "Scalper"]
    _cat_pick = st.selectbox(
        "Category", _bt_categories,
        index=_bt_categories.index(bt_mode) if bt_mode in _bt_categories else 0,
        key="bt_report_category_pick")
    _saved_reports = backtest_reports.names(_cat_pick)
    if _saved_reports:
        _load_pick = st.selectbox("Open a saved analysis", _saved_reports,
                                  key="bt_report_load_pick")
        lc1, lc2 = st.columns(2)
        if lc1.button("📂 Open", key="bt_report_open", use_container_width=True):
            st.session_state["bt_report_loaded"] = (_cat_pick, _load_pick)
        if lc2.button("🗑️ Delete", key="bt_report_delete", use_container_width=True):
            backtest_reports.delete(_cat_pick, _load_pick)
            st.session_state.pop("bt_report_loaded", None)
            st.rerun()

        _loaded = st.session_state.get("bt_report_loaded")
        if (isinstance(_loaded, tuple) and _loaded[0] == _cat_pick
                and _loaded[1] in backtest_reports.names(_cat_pick)):
            _loaded_cat, _loaded_name = _loaded
            _rep = backtest_reports.get(_loaded_cat, _loaded_name)
            st.caption(f"**{_rep['ticker']}** · {_rep['mode']} · {_rep['strategy']} "
                      f"· {_rep['start']} → {_rep['end']}")
            _rm = _rep.get("metrics", {})
            rk1, rk2, rk3, rk4, rk5 = st.columns(5)
            rk1.metric("Total Return %", _rm.get("Total Return %"))
            rk2.metric("Max Drawdown %", _rm.get("Max Drawdown %"))
            rk3.metric("Sharpe", _rm.get("Sharpe"))
            rk4.metric("Calmar", _rm.get("Calmar"))
            rk5.metric("Win Rate %", _rm.get("Win Rate %"))
            _rtrades = pd.DataFrame(_rep.get("trades", []))
            if not _rtrades.empty:
                _render_trade_analysis(
                    _rtrades, key_prefix=f"bt_saved_{_loaded_cat}_{_loaded_name}")
            else:
                st.info("This saved analysis has no trades to chart.")
    else:
        st.caption(f"No saved analyses in {_cat_pick} yet — run a backtest "
                   "above and save it.")

    # ----------------------------------------------------------------------- #
    #  Bulk / bucket backtest — same strategy + params across many instruments,
    #  all equity curves overlaid in one chart for side-by-side comparison.
    # ----------------------------------------------------------------------- #
    st.markdown("---")
    st.subheader("🧺 Bulk Backtest (compare a bucket)")
    st.caption("Pick several instruments and run the **same strategy with the "
               "same parameters** on all of them. Every equity curve is drawn "
               "in the one chart (a different colour each) so you can see which "
               "symbols the strategy performed best on.")

    # Load a saved watchlist straight into the bucket — the SAME watchlists the
    # live sidebar uses, so a set saved here is ready to trade live and vice-versa.
    def _apply_bulk_watchlist() -> None:
        name = st.session_state.get("bt_bulk_watchlist_pick", _WL_NONE)
        if not name or name == _WL_NONE:
            return
        st.session_state["bt_bulk_symbols"] = [
            s for s in watchlists.get(name) if s in config.INSTRUMENTS_BY_SYMBOL]

    _wl_names_bt = watchlists.names()
    wc1, wc2 = st.columns([2, 1])
    wc1.selectbox(
        "📋 Load a watchlist into the bucket", [_WL_NONE] + _wl_names_bt,
        key="bt_bulk_watchlist_pick", on_change=_apply_bulk_watchlist,
        help="Watchlists are shared with the live bot's sidebar.")

    bulk_symbols = st.multiselect(
        "Bucket — instruments to test together",
        [i.symbol for i in ALL_INSTRUMENTS],
        key="bt_bulk_symbols",
        help="All use the Mode, Strategy, capital and date range selected above.")

    # Save the current bucket as a watchlist so it can be reused here or live.
    sc1, sc2 = st.columns([2, 1])
    _bt_wl_new = sc1.text_input(
        "Save this bucket as a watchlist", key="bt_wl_new_name",
        placeholder="e.g. Swing Winners")
    if sc2.button("💾 Save bucket", key="bt_wl_save", use_container_width=True):
        if watchlists.save(_bt_wl_new, st.session_state.get("bt_bulk_symbols", [])):
            st.success(f"Saved '{_bt_wl_new.strip()}' "
                       f"({len(st.session_state.get('bt_bulk_symbols', []))} symbols).")
            st.rerun()
        else:
            st.error("Enter a name and add at least one instrument to the bucket.")
    norm_bulk = st.checkbox(
        "Normalise curves to % return (start all at 0%)", value=True,
        key="bt_bulk_norm",
        help="Recommended for comparison — removes the effect of each symbol "
             "starting from the same capital and lets you compare shapes.")

    if st.button("🚀 Run Bulk Backtest", key="bt_bulk_run"):
        if not bulk_symbols:
            st.error("Select at least one instrument for the bucket.")
        else:
            prog = st.progress(0.0, text="Starting…")

            def _cb(done, total, sym):
                prog.progress(done / total, text=f"{sym} ({done}/{total})")

            with st.spinner(f"Backtesting {bt_strategy.name} across "
                            f"{len(bulk_symbols)} instruments..."):
                bulk_results = backtester.run_bulk_backtest(
                    bulk_symbols, str(start_d), str(end_d), init_cap,
                    BT_MODES[bt_mode], strategy_key=bt_strategy.key,
                    progress_cb=_cb)
            prog.empty()

            summary = backtester.bulk_summary_frame(bulk_results)

            # Overlaid equity curves — one coloured line per instrument.
            fig = go.Figure()
            palette = px.colors.qualitative.Dark24
            plotted = 0
            for idx, (sym, res) in enumerate(bulk_results.items()):
                eq = res.equity_curve
                if eq.empty:
                    continue
                y = ((eq / init_cap - 1) * 100).values if norm_bulk else eq.values
                fig.add_trace(go.Scatter(
                    x=eq.index, y=y, mode="lines", name=sym,
                    line=dict(color=palette[idx % len(palette)], width=2)))
                plotted += 1
            if plotted:
                fig.update_layout(
                    title=f"Bulk Equity Curves — {bt_strategy.name}",
                    height=480, xaxis_title="Time",
                    yaxis_title="Return (%)" if norm_bulk else "Equity (₹)",
                    legend_title="Instrument", hovermode="x unified")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No equity curves to plot — none of the selected "
                        "instruments generated data in this window.")

            st.subheader("📊 Comparison")
            st.caption("Sorted by Total Return — the top row performed best.")
            st.dataframe(summary, use_container_width=True, hide_index=True)

            if (summary["Data Source"] == "synthetic").any():
                st.warning(
                    "⚠️ Some instruments fell back to **synthetic** random-walk "
                    "data (real history unavailable for this range) — their "
                    "curves are not comparable to the real ones. Check the "
                    "**Data Source** column.")
