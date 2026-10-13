"""
telegram_signals.py
Publishes every signal the platform decides to a Telegram channel, and keeps
a journal of them the admin can track and correct from the panel.

    StrategyRunner ──on_signal()──> journal row (OPEN) ──> "🟢 BUY X  SL  TP"
                   ──on_price()───> SL / TP touched    ──> reply "✅ Target hit"
                   ──on_exit()────> square-off / time  ──> reply "⏹ Closed"
    admin panel    ──edit/close/delete/manual post──> same journal, same channel

What a post contains — and what it never does
---------------------------------------------
Symbol, side, entry, stop-loss, target, time. NOTHING about how the signal was
found: no strategy name, no `Signal.reason`, no indicator values, no params.
`_render` is the only place a message is built and it reads only those fields.

Never in the trading path's way
-------------------------------
The runner calls in here from its tick loop, so every entry point does only
in-memory bookkeeping under a lock and hands the network call and the storage
write to ONE background worker. Telegram being slow, down, rate-limiting or
misconfigured can delay a post; it can never delay, stop or alter a trade.
Every entry point swallows its own exceptions for the same reason.

Never a fake signal
-------------------
A runner on a SIMULATED feed is trading invented prices. Those signals — and
SL/TP "hits" priced off them — are dropped here, so a public channel only ever
sees decisions made on real market data.

One signal, one post
--------------------
Several runners can decide the same thing (the admin's bot and the fleet
broadcast on the same strategy, or client runners). A post is keyed by
symbol + side + bar, and a symbol that already has an OPEN post is not posted
again until that one closes — a follower holds one position per symbol.

The journal tracks the PUBLISHED levels, which the admin may edit. It is the
channel's record, not any account's: accounts still size, fill and exit on
their own (strategy_runner Rule #4), including any partial or trail.
"""
from __future__ import annotations

import os
import queue
import threading
import uuid
from datetime import datetime
from html import escape
from typing import Any, Callable, Optional

import config_store
import credentials_vault
from config import now_ist

CONFIG_DOC = "telegram_config"
JOURNAL_DOC = "telegram_signals"
#: Oldest closed rows are dropped past this; OPEN rows are always kept.
MAX_ROWS = 500
API_URL = "https://api.telegram.org/bot{token}/{method}"
HTTP_TIMEOUT_SECONDS = 15

OPEN = "OPEN"
TARGET = "TARGET"
STOP = "STOP-LOSS"
CLOSED = "CLOSED"       # square-off, time exit, or closed by the admin
CANCELLED = "CANCELLED"  # withdrawn by the admin before it resolved

DEFAULT_CONFIG: dict = {
    "enabled": False,
    "chat_id": "",
    "post_exits": True,
    "header": "",
    "footer": "",
    "bot_token_enc": "",
}

_lock = threading.RLock()
_config: Optional[dict] = None
_rows: Optional[list[dict]] = None
_jobs: "queue.Queue[Callable[[], None]]" = queue.Queue()
_worker: Optional[threading.Thread] = None
#: Decrypted once per config change — on_price runs every tick per symbol.
_token_cache: Optional[str] = None


# --------------------------------------------------------------------------- #
#  Configuration
# --------------------------------------------------------------------------- #
def _cfg() -> dict:
    global _config
    with _lock:
        if _config is None:
            _config = {**DEFAULT_CONFIG, **config_store.load(CONFIG_DOC, {})}
        return _config


def _token() -> str:
    global _token_cache
    env = (os.getenv("TELEGRAM_BOT_TOKEN", "") or "").strip()
    if env:
        return env
    with _lock:
        if _token_cache is None:
            _token_cache = credentials_vault.decrypt(_cfg().get("bot_token_enc", ""))
        return _token_cache


def _chat_id() -> str:
    return (str(_cfg().get("chat_id") or "").strip()
            or (os.getenv("TELEGRAM_CHAT_ID", "") or "").strip())


def is_active() -> bool:
    return bool(_cfg().get("enabled")) and bool(_token()) and bool(_chat_id())


def public_config() -> dict:
    """Settings for the panel. The bot token has NO read path — only whether
    one is set and where it comes from."""
    cfg = _cfg()
    env_token = bool((os.getenv("TELEGRAM_BOT_TOKEN", "") or "").strip())
    return {
        "enabled": bool(cfg.get("enabled")),
        "chat_id": _chat_id(),
        "post_exits": bool(cfg.get("post_exits", True)),
        "header": cfg.get("header", ""),
        "footer": cfg.get("footer", ""),
        "token_set": bool(_token()),
        "token_source": "env" if env_token else ("panel" if cfg.get("bot_token_enc") else ""),
        "vault_ready": credentials_vault.is_configured(),
        "active": is_active(),
    }


def update_config(*, enabled: bool, chat_id: str, post_exits: bool,
                  header: str, footer: str, bot_token: str = "",
                  clear_token: bool = False) -> dict:
    global _config, _token_cache
    with _lock:
        cfg = dict(_cfg())
        cfg.update(enabled=bool(enabled), chat_id=(chat_id or "").strip(),
                   post_exits=bool(post_exits), header=(header or "").strip(),
                   footer=(footer or "").strip())
        if clear_token:
            cfg["bot_token_enc"] = ""
        elif bot_token.strip():
            # Raises VaultUnavailable without CREDENTIALS_KEY — a bot token is a
            # secret and is never stored in plaintext.
            cfg["bot_token_enc"] = credentials_vault.encrypt(bot_token.strip())
        _config, _token_cache = cfg, None
    config_store.save(CONFIG_DOC, cfg)
    return public_config()


# --------------------------------------------------------------------------- #
#  The journal
# --------------------------------------------------------------------------- #
def _journal() -> list[dict]:
    global _rows
    with _lock:
        if _rows is None:
            _rows = list(config_store.load(JOURNAL_DOC, {"rows": []}).get("rows", []))
        return _rows


def _persist() -> None:
    with _lock:
        rows = _journal()
        # Newest first, so the closed rows kept are the most recent ones.
        room = max(MAX_ROWS - sum(1 for r in rows if r["status"] == OPEN), 0)
        keep = set([r["id"] for r in rows if r["status"] != OPEN][:room])
        rows[:] = [r for r in rows if r["status"] == OPEN or r["id"] in keep]
        snapshot = [dict(r) for r in rows]
    config_store.save(JOURNAL_DOC, {"rows": snapshot})


def _find(signal_id: str) -> Optional[dict]:
    for r in _journal():
        if r["id"] == signal_id:
            return r
    return None


def _open_for(symbol: str) -> Optional[dict]:
    for r in _journal():
        if r["symbol"] == symbol and r["status"] == OPEN:
            return r
    return None


def list_signals() -> dict:
    with _lock:
        rows = [dict(r) for r in _journal()]
    done = [r for r in rows if r["status"] in (TARGET, STOP, CLOSED)]
    wins = [r for r in done if (r.get("pnl_pct") or 0) > 0]
    return {
        "rows": rows,
        "stats": {
            "total": len(rows),
            "open": sum(1 for r in rows if r["status"] == OPEN),
            "targets": sum(1 for r in rows if r["status"] == TARGET),
            "stops": sum(1 for r in rows if r["status"] == STOP),
            "closed": len(done),
            "win_rate": round(100.0 * len(wins) / len(done), 1) if done else None,
            "net_pct": round(sum(r.get("pnl_pct") or 0 for r in done), 2),
        },
    }


# --------------------------------------------------------------------------- #
#  Messages — the ONLY place post text is built
# --------------------------------------------------------------------------- #
def _px(v: Optional[float]) -> str:
    return "—" if v is None else f"{v:,.2f}"


def _pnl_pct(row: dict, price: float) -> float:
    e = row["entry"]
    if not e:
        return 0.0
    move = (price - e) if row["side"] == "BUY" else (e - price)
    return round(100.0 * move / e, 2)


def _outcome_line(row: dict) -> str:
    st, px, pct = row["status"], row.get("exit_price"), row.get("pnl_pct")
    sign = "" if pct is None else f" ({pct:+.2f}%)"
    if st == TARGET:
        return f"✅ <b>Target hit</b> @ {_px(px)}{sign}"
    if st == STOP:
        return f"❌ <b>Stop-loss hit</b> @ {_px(px)}{sign}"
    if st == CLOSED:
        why = escape(row.get("exit_reason") or "Closed")
        return f"⏹ <b>{why}</b> @ {_px(px)}{sign}"
    if st == CANCELLED:
        return "🚫 <b>Cancelled</b>"
    return ""


def _render(row: dict) -> str:
    """The signal post. Reads symbol/side/levels/time only — never anything
    describing how the signal was decided."""
    cfg = _cfg()
    dot = "🟢" if row["side"] == "BUY" else "🔴"
    lines = []
    if cfg.get("header"):
        lines.append(f"<b>{escape(cfg['header'])}</b>")
    lines += [
        f"{dot} <b>{row['side']}  {escape(row['symbol'])}</b>",
        "━━━━━━━━━━━━━━",
        f"Entry: <b>{_px(row['entry'])}</b>",
        f"Stop Loss: <b>{_px(row['stop'])}</b>",
        f"Target: <b>{_px(row['target'])}</b>",
        "━━━━━━━━━━━━━━",
        f"🕒 {escape(row['posted_label'])}",
    ]
    if row.get("edited"):
        lines.append("✏️ <i>Levels updated</i>")
    outcome = _outcome_line(row)
    if outcome:
        lines.append(outcome)
    if cfg.get("footer"):
        lines.append(f"<i>{escape(cfg['footer'])}</i>")
    return "\n".join(lines)


def _render_exit(row: dict) -> str:
    return f"{escape(row['symbol'])} {row['side']}\n{_outcome_line(row)}"


def _render_update(row: dict) -> str:
    return (f"✏️ <b>{escape(row['symbol'])} {row['side']} — levels updated</b>\n"
            f"Stop Loss: <b>{_px(row['stop'])}</b>\n"
            f"Target: <b>{_px(row['target'])}</b>")


# --------------------------------------------------------------------------- #
#  Telegram HTTP — always off the caller's thread except the explicit test
# --------------------------------------------------------------------------- #
class TelegramError(RuntimeError):
    pass


def _call(method: str, payload: dict, token: Optional[str] = None) -> dict:
    import requests
    tok = token or _token()
    if not tok:
        raise TelegramError("No bot token configured.")
    try:
        resp = requests.post(API_URL.format(token=tok, method=method),
                             json=payload, timeout=HTTP_TIMEOUT_SECONDS)
        body = resp.json()
    except Exception as exc:
        # requests puts the URL — which contains the token — into its errors.
        raise TelegramError(str(exc).replace(tok, "<token>")) from None
    if not body.get("ok"):
        raise TelegramError(body.get("description") or f"HTTP {resp.status_code}")
    return body.get("result") or {}


def _send(text: str, reply_to: Optional[int] = None) -> int:
    payload: dict[str, Any] = {"chat_id": _chat_id(), "text": text,
                               "parse_mode": "HTML",
                               "disable_web_page_preview": True}
    if reply_to:
        payload["reply_parameters"] = {"message_id": reply_to,
                                       "allow_sending_without_reply": True}
    return int(_call("sendMessage", payload).get("message_id") or 0)


def _ensure_worker() -> None:
    global _worker
    with _lock:
        if _worker is not None and _worker.is_alive():
            return
        _worker = threading.Thread(target=_work, daemon=True, name="telegram-signals")
        _worker.start()


def _work() -> None:
    while True:
        job = _jobs.get()
        try:
            job()
        except Exception as exc:  # a job must never kill the worker
            print(f"[telegram] job failed: {exc}")
        finally:
            _jobs.task_done()


def _enqueue(job: Callable[[], None]) -> None:
    _ensure_worker()
    _jobs.put(job)


def _job_post(row: dict) -> Callable[[], None]:
    def run() -> None:
        try:
            mid = _send(_render(row))
            with _lock:
                row["message_id"] = mid
                row["chat_id"] = _chat_id()
                row["error"] = ""
        except TelegramError as exc:
            with _lock:
                row["error"] = str(exc)
        _persist()
    return run


def _job_refresh(row: dict, reply: str = "") -> Callable[[], None]:
    """Re-render the original post with its current state, then optionally
    reply under it so followers get a notification."""
    def run() -> None:
        mid = row.get("message_id")
        try:
            if mid:
                try:
                    _call("editMessageText", {
                        "chat_id": row.get("chat_id") or _chat_id(),
                        "message_id": mid, "text": _render(row),
                        "parse_mode": "HTML", "disable_web_page_preview": True})
                except TelegramError as exc:
                    if "not modified" not in str(exc):
                        raise
            if reply:
                _send(reply, reply_to=mid)
            with _lock:
                row["error"] = ""
        except TelegramError as exc:
            with _lock:
                row["error"] = str(exc)
        _persist()
    return run


# --------------------------------------------------------------------------- #
#  Runner hooks — in-memory only on the caller's thread, never raise
# --------------------------------------------------------------------------- #
def _new_row(symbol: str, segment: str, side: str, entry: float, stop: float,
             target: float, source: str, bar_key: str = "") -> dict:
    now = now_ist()
    return {
        "id": uuid.uuid4().hex[:12],
        "symbol": symbol, "segment": segment, "side": side,
        "entry": round(float(entry), 2), "stop": round(float(stop), 2),
        "target": round(float(target), 2),
        "status": OPEN, "source": source, "bar_key": bar_key,
        "posted_at": now.isoformat(timespec="seconds"),
        "posted_label": now.strftime("%d %b %Y, %H:%M IST"),
        "closed_at": "", "exit_price": None, "pnl_pct": None,
        "exit_reason": "", "edited": False, "message_id": 0,
        "chat_id": "", "error": "",
    }


def on_signal(symbol: str, segment: str, side: str, entry: float,
              stop: float, target: float, bar_ts: Any,
              simulated: bool) -> None:
    """A runner decided an entry. Posts it once; see the module docstring."""
    try:
        if simulated or not is_active():
            return
        key = f"{symbol}|{side}|{bar_ts}"
        with _lock:
            rows = _journal()
            if _open_for(symbol) is not None:
                return
            if any(r.get("bar_key") == key for r in rows):
                return
            row = _new_row(symbol, segment, side, entry, stop, target, "auto", key)
            rows.insert(0, row)
        _enqueue(_job_post(row))
    except Exception as exc:
        print(f"[telegram] on_signal failed: {exc}")


def _close(row: dict, status: str, price: float, reason: str) -> None:
    """Caller holds _lock."""
    row["status"] = status
    row["exit_price"] = round(float(price), 2)
    row["pnl_pct"] = _pnl_pct(row, float(price))
    row["exit_reason"] = reason
    row["closed_at"] = now_ist().isoformat(timespec="seconds")


def _announce_close(row: dict) -> None:
    reply = _render_exit(row) if _cfg().get("post_exits", True) else ""
    _enqueue(_job_refresh(row, reply))


def on_price(symbol: str, price: float, simulated: bool) -> None:
    """Resolve an OPEN post whose published SL or TP the price has reached."""
    try:
        if simulated or not is_active():
            return
        with _lock:
            row = _open_for(symbol)
            if row is None:
                return
            long = row["side"] == "BUY"
            if (price <= row["stop"]) if long else (price >= row["stop"]):
                _close(row, STOP, row["stop"], "Stop-loss hit")
            elif (price >= row["target"]) if long else (price <= row["target"]):
                _close(row, TARGET, row["target"], "Target hit")
            else:
                return
        _announce_close(row)
    except Exception as exc:
        print(f"[telegram] on_price failed: {exc}")


def on_exit(symbol: str, price: float, reason: str, simulated: bool) -> None:
    """The runner flattened a symbol on the clock (square-off / time exit)."""
    try:
        if simulated or not is_active():
            return
        with _lock:
            row = _open_for(symbol)
            if row is None:
                return
            label = ("Squared off" if reason.startswith("SQUARE-OFF")
                     else "Time exit" if reason.startswith("TIME-EXIT")
                     else "Closed")
            _close(row, CLOSED, price, label)
        _announce_close(row)
    except Exception as exc:
        print(f"[telegram] on_exit failed: {exc}")


# --------------------------------------------------------------------------- #
#  Admin actions
# --------------------------------------------------------------------------- #
def _require_active() -> None:
    if not _token() or not _chat_id():
        raise ValueError("Set the bot token and channel first.")


def _check_levels(side: str, entry: float, stop: float, target: float) -> None:
    if side not in ("BUY", "SELL"):
        raise ValueError("Side must be BUY or SELL.")
    if min(entry, stop, target) <= 0:
        raise ValueError("Entry, stop-loss and target must be positive.")
    if side == "BUY" and not (stop < entry < target):
        raise ValueError("For a BUY: stop-loss < entry < target.")
    if side == "SELL" and not (target < entry < stop):
        raise ValueError("For a SELL: target < entry < stop-loss.")


def post_manual(symbol: str, side: str, entry: float, stop: float,
                target: float) -> dict:
    _require_active()
    symbol, side = symbol.strip().upper(), side.strip().upper()
    if not symbol:
        raise ValueError("Symbol is required.")
    _check_levels(side, entry, stop, target)
    with _lock:
        if _open_for(symbol) is not None:
            raise ValueError(f"{symbol} already has an open signal — close it first.")
        row = _new_row(symbol, "", side, entry, stop, target, "manual")
        _journal().insert(0, row)
    _enqueue(_job_post(row))
    return dict(row)


def edit_levels(signal_id: str, stop: Optional[float], target: Optional[float],
                notify: bool) -> dict:
    with _lock:
        row = _find(signal_id)
        if row is None:
            raise KeyError(signal_id)
        if row["status"] != OPEN:
            raise ValueError("Only an open signal's levels can be changed.")
        new_stop = row["stop"] if stop is None else round(float(stop), 2)
        new_target = row["target"] if target is None else round(float(target), 2)
        # Not _check_levels: trailing the stop past entry (to cost, or into
        # profit) is a normal edit. Only the two levels' order is fixed.
        if min(new_stop, new_target) <= 0:
            raise ValueError("Stop-loss and target must be positive.")
        if row["side"] == "BUY" and not new_stop < new_target:
            raise ValueError("For a BUY the stop-loss must be below the target.")
        if row["side"] == "SELL" and not new_stop > new_target:
            raise ValueError("For a SELL the stop-loss must be above the target.")
        row.update(stop=new_stop, target=new_target, edited=True)
    _enqueue(_job_refresh(row, _render_update(row) if notify else ""))
    return dict(row)


def close_signal(signal_id: str, price: Optional[float], cancel: bool) -> dict:
    with _lock:
        row = _find(signal_id)
        if row is None:
            raise KeyError(signal_id)
        if row["status"] != OPEN:
            raise ValueError("This signal is already closed.")
        if cancel:
            row["status"] = CANCELLED
            row["closed_at"] = now_ist().isoformat(timespec="seconds")
        else:
            if not price or price <= 0:
                raise ValueError("Give the exit price.")
            _close(row, CLOSED, price, "Closed")
    _announce_close(row)
    return dict(row)


def delete_signal(signal_id: str, from_channel: bool) -> None:
    with _lock:
        row = _find(signal_id)
        if row is None:
            raise KeyError(signal_id)
        _journal().remove(row)
    if from_channel and row.get("message_id"):
        def run() -> None:
            try:
                _call("deleteMessage", {"chat_id": row.get("chat_id") or _chat_id(),
                                        "message_id": row["message_id"]})
            except TelegramError as exc:
                print(f"[telegram] delete failed: {exc}")
            _persist()
        _enqueue(run)
    else:
        _enqueue(_persist)


def resend(signal_id: str) -> dict:
    """Post again a row whose first post failed (e.g. bad channel id)."""
    _require_active()
    with _lock:
        row = _find(signal_id)
        if row is None:
            raise KeyError(signal_id)
        if row.get("message_id"):
            raise ValueError("This signal is already in the channel.")
    _enqueue(_job_post(row))
    return dict(row)


def send_test() -> dict:
    """Synchronous on purpose: the admin is waiting to see if it worked."""
    _require_active()
    me = _call("getMe", {})
    _send("✅ Test message — this channel is connected and will receive "
          "trade signals.")
    return {"ok": True, "bot": me.get("username", "")}


def flush(timeout: float = 5.0) -> None:
    """Wait for queued posts (tests)."""
    import time
    end = time.time() + timeout
    while _jobs.unfinished_tasks and time.time() < end:
        time.sleep(0.01)


def _reset_for_tests() -> None:
    global _config, _rows, _token_cache
    with _lock:
        _config, _rows, _token_cache = None, None, None
