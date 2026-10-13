"""Telegram signal channel: one post per decision, nothing about the strategy
in it, never from a simulated feed, SL/TP tracked on the published levels,
and a Telegram failure never reaching the caller."""
import pytest

import config_store
import telegram_signals as tg


@pytest.fixture
def channel(monkeypatch):
    store: dict = {}
    calls: list[tuple[str, dict]] = []
    monkeypatch.setattr(config_store, "load",
                        lambda name, default=None: dict(store.get(name, default or {})))
    monkeypatch.setattr(config_store, "save",
                        lambda name, data: store.__setitem__(name, data))
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123:SECRET")
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)

    def fake_call(method, payload, token=None):
        calls.append((method, payload))
        return {"message_id": len(calls), "username": "testbot"}

    monkeypatch.setattr(tg, "_call", fake_call)
    tg._reset_for_tests()
    tg.update_config(enabled=True, chat_id="@chan", post_exits=True,
                     header="", footer="")
    yield calls, store
    tg.flush()
    tg._reset_for_tests()


def _signal(**kw):
    args = dict(symbol="RELIANCE", segment="NSE_EQ", side="BUY", entry=100.0,
                stop=98.0, target=104.0, bar_ts="2026-10-07 10:15",
                simulated=False)
    args.update(kw)
    tg.on_signal(**args)
    tg.flush()


def test_posts_symbol_side_and_levels_only(channel):
    calls, _ = channel
    _signal()
    sends = [p for m, p in calls if m == "sendMessage"]
    assert len(sends) == 1
    text = sends[0]["text"]
    for part in ("BUY", "RELIANCE", "100.00", "98.00", "104.00"):
        assert part in text
    # Nothing describing how it was found can be in the post: _render only
    # reads the row, and the row has no reason/strategy field at all.
    row = tg.list_signals()["rows"][0]
    assert not {"reason", "strategy", "params"} & set(row)


def test_same_decision_from_two_runners_posts_once(channel):
    calls, _ = channel
    _signal()
    _signal()
    _signal(bar_ts="2026-10-07 10:30")  # still OPEN on this symbol
    assert sum(1 for m, _ in calls if m == "sendMessage") == 1


def test_simulated_feed_never_posts_or_resolves(channel):
    calls, _ = channel
    _signal(simulated=True)
    assert calls == []
    _signal()
    tg.on_price("RELIANCE", 50.0, simulated=True)
    tg.flush()
    assert tg.list_signals()["rows"][0]["status"] == tg.OPEN


def test_disabled_posts_nothing(channel):
    calls, _ = channel
    tg.update_config(enabled=False, chat_id="@chan", post_exits=True,
                     header="", footer="")
    _signal()
    assert calls == []


@pytest.mark.parametrize("side,price,status,pct", [
    ("BUY", 104.5, tg.TARGET, 4.0),
    ("BUY", 97.0, tg.STOP, -2.0),
    ("SELL", 95.0, tg.TARGET, 4.0),
    ("SELL", 103.0, tg.STOP, -2.0),
])
def test_price_resolves_on_published_levels(channel, side, price, status, pct):
    calls, _ = channel
    stop, target = (98.0, 104.0) if side == "BUY" else (102.0, 96.0)
    _signal(side=side, stop=stop, target=target)
    tg.on_price("RELIANCE", 100.5 if side == "BUY" else 99.5, simulated=False)
    assert tg.list_signals()["rows"][0]["status"] == tg.OPEN
    tg.on_price("RELIANCE", price, simulated=False)
    tg.flush()
    row = tg.list_signals()["rows"][0]
    assert row["status"] == status and row["pnl_pct"] == pct
    methods = [m for m, _ in calls]
    assert "editMessageText" in methods
    reply = [p for m, p in calls if m == "sendMessage"][-1]
    assert reply["reply_parameters"]["message_id"] == row["message_id"]


def test_square_off_closes_and_frees_the_symbol(channel):
    _signal()
    tg.on_exit("RELIANCE", 101.0, "SQUARE-OFF (15:09)", simulated=False)
    tg.flush()
    row = tg.list_signals()["rows"][0]
    assert row["status"] == tg.CLOSED and row["exit_reason"] == "Squared off"
    _signal(bar_ts="2026-10-08 10:15")
    assert len(tg.list_signals()["rows"]) == 2


def test_admin_edit_moves_the_tracked_levels(channel):
    _signal()
    sid = tg.list_signals()["rows"][0]["id"]
    tg.edit_levels(sid, stop=100.5, target=None, notify=True)
    tg.on_price("RELIANCE", 100.2, simulated=False)  # below the NEW stop
    tg.flush()
    row = tg.list_signals()["rows"][0]
    assert row["status"] == tg.STOP and row["exit_price"] == 100.5
    with pytest.raises(ValueError):
        tg.edit_levels(sid, stop=99.0, target=None, notify=False)  # closed


def test_bad_levels_rejected(channel):
    with pytest.raises(ValueError):
        tg.post_manual("TCS", "BUY", entry=100, stop=101, target=105)
    with pytest.raises(ValueError):
        tg.post_manual("TCS", "SELL", entry=100, stop=99, target=95)


def test_telegram_failure_never_reaches_the_runner(channel, monkeypatch):
    def boom(method, payload, token=None):
        raise tg.TelegramError("chat not found")
    monkeypatch.setattr(tg, "_call", boom)
    _signal()  # must not raise
    row = tg.list_signals()["rows"][0]
    assert row["error"] == "chat not found" and row["message_id"] == 0


def test_token_never_returned(channel):
    cfg = tg.public_config()
    assert "SECRET" not in repr(cfg) and cfg["token_set"]
