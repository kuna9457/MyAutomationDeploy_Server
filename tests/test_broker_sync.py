"""The Live trade log rebuilt from Kite's own fills (broker_sync.py).

What is pinned: the round-trip reconstruction, that the day's P&L afterwards
equals Kite's, and — as much as the happy path — what the sync must NOT do:
hide OPEN rows, touch Paper or another account, double count a symbol that is
still open, or wipe a day when Kite hands back an empty tradebook.
"""
from __future__ import annotations

from datetime import datetime

import pytest

import broker_sync
import config
from config import Environment
from db_manager import DBManager

DAY = "2026-10-20"
NOW = datetime(2026, 10, 20, 16, 0)


def fill(sym, side, qty, price, hhmmss, order="o1", product="MIS",
         exchange="NSE", tid=None):
    return {"tradingsymbol": sym, "exchange": exchange, "product": product,
            "transaction_type": side, "quantity": qty, "average_price": price,
            "fill_timestamp": f"{DAY} {hhmmss}", "order_id": order,
            "trade_id": tid or f"{order}-{hhmmss}"}


def positions(day_realised=0.0, net=(), mults=None):
    day = [{"tradingsymbol": s, "exchange": "NSE", "product": "MIS",
            "realised": r, "multiplier": (mults or {}).get(s, 1)}
           for s, r in (day_realised if isinstance(day_realised, list) else [])]
    return {"day": day, "net": list(net)}


class FakeKite:
    def __init__(self, fills, pos):
        self.fills, self.pos = fills, pos

    def day_fills(self):
        return self.fills

    def day_positions(self):
        return self.pos


@pytest.fixture
def db(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "LOCAL_DB_DIR", str(tmp_path))
    d = DBManager.__new__(DBManager)
    d.client = None
    d.db = None
    return d


def bot_row(db, ticker, side, entry, exit_, qty, hhmmss, user="alice",
            status="CLOSED", env=Environment.LIVE, broker="Zerodha"):
    doc = DBManager.new_trade("Intraday", env.value, broker, ticker, side,
                              entry, entry * 0.99, entry * 1.01, qty, 100.0,
                              segment=config.Segment.EQUITY.value,
                              strategy="vwap_macd", user_id=user)
    # bot rows are stamped in UTC; 09:30 IST = 04:00 UTC
    h, m, s = map(int, hhmmss.split(":"))
    total = h * 3600 + m * 60 + s - 5 * 3600 - 30 * 60
    doc["timestamp"] = f"{DAY}T{total // 3600:02d}:{total % 3600 // 60:02d}:{total % 60:02d}"
    if status == "CLOSED":
        doc.update(status="CLOSED", exit_price=exit_,
                   realized_pnl=round(DBManager._pnl(doc, exit_), 2),
                   exit_reason="TARGET", exit_timestamp=doc["timestamp"])
    db.insert_trade(doc, env)
    return doc


# --------------------------------------------------------------------------- #
#  Reconstruction
# --------------------------------------------------------------------------- #
def test_a_scale_out_is_one_round_trip_at_vwap_prices():
    fills = broker_sync.normalize_fills([
        fill("INFY", "BUY", 10, 100.0, "09:30:00", "a"),
        fill("INFY", "BUY", 10, 101.0, "09:30:01", "a", tid="a2"),
        fill("INFY", "SELL", 10, 103.0, "10:00:00", "b"),
        fill("INFY", "SELL", 10, 102.0, "11:00:00", "c"),
    ], DAY)
    trips, unresolved = broker_sync.rebuild_round_trips(fills, {}, {})
    assert unresolved == set()
    [t] = trips
    assert (t["side"], t["quantity"]) == ("BUY", 20)
    assert t["entry_price"] == pytest.approx(100.5)
    assert t["exit_price"] == pytest.approx(102.5)
    assert t["realized_pnl"] == pytest.approx(40.0)
    assert t["entry_order_ids"] == ["a"] and t["exit_order_ids"] == ["b", "c"]


def test_a_fill_through_zero_closes_one_trip_and_opens_the_next():
    fills = broker_sync.normalize_fills([
        fill("SBIN", "SELL", 5, 200.0, "09:30:00", "a"),
        fill("SBIN", "BUY", 8, 198.0, "10:00:00", "b"),   # cover 5, long 3
        fill("SBIN", "SELL", 3, 199.0, "11:00:00", "c"),
    ], DAY)
    trips, unresolved = broker_sync.rebuild_round_trips(fills, {}, {})
    assert unresolved == set()
    assert [(t["side"], t["quantity"]) for t in trips] == [("SELL", 5), ("BUY", 3)]
    assert sum(t["realized_pnl"] for t in trips) == pytest.approx(10.0 + 3.0)


def test_still_open_or_carried_symbols_are_unresolved_not_half_counted():
    fills = broker_sync.normalize_fills([
        fill("INFY", "BUY", 10, 100.0, "09:30:00", "a"),
        fill("INFY", "SELL", 10, 101.0, "10:00:00", "b"),
        fill("INFY", "BUY", 5, 100.0, "11:00:00", "c"),   # open at the end
        fill("TCS", "SELL", 2, 3000.0, "09:40:00", "d"),  # closes yesterday's
    ], DAY)
    opening = {("TCS", "NSE", "MIS"): 2}
    trips, unresolved = broker_sync.rebuild_round_trips(fills, opening, {})
    assert trips == []
    assert unresolved == {("INFY", "NSE", "MIS"), ("TCS", "NSE", "MIS")}


def test_commodity_pnl_uses_the_broker_multiplier():
    fills = broker_sync.normalize_fills([
        fill("CRUDEOILM26NOVFUT", "BUY", 1, 6000.0, "19:00:00", "a",
             product="NRML", exchange="MCX"),
        fill("CRUDEOILM26NOVFUT", "SELL", 1, 6010.0, "20:00:00", "b",
             product="NRML", exchange="MCX"),
    ], DAY)
    [t] = broker_sync.rebuild_round_trips(
        fills, {}, {"CRUDEOILM26NOVFUT": 10})[0]
    assert t["realized_pnl"] == pytest.approx(100.0)


def test_fills_from_another_day_are_ignored():
    raw = [fill("INFY", "BUY", 1, 100.0, "09:30:00")]
    raw[0]["fill_timestamp"] = "2026-10-19 09:30:00"
    assert broker_sync.normalize_fills(raw, DAY) == []


# --------------------------------------------------------------------------- #
#  End to end against the local store
# --------------------------------------------------------------------------- #
def test_after_a_sync_the_day_reads_the_broker_numbers(db):
    # The bot booked INFY at assumed prices: 100 -> 102 on 10 = +20.
    bot_row(db, "INFY", "BUY", 100.0, 102.0, 10, "09:30:00")
    kite = FakeKite([fill("INFY", "BUY", 10, 100.2, "09:30:02", "a"),
                     fill("INFY", "SELL", 10, 101.6, "10:15:00", "b")],
                    positions([("INFY", 14.0)]))
    rep = broker_sync.sync_user("alice", db=db, broker=kite, now=NOW)
    assert rep["ok"] and rep["matches"], rep
    assert rep["broker_realised"] == 14.0 and rep["ledger_total"] == 14.0

    rows = db.get_trades(Environment.LIVE, user_id="alice")
    assert len(rows) == 1
    row = rows.iloc[0]
    assert row["source"] == "broker" and row["origin"] == "bot"
    assert row["entry_price"] == 100.2 and row["exit_price"] == 101.6
    # labels are carried over from the bot's own row
    assert row["strategy"] == "vwap_macd" and row["exit_reason"] == "TARGET"
    assert db.today_realized(Environment.LIVE, day=DAY, user_id="alice") == 14.0
    # the bot's own row still exists underneath
    raw = db.get_trades(Environment.LIVE, user_id="alice", broker_overlay=False)
    assert len(raw) == 2


def test_resync_replaces_rather_than_duplicates(db):
    kite = FakeKite([fill("INFY", "BUY", 10, 100.0, "09:30:00", "a"),
                     fill("INFY", "SELL", 10, 101.0, "10:00:00", "b")],
                    positions([("INFY", 10.0)]))
    broker_sync.sync_user("alice", db=db, broker=kite, now=NOW)
    broker_sync.sync_user("alice", db=db, broker=kite, now=NOW)
    assert len(db.get_trades(Environment.LIVE, user_id="alice")) == 1


def test_open_rows_unresolved_symbols_other_accounts_and_paper_are_untouched(db):
    bot_row(db, "INFY", "BUY", 100.0, 102.0, 10, "09:30:00")
    bot_row(db, "TCS", "BUY", 3000.0, 3010.0, 1, "09:45:00")          # closed...
    bot_row(db, "TCS", "BUY", 3000.0, None, 1, "11:00:00", status="OPEN")
    bot_row(db, "INFY", "BUY", 100.0, 102.0, 10, "09:30:00", user="bob")
    bot_row(db, "INFY", "BUY", 100.0, 102.0, 10, "09:30:00",
            env=Environment.PAPER)
    kite = FakeKite([fill("INFY", "BUY", 10, 100.0, "09:30:00", "a"),
                     fill("INFY", "SELL", 10, 101.0, "10:00:00", "b"),
                     fill("TCS", "BUY", 1, 3000.0, "09:45:00", "c"),
                     fill("TCS", "SELL", 1, 3010.0, "10:30:00", "d"),
                     fill("TCS", "BUY", 1, 3000.0, "11:00:00", "e")],
                    positions([("INFY", 10.0), ("TCS", 10.0)]))
    rep = broker_sync.sync_user("alice", db=db, broker=kite, now=NOW)
    assert rep["unresolved"] == ["TCS"]

    alice = db.get_trades(Environment.LIVE, user_id="alice")
    kinds = sorted((r["ticker"], r.get("source") or "bot", r["status"])
                   for r in alice.to_dict("records"))
    assert kinds == [("INFY", "broker", "CLOSED"),
                     ("TCS", "bot", "CLOSED"), ("TCS", "bot", "OPEN")]
    assert len(db.get_open_trades(Environment.LIVE, user_id="alice")) == 1
    assert len(db.get_trades(Environment.LIVE, user_id="bob")) == 1
    assert len(db.get_trades(Environment.PAPER)) == 1


def test_a_trade_placed_outside_the_bot_is_included_and_marked_manual(db):
    kite = FakeKite([fill("HDFCBANK", "BUY", 1, 1500.0, "12:00:00", "m1"),
                     fill("HDFCBANK", "SELL", 1, 1495.0, "12:30:00", "m2")],
                    positions([("HDFCBANK", -5.0)]))
    rep = broker_sync.sync_user("alice", db=db, broker=kite, now=NOW)
    assert rep["manual"] == 1 and rep["matches"]
    row = db.get_trades(Environment.LIVE, user_id="alice").iloc[0]
    assert row["origin"] == "manual" and row["exit_reason"] == "MANUAL"


def test_an_empty_tradebook_never_hides_the_bots_trades(db):
    bot_row(db, "INFY", "BUY", 100.0, 102.0, 10, "09:30:00")
    rep = broker_sync.sync_user("alice", db=db,
                                broker=FakeKite([], positions()), now=NOW)
    assert not rep["ok"]
    assert db.broker_sync_record("alice", DAY) is None
    assert len(db.get_trades(Environment.LIVE, user_id="alice")) == 1


def test_an_unreadable_broker_changes_nothing(db):
    bot_row(db, "INFY", "BUY", 100.0, 102.0, 10, "09:30:00")
    rep = broker_sync.sync_user("alice", db=db,
                                broker=FakeKite(None, positions()), now=NOW)
    assert not rep["ok"]
    assert len(db.get_trades(Environment.LIVE, user_id="alice")) == 1


def test_resetting_the_live_book_clears_its_sync_records(db):
    kite = FakeKite([fill("INFY", "BUY", 10, 100.0, "09:30:00", "a"),
                     fill("INFY", "SELL", 10, 101.0, "10:00:00", "b")],
                    positions([("INFY", 10.0)]))
    broker_sync.sync_user("alice", db=db, broker=kite, now=NOW)
    db.reset_environment(Environment.LIVE, user_id="alice")
    assert db.broker_sync_record("alice", DAY) is None
    # a trade booked after the reset is visible again
    bot_row(db, "INFY", "BUY", 100.0, 102.0, 10, "13:00:00")
    assert len(db.get_trades(Environment.LIVE, user_id="alice")) == 1


def test_the_scheduler_only_runs_in_the_trading_window():
    assert broker_sync.in_window(datetime(2026, 10, 20, 15, 45))
    assert broker_sync.in_window(datetime(2026, 10, 20, 23, 50))
    assert not broker_sync.in_window(datetime(2026, 10, 20, 8, 0))
    assert not broker_sync.in_window(datetime(2026, 10, 24, 12, 0))  # Saturday
