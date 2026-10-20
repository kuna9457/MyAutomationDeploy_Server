"""Booking the broker's REAL fill price, and withholding strategy from clients.

The bot used to book every trade at the price it asked for (an order call
returns an id, not a fill), so its weekly total drifted from the client's broker
P&L. These tests pin the fix and — just as important — that it is inert
everywhere it should be: Paper, a broker with no lookup, a lookup that fails.
"""
from __future__ import annotations

import pytest

import broker_api
import engine as engine_mod
from broker_api import poll_fill, vwap_of_fills
from config import Environment
from tests.test_engine_exit_path import (FakeBroker, FakeOrder, _engine, _open,
                                         BASE, inst)  # noqa: F401


# --------------------------------------------------------------------------- #
#  The price arithmetic.
# --------------------------------------------------------------------------- #
def test_vwap_weights_by_quantity():
    assert vwap_of_fills([(100.0, 3), (101.0, 1)]) == (100.25, 4)


def test_vwap_ignores_unusable_rows_and_empty_is_none():
    assert vwap_of_fills([(0, 5), (None, 2), ("x", 1), (100, 0)]) is None
    assert vwap_of_fills([]) is None


def test_poll_returns_as_soon_as_the_order_is_complete():
    calls = []

    def fetch():
        calls.append(1)
        return (100.5, 10)

    assert poll_fill(fetch, 10, timeout=5.0) == (100.5, 10)
    assert len(calls) == 1


def test_poll_returns_a_partial_fill_at_the_deadline():
    assert poll_fill(lambda: (100.5, 4), 10, timeout=0.0) == (100.5, 4)


def test_poll_treats_an_erroring_lookup_as_no_answer():
    def boom():
        raise RuntimeError("api down")

    assert poll_fill(boom, 10, timeout=0.0) is None


def test_base_broker_has_no_fill_lookup():
    b = broker_api.SimulatedBroker()
    assert b.confirm_fill("x", 1) is None
    assert b.find_exit_fill(None, "SELL") is None
    assert b.day_realized_pnl() is None


# --------------------------------------------------------------------------- #
#  Engine: which price gets booked.
# --------------------------------------------------------------------------- #
class FillBroker(FakeBroker):
    def __init__(self, fill=None, exit_fill=None, raises=False):
        super().__init__()
        self.fill, self.exit_fill, self.raises = fill, exit_fill, raises
        self.asked = []

    def confirm_fill(self, order_id, quantity, timeout=2.5):
        self.asked.append((order_id, quantity))
        if self.raises:
            raise RuntimeError("lookup failed")
        return self.fill

    def find_exit_fill(self, instrument, exit_side):
        return self.exit_fill


def _live(inst, **kw):
    eng = _engine(BASE)
    eng.environment = Environment.LIVE
    eng.broker = FillBroker(**kw)
    eng.broker.cancel_oco_exit = lambda _id: True
    return eng


def test_live_exit_is_booked_at_the_brokers_fill(inst):
    eng = _live(inst, fill=(101.7, 4))
    trade = _open(eng, inst, entry=100.0, qty=4)
    eng._finalize_close(inst, trade, 102.0, "TARGET")
    doc = eng.db.closed[0]
    assert doc["exit_price"] == 101.7                    # not the 102.0 asked for
    assert doc["realized_pnl"] == pytest.approx(6.8)    # (101.7-100)*4


def test_unreadable_fill_books_the_assumed_price_exactly_as_before(inst):
    for broker_kw in ({"fill": None}, {"raises": True}):
        eng = _live(inst, **broker_kw)
        trade = _open(eng, inst, entry=100.0, qty=4)
        eng._finalize_close(inst, trade, 102.0, "TARGET")
        assert eng.db.closed[0]["exit_price"] == 102.0


def test_paper_never_asks_the_broker(inst):
    eng = _engine(BASE)                                  # Environment.PAPER
    eng.broker = FillBroker(fill=(50.0, 4))
    trade = _open(eng, inst, entry=100.0, qty=4)
    eng._finalize_close(inst, trade, 102.0, "TARGET")
    assert eng.broker.asked == []
    assert eng.db.closed[0]["exit_price"] == 102.0


def test_exit_the_broker_placed_itself_is_booked_at_its_fill(inst):
    eng = _live(inst, exit_fill=98.4)
    eng._broker_position = lambda i, force=False: {"quantity": 0}   # flat
    trade = _open(eng, inst, entry=100.0, qty=4)
    eng._finalize_close(inst, trade, 98.0, "STOP-LOSS")
    assert eng.broker.square_offs == []                  # no double exit
    assert eng.db.closed[0]["exit_price"] == 98.4


def test_partial_is_booked_at_the_brokers_fill(inst):
    eng = _live(inst, fill=(101.9, 2))
    trade = _open(eng, inst, entry=100.0, qty=4)
    st = trade["_exit_state"]
    em = engine_mod.exit_manager
    act = em.Action(em.ActionType.PARTIAL, price=102.0, qty=2,
                    reason="PARTIAL-TARGET")
    st.stop = 100.0
    eng._book_partial(inst, trade, st, act)
    child = eng.db.closed[0]
    assert child["quantity"] == 2
    assert child["exit_price"] == 101.9
    assert eng.broker.asked == [("sim-1", 2)]


# --------------------------------------------------------------------------- #
#  Clients never receive strategy, even in the raw JSON.
# --------------------------------------------------------------------------- #
def test_client_status_payload_names_no_strategy():
    from api.routers import bot as bot_router

    out = {
        "groups": [{"strategy_key": "k", "strategy_name": "N"}],
        "strategy_pnl": [{"Strategy": "N"}],
        "daily_strategy_pnl": [{"Strategy": "N"}],
        "daily_pnl": [{"Date": "2026-10-19", "Strategies": "N", "Trades": 2}],
        "last_signals": [{"symbol": "X", "reason": "VWAP + MACD", "qty": 1}],
        "log": ["[candle_intraday] EXIT X BUY @ 10.00 [TARGET] PnL ₹1.00",
                "ENTER X BUY qty=1 @ 10.00 via MARKET | SL=9 TP=11 | Price>VWAP"],
    }
    bot_router._strip_strategy_for_client(out, ["candle_intraday"])
    assert "groups" not in out and "strategy_pnl" not in out
    assert "daily_strategy_pnl" not in out
    assert "Strategies" not in out["daily_pnl"][0]
    assert out["daily_pnl"][0]["Trades"] == 2
    assert "reason" not in out["last_signals"][0]
    assert out["log"][0].startswith("EXIT X")
    assert "VWAP" not in out["log"][1] and out["log"][1].startswith("ENTER X")
