"""The LIVE/paper side of the exit refactor, exercised through the real
engine methods.

exit_manager is tested on its own elsewhere; what is checked here is the half
that a pure decision test cannot reach — that engine._manage_open turns those
decisions into the right BROKER ORDERS and TRADE DOCUMENTS:

  * a partial squares off only the booked lots and writes them as their own
    closed child trade, leaving the runner open with a break-even stop and the
    runner target (Immutable Rule #1: the runner's quantity is what was left,
    never re-sized),
  * a trail persists each new stop as it moves,
  * the legs of one position add back up to the entry quantity (C-5),
  * a strategy with nothing configured exits exactly on its entry stop/target
    and writes exactly one document (C-2).

A TradingEngine is assembled field by field rather than constructed: __init__
opens a broker session and joins a shared market-data runner, neither of which
belongs in a unit test. The methods under test are the real ones.
"""
from __future__ import annotations

import threading
from dataclasses import replace

import pytest

import config
import engine as engine_mod
from config import Environment, Mode
from db_manager import DBManager


# --------------------------------------------------------------------------- #
#  Test doubles — only the surface the exit path actually touches.
# --------------------------------------------------------------------------- #
class FakeOrder:
    def __init__(self, order_id="sim-1"):
        self.ok = True
        self.order_id = order_id
        self.message = ""
        self.filled_price = None


class FakeBroker:
    name = "Simulated"

    def __init__(self):
        self.square_offs = []          # (symbol, side, qty, price)

    def square_off(self, inst, side, qty, price):
        self.square_offs.append((inst.symbol, side, int(qty), float(price)))
        return FakeOrder(f"sim-{len(self.square_offs)}")


class FakeDB:
    """In-memory stand-in for DBManager. new_trade is the REAL one, so the
    documents these tests assert on are the Section-5 schema documents the
    live bot would actually store."""

    new_trade = staticmethod(DBManager.new_trade)

    def __init__(self):
        self.docs: dict[str, dict] = {}
        self.closed: list[dict] = []

    def insert_trade(self, trade, environment):
        self.docs[trade["trade_id"]] = dict(trade)

    def close_trade(self, trade_id, exit_price, environment, exit_reason=""):
        doc = self.docs.get(trade_id)
        if doc is None:
            return None
        side = 1 if doc["side"] == "BUY" else -1
        pnl = ((float(exit_price) - float(doc["entry_price"]))
               * int(doc["quantity"]) * side
               * max(int(doc.get("contract_multiplier", 1) or 1), 1))
        doc.update(status="CLOSED", exit_price=float(exit_price),
                   realized_pnl=pnl, exit_reason=exit_reason)
        self.closed.append(doc)
        return doc

    def update_trade_fields(self, trade_id, environment, fields):
        self.docs.setdefault(trade_id, {}).update(fields)

    def today_realized(self, environment, user_id="admin"):
        return sum(d.get("realized_pnl", 0.0) for d in self.closed)


def _engine(params, symbol_rules=None):
    """A TradingEngine with only the fields the exit path reads."""
    eng = object.__new__(engine_mod.TradingEngine)
    eng.environment = Environment.PAPER
    eng.mode = Mode.INTRADAY
    eng.user_id = "test"
    eng.params = params
    eng.strategy = engine_mod.resolve_strategy(Mode.INTRADAY, "")
    eng.symbol_rules = dict(symbol_rules or {})
    eng.broker = FakeBroker()
    eng.db = FakeDB()
    eng.state = engine_mod.BotState()
    eng.state.lock = threading.RLock()
    # Both are dashboard bookkeeping with storage behind them; neither takes
    # part in an exit decision, so they are stubbed rather than faked.
    eng._refresh_daily = lambda: None
    eng._recompute_unrealized = lambda: None
    eng._broker_position = lambda inst, force=False: None
    return eng


def _open(eng, inst, side="BUY", entry=100.0, qty=4, risk=2.0, rr=1.0):
    """Put a position under management the way _enter leaves one."""
    stop = entry - risk if side == "BUY" else entry + risk
    target = entry + rr * risk if side == "BUY" else entry - rr * risk
    trade = eng.db.new_trade(
        mode=eng.mode.value, environment=eng.environment.value,
        user_id=eng.user_id, broker=eng.broker.name, ticker=inst.symbol,
        side=side, entry_price=entry, stop_loss=stop, target=target,
        quantity=qty, risk_amount=risk * qty, segment=inst.segment.value,
        contract_multiplier=inst.contract_multiplier,
        strategy=eng.strategy.key, entry_reason="test")
    eng.db.insert_trade(trade, eng.environment)
    trade["_live_price"] = entry
    trade["_entry_dt"] = None            # no time exit in these tests
    trade["_protected_stop"] = None
    trade["_margin"] = 0.0
    trade["_exit_state"] = engine_mod.exit_manager.ExitState(
        side=side, entry=entry, qty=qty, stop=stop, target=target,
        risk_dist=risk)
    eng.state.open_positions[inst.symbol] = trade
    return trade


@pytest.fixture
def inst():
    i = config.INSTRUMENTS_BY_SYMBOL.get("RELIANCE")
    if i is None:
        pytest.skip("RELIANCE is not in this deployment's instrument list")
    return i


BASE = config.params_for_mode(Mode.INTRADAY)


# --------------------------------------------------------------------------- #
#  C-2 — nothing configured, nothing changes.
# --------------------------------------------------------------------------- #
def test_unconfigured_strategy_exits_on_its_entry_target(inst):
    eng = _engine(config.apply_exit_style(BASE, "fixed"))
    _open(eng, inst)
    assert eng._manage_open(inst, 101.5, atr=0.4) is False   # nothing yet
    assert eng.broker.square_offs == []
    assert eng._manage_open(inst, 102.0, atr=0.4) is True
    # ONE square-off of the whole position, ONE closed document.
    assert eng.broker.square_offs == [("RELIANCE", "BUY", 4, 102.0)]
    assert len(eng.db.closed) == 1
    assert eng.db.closed[0]["exit_reason"] == "TARGET"
    assert eng.db.closed[0]["quantity"] == 4


def test_unconfigured_strategy_exits_on_its_entry_stop(inst):
    eng = _engine(config.apply_exit_style(BASE, "fixed"))
    trade = _open(eng, inst)
    assert eng._manage_open(inst, 97.9, atr=0.4) is True
    assert eng.broker.square_offs == [("RELIANCE", "BUY", 4, 98.0)]
    assert eng.db.closed[0]["exit_reason"] == "STOP-LOSS"
    # The stop never moved, so nothing was ever persisted over it.
    assert eng.db.docs[trade["trade_id"]]["stop_loss"] == 98.0


# --------------------------------------------------------------------------- #
#  Approach 2 through the real engine.
# --------------------------------------------------------------------------- #
def test_partial_books_a_child_trade_and_leaves_a_runner(inst):
    params = replace(config.apply_exit_style(BASE, "partial_trail"),
                     runner_rr_mult=3.0, trail_atr_mult=2.0)
    eng = _engine(params)
    trade = _open(eng, inst)

    closed = eng._manage_open(inst, 102.0, atr=0.2)
    assert closed is False, "the runner must still be open"

    # Only the booked lots were squared off — never the whole position.
    assert eng.broker.square_offs == [("RELIANCE", "BUY", 2, 102.0)]
    assert trade["quantity"] == 2
    assert inst.symbol in eng.state.open_positions

    # The partial is its OWN closed document, so every reader of `quantity`
    # sees a real 2-lot exit rather than a mutated parent.
    assert len(eng.db.closed) == 1
    child = eng.db.closed[0]
    assert child["quantity"] == 2
    assert child["exit_reason"] == "PARTIAL-TARGET"
    assert child["entry_price"] == 100.0
    assert child["trade_id"] != trade["trade_id"]

    # The parent was rewritten in place: runner target, break-even-plus stop
    # (or better, once the trail has run), and the partial marked spent so a
    # restart cannot book it twice.
    parent = eng.db.docs[trade["trade_id"]]
    assert parent["quantity"] == 2
    assert parent["partial_done"] is True
    assert parent["target"] == pytest.approx(106.0)
    assert parent["stop_loss"] >= 100.15


def test_runner_trails_and_every_stop_is_persisted(inst):
    params = replace(config.apply_exit_style(BASE, "partial_trail"),
                     runner_rr_mult=3.0, trail_atr_mult=2.0)
    eng = _engine(params)
    trade = _open(eng, inst)
    eng._manage_open(inst, 102.0, atr=0.2)          # partial + arm the trail

    stops = [trade["stop_loss"]]
    for px in (102.5, 103.0, 103.5, 104.0):
        eng._manage_open(inst, px, atr=0.2)
        stops.append(trade["stop_loss"])
        # What the bot manages to and what is stored can never disagree.
        assert eng.db.docs[trade["trade_id"]]["stop_loss"] == trade["stop_loss"]
    assert stops == sorted(stops)                   # ratchet only
    assert stops[-1] > stops[0]

    # Walk it back into the trailed stop: the runner exits, and the two legs
    # add back up to the four lots that were entered (C-5).
    assert eng._manage_open(inst, stops[-1] - 0.05, atr=0.2) is True
    assert sum(d["quantity"] for d in eng.db.closed) == 4
    assert len(eng.broker.square_offs) == 2
    assert eng.db.closed[-1]["exit_reason"] in {"TRAIL-PROFIT", "TRAIL-STOP",
                                                "BREAK-EVEN"}


def test_a_one_lot_position_runs_to_its_full_target(inst):
    """Immutable Rule #1's neighbour: a single lot is indivisible, so it takes
    the ordinary target exit rather than a partial that would be the whole
    position wearing a different name."""
    params = replace(config.apply_exit_style(BASE, "partial_trail"),
                     runner_rr_mult=3.0, trail_atr_mult=2.0)
    eng = _engine(params)
    _open(eng, inst, qty=1)
    eng._manage_open(inst, 102.0, atr=0.2)
    assert eng.broker.square_offs == []             # nothing booked at 1R
    assert not eng.db.closed


# --------------------------------------------------------------------------- #
#  Approach 1 through the real engine.
# --------------------------------------------------------------------------- #
def test_trail_from_entry_never_books_a_partial(inst):
    params = replace(config.apply_exit_style(BASE, "trail_full"),
                     trail_atr_mult=2.0)
    eng = _engine(params)
    trade = _open(eng, inst)
    for px in (100.5, 101.0, 101.5, 102.0, 103.0):
        eng._manage_open(inst, px, atr=0.2)
    assert eng.broker.square_offs == []             # never scaled out
    assert trade["quantity"] == 4
    assert trade["stop_loss"] > 100.0               # risk-free by now

    assert eng._manage_open(inst, trade["stop_loss"] - 0.05, atr=0.2) is True
    assert eng.broker.square_offs == [("RELIANCE", "BUY", 4,
                                       pytest.approx(trade["stop_loss"]))]
    assert len(eng.db.closed) == 1
    assert eng.db.closed[0]["quantity"] == 4
    assert eng.db.closed[0]["exit_reason"] == "TRAIL-PROFIT"


# --------------------------------------------------------------------------- #
#  Rehydration — a restart mid-runner must not re-partial.
# --------------------------------------------------------------------------- #
def test_a_rehydrated_runner_does_not_partial_again(inst):
    params = replace(config.apply_exit_style(BASE, "partial_trail"),
                     runner_rr_mult=3.0, trail_atr_mult=2.0)
    eng = _engine(params)
    trade = _open(eng, inst)
    eng._manage_open(inst, 102.0, atr=0.2)

    # What _rehydrate reconstructs: the persisted document plus the two flags,
    # and NO in-process exit state.
    doc = dict(eng.db.docs[trade["trade_id"]])
    doc.update(side="BUY", entry_price=100.0, ticker=inst.symbol,
               _live_price=100.0, _entry_dt=None, _protected_stop=None,
               _margin=0.0, _partial_done=True)
    doc.pop("_exit_state", None)

    eng2 = _engine(params)
    eng2.db.insert_trade(doc, eng2.environment)   # as _rehydrate reads it back
    eng2.state.open_positions[inst.symbol] = doc
    eng2._manage_open(inst, 106.5, atr=0.2)     # straight through the runner target
    # It exits — it does NOT book a second partial at the runner target.
    assert all(qty == doc["quantity"]
               for _sym, _side, qty, _px in eng2.broker.square_offs)
    assert [d["exit_reason"] for d in eng2.db.closed] == ["TARGET"]
