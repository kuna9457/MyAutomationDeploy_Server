"""Automatic MCX contract roll (mcx_rollover.py).

The safety property under test: a symbol's instrument_key is swapped ONLY when
nothing still holds the old contract — no open trade, no running runner —
because every exit path looks the key up by symbol and would otherwise send
the closing order to the NEW contract.
"""
from __future__ import annotations

import json
from dataclasses import replace
from datetime import date, datetime, timedelta

import pytest

import config
import mcx_rollover
import strategy_runner
from config import Instrument, Segment


def _ms(d: date) -> float:
    return datetime(d.year, d.month, d.day, 23, 59,
                    tzinfo=config.IST).timestamp() * 1000


TODAY = config.now_ist().date()
OLD = Instrument("CRUDEOIL", Segment.MCX, "MCX_FO|OLD", 100, 1.0, 7580.0, 100,
                 expiry=(TODAY + timedelta(days=2)).isoformat())
NEXT_EXP = TODAY + timedelta(days=33)
MASTER = [
    # the dying contract itself — inside the window, must be skipped
    {"instrument_type": "FUT", "asset_symbol": "CRUDEOIL",
     "instrument_key": "MCX_FO|OLD", "expiry": _ms(TODAY + timedelta(days=2)),
     "lot_size": 100, "tick_size": 100.0, "qty_multiplier": 100},
    # the one after next — must lose to the nearer one
    {"instrument_type": "FUT", "asset_symbol": "CRUDEOIL",
     "instrument_key": "MCX_FO|FAR", "expiry": _ms(TODAY + timedelta(days=64)),
     "lot_size": 100, "tick_size": 100.0, "qty_multiplier": 100},
    {"instrument_type": "FUT", "asset_symbol": "CRUDEOIL",
     "instrument_key": "MCX_FO|NEXT", "expiry": _ms(NEXT_EXP),
     "lot_size": 100, "tick_size": 100.0, "qty_multiplier": 100},
    # an option on the same root — never a candidate
    {"instrument_type": "CE", "asset_symbol": "CRUDEOIL",
     "instrument_key": "MCX_FO|OPT", "expiry": _ms(TODAY + timedelta(days=20))},
]


@pytest.fixture
def registry(monkeypatch, tmp_path):
    """A registry holding only OLD, restored afterwards; rolls saved to tmp."""
    monkeypatch.setenv("MCX_AUTO_ROLL", "true")
    saved = (list(config.MCX_INSTRUMENTS), list(config.ALL_INSTRUMENTS),
             dict(config.INSTRUMENTS_BY_SYMBOL))
    config.MCX_INSTRUMENTS[:] = [OLD]
    config.ALL_INSTRUMENTS[:] = [i for i in saved[1] if i.segment != Segment.MCX] + [OLD]
    config.INSTRUMENTS_BY_SYMBOL["CRUDEOIL"] = OLD
    rolls = tmp_path / "mcx_rolls.json"
    monkeypatch.setattr(mcx_rollover, "ROLLS_FILE", str(rolls))
    monkeypatch.setattr(mcx_rollover, "_fetch_master", lambda force=False: MASTER)
    monkeypatch.setattr(mcx_rollover, "_open_symbols", lambda: set())
    yield rolls
    config.MCX_INSTRUMENTS[:] = saved[0]
    config.ALL_INSTRUMENTS[:] = saved[1]
    config.INSTRUMENTS_BY_SYMBOL.clear()
    config.INSTRUMENTS_BY_SYMBOL.update(saved[2])


def test_picks_the_nearest_future_beyond_the_window():
    new = mcx_rollover.pick_next(OLD, MASTER, TODAY)
    assert new.instrument_key == "MCX_FO|NEXT"
    assert new.expiry == NEXT_EXP.isoformat()
    assert new.tick_size == 1.0            # master is in paise
    assert (new.symbol, new.lot_size, new.contract_multiplier) == ("CRUDEOIL", 100, 100)


def test_rolls_a_due_contract_and_saves_it(registry):
    rep = mcx_rollover.roll_if_due()
    assert len(rep["rolled"]) == 1 and not rep["deferred"]
    assert config.INSTRUMENTS_BY_SYMBOL["CRUDEOIL"].instrument_key == "MCX_FO|NEXT"
    assert config.MCX_INSTRUMENTS[0].instrument_key == "MCX_FO|NEXT"
    assert any(i.instrument_key == "MCX_FO|NEXT" for i in config.ALL_INSTRUMENTS)
    saved = json.loads(registry.read_text())
    assert saved["CRUDEOIL"]["instrument_key"] == "MCX_FO|NEXT"


def test_never_rolls_under_an_open_position(registry, monkeypatch):
    monkeypatch.setattr(mcx_rollover, "_open_symbols", lambda: {"CRUDEOIL"})
    rep = mcx_rollover.roll_if_due()
    assert not rep["rolled"] and len(rep["deferred"]) == 1
    assert config.INSTRUMENTS_BY_SYMBOL["CRUDEOIL"].instrument_key == "MCX_FO|OLD"
    assert not registry.exists()


def test_defers_everything_when_open_trades_cannot_be_read(registry, monkeypatch):
    monkeypatch.setattr(mcx_rollover, "_open_symbols", lambda: None)
    rep = mcx_rollover.roll_if_due()
    assert not rep["rolled"]
    assert config.INSTRUMENTS_BY_SYMBOL["CRUDEOIL"].instrument_key == "MCX_FO|OLD"


def test_never_rolls_underneath_a_running_bot(registry, monkeypatch):
    class _Runner:
        instruments = [OLD]
        _accounts = []
    monkeypatch.setitem(strategy_runner._runners, "test-runner", _Runner())
    rep = mcx_rollover.roll_if_due()
    assert not rep["rolled"] and len(rep["deferred"]) == 1
    assert config.INSTRUMENTS_BY_SYMBOL["CRUDEOIL"].instrument_key == "MCX_FO|OLD"


def test_nothing_due_means_no_download(registry, monkeypatch):
    fresh = replace(OLD, expiry=(TODAY + timedelta(days=30)).isoformat())
    config.MCX_INSTRUMENTS[:] = [fresh]

    def boom(force=False):
        raise AssertionError("downloaded the master with nothing due")
    monkeypatch.setattr(mcx_rollover, "_fetch_master", boom)
    assert mcx_rollover.roll_if_due() == {"rolled": [], "deferred": [], "unavailable": []}


def test_disabled_flag_is_a_no_op(registry, monkeypatch):
    monkeypatch.setenv("MCX_AUTO_ROLL", "false")
    assert not mcx_rollover.roll_if_due()["rolled"]
    assert config.INSTRUMENTS_BY_SYMBOL["CRUDEOIL"].instrument_key == "MCX_FO|OLD"


def test_current_reresolves_a_stale_lookup_to_the_rolled_contract(registry):
    eq = next(i for i in config.ALL_INSTRUMENTS if i.segment == Segment.EQUITY)
    out = mcx_rollover.current([OLD, eq])
    assert out[0].instrument_key == "MCX_FO|NEXT"
    assert out[1] is eq


def test_config_overlay_keeps_the_later_expiry(tmp_path, monkeypatch):
    f = tmp_path / "mcx_rolls.json"
    f.write_text(json.dumps({"CRUDEOIL": {
        "instrument_key": "MCX_FO|NEXT", "lot_size": 100, "tick_size": 1.0,
        "contract_multiplier": 100, "expiry": NEXT_EXP.isoformat()}}))
    monkeypatch.setattr(config, "MCX_ROLLS_FILE", str(f))
    assert config._apply_saved_mcx_rolls([OLD])[0].instrument_key == "MCX_FO|NEXT"
    # a newer committed contract wins over an older saved roll
    newer = replace(OLD, instrument_key="MCX_FO|COMMITTED",
                    expiry=(NEXT_EXP + timedelta(days=30)).isoformat())
    assert config._apply_saved_mcx_rolls([newer])[0].instrument_key == "MCX_FO|COMMITTED"
