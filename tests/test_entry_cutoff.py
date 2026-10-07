"""The entry cutoff, and the measured Candlestick Phase 1 defaults.

An entry taken with no session left cannot reach its target: it is squared off
flat having paid a full round trip. Measured on a real Intraday log, every
position opened on the last tradeable bar lost money — six for six, on a
combined +Rs24 of gross movement.

`entry_cutoff_before_close` has existed as a field since the CRUDEOIL strategy
was written, but only that strategy honoured it, so setting it on anything else
did nothing. It is now enforced centrally, in the backtester AND the live
runner, from one shared derivation.
"""
from __future__ import annotations

from dataclasses import replace
from datetime import time

import pytest

import config
from config import Mode, Segment


P = config.CANDLE_INTRADAY_PARAMS
# Candlestick Intraday no longer sets a cutoff (removed 2026-10-07), so the
# derivation is exercised on a copy that does — the mechanism is unchanged.
PC = replace(P, entry_cutoff_before_close=190)


# --------------------------------------------------------------------------- #
#  The derivation.
# --------------------------------------------------------------------------- #
def test_cutoff_is_derived_from_the_flat_out_not_configured_as_a_clock():
    assert config.square_off_time_for(Segment.EQUITY, Mode.INTRADAY) == time(15, 9)
    assert config.entry_cutoff_for(Segment.EQUITY, Mode.INTRADAY, PC) == time(11, 59)


def test_moving_the_square_off_moves_the_cutoff_with_it():
    """The right coupling: the runway a trade needs does not shrink just
    because the day was shortened."""
    got = config.entry_cutoff_for(Segment.EQUITY, Mode.INTRADAY, PC, "14:00")
    assert got == time(10, 50)                  # 14:00 - 190 min


def test_inert_for_a_strategy_that_sets_no_cutoff():
    assert config.entry_cutoff_for(Segment.EQUITY, Mode.INTRADAY,
                                   config.INTRADAY_PARAMS) is None
    for name in ("SWING_PARAMS", "SCALPER_VWAP_PARAMS", "CANDLE_SWING_PARAMS"):
        p = getattr(config, name)
        assert p.entry_cutoff_before_close == 0, name


def test_swing_has_no_cutoff_because_it_holds_overnight():
    assert config.entry_cutoff_for(Segment.EQUITY, Mode.SWING,
                                   replace(PC, mode=Mode.SWING)) is None


def test_crudeoil_keeps_the_time_its_own_strategy_already_enforced():
    """CRUDEOIL honoured this field internally before it was central. The
    central gate must agree with it, not fight it — 23:15 MCX flat-out minus
    45 minutes is the ~22:30 its own comment names."""
    assert config.entry_cutoff_for(Segment.MCX, Mode.INTRADAY,
                                   config.CRUDEOIL_PARAMS) == time(22, 30)


# --------------------------------------------------------------------------- #
#  Both halves enforce it. A cutoff only one of them honoured would make the
#  backtest describe a bot that does not exist.
# --------------------------------------------------------------------------- #
def test_candlestick_intraday_has_no_cutoff_any_more():
    """Removed on request 2026-10-07: entries run to the session end."""
    assert config.entry_cutoff_for(Segment.EQUITY, Mode.INTRADAY, P) is None


def test_the_live_runner_enforces_it_too():
    import inspect

    import strategy_runner
    src = inspect.getsource(strategy_runner.StrategyRunner._poll_once
                            if hasattr(strategy_runner.StrategyRunner, "_poll_once")
                            else strategy_runner.StrategyRunner)
    assert "entry_cutoff_for" in src, (
        "the live runner must gate entries on the same cutoff the backtest does")


# --------------------------------------------------------------------------- #
#  The measured defaults, pinned. These are not taste — each one was chosen
#  from a backtest recorded in the CLAUDE.md changelog, and a silent change to
#  any of them invalidates that measurement.
# --------------------------------------------------------------------------- #
def test_candlestick_intraday_carries_the_measured_settings():
    # cs_min_score is a base floor; the admin picks the real one in the panel.
    assert P.cs_min_score == 3.0
    assert P.entry_cutoff_before_close == 0     # cutoff removed 2026-10-07
    assert P.min_stop_pct == 0.8            # widen stops, +0.19pp
    assert P.max_stop_pct == 0.0            # capping them costs ~2pp — leave it


def test_none_of_it_touched_risk_per_trade():
    """Immutable Rule #1: these settings change WHICH trades are taken and
    where their levels sit. They must not have moved the risk cap."""
    assert P.risk_per_trade == 0.01         # 1%, inside the 2% ceiling
    assert P.risk_reward == 1.0
    assert P.max_leverage == 15.0
