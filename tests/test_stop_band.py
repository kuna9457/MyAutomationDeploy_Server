"""The stop-distance band — bounding the ATR stop to a percent of price.

The complaint it answers: "sometimes it selects a way too high target which
does not hit on either side and exits sideways." Measured on a real Intraday
log, one symbol produced risk distances from 0.64% to 3.9% of price. At 3.9%
a 1:1 target is a 3.9% move, which an ordinary session will not deliver, so
the trade resolves neither way and dies at the square-off.

The band bounds the answer WITHOUT changing how it is computed: the strategy
still derives the stop from ATR (and its structural swing, where it uses one),
and the target moves with the stop so the reward:risk is untouched.
"""
from __future__ import annotations

from dataclasses import replace

import pytest

import config
from config import Mode
from strategy import Signal, clamp_signal_risk


def _params(**kw):
    return replace(config.params_for_mode(Mode.INTRADAY), **kw)


def _sig(side="BUY", entry=1100.0, risk=40.0, rr=1.0):
    s = 1 if side == "BUY" else -1
    return Signal(side=side, entry_price=entry, stop_loss=entry - s * risk,
                  target=entry + s * rr * risk, reason="test")


# --------------------------------------------------------------------------- #
#  Inert unless configured.
# --------------------------------------------------------------------------- #
def test_no_band_returns_the_very_same_object():
    """Not an equal copy — the SAME object, so an unconfigured strategy is
    provably on the identical code path it was before this existed."""
    sig = _sig()
    assert clamp_signal_risk(sig, _params()) is sig


def test_none_passes_through():
    assert clamp_signal_risk(None, _params(max_stop_pct=1.0)) is None


def test_a_stop_already_inside_the_band_is_untouched():
    sig = _sig(entry=1100.0, risk=6.0)          # 0.55% of price
    assert clamp_signal_risk(sig, _params(max_stop_pct=1.0)) is sig


# --------------------------------------------------------------------------- #
#  The owner's example: INFY at 1100, cap at 1%.
# --------------------------------------------------------------------------- #
def test_a_too_wide_stop_is_capped_to_one_percent():
    sig = _sig(side="BUY", entry=1100.0, risk=40.0, rr=1.0)   # 3.64% of price
    out = clamp_signal_risk(sig, _params(max_stop_pct=1.0))
    assert out.stop_loss == pytest.approx(1089.0)   # 1100 - 1%
    assert out.target == pytest.approx(1111.0)      # 1100 + 1%, RR preserved
    assert out.entry_price == sig.entry_price
    assert out.side == sig.side and out.reason == sig.reason


def test_the_short_side_is_the_exact_mirror():
    """'it should be same replicated in sell side as well'."""
    sig = _sig(side="SELL", entry=1100.0, risk=40.0, rr=1.0)
    out = clamp_signal_risk(sig, _params(max_stop_pct=1.0))
    assert out.stop_loss == pytest.approx(1111.0)   # stop ABOVE for a short
    assert out.target == pytest.approx(1089.0)      # target BELOW
    assert out.stop_loss > out.entry_price > out.target


@pytest.mark.parametrize("rr", [0.5, 1.0, 1.5, 3.0])
@pytest.mark.parametrize("side", ["BUY", "SELL"])
def test_the_reward_to_risk_is_preserved_exactly(rr, side):
    """Moving the stop without moving the target would silently re-rate every
    trade the clamp touched — a 1:3 swing quietly becoming 1:11."""
    sig = _sig(side=side, entry=1100.0, risk=40.0, rr=rr)
    out = clamp_signal_risk(sig, _params(max_stop_pct=1.0))
    assert out.risk_reward == pytest.approx(rr, rel=1e-3)
    assert sig.risk_reward == pytest.approx(rr, rel=1e-3)


def test_the_floor_widens_a_stop_that_is_inside_the_noise():
    sig = _sig(entry=1100.0, risk=1.10)             # 0.1% — inside the spread
    out = clamp_signal_risk(sig, _params(min_stop_pct=0.4))
    assert out.stop_loss == pytest.approx(1095.6)   # 1100 - 0.4%
    assert out.target == pytest.approx(1104.4)


def test_the_cap_wins_over_a_misconfigured_floor():
    """min above max is a misconfiguration; letting the floor push the stop
    past the cap would defeat the only thing the cap is there to do."""
    sig = _sig(entry=1100.0, risk=40.0)
    out = clamp_signal_risk(sig, _params(min_stop_pct=2.0, max_stop_pct=1.0))
    assert abs(out.entry_price - out.stop_loss) == pytest.approx(11.0)


@pytest.mark.parametrize("bad", [
    dict(entry=1100.0, risk=0.0),        # degenerate: stop on entry
    dict(entry=0.0, risk=5.0),           # degenerate: no price
])
def test_degenerate_signals_are_passed_through_untouched(bad):
    """The clamp must not manufacture a level out of a broken signal; the
    caller (engine._enter) already rejects these with a real message."""
    sig = _sig(**bad)
    assert clamp_signal_risk(sig, _params(max_stop_pct=1.0)) is sig


# --------------------------------------------------------------------------- #
#  Both callers run it — the whole point of putting it in one function.
# --------------------------------------------------------------------------- #
def test_backtest_honours_the_band_end_to_end():
    import backtester
    if "INFY" not in config.INSTRUMENTS_BY_SYMBOL:
        pytest.skip("INFY is not in this deployment's instrument list")
    kw = dict(strategy_key="candlestick_engine", exit_style="partial_ladder",
              min_score=7.0, include_costs=True)
    wide = backtester.run_backtest("INFY", "2026-01-01", "2026-08-31",
                                   200_000.0, Mode.INTRADAY, **kw).trades
    capped = backtester.run_backtest("INFY", "2026-01-01", "2026-08-31",
                                     200_000.0, Mode.INTRADAY,
                                     max_stop_pct=0.5, **kw).trades
    if wide.empty or capped.empty:
        pytest.skip("no trades in that window")
    wide_pct = (wide.risk_dist / wide.entry * 100)
    capped_pct = (capped.risk_dist / capped.entry * 100)
    assert wide_pct.max() > 0.5, "the window has no stop wide enough to test"
    # Every capped trade is inside the band, allowing for the tick rounding
    # engine._enter and the strategies apply.
    assert capped_pct.max() <= 0.5 + 0.02


def test_live_path_runs_the_same_clamp():
    """strategy.run_strategy is what the engine's runner calls; it must apply
    the identical function the backtester does, or the cap would be a
    backtest-only fiction."""
    import inspect

    import strategy
    src = inspect.getsource(strategy.run_strategy)
    assert "clamp_signal_risk" in src
    import backtester
    assert "clamp_signal_risk" in inspect.getsource(backtester.run_backtest)
