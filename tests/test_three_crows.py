"""Identical Three Crows (newstrategy.md), as registered by
strategies/three_crows_engine.py.

Two things are being asserted, and the second matters more than the first:

  1. The STRATEGY is the script — the pattern's geometry, the simple-mean ATR
     stop, short-only, TP1 at 1R, and both of the script's exit rules reachable
     through the exit-style control that already exists.

  2. It reached NOTHING ELSE. A new strategy that quietly re-tunes a shared
     default would be worse than no strategy at all: every recorded backtest
     number and every running preset would move underneath the operator with
     no way to tell. The last section pins the shared values this file could
     plausibly have touched.
"""
from __future__ import annotations

from dataclasses import replace
from datetime import time

import pandas as pd
import pytest

import config
import exit_manager as em
import strategy
from config import Mode, Segment
from strategies import three_crows_engine as tc

KEY = "identical_three_crows"
TICK = 0.05
IDX = pd.date_range("2026-09-01 09:15", periods=64, freq="15min")


def _frame(tail):
    """Flat warm-up bars with `tail` (open, high, low, close) tuples appended."""
    rows = [(1000.0, 1003.0, 997.0, 1000.0)] * (40 - len(tail)) + list(tail)
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close"],
                      index=IDX[:len(rows)])
    df["volume"] = 100_000.0
    return df


#: c1 1000 -> 994, c2 opens 994.5 (0.05% away) -> 988, c3 opens 988.2 -> 982.
CROWS = [(1000.0, 1001.0, 993.0, 994.0),
         (994.5, 995.0, 987.0, 988.0),
         (988.2, 988.5, 981.0, 982.0)]


# --------------------------------------------------------------------------- #
#  1. Registration
# --------------------------------------------------------------------------- #
def test_it_is_registered_and_the_plugin_loader_is_clean():
    assert not strategy.PLUGIN_ERRORS, strategy.PLUGIN_ERRORS
    assert KEY in {s.key for s in strategy.all_strategies()}


def test_intraday_and_scalper_only():
    """SWING is deliberately absent: an overnight NSE cash short is a delivery
    you cannot make, so registering a short-only strategy there would offer a
    mode that can only ever return zero trades."""
    sd = next(s for s in strategy.all_strategies() if s.key == KEY)
    assert set(sd.modes) == {Mode.INTRADAY, Mode.SCALPER}


def test_it_does_not_claim_to_use_the_signal_score():
    """One pattern IS the signal, so cs_min_score is never read. Declaring
    otherwise would put a control in the admin panel that silently does
    nothing."""
    sd = next(s for s in strategy.all_strategies() if s.key == KEY)
    assert sd.uses_min_score is False


# --------------------------------------------------------------------------- #
#  2. The pattern
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("tail,want", [
    (CROWS, True),
    # third candle closes green
    (CROWS[:2] + [(988.2, 995.0, 987.0, 992.0)], False),
    # third candle opens 1% above the second's close — a bounce happened
    (CROWS[:2] + [(998.0, 998.5, 981.0, 982.0)], False),
    # closes do not step strictly down
    (CROWS[:2] + [(988.2, 990.0, 987.5, 989.0)], False),
    # second candle gaps well below the first's close
    ([CROWS[0], (980.0, 981.0, 974.0, 975.0), (975.1, 975.5, 969.0, 970.0)],
     False),
])
def test_geometry(tail, want):
    df = _frame(tail)
    assert tc.is_identical_three_crows(df.iloc[-3], df.iloc[-2],
                                       df.iloc[-1]) is want


def test_tolerance_is_a_fraction_of_PRICE_not_of_the_candle_range():
    """The script's unit, and it is NOT candlestick_engine's. That engine
    measures the same gap against the candle range; this one against the close.
    A 0.3% gap must fail at the 0.2% default and pass when the tolerance is
    widened, which is only true if the unit is price."""
    tail = CROWS[:2] + [(985.0, 988.5, 981.0, 982.0)]     # 988 -> 985 = 0.30%
    df = _frame(tail)
    c1, c2, c3 = df.iloc[-3], df.iloc[-2], df.iloc[-1]
    assert tc.is_identical_three_crows(c1, c2, c3, 0.002) is False
    assert tc.is_identical_three_crows(c1, c2, c3, 0.004) is True


# --------------------------------------------------------------------------- #
#  3. The signal, through the LIVE path (enrich + clamp)
# --------------------------------------------------------------------------- #
def _signal(df, params=None):
    sd = strategy.resolve_strategy(Mode.INTRADAY, KEY)
    if params is not None:
        sd = replace(sd, params=params)
    return strategy.run_strategy(sd, df)


def test_short_at_the_third_close_with_a_1_5_atr_stop_and_a_1R_target():
    df = _frame(CROWS)
    sig = _signal(df)
    assert sig is not None and sig.side == "SELL"
    atr = tc.sma_atr(df, 14)
    r = sig.stop_loss - sig.entry_price
    assert sig.entry_price == pytest.approx(float(df.iloc[-1]["close"]))
    assert r == pytest.approx(1.5 * atr)
    assert sig.entry_price - sig.target == pytest.approx(r)   # TP1 = 1R


def test_the_atr_is_the_simple_mean_not_wilder():
    """A real difference, not a rounding one: strategy.atr() is an EMA with
    alpha = 1/N and reacts slower, so the two put the stop in different
    places. The script measures the simple mean, so that is what is traded."""
    df = _frame(CROWS)
    assert tc.sma_atr(df, 14) != pytest.approx(
        float(strategy.atr(df, 14).iloc[-1]), abs=1e-6)


def test_nothing_fires_on_flat_tape():
    assert _signal(_frame([(1000.0, 1003.0, 997.0, 1000.0)] * 3)) is None


def test_short_only_even_if_allow_short_is_turned_off():
    """`allow_short=False` must mean "do not sell", never "sell anyway" — an
    admin can reach that field, and a strategy with no long reading has to
    stand down rather than invent one."""
    p = replace(strategy.resolve_strategy(Mode.INTRADAY, KEY).params,
                allow_short=False)
    assert _signal(_frame(CROWS), p) is None


# --------------------------------------------------------------------------- #
#  4. Exit rule 1 — the script's "Trail SL TP", run on the SHARED machine.
#
#  entry 1000, R = 10  =>  SL 1010, TP1 990, midpoint 985, TP2 980.
# --------------------------------------------------------------------------- #
ENTRY, RISK, QTY = 1000.0, 10.0, 100
TP1, TP2, MID = 990.0, 980.0, 985.0
BE = 999.0                                   # entry * (1 - 0.001), the script's


def _ep(style="strategy"):
    p = strategy.resolve_strategy(Mode.INTRADAY, KEY).params
    return em.params_from_strategy(config.apply_exit_style(p, style), TICK, 1)


def _walk(prices, qty=QTY, style="strategy"):
    st = em.ExitState(side="SELL", entry=ENTRY, qty=qty, stop=ENTRY + RISK,
                      target=TP1, risk_dist=RISK)
    p, out = _ep(style), []
    for px in prices:
        if st.qty <= 0:
            break
        out += em.step_tick(st, px, atr=0.0, p=p)
    return st, out


def test_the_break_even_stop_is_the_scripts_entry_times_0_999():
    _, acts = _walk([TP1])
    moves = [a for a in acts if a.kind is em.ActionType.MOVE_STOP]
    assert moves and moves[0].new_stop == pytest.approx(BE)
    assert BE == pytest.approx(ENTRY * (1 - 0.001))


def test_case_2_stop_before_TP1_closes_the_whole_position():
    _, acts = _walk([1005.0, 1010.0])
    assert [a.kind for a in acts] == [em.ActionType.EXIT]
    assert acts[0].qty == QTY and acts[0].reason == "STOP-LOSS"


def test_case_1_TP1_books_half_and_leaves_a_runner():
    st, acts = _walk([TP1])
    part = [a for a in acts if a.kind is em.ActionType.PARTIAL]
    assert len(part) == 1 and part[0].qty == QTY // 2
    assert st.qty == QTY - QTY // 2


def test_case_1b_a_stall_short_of_the_midpoint_pays_only_break_even():
    """The stop must NOT be trailed anywhere between TP1 and the midpoint —
    that is the whole difference between this rule and locking at TP1."""
    _, acts = _walk([TP1, 988.0, 995.0, 999.5])
    ex = [a for a in acts if a.kind is em.ActionType.EXIT]
    assert ex and ex[-1].price == pytest.approx(BE)
    assert not any(a.kind is em.ActionType.MOVE_STOP and a.reason == "lock TP1"
                   for a in acts)


def test_case_1c_the_midpoint_promotes_the_stop_to_TP1():
    _, acts = _walk([TP1, MID, TP1])
    assert any(a.kind is em.ActionType.MOVE_STOP and a.reason == "lock TP1"
               and a.new_stop == pytest.approx(TP1) for a in acts)
    ex = [a for a in acts if a.kind is em.ActionType.EXIT]
    assert ex and ex[-1].price == pytest.approx(TP1)
    assert ex[-1].reason == "TP1-LOCK"


def test_case_1a_TP2_pays_the_runner():
    _, acts = _walk([TP1, MID, TP2])
    ex = [a for a in acts if a.kind is em.ActionType.EXIT]
    assert ex and ex[-1].price == pytest.approx(TP2) and ex[-1].reason == "TARGET"


def test_a_dip_and_recover_before_the_midpoint_survives():
    """The reason the promotion is delayed at all. Under an immediate TP1 lock
    this runner would have been squared off at 990 on the second price."""
    _, acts = _walk([TP1, 991.0, 995.0, MID, TP2])
    ex = [a for a in acts if a.kind is em.ActionType.EXIT]
    assert ex and ex[-1].price == pytest.approx(TP2)


def test_one_indivisible_unit_is_never_split():
    """The script's `is_single_lot`: nothing to halve, so the stop still moves
    to break-even at TP1 and the whole unit runs for TP2."""
    _, acts = _walk([TP1, MID, TP2], qty=1)
    assert not any(a.kind is em.ActionType.PARTIAL for a in acts)
    ex = [a for a in acts if a.kind is em.ActionType.EXIT]
    assert ex and ex[-1].qty == 1 and ex[-1].price == pytest.approx(TP2)


# --------------------------------------------------------------------------- #
#  5. Exit rule 2 — "Fixed RR", i.e. the existing `fixed` style.
# --------------------------------------------------------------------------- #
def test_fixed_style_is_all_in_all_out():
    ep = _ep("fixed")
    assert ep.partial_exit_fraction == 0.0
    assert ep.trail_remainder is False and ep.trail_from_entry is False
    st = em.ExitState(side="SELL", entry=ENTRY, qty=QTY, stop=ENTRY + RISK,
                      target=TP2, risk_dist=RISK)
    acts = []
    for px in (995.0, 985.0, TP2):
        acts += em.step_tick(st, px, atr=0.0, p=ep)
    assert len(acts) == 1
    assert acts[0].kind is em.ActionType.EXIT and acts[0].qty == QTY
    assert acts[0].reason == "TARGET"


@pytest.mark.parametrize("rr", config.RR_CHOICES)
def test_the_RR_control_moves_the_target_and_nothing_else(rr):
    """RR moves the TARGET only (Immutable Rule #1). The stop is ATR's, so the
    risk distance — and therefore the position size — is identical at every
    ratio."""
    p = replace(strategy.resolve_strategy(Mode.INTRADAY, KEY).params,
                risk_reward=rr)
    sig = _signal(_frame(CROWS), p)
    assert sig is not None
    r = sig.stop_loss - sig.entry_price
    assert r == pytest.approx(1.5 * tc.sma_atr(_frame(CROWS), 14))
    assert sig.entry_price - sig.target == pytest.approx(rr * r)


# --------------------------------------------------------------------------- #
#  6. Session rules
# --------------------------------------------------------------------------- #
def test_no_new_entries_after_15_00():
    """The script's `if current_time.hour >= 15: continue`, expressed the way
    this bot expresses it — derived from the flat-out, so shortening the day
    moves it too."""
    p = strategy.resolve_strategy(Mode.INTRADAY, KEY).params
    assert config.entry_cutoff_for(Segment.EQUITY, Mode.INTRADAY, p) == time(15, 0)
    assert config.entry_cutoff_for(Segment.EQUITY, Mode.INTRADAY, p,
                                   "14:00") == time(13, 51)


# --------------------------------------------------------------------------- #
#  7. IT REACHED NOTHING ELSE.
#
#  The point of the whole file. These are the shared values this strategy
#  could plausibly have edited on its way in; if any of them moves, every
#  recorded backtest and every running preset moves with it.
# --------------------------------------------------------------------------- #
def test_the_shipped_exit_styles_are_untouched():
    ladder = config._EXIT_STYLE_FIELDS["partial_ladder"]
    assert ladder["partial_exit_fraction"] == config.LADDER_PARTIAL_FRACTION == 0.30
    assert ladder["breakeven_buffer_bps"] == config.BREAKEVEN_BUFFER_BPS == 20.0
    assert ladder["lock_trigger_atr_mult"] == config.LOCK_TRIGGER_ATR == 1.5
    assert config._EXIT_STYLE_FIELDS["partial_lock"]["lock_trigger_frac"] == 0.0
    assert config.RUNNER_TARGET_RR == 2.0
    assert set(config.EXIT_STYLES) == {"strategy", "fixed", "trail_full",
                                       "partial_trail", "partial_lock",
                                       "partial_ladder"}


def test_the_candlestick_strategies_are_untouched():
    assert config.CANDLE_INTRADAY_PARAMS.cs_min_score == 3.0
    assert config.CANDLE_INTRADAY_PARAMS.entry_cutoff_before_close == 0
    assert config.CANDLE_INTRADAY_PARAMS.min_stop_pct == 0.8
    assert config.CANDLE_INTRADAY_PARAMS.partial_exit_fraction == 0.0
    assert config.CANDLE_SWING_PARAMS.allow_short is False


def test_this_strategy_carries_its_own_exit_block_rather_than_a_shared_one():
    """It configures 2c on ITS OWN params — 50% at 10 bps on the midpoint
    trigger — which is a different experiment from the shipped `partial_ladder`
    and must not be confused with it."""
    p = strategy.resolve_strategy(Mode.INTRADAY, KEY).params
    assert p.partial_exit_fraction == 0.5
    assert p.breakeven_buffer_bps == 10.0
    assert p.breakeven_buffer_ticks == 0.0
    assert p.runner_rr_mult == 2.0
    assert p.lock_first_target is True
    assert p.lock_trigger_frac == 0.5
    assert p.lock_trigger_atr_mult == 0.0      # the midpoint, not volatility
    assert p.trail_remainder is False and p.trail_from_entry is False


def test_risk_stays_inside_immutable_rule_1():
    for mode in (Mode.INTRADAY, Mode.SCALPER):
        p = strategy.resolve_strategy(mode, KEY).params
        assert p.risk_per_trade == 0.01 <= 0.02
        assert p.max_stop_pct == 0.0 and p.min_stop_pct == 0.0
        assert config.is_valid_rr(p.risk_reward)
