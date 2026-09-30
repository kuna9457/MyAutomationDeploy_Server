"""Approach 2c ("partial_ladder") — the owner's two-stage stop ladder.

The rule, as specified:

    Buy 100, TP 101, SL 99, qty 100.
    Case 1   price reaches TP (101): sell half (50), move the stop to
             buy + brokerage, and hold the rest for TP2.
    Case 1a  TP2 reached          -> exit the rest there.
    Case 1b  price falls back WITHOUT ever reaching the midpoint of TP..TP2
             -> the stop stays at buy + brokerage. Do not trail it anywhere.
    Case 1c  price reaches the MIDPOINT of TP..TP2 -> promote the stop to TP,
             then either TP2 pays or the promoted stop does.
    Case 2   SL hit before TP     -> exit the whole position.

What makes 2c different from 2b is only Case 1c's TIMING: 2b promotes the stop
to TP the instant the partial books, so any pullback to TP ends the runner. 2c
leaves it on break-even + friction until the runner has travelled half way to
TP2, so a dip-and-recover survives. Everything else — the partial, the buffer,
the second target, the absence of any ATR trailing — is shared with 2b, and is
covered by test_tp1_lock.py.
"""
from __future__ import annotations

import pytest

import config
import exit_manager as em
from config import Mode
from exit_manager import ActionType, ExitState


TICK = 0.05
ENTRY, RISK, QTY = 100.00, 1.00, 100        # buy 100, SL 99, TP 101
TP1, TP2 = 101.00, 102.00                   # 1R and 2R
MID = 101.50                                # midpoint of TP1..TP2


def _p(style="partial_ladder"):
    return em.params_from_strategy(
        config.apply_exit_style(config.params_for_mode(Mode.INTRADAY), style),
        TICK, 1)


def _st(side="BUY"):
    s = 1 if side == "BUY" else -1
    return ExitState(side=side, entry=ENTRY, qty=QTY,
                     stop=ENTRY - s * RISK, target=ENTRY + s * RISK,
                     risk_dist=RISK)


def _walk(st, p, prices, atr=0.0):
    """atr=0 is deliberate in most tests: with no ATR the volatility trigger
    cannot be computed and the machine falls back to `lock_trigger_frac`, so
    these exercise the FALLBACK path. The ATR trigger has its own section."""
    out = []
    for px in prices:
        if st.qty <= 0:
            break
        out += em.step_tick(st, round(px, 2), atr=atr, p=p)
    return out


#: The booked fraction and the runner it leaves, for qty=100.
BOOKED = int(QTY * config.LADDER_PARTIAL_FRACTION)      # 30
RUNNER = QTY - BOOKED                                   # 70


def _tp1(side):
    return ENTRY + (RISK if side == "BUY" else -RISK)


def _tp2(side):
    return ENTRY + (2 * RISK if side == "BUY" else -2 * RISK)


def _be_level():
    """buy + brokerage, the level the spec asks the stop to move to."""
    return ENTRY + config.BREAKEVEN_BUFFER_BPS / 1e4 * ENTRY


# --------------------------------------------------------------------------- #
#  Case 1 — the partial and the first stop move.
# --------------------------------------------------------------------------- #
def test_case1_books_the_partial_and_parks_the_stop_above_cost():
    p, st = _p(), _st()
    acts = _walk(st, p, [TP1])

    partial = [a for a in acts if a.kind == ActionType.PARTIAL]
    assert len(partial) == 1
    assert partial[0].qty == BOOKED and partial[0].price == pytest.approx(TP1)
    assert st.qty == RUNNER                   # the runner — 70%, not half
    assert st.target == pytest.approx(TP2)    # now waiting for TP2

    # Stop is at buy + brokerage — ABOVE the buy price, so a stop-out here is
    # a small profit rather than a scratch. NOT at TP1: that promotion is
    # Case 1c's job and has not been earned yet.
    move = [a for a in acts if a.kind == ActionType.MOVE_STOP]
    assert move and move[-1].reason == "break-even+buffer"
    assert ENTRY < st.stop < TP1
    assert st.stop == pytest.approx(_be_level(), abs=TICK)
    assert not [a for a in acts if a.kind == ActionType.EXIT]


def test_the_buffer_covers_the_round_trip_at_a_realistic_size():
    """'SL should be placed a little above bought price to cover brokerages'
    — checked against the actual cost model, on the RUNNER's own quantity, at
    a position size the buffer was derived for (~Rs1L notional)."""
    from cost_model import INTRADAY_EQUITY as cm
    p = _p()
    st = ExitState(side="BUY", entry=ENTRY, qty=1000, stop=ENTRY - RISK,
                   target=TP1, risk_dist=RISK)          # Rs1L notional
    _walk(st, p, [TP1])
    locked_per_share = st.stop - ENTRY
    # The RUNNER is what this stop protects — the 70% left after the partial,
    # not the 30% booked. A bigger runner spreads the flat fee further, so the
    # 30% split makes the buffer MORE than adequate, not less.
    runner = 1000 - int(1000 * config.LADDER_PARTIAL_FRACTION)
    runner_cost_per_share = cm.round_trip_cost(ENTRY, st.stop, runner) / runner
    assert locked_per_share > runner_cost_per_share - TICK


def test_at_a_small_position_size_no_buffer_can_cover_the_costs():
    """THE SIZE TRAP, pinned so nobody rediscovers it the expensive way.

    The flat Rs20-per-order brokerage does not scale with the trade. On a
    100-share position in a Rs100 stock the runner is 70 shares — Rs7,000 of
    notional — and its own round trip costs ~Rs0.77 a share, roughly 77 bps.
    That is nearly FOUR TIMES the 20 bps buffer, and three quarters of the
    entire 1R move the trade is aiming at. No "stop a little above the buy
    price" can cover costs at that size.

    (Booking 30% rather than 50% genuinely helps here — the larger runner
    spreads the flat fee over more shares, taking the per-share cost from
    ~Rs1.04 to ~Rs0.77. It does not come close to fixing it.)

    The exit manager does the only honest thing available: it clamps the stop
    one tick inside the printed price rather than inventing a level above the
    market. But the real fix is not in this file — it is a bigger position.
    """
    from cost_model import INTRADAY_EQUITY as cm
    runner_cost_per_share = cm.round_trip_cost(ENTRY, TP1, RUNNER) / RUNNER
    buffer_per_share = config.BREAKEVEN_BUFFER_BPS / 1e4 * ENTRY
    assert runner_cost_per_share > 3 * buffer_per_share, (
        "the runner's round trip should still dwarf the buffer at this size")
    p, st = _p(), _st()                                   # the owner's 100 sh
    _walk(st, p, [TP1])
    # Still above the buy price, still below the fill, still never above the
    # printed price — but nowhere near cost-free.
    assert ENTRY < st.stop <= TP1 - TICK + 1e-9
    assert (st.stop - ENTRY) < runner_cost_per_share


# --------------------------------------------------------------------------- #
#  Case 1a / 1b / 1c
# --------------------------------------------------------------------------- #
def test_case1a_reaching_tp2_exits_the_rest_there():
    p, st = _p(), _st()
    acts = _walk(st, p, [TP1, 101.4, MID, 101.8, TP2])
    ex = [a for a in acts if a.kind == ActionType.EXIT]
    assert len(ex) == 1
    assert ex[0].price == pytest.approx(TP2) and ex[0].qty == RUNNER
    assert ex[0].reason == "TARGET"
    booked = sum(a.qty for a in acts
                 if a.kind in (ActionType.PARTIAL, ActionType.EXIT))
    assert booked == QTY


def test_case1b_falling_back_before_the_midpoint_keeps_the_break_even_stop():
    """The runner never reached 101.50, so the stop must NOT have moved to
    101. It stops out at buy + brokerage — the whole point of the rule."""
    p, st = _p(), _st()
    acts = _walk(st, p, [TP1, 101.3, 101.45, 100.8, 100.3, _be_level() - 0.05])

    assert not any(a.reason == "lock TP1" for a in acts)
    ex = [a for a in acts if a.kind == ActionType.EXIT]
    assert len(ex) == 1
    assert ex[0].price == pytest.approx(st.stop)
    assert ex[0].price > ENTRY               # still a profit, after costs
    assert ex[0].price < TP1                 # but NOT the locked 1R
    assert ex[0].reason == "BREAK-EVEN"


def test_case1c_crossing_the_midpoint_promotes_the_stop_to_tp():
    p, st = _p(), _st()
    acts = _walk(st, p, [TP1, 101.3, MID])
    assert any(a.reason == "lock TP1" for a in acts)
    assert st.stop == pytest.approx(TP1)

    # ...and it STAYS there when price falls back — the level was earned.
    _walk(st, p, [101.8, 101.2])
    assert st.stop == pytest.approx(TP1)
    acts = _walk(st, p, [TP1])
    ex = [a for a in acts if a.kind == ActionType.EXIT]
    assert len(ex) == 1 and ex[0].price == pytest.approx(TP1)
    assert ex[0].reason == "TP1-LOCK"


def test_case1c_survives_a_dip_that_would_have_ended_2b():
    """The exact trade the two approaches disagree about: up to TP, back to
    just above break-even, then on to TP2. 2c collects 2R on the runner; 2b
    was stopped out at 1R on the way down."""
    path = [TP1, 101.2, 100.4, 100.9, MID, 101.9, TP2]

    st_c, p_c = _st(), _p("partial_ladder")
    acts_c = _walk(st_c, p_c, path)
    st_b, p_b = _st(), _p("partial_lock")
    acts_b = _walk(st_b, p_b, path)

    out_c = next(a for a in acts_c if a.kind == ActionType.EXIT)
    out_b = next(a for a in acts_b if a.kind == ActionType.EXIT)
    assert out_c.price == pytest.approx(TP2) and out_c.reason == "TARGET"
    assert out_b.price == pytest.approx(TP1) and out_b.reason == "TP1-LOCK"


def test_the_reverse_case_2b_wins():
    """The mirror, so neither approach is sold as strictly better: up to TP,
    then straight down without recovering. 2b banked the 1R; 2c gives it back
    to break-even."""
    path = [TP1, 101.2, 100.6, 100.1, 99.5]

    st_c, st_b = _st(), _st()
    out_c = next(a for a in _walk(st_c, _p("partial_ladder"), path)
                 if a.kind == ActionType.EXIT)
    out_b = next(a for a in _walk(st_b, _p("partial_lock"), path)
                 if a.kind == ActionType.EXIT)
    assert out_b.price == pytest.approx(TP1)          # kept the 1R
    assert ENTRY < out_c.price < TP1                  # gave it back
    assert out_c.reason == "BREAK-EVEN"


# --------------------------------------------------------------------------- #
#  Case 2 — and the invariants.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("side", ["BUY", "SELL"])
def test_case2_stop_before_the_target_closes_everything(side):
    p, st = _p(), _st(side)
    s = 1 if side == "BUY" else -1
    acts = _walk(st, p, [ENTRY + s * 0.4, ENTRY - s * RISK - s * 0.1])
    assert not [a for a in acts if a.kind == ActionType.PARTIAL]
    ex = [a for a in acts if a.kind == ActionType.EXIT]
    assert len(ex) == 1
    assert ex[0].qty == QTY                            # the WHOLE position
    assert ex[0].price == pytest.approx(ENTRY - s * RISK)
    assert ex[0].reason == "STOP-LOSS"


def test_short_is_the_exact_mirror():
    p, st = _p(), _st("SELL")
    acts = _walk(st, p, [99.0, 98.7, 98.5])            # TP, then the midpoint
    assert any(a.kind == ActionType.PARTIAL for a in acts)
    assert any(a.reason == "lock TP1" for a in acts)
    assert st.stop == pytest.approx(99.0)
    assert st.target == pytest.approx(98.0)


def test_it_never_trails_between_the_two_stages():
    """'if it fails to reach the midpoint then don't trail SL, keep the old
    one' — there must be exactly two stop levels, never a creeping third."""
    p, st = _p(), _st()
    stops = []
    for px in (TP1, 101.1, 101.2, 101.3, 101.4, 101.45):
        for a in em.step_tick(st, px, 5.0, p):        # a big ATR, deliberately
            if a.kind == ActionType.MOVE_STOP:
                stops.append(round(a.new_stop, 2))
    assert len(stops) == 1, f"the stop moved more than once: {stops}"


def test_other_styles_are_untouched():
    assert _p("partial_lock").lock_trigger_frac == 0.0
    assert _p("partial_trail").lock_first_target is False
    assert _p("fixed").partial_exit_fraction == 0.0
    assert config.CRUDEOIL_PARAMS.lock_trigger_frac == 0.0


# --------------------------------------------------------------------------- #
#  The VOLATILITY trigger — promote on ATR travelled, not on a share of the
#  TP1..TP2 span.
#
#  A fraction is a slice of a distance the strategy picked; it knows nothing
#  about how much the instrument is moving today. On a quiet session half a 1R
#  span can sit inside the spread, so the promotion fires on noise. ATR is a
#  direct reading of that noise, so the same multiple means the same thing on
#  every symbol.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("side", ["BUY", "SELL"])
def test_the_atr_trigger_fires_at_tp1_plus_n_atr(side):
    p, st = _p(), _st(side)
    sign = 1 if side == "BUY" else -1
    atr = 0.40
    want = _tp1(side) + sign * config.LOCK_TRIGGER_ATR * atr   # TP1 +/- 0.30

    # Just short of the trigger: booked, but NOT promoted.
    acts = _walk(st, p, [_tp1(side), want - sign * 0.05], atr=atr)
    assert any(a.kind == ActionType.PARTIAL for a in acts)
    assert not any(a.reason == "lock TP1" for a in acts)

    # At the trigger: promoted to TP1.
    acts = _walk(st, p, [want], atr=atr)
    assert any(a.reason == "lock TP1" for a in acts)
    assert st.stop == pytest.approx(_tp1(side))


def test_the_atr_trigger_beats_the_fraction_when_both_are_set():
    """Both are configured on this style; ATR must win whenever it can be
    computed, or the volatility reading would be decorative."""
    p = _p()
    assert p.lock_trigger_atr_mult > 0 and p.lock_trigger_frac > 0
    # Pick an ATR small enough that its trigger lands INSIDE the midpoint, so
    # the two rules genuinely disagree about this price. Derived from the
    # configured multiple rather than hardcoded, so it survives a retune.
    atr = 0.5 * (MID - TP1) / p.lock_trigger_atr_mult
    trigger = TP1 + p.lock_trigger_atr_mult * atr
    assert trigger < MID, "the test needs the ATR trigger inside the midpoint"
    st = _st()
    _walk(st, p, [TP1, trigger], atr=atr)
    assert st.stop == pytest.approx(TP1), "the fraction was used, not the ATR"


def test_a_bigger_atr_demands_a_bigger_move():
    """The whole point: on a noisier session the runner has to travel further
    before it gives up the right to retrace."""
    p = _p()
    quiet, wild = 0.10, 1.00
    # A price that clears the quiet session's trigger and not the wild one.
    price = TP1 + p.lock_trigger_atr_mult * quiet
    assert price < TP1 + p.lock_trigger_atr_mult * wild
    st_q, st_w = _st(), _st()
    _walk(st_q, p, [TP1, price], atr=quiet)
    _walk(st_w, p, [TP1, price], atr=wild)
    assert st_q.stop == pytest.approx(TP1)       # promoted
    assert st_w.stop < TP1                       # still on break-even


def test_no_atr_falls_back_to_the_fraction_rather_than_disarming():
    """A short series or a NaN must degrade to the old behaviour, not leave the
    runner sitting on break-even for ever."""
    p, st = _p(), _st()
    _walk(st, p, [TP1, 101.40], atr=0.0)         # short of the 101.50 midpoint
    assert st.stop < TP1
    _walk(st, p, [MID], atr=0.0)
    assert st.stop == pytest.approx(TP1)


def test_the_shipped_multiple_sits_at_the_edge_of_the_span_on_purpose():
    """The stop is itself ATR-derived, so with atr_sl_mult=1.5 and a 1:1
    target the whole TP1..TP2 span is ~1.5 x ATR. The shipped trigger sits AT
    that edge, which means the lock hardly ever fires — and that is the
    measured optimum, not an oversight.

    Over 8 names the result is monotonic in how often the lock fires: 0.25 x
    ATR (60% of runners locked) nets -0.38%, 1.5 x ATR (10%) nets -0.19%, and
    switching the lock off entirely also nets -0.19%. Locking at TP1 truncates
    the runners that would have reached TP2.

    This test exists so that anyone LOWERING the multiple has to come here and
    read why it was set high.
    """
    base = config.params_for_mode(Mode.INTRADAY)
    span_in_atr = base.risk_reward * base.atr_sl_mult      # TP1..TP2, in ATR
    assert config.LOCK_TRIGGER_ATR >= span_in_atr, (
        "the trigger was lowered below the TP1..TP2 span — that makes the "
        "lock fire often, which measured WORSE. See LOCK_TRIGGER_ATR.")
