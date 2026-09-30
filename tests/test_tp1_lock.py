"""Approach 2b — book half at TP1, then LOCK the runner's floor at TP1.

The rule, stated as the owner stated it:

    take the trade at 1:1 with TP and SL marked; at TP1 book half; move to TP2
    and wait. If price goes above TP1, does NOT reach TP2, and comes back to
    TP1 — square the rest off there.

So TP1, not break-even, is the runner's floor: the 1R already earned can never
be handed back. What makes this non-trivial is that the partial fills AT TP1,
so a stop naively set to TP1 on that same tick fires immediately and the
runner never exists at all. The stop is clamped one tick inside the printed
price on the event tick and raised to TP1 exactly once price moves away —
which is also what keeps the eventual fill honest, since price must trade
back through TP1 to trigger it.
"""
from __future__ import annotations

import pytest

import config
import exit_manager as em
from config import Mode
from exit_manager import ActionType, ExitState


TICK = 0.05
ENTRY = 1000.0
RISK = 10.0                      # 1R; RR is 1:1 so TP1 = entry +/- 10
QTY = 10


def _p(style="partial_lock"):
    return em.params_from_strategy(
        config.apply_exit_style(config.params_for_mode(Mode.INTRADAY), style),
        TICK, 1)


def _st(side="BUY"):
    sign = 1 if side == "BUY" else -1
    return ExitState(side=side, entry=ENTRY, qty=QTY,
                     stop=ENTRY - sign * RISK, target=ENTRY + sign * RISK,
                     risk_dist=RISK)


def _tp1(side):
    return ENTRY + (RISK if side == "BUY" else -RISK)


def _tp2(side):
    return ENTRY + (2 * RISK if side == "BUY" else -2 * RISK)


def _walk(st, p, prices, atr=3.0):
    """Feed prices one at a time, returning every action in order."""
    out = []
    for px in prices:
        if st.qty <= 0:
            break
        out += em.step_tick(st, round(px, 2), atr, p)
    return out


# --------------------------------------------------------------------------- #
#  THE RULE, exactly as described.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("side", ["BUY", "SELL"])
def test_above_tp1_then_back_to_tp1_squares_the_runner_off_there(side):
    p, st = _p(), _st(side)
    sign = 1 if side == "BUY" else -1
    tp1 = _tp1(side)

    # 1) up to TP1 -> half booked, runner armed, TP2 set.
    acts = _walk(st, p, [ENTRY + sign * RISK])
    partial = [a for a in acts if a.kind == ActionType.PARTIAL]
    assert len(partial) == 1 and partial[0].qty == 5
    assert partial[0].price == pytest.approx(tp1)
    assert st.qty == 5
    assert st.target == pytest.approx(_tp2(side))       # waiting for TP2
    # ...and the runner is NOT closed on the same tick at a price that would
    # have been the fill it just took.
    assert not [a for a in acts if a.kind == ActionType.EXIT]

    # 2) above TP1, but never as far as TP2.
    acts = _walk(st, p, [ENTRY + sign * RISK * 1.3,
                         ENTRY + sign * RISK * 1.7,
                         ENTRY + sign * RISK * 1.5])
    assert not [a for a in acts if a.kind == ActionType.EXIT]
    # The floor has now been raised the rest of the way to TP1 exactly.
    assert st.stop == pytest.approx(tp1)
    assert any(a.reason == "lock TP1" for a in acts
               if a.kind == ActionType.MOVE_STOP)

    # 3) back to TP1 -> the rest is squared off THERE.
    acts = _walk(st, p, [tp1])
    exits = [a for a in acts if a.kind == ActionType.EXIT]
    assert len(exits) == 1
    assert exits[0].price == pytest.approx(tp1)
    assert exits[0].qty == 5
    assert exits[0].reason == "TP1-LOCK"
    assert st.qty == 0


@pytest.mark.parametrize("side", ["BUY", "SELL"])
def test_reaching_tp2_pays_the_runner_at_tp2(side):
    """The other branch: if it does get there, the runner is worth 2R."""
    p, st = _p(), _st(side)
    sign = 1 if side == "BUY" else -1
    acts = _walk(st, p, [ENTRY + sign * RISK,          # TP1, book half
                         ENTRY + sign * RISK * 1.5,
                         ENTRY + sign * RISK * 2.0])   # TP2
    exits = [a for a in acts if a.kind == ActionType.EXIT]
    assert len(exits) == 1
    assert exits[0].price == pytest.approx(_tp2(side))
    assert exits[0].reason == "TARGET"
    assert exits[0].qty == 5
    # Both legs together = the entry quantity (C-5), booked at 1R and 2R.
    booked = sum(a.qty for a in acts
                 if a.kind in (ActionType.PARTIAL, ActionType.EXIT))
    assert booked == QTY


@pytest.mark.parametrize("side", ["BUY", "SELL"])
def test_the_lock_never_books_a_price_the_market_did_not_print(side):
    """The failure the clamp exists to prevent: a stop parked beyond the fill
    would be 'hit' on the tick that set it, booking a fill nobody offered."""
    p, st = _p(), _st(side)
    sign = 1 if side == "BUY" else -1
    tp1 = _tp1(side)

    acts = em.step_tick(st, tp1, 3.0, p)
    assert not [a for a in acts if a.kind == ActionType.EXIT]
    if side == "BUY":
        assert st.stop <= tp1 - TICK + 1e-9
    else:
        assert st.stop >= tp1 + TICK - 1e-9
    # One tick, and no more, is given back before the floor engages.
    assert abs(st.stop - tp1) <= TICK + 1e-9


@pytest.mark.parametrize("side", ["BUY", "SELL"])
def test_a_reversal_straight_off_tp1_gives_back_at_most_one_tick(side):
    """The worst case for this rule: price touches TP1 and turns immediately,
    never trading above it, so the floor never gets its chance to reach TP1."""
    p, st = _p(), _st(side)
    sign = 1 if side == "BUY" else -1
    acts = _walk(st, p, [_tp1(side), _tp1(side) - sign * TICK])
    exits = [a for a in acts if a.kind == ActionType.EXIT]
    assert len(exits) == 1
    # Out one tick below TP1 — the whole 1R kept bar a single tick.
    assert abs(exits[0].price - _tp1(side)) <= TICK + 1e-9
    booked = sum(a.qty for a in acts
                 if a.kind in (ActionType.PARTIAL, ActionType.EXIT))
    assert booked == QTY


# --------------------------------------------------------------------------- #
#  It must not leak into anything that did not ask for it.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("style", ["fixed", "trail_full", "partial_trail"])
def test_other_styles_do_not_lock(style):
    p = _p(style)
    assert p.lock_first_target is False
    st = _st("BUY")
    _walk(st, p, [ENTRY + RISK])
    if p.partial_exit_fraction > 0 or p.trail_from_entry:
        # break-even + buffer, which is far BELOW TP1 — the whole difference
        # between Approach 2 and 2b.
        assert st.stop < _tp1("BUY") - TICK
    assert config.CRUDEOIL_PARAMS.lock_first_target is False


def test_partial_trail_and_partial_lock_differ_only_where_they_should():
    a = config.INTRADAY_PARTIAL_TRAIL_PARAMS
    b = config.INTRADAY_PARTIAL_LOCK_PARAMS
    differ = {f for f in type(a).__dataclass_fields__
              if getattr(a, f) != getattr(b, f)}
    assert differ == {"lock_first_target", "runner_rr_mult", "trail_remainder"}
    assert b.runner_rr_mult == config.RUNNER_TARGET_RR
    # Neither touches anything that sizes or selects a trade.
    base = config.params_for_mode(Mode.INTRADAY)
    for field in ("risk_per_trade", "risk_reward", "atr_sl_mult",
                  "max_leverage", "max_capital_per_trade_pct",
                  "risk_per_trade_cash"):
        assert getattr(b, field) == getattr(base, field), field


# --------------------------------------------------------------------------- #
#  The bar path has to agree with the tick path here too (C-3).
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("side", ["BUY", "SELL"])
def test_one_bar_that_spikes_past_tp1_and_closes_back_on_it(side):
    """A single 15m bar can contain the whole story. Pessimistic ordering means
    the low is tested first, then the high books the partial and raises the
    floor, then the close comes back to TP1 and takes the runner out."""
    p, st = _p(), _st(side)
    sign = 1 if side == "BUY" else -1
    o = ENTRY
    hi = ENTRY + sign * RISK * 1.6
    lo = ENTRY - sign * RISK * 0.2
    close = ENTRY + sign * RISK          # back on TP1
    acts = (em.step_bar(st, o, hi, lo, close, 3.0, p) if side == "BUY"
            else em.step_bar(st, o, lo, hi, close, 3.0, p))
    kinds = [a.kind for a in acts]
    assert ActionType.PARTIAL in kinds and ActionType.EXIT in kinds
    exit_act = next(a for a in acts if a.kind == ActionType.EXIT)
    assert exit_act.reason == "TP1-LOCK"
    assert exit_act.price == pytest.approx(_tp1(side))
    assert sum(a.qty for a in acts
               if a.kind in (ActionType.PARTIAL, ActionType.EXIT)) == QTY
