"""Unit tests for exit_manager — the ONE exit state-machine that live, paper
and backtest all run.

The acceptance criteria these cover (implementation brief §7):

    C-1  no regression — a reference implementation of the OLD engine
         (_maybe_scale_out + _apply_trail + the inline stop/target check)
         produces byte-identical decisions to exit_manager on CRUDEOIL's
         Approach-2 parameters, over a long random price path.
    C-2  baseline unchanged — with nothing configured a position exits on the
         fixed ATR stop/target, with no trailing and no partial.
    C-5  quantity conservation — every leg booked, summed, equals the entry
         quantity.
    C-6  invariants — the trail ratchets one way only, is never moved past the
         live price or the target, and never widens risk.
"""
from __future__ import annotations

import math
import random

import pytest

import exit_manager as em
from exit_manager import Action, ActionType, ExitParams, ExitState


TICK = 0.05


def _state(side="BUY", entry=100.0, qty=4, risk=2.0, rr=1.0) -> ExitState:
    stop = entry - risk if side == "BUY" else entry + risk
    target = entry + rr * risk if side == "BUY" else entry - rr * risk
    return ExitState(side=side, entry=entry, qty=qty, stop=stop,
                     target=target, risk_dist=risk)


BASELINE = ExitParams(tick_size=TICK)
APPROACH_1 = ExitParams(tick_size=TICK, trail_atr_mult=2.0,
                        trail_from_entry=True)
APPROACH_2 = ExitParams(tick_size=TICK, partial_exit_fraction=0.5,
                        breakeven_buffer_ticks=3.0, runner_rr_mult=3.0,
                        trail_remainder=True, trail_atr_mult=2.0)


# --------------------------------------------------------------------------- #
#  C-2 — baseline: nothing configured means nothing changes.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("side", ["BUY", "SELL"])
def test_baseline_exits_on_the_fixed_target(side):
    st = _state(side)
    hit = st.target + (0.5 if side == "BUY" else -0.5)
    acts = em.step_tick(st, hit, atr=1.0, p=BASELINE)
    assert [a.kind for a in acts] == [ActionType.EXIT]
    assert acts[0].reason == "TARGET"
    assert acts[0].price == st.target        # filled AT the level, not beyond
    assert acts[0].qty == 4


@pytest.mark.parametrize("side", ["BUY", "SELL"])
def test_baseline_exits_on_the_fixed_stop(side):
    st = _state(side)
    stop = st.stop
    hit = stop - 0.5 if side == "BUY" else stop + 0.5
    acts = em.step_tick(st, hit, atr=1.0, p=BASELINE)
    assert [a.kind for a in acts] == [ActionType.EXIT]
    assert acts[0].reason == "STOP-LOSS"
    assert acts[0].price == stop


def test_baseline_never_trails_or_partials():
    """A favourable walk that never reaches the target must produce NO actions
    at all — the position keeps the stop it was entered with."""
    st = _state()
    for px in (100.5, 101.0, 101.4, 101.9):
        assert em.step_tick(st, px, atr=1.0, p=BASELINE) == []
    assert st.stop == 98.0
    assert st.qty == 4


# --------------------------------------------------------------------------- #
#  Approach 2 — book part at the first target, trail the runner.
# --------------------------------------------------------------------------- #
def test_approach2_long_books_half_and_arms_the_runner():
    st = _state("BUY", entry=100.0, qty=4, risk=2.0, rr=1.0)   # target 102
    acts = em.step_tick(st, 102.0, atr=0.5, p=APPROACH_2)

    partial = [a for a in acts if a.kind == ActionType.PARTIAL]
    assert len(partial) == 1
    assert partial[0].qty == 2 and partial[0].price == 102.0
    assert partial[0].reason == "PARTIAL-TARGET"

    assert st.qty == 2                       # the runner
    assert st.first_target_done and st.partial_taken and st.trail_active
    # break-even + 3 ticks, floored onto the tick grid for a long. Asserted on
    # the ACTION, not on st.stop: the chandelier runs on the same tick and, at
    # 2xATR behind a 102 peak, immediately ratchets past break-even to 101.
    be = [a for a in acts
          if a.kind == ActionType.MOVE_STOP and a.reason == "break-even+buffer"]
    assert len(be) == 1 and be[0].new_stop == pytest.approx(100.15)
    # runner target = entry + 3 x the ORIGINAL risk distance
    assert st.target == pytest.approx(106.0)
    assert st.stop == pytest.approx(101.0)   # 102 peak - 2 x 0.5 ATR


def test_approach2_short_is_the_exact_mirror():
    st = _state("SELL", entry=100.0, qty=4, risk=2.0, rr=1.0)  # target 98
    acts = em.step_tick(st, 98.0, atr=0.5, p=APPROACH_2)
    partial = [a for a in acts if a.kind == ActionType.PARTIAL]
    assert len(partial) == 1 and partial[0].qty == 2
    assert st.qty == 2
    be = [a for a in acts
          if a.kind == ActionType.MOVE_STOP and a.reason == "break-even+buffer"]
    assert len(be) == 1 and be[0].new_stop == pytest.approx(99.85)  # CEILed
    assert st.target == pytest.approx(94.0)  # entry - 3R
    assert st.stop == pytest.approx(99.0)    # 98 trough + 2 x 0.5 ATR


def test_one_lot_never_partials():
    """A single lot is indivisible: 'half' of it is all of it, so a partial
    would just be an ordinary target exit that also gave up the runner."""
    st = _state("BUY", qty=1)
    acts = em.step_tick(st, 102.0, atr=0.5, p=APPROACH_2)
    assert not [a for a in acts if a.kind == ActionType.PARTIAL]
    assert st.qty == 1
    # It still enters the managed lifecycle (stop to break-even, trail armed),
    # which is what makes it a runner rather than a closed trade.
    assert st.first_target_done and not st.partial_taken


def test_runner_rr_zero_means_pure_trail():
    p = ExitParams(tick_size=TICK, partial_exit_fraction=0.5,
                   breakeven_buffer_ticks=3.0, runner_rr_mult=0.0,
                   trail_remainder=True, trail_atr_mult=2.0)
    st = _state("BUY")
    em.step_tick(st, 102.0, atr=0.5, p=p)
    assert st.target == math.inf             # only the trail can exit it


# --------------------------------------------------------------------------- #
#  Approach 1 — trail the whole position from entry.
# --------------------------------------------------------------------------- #
def test_approach1_trails_the_full_position_and_never_partials():
    st = _state("BUY", entry=100.0, qty=4, risk=2.0)
    stops = []
    for px in (100.5, 101.0, 102.0, 103.0, 104.0):
        for a in em.step_tick(st, px, atr=0.5, p=APPROACH_1):
            assert a.kind != ActionType.PARTIAL
            if a.kind == ActionType.MOVE_STOP:
                stops.append(a.new_stop)
    assert stops == sorted(stops)            # ratchet only
    assert st.qty == 4                       # nothing was ever booked
    assert st.stop > 100.0                   # the runner is risk-free


# --------------------------------------------------------------------------- #
#  C-6 — the trail's invariants, checked on a random walk rather than by
#  eyeballing one path.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("side", ["BUY", "SELL"])
def test_trail_invariants_hold_on_a_random_walk(side):
    rng = random.Random(7)
    st = _state(side, qty=4)
    p = APPROACH_2
    prev_stop = st.stop
    price = st.entry
    for _ in range(500):
        price = round(price + rng.uniform(-0.6, 0.7) * (1 if side == "BUY" else -1), 2)
        acts = em.step_tick(st, price, atr=0.5, p=p)
        if st.side == "BUY":
            assert st.stop >= prev_stop - 1e-9          # never moves away
            if not any(a.kind == ActionType.EXIT for a in acts):
                assert st.stop < price + 1e-9           # never past the price
                assert st.stop <= st.target             # never past the target
        else:
            assert st.stop <= prev_stop + 1e-9
            if not any(a.kind == ActionType.EXIT for a in acts):
                assert st.stop > price - 1e-9
                assert st.stop >= st.target
        prev_stop = st.stop
        if any(a.kind == ActionType.EXIT for a in acts):
            break
    # Risk can only ever have shrunk: quantity is fixed at entry and the stop
    # only ratcheted toward price.
    assert st.qty <= 4


# --------------------------------------------------------------------------- #
#  step_bar — pessimistic intrabar ordering.
# --------------------------------------------------------------------------- #
def test_bar_that_hits_both_stop_and_target_books_the_stop():
    st = _state("BUY", entry=100.0, qty=4, risk=2.0)     # stop 98, target 102
    acts = em.step_bar(st, o=100.0, h=103.0, l=97.0, c=101.0,
                       atr=0.5, p=BASELINE)
    assert [a.kind for a in acts] == [ActionType.EXIT]
    assert acts[0].price == 98.0 and acts[0].reason == "STOP-LOSS"


def test_bar_credits_the_favourable_move_only_after_the_adverse_one():
    """The same bar, with a low that does NOT reach the stop, may partial.

    And the whole lifecycle can complete inside one bar: high 103 books the
    partial and drags the chandelier to 102, and the close at 101 is then
    below it, so the runner is stopped out on the same bar. Both legs are
    emitted, and they add back up to the entry quantity.
    """
    st = _state("BUY", entry=100.0, qty=4, risk=2.0)
    acts = em.step_bar(st, o=100.0, h=103.0, l=99.0, c=101.0,
                       atr=0.5, p=APPROACH_2)
    kinds = [a.kind for a in acts]
    assert ActionType.PARTIAL in kinds and ActionType.EXIT in kinds
    booked = sum(a.qty for a in acts
                 if a.kind in (ActionType.PARTIAL, ActionType.EXIT))
    assert booked == 4 and st.qty == 0
    exit_act = next(a for a in acts if a.kind == ActionType.EXIT)
    assert exit_act.price == pytest.approx(102.0)
    assert exit_act.reason == "TRAIL-PROFIT"


def test_closed_position_is_never_acted_on_twice():
    st = _state("BUY")
    em.step_tick(st, 90.0, atr=0.5, p=BASELINE)          # stopped out
    assert st.qty == 0
    assert em.step_tick(st, 89.0, atr=0.5, p=BASELINE) == []
    assert em.step_bar(st, 89, 89, 88, 88, 0.5, BASELINE) == []


# --------------------------------------------------------------------------- #
#  C-5 — quantity conservation across every leg, both approaches, both sides.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("p", [BASELINE, APPROACH_1, APPROACH_2],
                         ids=["baseline", "approach1", "approach2"])
@pytest.mark.parametrize("side", ["BUY", "SELL"])
def test_quantity_conservation(p, side):
    rng = random.Random(11)
    st = _state(side, qty=6)
    booked = 0
    price = st.entry
    for _ in range(2000):
        price = round(price + rng.uniform(-0.8, 0.8), 2)
        for a in em.step_tick(st, price, atr=0.5, p=p):
            if a.kind in (ActionType.PARTIAL, ActionType.EXIT):
                booked += a.qty
        if st.qty == 0:
            break
    assert st.qty == 0, "the walk never resolved — widen it"
    assert booked == 6


# --------------------------------------------------------------------------- #
#  C-1 — no regression. A faithful reimplementation of the PRE-REFACTOR engine
#  (_maybe_scale_out + _apply_trail + the inline stop/target check, in that
#  order) must agree with exit_manager decision for decision.
#
#  Written out longhand rather than imported, on purpose: the old code is gone
#  from engine.py, and a test that imported the new code to check the new code
#  would prove nothing.
# --------------------------------------------------------------------------- #
class _LegacyEngine:
    """engine.py as it stood before exit_manager existed. CRUDEOIL's path:
    partial_exit_fraction=0.5, breakeven_buffer_ticks=3, runner_rr_mult=3,
    trail_remainder=True, no per-symbol rule (so the runner trails at
    atr_sl_mult)."""

    def __init__(self, side, entry, qty, stop, target, tick, rr,
                 frac, buf_ticks, runner_mult, trail_remainder, atr_sl_mult):
        self.side, self.entry, self.qty = side, entry, qty
        self.stop, self.target, self.tick, self.rr = stop, target, tick, rr
        self.frac, self.buf_ticks = frac, buf_ticks
        self.runner_mult, self.trail_remainder = runner_mult, trail_remainder
        self.atr_sl_mult = atr_sl_mult
        self.partial_done = False
        self.trail_active = False
        self.peak = entry

    def step(self, price, atr):
        """Returns (partial_qty_or_0, stop_after, exit_price_or_None)."""
        partial_qty = 0
        # ---- _maybe_scale_out ------------------------------------------- #
        if (self.frac > 0 and not self.partial_done and self.qty > 1
                and ((self.side == "BUY" and price >= self.target)
                     or (self.side == "SELL" and price <= self.target))):
            exit_qty = max(1, int(math.floor(self.qty * self.frac)))
            exit_qty = min(exit_qty, self.qty - 1)
            partial_qty = exit_qty
            self.qty -= exit_qty
            buf = self.buf_ticks * self.tick
            if self.side == "BUY":
                self.stop = round(math.floor((self.entry + buf) / self.tick)
                                  * self.tick, 2)
            else:
                self.stop = round(math.ceil((self.entry - buf) / self.tick)
                                  * self.tick, 2)
            risk_dist = abs(self.target - self.entry) / self.rr
            mult = self.runner_mult or (self.rr * 2.0)
            if self.side == "BUY":
                self.target = round(self.entry + mult * risk_dist, 2)
            else:
                self.target = round(self.entry - mult * risk_dist, 2)
            self.partial_done = True
            self.trail_active = bool(self.trail_remainder)
            self.peak = price
        # ---- _apply_trail ------------------------------------------------ #
        if atr > 0 and self.trail_active and self.atr_sl_mult > 0:
            m = self.atr_sl_mult
            if self.side == "BUY":
                self.peak = max(self.peak, price)
                new_stop = math.floor((self.peak - m * atr) / self.tick) * self.tick
                new_stop = min(new_stop, price - self.tick, self.target - self.tick)
                if new_stop > self.stop + self.tick / 2:
                    self.stop = round(new_stop, 2)
            else:
                self.peak = min(self.peak, price)
                new_stop = math.ceil((self.peak + m * atr) / self.tick) * self.tick
                new_stop = max(new_stop, price + self.tick, self.target + self.tick)
                if new_stop < self.stop - self.tick / 2:
                    self.stop = round(new_stop, 2)
        # ---- the inline stop/target check -------------------------------- #
        exit_price = None
        if self.side == "BUY":
            if price <= self.stop:
                exit_price = self.stop
            elif price >= self.target:
                exit_price = self.target
        else:
            if price >= self.stop:
                exit_price = self.stop
            elif price <= self.target:
                exit_price = self.target
        return partial_qty, self.stop, exit_price


@pytest.mark.parametrize("side", ["BUY", "SELL"])
@pytest.mark.parametrize("seed", range(12))
def test_c1_matches_the_pre_refactor_engine_on_crudeoil_params(side, seed):
    import config

    p_cfg = config.CRUDEOIL_PARAMS
    tick, rr = 1.0, 1.5            # CRUDEOIL's tick grid and its RR
    entry, qty = 7500.0, 4
    risk = 30.0
    stop = entry - risk if side == "BUY" else entry + risk
    target = entry + rr * risk if side == "BUY" else entry - rr * risk

    legacy = _LegacyEngine(side, entry, qty, stop, target, tick, rr,
                           p_cfg.partial_exit_fraction,
                           p_cfg.breakeven_buffer_ticks,
                           p_cfg.runner_rr_mult, p_cfg.trail_remainder,
                           p_cfg.atr_sl_mult)
    st = ExitState(side=side, entry=entry, qty=qty, stop=stop, target=target,
                   risk_dist=risk)
    p = ExitParams(tick_size=tick, contract_multiplier=100,
                   partial_exit_fraction=p_cfg.partial_exit_fraction,
                   breakeven_buffer_ticks=p_cfg.breakeven_buffer_ticks,
                   runner_rr_mult=p_cfg.runner_rr_mult,
                   trail_remainder=p_cfg.trail_remainder,
                   trail_atr_mult=p_cfg.atr_sl_mult)

    rng = random.Random(seed)
    price, atr = entry, 20.0
    for _ in range(400):
        price = round(price + rng.uniform(-12, 12), 2)
        atr = max(5.0, round(atr + rng.uniform(-1, 1), 2))

        want_partial, want_stop, want_exit = legacy.step(price, atr)
        acts = em.step_tick(st, price, atr, p)
        got_partial = sum(a.qty for a in acts if a.kind == ActionType.PARTIAL)
        got_exit = next((a.price for a in acts
                         if a.kind == ActionType.EXIT), None)

        assert got_partial == want_partial, f"partial qty diverged @ {price}"
        assert st.stop == pytest.approx(want_stop), f"stop diverged @ {price}"
        if want_exit is None:
            assert got_exit is None, f"spurious exit @ {price}"
        else:
            assert got_exit == pytest.approx(want_exit), f"exit diverged @ {price}"
            break
