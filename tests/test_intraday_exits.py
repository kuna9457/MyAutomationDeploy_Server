"""Intraday-specific integration of the two profit-booking approaches.

Intraday is not CRUDEOIL with different numbers. The scale-out and trail were
written for a single commodity contract — one tick grid, one price, one lot
size — and Intraday trades equity across a 10x price range in share counts.
Two things break when the same settings are pointed at it, and both are
checked here:

  1. THE BREAK-EVEN BUFFER CANNOT BE A TICK COUNT. Round-trip friction is
     mostly proportional to turnover, so the buffer that clears a runner's own
     round trip is ~19 bps on every symbol but anywhere from 7 to 92 ticks
     depending on the price. A runner handed CRUDEOIL's 3-tick stop on a
     Rs3,200 share is not free after costs, it is down ~Rs6 a share — the
     exact failure the buffer exists to prevent.
  2. THE BUFFER CAN EXCEED THE WHOLE 1R MOVE. On a tight stop, 20 bps of a
     Rs3,200 share is Rs6.40 against a Rs1.60 risk distance, so the naive
     break-even stop lands BEYOND the price that just filled the partial. It
     must be clamped to the safe side of the printed price, exactly as the
     trail is, or the very next check books a fill that never printed.

The last test is the one that matters most: CRUDEOIL's own settings must come
through all of this untouched.
"""
from __future__ import annotations

import pytest

import config
import exit_manager as em
from config import Mode
from cost_model import INTRADAY_EQUITY
from exit_manager import ActionType, ExitState


#: Real Intraday equity shapes: a mid-priced name, an expensive one, a cheap
#: one. ~Rs1L of notional each, which is what a Rs1L account at 1x deploys.
EQUITY_CASES = [
    ("mid",   1400.0, 71),
    ("dear",  3200.0, 31),
    ("cheap",  250.0, 400),
]


def _params(style="partial_trail"):
    return config.apply_exit_style(config.params_for_mode(Mode.INTRADAY), style)


def _exit_params(style="partial_trail", tick=0.05):
    return em.params_from_strategy(_params(style), tick, 1)


# --------------------------------------------------------------------------- #
#  1. The buffer has to clear real round-trip costs on every price.
# --------------------------------------------------------------------------- #
def _runner_cost_per_share(price, qty):
    """Round-trip cost of the RUNNER leg, per share.

    The runner is what the break-even stop protects, and it is the smaller
    half — so the flat per-order brokerage is spread over fewer shares and the
    per-share cost is HIGHER than the whole position's. Sizing the buffer
    against the full position (the intuitive thing) under-protects it.
    """
    runner = max(qty // 2, 1)
    return INTRADAY_EQUITY.round_trip_cost(price, price, runner) / runner


@pytest.mark.parametrize("label,price,qty", EQUITY_CASES)
def test_breakeven_buffer_clears_round_trip_costs(label, price, qty):
    """The runner's stop must leave the position PROFITABLE after costs, not
    merely level before them.

    Allowed one tick of slack, and no more: the stop is floored onto the tick
    grid (the looser side, as everywhere here), which on a cheap share is the
    difference between covering the round trip and missing it by a rupee-cent.
    That limitation is stated in config.BREAKEVEN_BUFFER_BPS; this test is
    what stops it growing past one tick.
    """
    p = _exit_params()
    risk = price * 0.006                      # a normal ~0.6% intraday stop
    st = ExitState(side="BUY", entry=price, qty=qty, stop=price - risk,
                   target=price + risk, risk_dist=risk)
    em.step_tick(st, price + risk, atr=0.0, p=p)   # atr=0 => no trail, buffer only

    locked_per_share = st.stop - price
    cost_per_share = _runner_cost_per_share(price, qty)
    assert locked_per_share > cost_per_share - p.tick_size, (
        f"{label}: stop locks Rs{locked_per_share:.3f}/share but the runner's "
        f"round trip costs Rs{cost_per_share:.3f}/share — not free even "
        f"allowing a tick of grid rounding")


@pytest.mark.parametrize("label,price,qty", EQUITY_CASES)
def test_a_tick_buffer_would_not_have(label, price, qty):
    """The counter-test, so the one above cannot pass vacuously: CRUDEOIL's
    3-tick buffer fails on Intraday equity at every price. This is why the
    field is in bps."""
    p = em.params_from_strategy(
        config.replace(_params(), breakeven_buffer_bps=0.0,
                       breakeven_buffer_ticks=3.0), 0.05, 1)
    risk = price * 0.006
    st = ExitState(side="BUY", entry=price, qty=qty, stop=price - risk,
                   target=price + risk, risk_dist=risk)
    em.step_tick(st, price + risk, atr=0.0, p=p)
    locked = st.stop - price
    cost = _runner_cost_per_share(price, qty)
    assert locked < cost, (
        f"{label}: 3 ticks unexpectedly covered the round trip — re-derive "
        f"config.BREAKEVEN_BUFFER_BPS, the cost model has changed")


def test_buffer_is_the_same_in_bps_across_the_whole_price_range():
    """One setting, one meaning, whatever the symbol costs — the property a
    tick count does not have."""
    p = _exit_params()
    bps = []
    for _label, price, qty in EQUITY_CASES:
        risk = price * 0.006
        st = ExitState(side="BUY", entry=price, qty=qty, stop=price - risk,
                       target=price + risk, risk_dist=risk)
        em.step_tick(st, price + risk, atr=0.0, p=p)
        # The grid is coarser in bps on a cheap share, so express the
        # tolerance in ticks rather than as a flat number of bps.
        bps.append((1e4 * (st.stop - price) / price, 1e4 * p.tick_size / price))
    # Every symbol lands within ONE TICK below the constant — never above it,
    # because flooring only ever moves toward entry.
    for got, tick_bps in bps:
        assert config.BREAKEVEN_BUFFER_BPS - tick_bps <= got \
               <= config.BREAKEVEN_BUFFER_BPS + 1e-6


# --------------------------------------------------------------------------- #
#  2. The buffer must never be allowed past the printed price.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("side", ["BUY", "SELL"])
def test_a_buffer_wider_than_1R_is_clamped_to_the_fill(side):
    """20 bps of a Rs3,200 share is Rs6.40; a tight 0.05% stop is Rs1.60. The
    unclamped break-even stop would sit Rs4.80 BEYOND the price that just
    printed, and the exit check on this very tick would then book a fill the
    market never offered.

    Clamped, the stop lands one tick inside the fill: it protects 1.55 of the
    1.60 move rather than the 6.40 the buffer asked for. That is the honest
    outcome — when friction is wider than the whole move there is no
    cost-free runner to be had, and the most the position can be given is
    everything the market actually printed."""
    p = _exit_params()
    price, qty, risk = 3200.0, 31, 1.60
    entry = price
    target = entry + risk if side == "BUY" else entry - risk
    st = ExitState(side=side, entry=entry, qty=qty,
                   stop=entry - risk if side == "BUY" else entry + risk,
                   target=target, risk_dist=risk)
    acts = em.step_tick(st, target, atr=0.0, p=p)

    assert any(a.kind == ActionType.PARTIAL for a in acts)
    eps = 1e-9
    # One tick inside the printed fill, on the side that PROTECTS: above entry
    # for a long, below it for a short. Never beyond the fill itself.
    if side == "BUY":
        assert entry < st.stop <= target - p.tick_size + eps
    else:
        assert target + p.tick_size - eps <= st.stop < entry
    # ...and the buffer it actually got is LESS than the one it asked for,
    # because the move was not big enough to pay for it.
    asked = config.BREAKEVEN_BUFFER_BPS / 1e4 * entry
    assert abs(st.stop - entry) < asked
    # ...and therefore the runner is NOT closed on the same tick at a
    # fictitious price.
    assert not any(a.kind == ActionType.EXIT for a in acts)
    assert st.qty > 0


# --------------------------------------------------------------------------- #
#  3. Both approaches actually run on Intraday, end to end.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("style", ["trail_full", "partial_trail"])
def test_intraday_preset_matches_the_named_style(style):
    """The documented constants and the style selector cannot drift apart."""
    named = {"trail_full": config.INTRADAY_TRAIL_FULL_PARAMS,
             "partial_trail": config.INTRADAY_PARTIAL_TRAIL_PARAMS}[style]
    assert named == _params(style)
    # ...and neither approach touched anything that sizes or selects a trade.
    base = config.params_for_mode(Mode.INTRADAY)
    for field in ("risk_per_trade", "risk_reward", "atr_sl_mult", "atr_period",
                  "max_leverage", "max_capital_per_trade_pct", "cs_min_score",
                  "risk_per_trade_cash"):
        assert getattr(named, field) == getattr(base, field), field


@pytest.mark.parametrize("side", ["BUY", "SELL"])
def test_intraday_partial_then_trail_on_a_realistic_move(side):
    p = _exit_params("partial_trail")
    entry, qty, risk, atr = 1400.0, 71, 8.4, 5.6   # ~0.6% stop, 1.5xATR(14)
    sign = 1 if side == "BUY" else -1
    st = ExitState(side=side, entry=entry, qty=qty, stop=entry - sign * risk,
                   target=entry + sign * risk, risk_dist=risk)

    booked = 0
    for step in range(1, 12):                      # walk it out to ~2R
        price = round(entry + sign * risk * step / 4.0, 2)
        for a in em.step_tick(st, price, atr, p):
            if a.kind in (ActionType.PARTIAL, ActionType.EXIT):
                booked += a.qty
        if st.qty == 0:
            break
    assert st.partial_taken and st.trail_active
    # Walk it back into the trailed stop and check the books balance.
    while st.qty:
        for a in em.step_tick(st, st.stop - sign * 0.05, atr, p):
            if a.kind == ActionType.EXIT:
                booked += a.qty
    assert booked == qty                            # C-5, on Intraday
    assert (st.stop - entry) * sign > 0             # closed above break-even


# --------------------------------------------------------------------------- #
#  4. CRUDEOIL comes through untouched.
# --------------------------------------------------------------------------- #
def test_crudeoil_keeps_its_tick_buffer():
    """Its 3 ticks are Rs3 on a Rs1.0 grid x100 multiplier = Rs300 a lot, which
    IS the right shape for that contract. The bps field is additive and stays
    at zero there, so nothing about CRUDEOIL moved."""
    c = config.CRUDEOIL_PARAMS
    assert c.breakeven_buffer_ticks == 3.0
    assert c.breakeven_buffer_bps == 0.0
    assert c.runner_rr_mult == 3.0

    inst = config.INSTRUMENTS_BY_SYMBOL.get("CRUDEOIL")
    if inst is None:
        pytest.skip("CRUDEOIL is not in this deployment's instrument list")
    p = em.params_from_strategy(c, inst.tick_size, inst.contract_multiplier)
    entry, risk = 7500.0, 30.0
    st = ExitState(side="BUY", entry=entry, qty=4, stop=entry - risk,
                   target=entry + 1.5 * risk, risk_dist=risk)
    em.step_tick(st, entry + 1.5 * risk, atr=0.0, p=p)
    assert st.stop == pytest.approx(entry + 3.0)    # exactly 3 ticks, as before
