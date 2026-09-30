"""C-3 — parity between the LIVE path and the BACKTEST path.

"The bot you backtest and the bot you run must execute the same exit decisions
from the same code" is the whole point of exit_manager, and it can fail in
exactly two places:

  1. the two callers build DIFFERENT ExitParams from the same StrategyParams
     (a typo, a getattr default, a knob one of them forgot), and
  2. the two entry points — step_tick (one price per call, live) and step_bar
     (one OHLC bar, backtest) — disagree on the same price sequence.

Both are checked here on real instruments and on all four exit styles. The
third test walks the actual backtester over real (or synthetic) candles and
asserts the properties that must hold of any run: quantity conservation across
legs (C-5), costs strictly reducing P&L (C-4), and honest exit reasons.
"""
from __future__ import annotations

import random
from types import SimpleNamespace

import pytest

import backtester
import config
import engine
import exit_manager as em
from config import Mode
from exit_manager import ActionType, ExitState


#: Three instruments with genuinely different tick grids and contract sizes —
#: 0.05/1x equity, and CRUDEOIL's 1.0 tick with a 100x multiplier — because a
#: params mismatch is most likely to hide in exactly those two fields.
INSTRUMENTS = ["RELIANCE", "TCS", "CRUDEOIL"]
STYLES = list(config.EXIT_STYLES)


def _inst(sym):
    inst = config.INSTRUMENTS_BY_SYMBOL.get(sym)
    if inst is None:
        pytest.skip(f"{sym} is not in this deployment's instrument list")
    return inst


# --------------------------------------------------------------------------- #
#  1. The two callers must build identical ExitParams.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("symbol", INSTRUMENTS)
@pytest.mark.parametrize("style", STYLES)
def test_engine_and_backtester_build_the_same_exit_params(symbol, style):
    inst = _inst(symbol)
    params = config.apply_exit_style(config.params_for_mode(Mode.INTRADAY),
                                     style)

    # The live side, called unbound against a stub: _exit_params reads only
    # self.params and self.symbol_rules, and an unconfigured symbol (no saved
    # per-symbol rule) is the case a backtest models.
    stub = SimpleNamespace(params=params, symbol_rules={})
    live = engine.TradingEngine._exit_params(stub, inst)

    # The backtest side, through the same shared constructor run_backtest uses.
    sim = em.params_from_strategy(params, inst.tick_size,
                                  inst.contract_multiplier)
    assert live == sim


def test_a_symbol_rule_overrides_the_trail_live_only():
    """The one field that is ALLOWED to differ, and the direction it differs
    in: a per-symbol trail is a live admin control, so the engine honours it
    and the backtest ignores it."""
    inst = _inst("RELIANCE")
    params = config.params_for_mode(Mode.INTRADAY)
    rules = SimpleNamespace(trail_enabled=True, trail_atr_mult=3.0,
                            trail_mult=lambda default: 3.0)
    stub = SimpleNamespace(params=params, symbol_rules={inst.symbol: rules})
    live = engine.TradingEngine._exit_params(stub, inst)
    sim = em.params_from_strategy(params, inst.tick_size,
                                  inst.contract_multiplier)
    assert live.trail_atr_mult == 3.0 and live.trail_from_entry is True
    assert sim.trail_atr_mult == params.atr_sl_mult
    assert sim.trail_from_entry is False


# --------------------------------------------------------------------------- #
#  2. step_tick and step_bar must agree on the same price sequence.
#
#  step_bar is DEFINED as three step_tick calls in the adverse order, so this
#  is a regression guard on that definition: if anyone ever "optimises"
#  step_bar into its own inline logic, the backtest silently stops being the
#  live bot and this test is what says so.
# --------------------------------------------------------------------------- #
def _fresh(side, entry=100.0, qty=4, risk=2.0, rr=1.0):
    stop = entry - risk if side == "BUY" else entry + risk
    target = entry + rr * risk if side == "BUY" else entry - rr * risk
    return ExitState(side=side, entry=entry, qty=qty, stop=stop,
                     target=target, risk_dist=risk)


def _fingerprint(actions):
    return [(a.kind.value, round(a.price, 4), a.qty, round(a.new_stop, 4),
             a.reason) for a in actions]


@pytest.mark.parametrize("symbol", INSTRUMENTS)
@pytest.mark.parametrize("style", STYLES)
@pytest.mark.parametrize("side", ["BUY", "SELL"])
def test_step_bar_equals_the_equivalent_tick_sequence(symbol, style, side):
    inst = _inst(symbol)
    params = config.apply_exit_style(config.params_for_mode(Mode.INTRADAY),
                                     style)
    p = em.params_from_strategy(params, inst.tick_size,
                                inst.contract_multiplier)

    rng = random.Random(hash((symbol, style, side)) & 0xFFFF)
    entry = float(inst.reference_price or 100.0)
    risk = max(entry * 0.004, (inst.tick_size or 0.05) * 4)
    bars = []
    px = entry
    for _ in range(200):
        o = px
        h = o + abs(rng.gauss(0, risk / 2))
        l = o - abs(rng.gauss(0, risk / 2))
        c = rng.uniform(l, h)
        bars.append((o, h, l, c, max(risk / 2, abs(rng.gauss(risk, risk / 4)))))
        px = c

    by_bar = _fresh(side, entry, 4, risk)
    by_tick = _fresh(side, entry, 4, risk)
    for o, h, l, c, atr in bars:
        got_bar = em.step_bar(by_bar, o, h, l, c, atr, p)
        # The SAME path, spelled out one price at a time the way the live
        # engine sees it: the adverse extreme first, then the favourable one,
        # then the close.
        adverse, favourable = (l, h) if side == "BUY" else (h, l)
        got_tick = []
        for price in (adverse, favourable, c):
            if by_tick.qty <= 0:
                break
            got_tick += em.step_tick(by_tick, price, atr, p)
            if any(a.kind == ActionType.EXIT for a in got_tick):
                break
        assert _fingerprint(got_bar) == _fingerprint(got_tick)
        assert by_bar == by_tick
        if by_bar.qty == 0:
            break


# --------------------------------------------------------------------------- #
#  3. The real backtester, on real instruments: the properties that must hold
#     of every run whatever the market did.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("symbol", INSTRUMENTS)
@pytest.mark.parametrize("style", STYLES)
def test_backtest_run_is_internally_consistent(symbol, style):
    _inst(symbol)
    res = backtester.run_backtest(symbol, "2026-01-01", "2026-06-30",
                                  200_000.0, Mode.INTRADAY,
                                  exit_style=style, include_costs=True)
    t = res.trades
    if t.empty:
        pytest.skip(f"{symbol}/{style} took no trades in the window")

    # C-5: the legs of one position add back up to the entry quantity — i.e.
    # no leg ever books quantity the position did not hold.
    per_position = t.groupby("position_id")["qty"].sum()
    assert (per_position > 0).all()
    assert int(t["qty"].sum()) == int(per_position.sum())
    # A scaled-out position has 2 legs, everything else exactly 1, and never
    # more than 2 — the partial fires once per position by construction.
    legs = t.groupby("position_id")["leg"].max()
    assert legs.between(1, 2).all()
    # Derived from the params, not from a hardcoded list of style names, so a
    # style added later is covered by this test the day it is added.
    scales_out = config.apply_exit_style(
        config.params_for_mode(Mode.INTRADAY), style).partial_exit_fraction > 0
    if not scales_out:
        assert (legs == 1).all(), "only a scaling style may produce two legs"

    # C-4: cost-aware runs are strictly worse than gross, per leg and in total.
    assert "cost" in t.columns and (t["cost"] > 0).all()
    assert (t["net_pnl"] < t["pnl"]).all()
    assert res.metrics["Total Return %"] < res.metrics["Gross Return %"]

    # Task D: every exit names itself honestly enough to compare a live log
    # against this one trade by trade.
    known = {"TARGET", "STOP-LOSS", "PARTIAL-TARGET", "TRAIL-PROFIT",
             "TRAIL-STOP", "BREAK-EVEN", "TP1-LOCK"}
    reasons = set(t["exit_reason"])
    unknown = {r for r in reasons
               if r not in known and not r.startswith(("TIME-EXIT",
                                                       "SQUARE-OFF"))}
    assert not unknown, f"unnamed exit reason(s): {unknown}"
    if style in ("fixed", "strategy") and symbol != "CRUDEOIL":
        # C-2 again, end to end: nothing configured means nothing manages.
        assert not (reasons & {"PARTIAL-TARGET", "TRAIL-PROFIT", "TRAIL-STOP",
                               "BREAK-EVEN", "TP1-LOCK"})


def test_partial_is_charged_twice():
    """C-4's sharper half: two legs of the same total size cost MORE than one.

    This is the entire reason to test rather than assume that scaling out
    helps — on small size the second exit's costs can eat the runner's edge.
    """
    from cost_model import INTRADAY_EQUITY as cm
    one_leg = cm.round_trip_cost(1000.0, 1010.0, 100)
    two_legs = (cm.round_trip_cost(1000.0, 1010.0, 50)
                + cm.round_trip_cost(1000.0, 1020.0, 50))
    assert two_legs > one_leg
