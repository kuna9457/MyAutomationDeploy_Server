"""Every backtest surface must measure the SAME bot.

The failure this guards against is quiet and expensive: a bulk screen ranks
50 symbols on the plain fixed exit while the single-symbol tab reports the
managed one, so you pick your "best stock" from a list that was scored against
a bot you are not going to trade. Measured on Candlestick Phase 1 the gap is
about 0.5 percentage points — small enough to look like noise, big enough to
reorder a ranking.

So: the run-shape knobs (exit style, its three overrides, and the stop band)
must exist on every request shape, be forwarded by every runner, and produce
identical numbers for one symbol however it is reached.
"""
from __future__ import annotations

import inspect

import pytest

import backtester
import bulk_backtester
import config
from advanced_backtest.search import SearchSpec
from config import Mode

#: The knobs that decide WHAT BOT is being measured, as opposed to which
#: symbol or window.
RUN_SHAPE = ("exit_style", "trail_atr_mult", "partial_exit_fraction",
             "runner_rr_mult", "max_stop_pct", "min_stop_pct")


# --------------------------------------------------------------------------- #
#  Every entry point carries them.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", ["BacktestRequest", "BulkBacktestRequest",
                                  "RRSweepRequest"])
def test_request_schemas_carry_the_run_shape(name):
    import api.schemas as schemas
    fields = getattr(schemas, name).model_fields
    missing = [k for k in RUN_SHAPE if k not in fields]
    assert not missing, f"{name} cannot express: {missing}"


def test_the_advanced_search_request_carries_them_too():
    from api.routers.advanced_backtest import SearchRequest
    missing = [k for k in RUN_SHAPE if k not in SearchRequest.model_fields]
    assert not missing, f"SearchRequest cannot express: {missing}"


@pytest.mark.parametrize("fn", [backtester.run_backtest,
                                backtester.run_bulk_backtest,
                                backtester.run_rr_sweep,
                                bulk_backtester.run_with_costs,
                                bulk_backtester.run_bulk_with_costs])
def test_every_runner_accepts_the_run_shape(fn):
    params = inspect.signature(fn).parameters
    missing = [k for k in RUN_SHAPE if k not in params]
    assert not missing, f"{fn.__module__}.{fn.__name__} drops: {missing}"


def test_the_search_spec_forwards_them_as_one_block():
    """run_shape() exists so the three call sites in search.py cannot each
    forget a different knob."""
    spec = SearchSpec(symbols=["INFY"], start="2026-01-01", end="2026-08-31",
                      exit_style="partial_ladder", min_stop_pct=0.8)
    shape = spec.run_shape()
    assert set(shape) == set(RUN_SHAPE)
    assert shape["exit_style"] == "partial_ladder"
    assert shape["min_stop_pct"] == 0.8
    src = inspect.getsource(__import__("advanced_backtest.search",
                                       fromlist=["search"]))
    assert src.count("**spec.run_shape()") == 3, (
        "a call site in search.py is not forwarding the run shape")


# --------------------------------------------------------------------------- #
#  And they actually change the answer, identically, on every path.
# --------------------------------------------------------------------------- #
def _skip_unless(sym="INFY"):
    if sym not in config.INSTRUMENTS_BY_SYMBOL:
        pytest.skip(f"{sym} is not in this deployment's instrument list")


WINDOW = ("2026-01-01", "2026-08-31")
CAP = 200_000.0
KW = dict(strategy_key="candlestick_engine", include_costs=True)


def test_the_style_actually_reaches_the_bulk_runner():
    """The regression that prompted this file: bulk ignored exit_style, so a
    managed run and a plain one returned the same number."""
    _skip_unless()
    plain = backtester.run_bulk_backtest(
        ["INFY"], *WINDOW, CAP, Mode.INTRADAY, exit_style="fixed", **KW)
    managed = backtester.run_bulk_backtest(
        ["INFY"], *WINDOW, CAP, Mode.INTRADAY, exit_style="partial_ladder", **KW)
    p, m = plain["INFY"], managed["INFY"]
    if p.trades.empty or m.trades.empty:
        pytest.skip("no trades in that window")
    # A managed run scales out, so it books MORE legs on the same setups.
    assert len(m.trades) > len(p.trades)
    assert m.metrics["Total Return %"] != p.metrics["Total Return %"]


def test_bulk_and_single_agree_on_the_same_symbol_and_style():
    """One symbol, one style, two code paths — the numbers must match, or the
    screen is ranking on something the detail view will not reproduce."""
    _skip_unless()
    style = "partial_ladder"
    single = backtester.run_backtest("INFY", *WINDOW, CAP, Mode.INTRADAY,
                                     exit_style=style, **KW)
    bulk = backtester.run_bulk_backtest(["INFY"], *WINDOW, CAP, Mode.INTRADAY,
                                        exit_style=style, **KW)["INFY"]
    if single.trades.empty:
        pytest.skip("no trades in that window")
    assert bulk.metrics["Total Return %"] == single.metrics["Total Return %"]
    assert len(bulk.trades) == len(single.trades)


def test_the_stop_band_reaches_bulk_too():
    _skip_unless()
    a = backtester.run_bulk_backtest(["INFY"], *WINDOW, CAP, Mode.INTRADAY,
                                     exit_style="fixed", **KW)["INFY"]
    b = backtester.run_bulk_backtest(["INFY"], *WINDOW, CAP, Mode.INTRADAY,
                                     exit_style="fixed", max_stop_pct=0.5,
                                     **KW)["INFY"]
    if a.trades.empty or b.trades.empty:
        pytest.skip("no trades in that window")
    widest = (b.trades.risk_dist / b.trades.entry * 100).max()
    assert widest <= 0.5 + 0.02, "max_stop_pct was ignored by the bulk path"


# --------------------------------------------------------------------------- #
#  The two job-based surfaces: the combination search and the multi-axis
#  funnel. Both run for minutes and both rank symbols, so a default that
#  measures the wrong bot is expensive in a way a single run is not.
# --------------------------------------------------------------------------- #
def test_the_funnel_request_and_spec_carry_the_run_shape():
    from api.routers.bulk_backtest import FunnelRequest
    from bulk_backtest.funnel import FunnelSpec
    missing = [k for k in RUN_SHAPE if k not in FunnelRequest.model_fields]
    assert not missing, f"FunnelRequest cannot express: {missing}"
    spec = FunnelSpec(symbols=["INFY"], strategy_keys=["candlestick_engine"],
                      start="2026-01-01", end="2026-08-31",
                      exit_style="partial_ladder", min_stop_pct=0.8)
    assert set(spec.run_shape()) == set(RUN_SHAPE)
    assert spec.run_shape()["exit_style"] == "partial_ladder"


def test_every_funnel_round_forwards_the_shape():
    """Five rounds each call run_with_costs. One that forgets would screen a
    different bot from the round before it, and the funnel's whole output is a
    comparison ACROSS rounds."""
    import bulk_backtest.funnel as funnel
    src = inspect.getsource(funnel)
    calls = src.count("bulk_backtester.run_with_costs(")
    forwards = src.count("**spec.run_shape()")
    assert calls > 0
    assert forwards == calls, (
        f"{calls} run_with_costs calls but only {forwards} forward the shape")


def test_both_job_routers_send_it_on():
    for mod, req in (("api.routers.advanced_backtest", "SearchRequest"),
                     ("api.routers.bulk_backtest", "FunnelRequest")):
        m = __import__(mod, fromlist=["x"])
        src = inspect.getsource(m)
        assert "exit_style=req.exit_style" in src, f"{mod} drops exit_style"
