"""The backtest chart endpoint — candles + the run's own trades drawn on them.

The feature's whole value rests on ONE property: the chart must show the run
the metrics table is describing. It is served by a second call that re-runs the
simulation, so the two could in principle disagree — a different bar size, a
different exit style, a stale cache. These tests pin that down:

  * the chart's trades are the SAME trades /backtest/run reports,
  * they are drawn on the SAME timeframe the simulation walked,
  * every timestamp lands on a real candle rather than 5.5 hours off it (the
    IST-labelled-as-UTC convention lightweight-charts forces), and
  * the payload is plain JSON — a numpy scalar leaking through would 500 the
    endpoint at render time, long after the tests passed.
"""
from __future__ import annotations

import json

import pytest

import backtester
import config
from api.routers.backtest import backtest_chart
from api.schemas import BacktestRequest
from config import Mode


WINDOW = dict(start="2026-01-01", end="2026-03-31", initial_capital=200_000.0)


def _req(**kw):
    base = dict(ticker="RELIANCE", mode="Intraday", **WINDOW)
    base.update(kw)
    return BacktestRequest(**base)


@pytest.fixture(scope="module")
def payload():
    if "RELIANCE" not in config.INSTRUMENTS_BY_SYMBOL:
        pytest.skip("RELIANCE is not in this deployment's instrument list")
    out = backtest_chart(_req(exit_style="partial_lock"))
    if not out["trades"]:
        pytest.skip("no trades in that window")
    return out


def test_chart_trades_match_the_run_it_claims_to_draw(payload):
    """The one property the feature lives or dies on."""
    run = backtester.run_backtest(
        "RELIANCE", WINDOW["start"], WINDOW["end"], WINDOW["initial_capital"],
        Mode.INTRADAY, exit_style="partial_lock", include_costs=True)
    assert len(payload["trades"]) == len(run.trades)
    for drawn, (_i, actual) in zip(payload["trades"], run.trades.iterrows()):
        assert drawn["entry_price"] == pytest.approx(actual["entry"])
        assert drawn["exit_price"] == pytest.approx(actual["exit"])
        assert drawn["side"] == actual["side"]
        assert drawn["exit_reason"] == actual["exit_reason"]
        assert drawn["quantity"] == int(actual["qty"])
        # Cost-aware run -> the chart's win/lose colouring follows the NET
        # money, so it can never contradict the table printed beside it.
        assert drawn["pnl"] == pytest.approx(round(actual["net_pnl"], 2))


def test_drawn_on_the_timeframe_the_simulation_walked(payload):
    assert payload["interval"] == backtester.interval_for(Mode.INTRADAY, 0)


def test_a_five_minute_run_is_drawn_on_five_minute_bars():
    """The failure this prevents: a 5m run drawn on 15m candles puts every
    marker on the wrong bar, and the chart quietly lies about the entry."""
    out = backtest_chart(_req(timeframe_minutes=5))
    assert out["interval"] == "5m"
    if out["trades"]:
        times = {c["time"] for c in out["candles"]}
        # 5m bars are 300s apart; an entry must land ON one of them.
        assert out["trades"][0]["entry_time"] in times


def test_every_marker_lands_on_a_real_candle(payload):
    """Candles and trades must share one time convention. If they drift, the
    markers are silently hours away from the bars they describe."""
    times = {c["time"] for c in payload["candles"]}
    for t in payload["trades"]:
        assert t["entry_time"] in times, "entry is not on any candle"
        assert t["exit_time"] in times, "exit is not on any candle"


def test_levels_are_the_ones_the_trade_opened_with(payload):
    """stop/target are reconstructed from risk_dist and rr, so a wrong sign
    would draw the risk box on the profitable side of a short."""
    for t in payload["trades"]:
        if t["stop_loss"] is None or t["target"] is None:
            continue
        if t["side"] == "BUY":
            assert t["stop_loss"] < t["entry_price"] < t["target"]
        else:
            assert t["target"] < t["entry_price"] < t["stop_loss"]


def test_payload_is_plain_json(payload):
    """FastAPI serialises at RESPONSE time, so a numpy scalar that survives
    every assertion above still 500s the endpoint in the browser."""
    json.dumps(payload)          # raises TypeError on numpy/pandas scalars
    for t in payload["trades"]:
        assert type(t["quantity"]) is int
        assert type(t["win"]) is bool
        assert type(t["entry_time"]) is int


def test_unknown_ticker_and_bad_style_are_rejected():
    from fastapi import HTTPException
    with pytest.raises(HTTPException) as e:
        backtest_chart(_req(ticker="NOT_A_SYMBOL"))
    assert e.value.status_code == 400
    with pytest.raises(HTTPException) as e:
        backtest_chart(_req(exit_style="nonsense"))
    assert e.value.status_code == 400
