"""
api/routers/backtest.py
Wraps backtester.run_backtest exactly as app.py's "Run Backtest" button does
— same inputs, same BacktestResult, only the rendering differs (JSON instead
of st.dataframe/plotly). Bulk/save-report parity (app.py's "Bulk Backtest"
and "Saved Backtest Analyses" sections) is deferred to a follow-up pass —
see frontend_migration_plan.md.
"""
from __future__ import annotations

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException

import backtester
import config
from api.auth import require_admin
from api.schemas import BacktestRequest, BulkBacktestRequest, RRSweepRequest

#: Ceiling on one bulk request. Each symbol is a full simulation; the executor
#: runs 5 at a time, so 40 is a few minutes rather than an unbounded wait.
BULK_MAX_TICKERS = 40
from config import Mode

router = APIRouter(prefix="/backtest", tags=["backtest"], dependencies=[Depends(require_admin)])


def _jsonable_trades(trades: pd.DataFrame) -> list[dict]:
    if trades.empty:
        return []
    t = trades.copy()
    for col in ("entry_time", "exit_time"):
        if col in t.columns:
            t[col] = t[col].astype(str)
    return t.to_dict("records")


#: Bounds on a requested Intraday bar-size override. 1 = effectively Scalper's
#: own grain (that mode exists for a reason — this endpoint doesn't stop you,
#: but it won't silently accept nonsense either); 60 = a bar coarser than this
#: is Swing's job, not an Intraday override.
_MIN_TF_MINUTES, _MAX_TF_MINUTES = 1, 60


def _check_exit_style(req) -> None:
    """Reject a typo'd style here rather than deep inside a run."""
    if not config.is_valid_exit_style(req.exit_style):
        raise HTTPException(
            400, f"Unknown exit_style {req.exit_style!r} — use one of "
                 f"{', '.join(config.EXIT_STYLES)}.")
    if not (req.partial_exit_fraction < 0 or 0 <= req.partial_exit_fraction < 1):
        raise HTTPException(
            400, "partial_exit_fraction must be between 0 and 1 (or -1 to "
                 "leave the style's own).")
    for name, v in (("max_stop_pct", req.max_stop_pct),
                    ("min_stop_pct", req.min_stop_pct)):
        if v and not (0 < v <= 20):
            raise HTTPException(
                400, f"{name} is a PERCENT of price — use 0 to leave the "
                     f"strategy's ATR stop alone, or a value up to 20.")
    if req.max_stop_pct and req.min_stop_pct and req.min_stop_pct > req.max_stop_pct:
        raise HTTPException(400, "min_stop_pct cannot exceed max_stop_pct.")


def _check_timeframe(mode: "Mode", timeframe_minutes: int) -> None:
    if not timeframe_minutes:
        return
    if mode != Mode.INTRADAY:
        raise HTTPException(
            400, "timeframe_minutes only overrides Intraday's bar size — "
                 f"{mode.value} is not Intraday.")
    if not (_MIN_TF_MINUTES <= timeframe_minutes <= _MAX_TF_MINUTES):
        raise HTTPException(
            400, f"timeframe_minutes must be between {_MIN_TF_MINUTES} and "
                 f"{_MAX_TF_MINUTES}.")


def _epoch(ts) -> int:
    """A tz-naive IST timestamp -> the UNIX second lightweight-charts must be
    given for that wall-clock to APPEAR on its axis.

    The chart library renders UNIX seconds in UTC and has no timezone support,
    so the convention (identical to api/routers/chart.py, which draws live
    trades the same way) is to label IST wall-clock AS IF it were UTC. Candles
    and trades get the same treatment, so a 09:20 entry lands on the 09:15 bar
    it belongs to rather than 5-and-a-half hours away from it.
    """
    return int(pd.Timestamp(ts).tz_localize("UTC").timestamp())


def _chart_trades(trades: pd.DataFrame, strategy_key: str,
                  mode: str) -> list[dict]:
    """Backtest legs -> the ChartTrade shape TradeChart already understands.

    The two levels the chart draws its risk and reward boxes from are
    RECONSTRUCTED rather than stored, from columns the trade log already
    carries:

        stop  = entry -/+ risk_dist          (risk_dist IS 1R, written per leg)
        target= entry +/- rr x risk_dist     (rr is the run's risk_reward)

    They are the levels the position was OPENED with, which is what you want
    to see: where the trade was risking to and aiming at when it was taken.
    A managed style moves both afterwards — that movement is visible as the
    gap between the drawn target and where the exit arrow actually lands.

    `pnl` is the NET figure when the run was cost-aware, so the chart's
    win/lose colouring can never disagree with the P&L in the table beside it.
    """
    if trades.empty:
        return []
    out = []
    for i, t in trades.reset_index(drop=True).iterrows():
        side = str(t.get("side", "BUY"))
        sign = 1 if side == "BUY" else -1
        entry = float(t["entry"])
        risk = float(t.get("risk_dist") or 0.0)
        rr = float(t.get("rr") or 0.0)
        pnl = float(t["net_pnl"] if "net_pnl" in trades.columns else t["pnl"])
        out.append({
            # position_id groups the legs of one trade; leg orders them. Both
            # are in the id so the two rows of a scaled-out trade are distinct
            # to the chart but obviously related to a reader.
            "id": f"{int(t.get('position_id', i))}-{int(t.get('leg', 1))}",
            "side": side,
            "entry_time": _epoch(t["entry_time"]),
            "entry_price": entry,
            "exit_time": _epoch(t["exit_time"]),
            "exit_price": float(t["exit"]),
            "stop_loss": round(entry - sign * risk, 2) if risk else None,
            "target": round(entry + sign * rr * risk, 2) if risk and rr else None,
            "quantity": int(t.get("qty") or 0),
            "pnl": round(pnl, 2),
            "win": bool(t.get("win", pnl > 0)),
            "strategy": strategy_key,
            "mode": mode,
            "entry_reason": str(t.get("entry_reason", "") or ""),
            "exit_reason": str(t.get("exit_reason", "") or ""),
        })
    return out


@router.post("/run")
def run_backtest(req: BacktestRequest):
    if req.ticker not in config.INSTRUMENTS_BY_SYMBOL:
        raise HTTPException(400, f"Unknown ticker: {req.ticker}")
    try:
        mode = Mode(req.mode)
    except ValueError:
        raise HTTPException(400, "Invalid mode.")
    inst = config.INSTRUMENTS_BY_SYMBOL[req.ticker]
    try:
        filters = backtester.parse_filters(req.trade_days, req.trade_hours,
                                           req.side)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    _check_timeframe(mode, req.timeframe_minutes)
    _check_exit_style(req)
    result = backtester.run_backtest(
        req.ticker, req.start, req.end, req.initial_capital, mode,
        lot_size=inst.lot_size, strategy_key=req.strategy_key,
        risk_reward=req.risk_reward, min_score=req.min_score,
        filters=filters, patterns=req.patterns,
        timeframe_minutes=req.timeframe_minutes,
        # What happens AFTER the first target — the partial, the break-even
        # move, the trail. Decided by the same exit_manager the live engine
        # runs, so this measures the bot rather than an idealised version of
        # it. "strategy" (the default) changes nothing.
        exit_style=req.exit_style, trail_atr_mult=req.trail_atr_mult,
        partial_exit_fraction=req.partial_exit_fraction,
        runner_rr_mult=req.runner_rr_mult,
        # Bounds on the ATR stop distance. Unlike the exit style these
        # move the ENTRY levels, so a run at a different band is a
        # different strategy — see strategy.clamp_signal_risk.
        max_stop_pct=req.max_stop_pct, min_stop_pct=req.min_stop_pct,
        # Let the position live past 15:09 and close on its own stop, target
        # or trail instead. Off by default; see run_backtest's docstring for
        # why the resulting net figure is optimistic (delivery costs and
        # overnight gap fills are both unpriced).
        hold_overnight=req.hold_overnight,
        # Independent knob: this one changes how many trades are TAKEN,
        # hold_overnight changes how they CLOSE. Moving both in one run
        # attributes the difference to neither.
        ignore_entry_cutoff=req.ignore_entry_cutoff,
        # The tab reports what the account would actually have kept. Brokerage,
        # STT, GST, stamp, SEBI and an estimated slippage come out of the
        # compounding capital, so Total Return / Max Drawdown / Sharpe here are
        # all NET; "Gross Return %" beside them is the old number.
        include_costs=True,
    )
    equity = result.equity_curve
    return {
        "metrics": result.metrics,
        "equity_curve": [{"t": str(ts), "equity": float(v)}
                         for ts, v in equity.items()] if not equity.empty else [],
        "trades": _jsonable_trades(result.trades),
        # Derived from the very same trade rows returned above, so the charts
        # and the table can never disagree.
        "analytics": backtester.trade_analytics(result.trades),
        "filters": filters.describe() if filters else "",
    }


@router.post("/chart")
def backtest_chart(req: BacktestRequest):
    """The same run as /backtest/run, drawn on its own candles.

    WHY THIS EXISTS. A metrics table tells you a strategy lost money; it cannot
    tell you WHERE it went wrong. Seeing the entries and exits on the bars they
    were decided on is how you find out that (say) half the trades never move
    and die at the square-off, or that the stop sits inside the noise. This
    feeds the SAME TradeChart component the live Trade Charts tab uses, so a
    backtested trade and a real one are read the same way.

    It re-runs the simulation rather than having /run carry candles in its
    response. run_backtest is deterministic for a given request — the history
    is disk-cached and the synthetic fallback is seeded per ticker — so the
    trades drawn here are byte-for-byte the ones /run reported. The
    alternative, shipping a few thousand bars with every /run, would weigh
    down the metrics path that most requests actually want.
    """
    if req.ticker not in config.INSTRUMENTS_BY_SYMBOL:
        raise HTTPException(400, f"Unknown ticker: {req.ticker}")
    try:
        mode = Mode(req.mode)
    except ValueError:
        raise HTTPException(400, "Invalid mode.")
    inst = config.INSTRUMENTS_BY_SYMBOL[req.ticker]
    try:
        filters = backtester.parse_filters(req.trade_days, req.trade_hours,
                                           req.side)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    _check_timeframe(mode, req.timeframe_minutes)
    _check_exit_style(req)

    result = backtester.run_backtest(
        req.ticker, req.start, req.end, req.initial_capital, mode,
        lot_size=inst.lot_size, strategy_key=req.strategy_key,
        risk_reward=req.risk_reward, min_score=req.min_score,
        filters=filters, patterns=req.patterns,
        timeframe_minutes=req.timeframe_minutes,
        exit_style=req.exit_style, trail_atr_mult=req.trail_atr_mult,
        partial_exit_fraction=req.partial_exit_fraction,
        runner_rr_mult=req.runner_rr_mult,
        # Bounds on the ATR stop distance. Unlike the exit style these
        # move the ENTRY levels, so a run at a different band is a
        # different strategy — see strategy.clamp_signal_risk.
        max_stop_pct=req.max_stop_pct, min_stop_pct=req.min_stop_pct,
        hold_overnight=req.hold_overnight,
        # Independent knob: this one changes how many trades are TAKEN,
        # hold_overnight changes how they CLOSE. Moving both in one run
        # attributes the difference to neither.
        ignore_entry_cutoff=req.ignore_entry_cutoff,
        include_costs=True)

    # The SAME bars the simulation walked. interval_for is shared with
    # run_backtest precisely so these cannot diverge — a chart drawn on 15m
    # bars for a 5m run would put every marker on the wrong candle.
    interval = backtester.interval_for(mode, req.timeframe_minutes)
    token = config.UPSTOX_LIVE_ACCESS_TOKEN or config.UPSTOX_SANDBOX_TOKEN
    try:
        bars_df, source = backtester.fetch_history(
            req.ticker, req.start, req.end, interval,
            instrument_key=inst.instrument_key, token=token)
    except Exception as exc:
        raise HTTPException(502, f"Could not fetch history: {exc}")
    if bars_df is None or bars_df.empty:
        raise HTTPException(404, "No candles for that window.")

    candles = [{
        "time": _epoch(ts),
        "open": round(float(r["open"]), 2), "high": round(float(r["high"]), 2),
        "low": round(float(r["low"]), 2), "close": round(float(r["close"]), 2),
        "volume": int(r["volume"]) if pd.notna(r.get("volume")) else 0,
    } for ts, r in bars_df.iterrows()]

    return {
        "symbol": req.ticker, "interval": interval, "mode": mode.value,
        "source": source, "candles": candles,
        "trades": _chart_trades(result.trades, req.strategy_key, mode.value),
        # Surfaced so a chart drawn on a random walk is never mistaken for one
        # drawn on the market — fetch_history falls back to synthetic data.
        "is_real_data": source in ("upstox", "yfinance", "binance"),
    }


@router.post("/bulk")
def bulk_backtest(req: BulkBacktestRequest):
    """Run one strategy across many symbols and rank them.

    Ranked by return, and every row carries its trade count so a chart-topping
    symbol with four trades is visibly not a finding. Analytics are computed on
    the POOLED trade log across all symbols — the day/hour/setup edges worth
    acting on are the ones that survive a whole bucket, not one lucky name.
    """
    if not req.tickers:
        raise HTTPException(400, "Select at least one symbol.")
    unknown = [t for t in req.tickers if t not in config.INSTRUMENTS_BY_SYMBOL]
    if unknown:
        raise HTTPException(400, f"Unknown ticker(s): {', '.join(unknown)}")
    _check_exit_style(req)
    if len(req.tickers) > BULK_MAX_TICKERS:
        raise HTTPException(
            400, f"{len(req.tickers)} symbols requested; the limit is "
                 f"{BULK_MAX_TICKERS} per run.")
    try:
        mode = Mode(req.mode)
    except ValueError:
        raise HTTPException(400, "Invalid mode.")
    try:
        filters = backtester.parse_filters(req.trade_days, req.trade_hours,
                                           req.side)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    _check_timeframe(mode, req.timeframe_minutes)

    results = backtester.run_bulk_backtest(
        req.tickers, req.start, req.end, req.initial_capital, mode,
        strategy_key=req.strategy_key, risk_reward=req.risk_reward,
        min_score=req.min_score, filters=filters, patterns=req.patterns,
        exit_style=req.exit_style, trail_atr_mult=req.trail_atr_mult,
        partial_exit_fraction=req.partial_exit_fraction,
        runner_rr_mult=req.runner_rr_mult,
        max_stop_pct=req.max_stop_pct, min_stop_pct=req.min_stop_pct,
        hold_overnight=req.hold_overnight,
        # Independent knob: this one changes how many trades are TAKEN,
        # hold_overnight changes how they CLOSE. Moving both in one run
        # attributes the difference to neither.
        ignore_entry_cutoff=req.ignore_entry_cutoff,
        timeframe_minutes=req.timeframe_minutes,
        # NET, like /backtest/run. Ranking symbols on GROSS would order them by
        # a number this book has repeatedly shown is dominated by the very cost
        # the ranking then ignores.
        include_costs=True,
    )
    summary = backtester.bulk_summary_frame(results)

    pooled = [r.trades for r in results.values()
              if r.trades is not None and not r.trades.empty]
    all_trades = pd.concat(pooled, ignore_index=True) if pooled else pd.DataFrame()

    return {
        "ranking": summary.to_dict("records") if not summary.empty else [],
        "analytics": backtester.trade_analytics(all_trades),
        "filters": filters.describe() if filters else "",
        "tickers": len(req.tickers),
    }


@router.post("/rr-sweep")
def rr_sweep(req: RRSweepRequest):
    """The same backtest at every RR in a ladder — one row each.

    Answers "which risk:reward actually suits this strategy on this symbol"
    in one request instead of re-running the form by hand. Only RR moves;
    every other input is held constant.
    """
    if req.ticker not in config.INSTRUMENTS_BY_SYMBOL:
        raise HTTPException(400, f"Unknown ticker: {req.ticker}")
    try:
        mode = Mode(req.mode)
    except ValueError:
        raise HTTPException(400, "Invalid mode.")
    inst = config.INSTRUMENTS_BY_SYMBOL[req.ticker]
    _check_timeframe(mode, req.timeframe_minutes)
    try:
        rows = backtester.run_rr_sweep(
            req.ticker, req.start, req.end, req.initial_capital, mode,
            rr_start=req.rr_start, rr_step=req.rr_step, rr_end=req.rr_end,
            lot_size=inst.lot_size, strategy_key=req.strategy_key,
            min_score=req.min_score, patterns=req.patterns,
            exit_style=req.exit_style, trail_atr_mult=req.trail_atr_mult,
            partial_exit_fraction=req.partial_exit_fraction,
            runner_rr_mult=req.runner_rr_mult,
            max_stop_pct=req.max_stop_pct, min_stop_pct=req.min_stop_pct,
            hold_overnight=req.hold_overnight,
            # Independent knob: this one changes how many trades are TAKEN,
            # hold_overnight changes how they CLOSE. Moving both in one run
            # attributes the difference to neither.
            ignore_entry_cutoff=req.ignore_entry_cutoff,
            include_costs=True,
            timeframe_minutes=req.timeframe_minutes,
        )
    except ValueError as exc:
        # Bad ladder (start<=0, end<start, step too small, too many runs) —
        # the message is written to be shown to the user as-is.
        raise HTTPException(400, str(exc))
    # `best` is by RETURN, and named rather than implied, so the table can
    # highlight it without the UI re-deriving a different notion of "best".
    ok = [r for r in rows if not r["error"]]
    best = max(ok, key=lambda r: r["return_pct"])["risk_reward"] if ok else None
    return {"rows": rows, "best_by_return": best}
