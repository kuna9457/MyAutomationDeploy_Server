"""
bulk_backtester.py
A SEPARATE backtesting wrapper that adds cost-awareness and enriched trade
logging to the existing backtester — without modifying backtester.py at all.

Design contract:
  * This file IMPORTS backtester.run_backtest as a READ-ONLY black box. It
    never patches, monkey-patches, subclasses, or modifies it.
  * The existing BacktestTab, AdvancedBacktestTab, and all their code paths
    are UNTOUCHED. They do not know this file exists.
  * New columns are added POST-HOC to the trade DataFrame that run_backtest
    already returns: stop, target, risk_amt, mfe_r, mae_r, bars_to_mfe,
    bars_to_1r, bars_held, weekday, hour, costs, net_pnl.

The funnel/attribution engine (bulk_backtest/attribute.py) reads these
columns — this is where they come from.
"""
from __future__ import annotations

import math
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
from typing import Optional

import numpy as np
import pandas as pd

import backtester
import config
from config import Mode, Segment
from cost_model import CostModel, INTRADAY_EQUITY, MCX_COMMODITY

# Match backtester's worker count.
BULK_MAX_WORKERS = backtester.BULK_MAX_WORKERS


# --------------------------------------------------------------------------- #
#  Enriched result
# --------------------------------------------------------------------------- #
@dataclass
class BulkBacktestResult:
    """Like BacktestResult, but with cost-adjusted metrics and enriched trades."""
    gross_metrics: dict          # the original, unmodified backtester metrics
    net_metrics: dict            # with costs deducted
    equity_curve_gross: pd.Series
    equity_curve_net: pd.Series
    trades: pd.DataFrame         # enriched with extra columns
    data_source: str


# --------------------------------------------------------------------------- #
#  Trade enrichment — post-process the trade DataFrame
# --------------------------------------------------------------------------- #
def _enrich_trades(
    trades: pd.DataFrame,
    history: pd.DataFrame,
    cost_model: CostModel,
    contract_multiplier: float = 1.0,
) -> pd.DataFrame:
    """Add columns to a trade log WITHOUT re-running the simulation.

    New columns:
      stop, target     — reconstructed from entry_reason (entry ± risk × RR)
      risk_amt         — |entry - stop| × qty × contract_multiplier
      mfe_r, mae_r     — max favourable/adverse excursion in R-multiples
      bars_held        — count of bars between entry and exit
      weekday, hour    — from entry_time, for attribution
      trade_cost       — round-trip cost from the cost model
      net_pnl          — pnl minus trade_cost
      turnover         — buy_turnover + sell_turnover
    """
    if trades is None or trades.empty:
        return trades

    df = trades.copy()

    # -- Timing columns for attribution -----------------------------------
    entry_ts = pd.to_datetime(df["entry_time"], errors="coerce")
    exit_ts = pd.to_datetime(df["exit_time"], errors="coerce")
    df["weekday"] = entry_ts.dt.weekday.values
    df["hour"] = entry_ts.dt.hour.values

    # -- Bars held --------------------------------------------------------
    if not history.empty and entry_ts.notna().any() and exit_ts.notna().any():
        idx = history.index
        bars = []
        for et, xt in zip(entry_ts, exit_ts):
            if pd.isna(et) or pd.isna(xt):
                bars.append(0)
                continue
            # Count bars between entry and exit timestamps
            mask = (idx >= et) & (idx <= xt)
            bars.append(int(mask.sum()))
        df["bars_held"] = bars
    else:
        df["bars_held"] = 0

    # -- Stop/target reconstruction ---------------------------------------
    # The position dict stores stop and target but run_backtest doesn't write
    # them to the trade row. We reconstruct from entry, exit, rr, and
    # exit_reason. If exit was STOP-LOSS, then exit == stop.
    # If exit was TARGET, then exit == target.
    stops = []
    targets = []
    risk_amts = []
    for _, row in df.iterrows():
        entry = float(row["entry"])
        exit_ = float(row["exit"])
        rr = float(row.get("rr", 2.0) or 2.0)
        side = str(row.get("side", "BUY"))
        exit_reason = str(row.get("exit_reason", ""))
        qty = float(row.get("qty", 1))

        if exit_reason == "STOP-LOSS":
            # exit == stop, target = entry + (entry - stop) * rr for long
            stop = exit_
            risk_dist = abs(entry - stop)
            if side == "BUY":
                target = entry + risk_dist * rr
            else:
                target = entry - risk_dist * rr
        elif exit_reason == "TARGET":
            # exit == target, stop = entry - (target - entry) / rr for long
            target = exit_
            if rr > 0:
                risk_dist = abs(target - entry) / rr
            else:
                risk_dist = abs(target - entry)
            if side == "BUY":
                stop = entry - risk_dist
            else:
                stop = entry + risk_dist
        else:
            # Time exit or other — estimate from RR
            # Try to infer from PnL direction
            if side == "BUY":
                # Approximate: assume typical ATR-based stop
                risk_dist = abs(entry - exit_)  # fallback
                stop = entry - risk_dist
                target = entry + risk_dist * rr
            else:
                risk_dist = abs(entry - exit_)
                stop = entry + risk_dist
                target = entry - risk_dist * rr

        risk_amt = risk_dist * qty * contract_multiplier
        stops.append(stop)
        targets.append(target)
        risk_amts.append(risk_amt)

    df["stop"] = stops
    df["target"] = targets
    df["risk_amt"] = risk_amts

    # -- MFE / MAE in R-multiples, plus HOW FAST (velocity, not just magnitude) --
    # Walk the history bars between entry and exit to find the extreme moves.
    # Two scrips can share an mfe_r of 2.0 and mean opposite things for RR: one
    # got there in 3 bars, the other crawled there in 40 and would have been
    # cut off by a time exit / EOD square-off on a faster-moving symbol's
    # schedule. bars_to_mfe / bars_to_1r capture that — in BAR units (multiply
    # by the mode's timeframe minutes for wall-clock time).
    mfe_rs = []
    mae_rs = []
    bars_to_mfe_list = []
    bars_to_1r_list = []
    for i, row in df.iterrows():
        et = pd.to_datetime(row["entry_time"])
        xt = pd.to_datetime(row["exit_time"])
        entry = float(row["entry"])
        stop = float(row["stop"])
        risk_dist = abs(entry - stop)
        side = str(row["side"])

        if risk_dist < 1e-9 or history.empty or pd.isna(et) or pd.isna(xt):
            mfe_rs.append(0.0)
            mae_rs.append(0.0)
            bars_to_mfe_list.append(None)
            bars_to_1r_list.append(None)
            continue

        # Slice the history to [entry_time, exit_time]
        mask = (history.index >= et) & (history.index <= xt)
        window = history[mask]
        if window.empty:
            mfe_rs.append(0.0)
            mae_rs.append(0.0)
            bars_to_mfe_list.append(None)
            bars_to_1r_list.append(None)
            continue

        # fav_r[k] = the R-multiple this trade was favourably extended to as of
        # bar k of the window — the same quantity mfe_r takes the max of, kept
        # per-bar here so we can also find WHEN the max (and 1R) first happened.
        if side == "BUY":
            fav_r = (window["high"].to_numpy(dtype=float) - entry) / risk_dist
            mae = float((entry - window["low"].min()) / risk_dist)
        else:
            fav_r = (entry - window["low"].to_numpy(dtype=float)) / risk_dist
            mae = float((window["high"].max() - entry) / risk_dist)

        mfe = float(fav_r.max()) if len(fav_r) else 0.0
        mfe_rs.append(round(max(0.0, mfe), 4))
        mae_rs.append(round(max(0.0, mae), 4))

        # None (not 0) when the milestone was never reached — 0 would read as
        # "reached instantly", which is a different fact.
        bars_to_mfe_list.append(int(fav_r.argmax()) if mfe > 0 else None)
        reached_1r = np.flatnonzero(fav_r >= 1.0)
        bars_to_1r_list.append(int(reached_1r[0]) if reached_1r.size else None)

    df["mfe_r"] = mfe_rs
    df["mae_r"] = mae_rs
    df["bars_to_mfe"] = bars_to_mfe_list
    df["bars_to_1r"] = bars_to_1r_list

    # -- Costs ------------------------------------------------------------
    costs = []
    for _, row in df.iterrows():
        entry = float(row["entry"])
        exit_ = float(row["exit"])
        qty = float(row["qty"])
        c = cost_model.round_trip_cost(entry, exit_, qty, contract_multiplier)
        costs.append(round(c, 2))

    df["trade_cost"] = costs
    df["net_pnl"] = df["pnl"].astype(float) - df["trade_cost"].astype(float)
    df["turnover"] = (
        df["entry"].astype(float) * df["qty"].astype(float) * contract_multiplier
        + df["exit"].astype(float) * df["qty"].astype(float) * contract_multiplier
    )

    return df


# --------------------------------------------------------------------------- #
#  Net metrics from enriched trades
# --------------------------------------------------------------------------- #
def _net_metrics(
    gross_metrics: dict,
    trades: pd.DataFrame,
    initial_capital: float,
) -> dict:
    """Build the net-of-costs metric set from enriched trades."""
    m = dict(gross_metrics)  # start from gross
    m["Gross Return %"] = m.pop("Total Return %", 0.0)

    if trades is None or trades.empty or "net_pnl" not in trades.columns:
        m["Net Return %"] = m["Gross Return %"]
        m["Total Costs"] = 0.0
        m["Turnover"] = 0.0
        m["Cost per Trade"] = 0.0
        return m

    total_costs = float(trades["trade_cost"].sum())
    total_net_pnl = float(trades["net_pnl"].sum())
    turnover = float(trades["turnover"].sum()) if "turnover" in trades.columns else 0.0
    n = len(trades)

    net_return_pct = round((total_net_pnl / initial_capital) * 100, 2)
    net_win_rate = (
        round(100.0 * (trades["net_pnl"] > 0).sum() / n, 2) if n > 0 else 0.0
    )

    m["Net Return %"] = net_return_pct
    m["Net Win Rate %"] = net_win_rate
    m["Total Costs"] = round(total_costs, 2)
    m["Turnover"] = round(turnover, 2)
    m["Cost per Trade"] = round(total_costs / n, 2) if n > 0 else 0.0
    m["Final Equity (Net)"] = round(initial_capital + total_net_pnl, 2)

    return m


# --------------------------------------------------------------------------- #
#  Main entry point: single-symbol, cost-aware backtest
# --------------------------------------------------------------------------- #
def run_with_costs(
    ticker: str,
    start: str,
    end: str,
    initial_capital: float,
    mode: Mode,
    strategy_key: str = "",
    risk_reward: float = 0.0,
    min_score: float = 0.0,
    lot_size: int = 1,
    filters: Optional[backtester.TradeFilters] = None,
    patterns: Optional[list[str]] = None,
    ignore_saved_patterns: bool = False,
    cost_model: Optional[CostModel] = None,
    timeframe_minutes: int = 0,
    exit_style: str = "strategy",
    trail_atr_mult: float = 0.0,
    partial_exit_fraction: float = -1.0,
    runner_rr_mult: float = -1.0,
    max_stop_pct: float = 0.0,
    min_stop_pct: float = 0.0,
    hold_overnight: bool = False,
    ignore_entry_cutoff: bool = False,
) -> BulkBacktestResult:
    """Run an existing backtest and post-process with costs + enrichment.

    Calls backtester.run_backtest() unchanged, then:
      1. Fetches the same history (from cache — instant) for MFE/MAE
      2. Enriches every trade row with extra columns
      3. Computes net metrics
    """
    # 1. Run the unmodified backtester
    result = backtester.run_backtest(
        ticker, start, end, initial_capital, mode,
        lot_size=lot_size, strategy_key=strategy_key,
        risk_reward=risk_reward, min_score=min_score,
        filters=filters, patterns=patterns,
        ignore_saved_patterns=ignore_saved_patterns,
        timeframe_minutes=timeframe_minutes,
        # The exit style decides WHAT BOT this is measuring. Left at
        # "strategy" a bulk screen ranks symbols on the plain fixed exit while
        # the single-symbol tab reports a managed one, and the two tabs then
        # disagree about the same symbol.
        exit_style=exit_style, trail_atr_mult=trail_atr_mult,
        partial_exit_fraction=partial_exit_fraction,
        runner_rr_mult=runner_rr_mult,
        max_stop_pct=max_stop_pct, min_stop_pct=min_stop_pct,
        # Removes the end-of-session flat-out, so a position runs to its own
        # stop/target/trail across sessions. Off by default; see
        # backtester.run_backtest's docstring for what the resulting number
        # does and does not mean (in particular: costs below are still priced
        # as INTRADAY, and a held-overnight position is delivery).
        hold_overnight=hold_overnight,
        # Independent of the flag above: this one changes how many trades are
        # taken, that one changes how they close. See run_backtest's docstring.
        ignore_entry_cutoff=ignore_entry_cutoff,
    )

    # 2. Determine cost model and contract multiplier
    inst = config.INSTRUMENTS_BY_SYMBOL.get(ticker)
    is_mcx = inst is not None and inst.segment == Segment.MCX
    cm = contract_multiplier = inst.contract_multiplier if inst else 1
    if cost_model is None:
        cost_model = MCX_COMMODITY if is_mcx else INTRADAY_EQUITY

    # 3. Load history from cache for MFE/MAE computation. MUST resolve to the
    # exact same interval run_backtest just traded on above — if this drifted
    # (e.g. always assuming 15m) MFE/MAE would be measured over a candle set
    # different from the one that produced the trades, silently.
    if mode == Mode.INTRADAY and timeframe_minutes and timeframe_minutes > 0:
        interval = f"{int(timeframe_minutes)}m"
    else:
        interval = {Mode.SWING: "1d", Mode.INTRADAY: "15m", Mode.SCALPER: "1m"}[mode]
    instrument_key = inst.instrument_key if inst else ""
    token = config.UPSTOX_LIVE_ACCESS_TOKEN or config.UPSTOX_SANDBOX_TOKEN
    try:
        history, source = backtester.fetch_history(
            ticker, start, end, interval,
            instrument_key=instrument_key, token=token)
    except Exception:
        history = pd.DataFrame()
        source = result.metrics.get("Data Source", "unknown")

    # 4. Enrich trades
    enriched = _enrich_trades(result.trades, history, cost_model, cm)

    # 5. Compute net metrics
    net_metrics = _net_metrics(result.metrics, enriched, initial_capital)

    # 6. Build net equity curve
    if enriched is not None and not enriched.empty and "net_pnl" in enriched.columns:
        net_cumulative = enriched["net_pnl"].cumsum()
        net_equity = initial_capital + net_cumulative
        net_equity.index = range(len(net_equity))
    else:
        net_equity = result.equity_curve.copy()

    return BulkBacktestResult(
        gross_metrics=result.metrics,
        net_metrics=net_metrics,
        equity_curve_gross=result.equity_curve,
        equity_curve_net=net_equity,
        trades=enriched,
        data_source=result.metrics.get("Data Source", "unknown"),
    )


# --------------------------------------------------------------------------- #
#  Multi-symbol bulk run with costs
# --------------------------------------------------------------------------- #
def run_bulk_with_costs(
    tickers: list[str],
    start: str,
    end: str,
    initial_capital: float,
    mode: Mode,
    strategy_key: str = "",
    risk_reward: float = 0.0,
    min_score: float = 0.0,
    filters: Optional[backtester.TradeFilters] = None,
    patterns: Optional[list[str]] = None,
    ignore_saved_patterns: bool = False,
    cost_model: Optional[CostModel] = None,
    progress_cb=None,
    max_workers: int = BULK_MAX_WORKERS,
    timeframe_minutes: int = 0,
    exit_style: str = "strategy",
    trail_atr_mult: float = 0.0,
    partial_exit_fraction: float = -1.0,
    runner_rr_mult: float = -1.0,
    max_stop_pct: float = 0.0,
    min_stop_pct: float = 0.0,
) -> dict[str, BulkBacktestResult]:
    """Run cost-aware backtests across multiple symbols concurrently.

    Same parallelism model as backtester.run_bulk_backtest — one thread per
    symbol, up to max_workers at a time.
    """
    total = len(tickers)

    def _run_one(ticker: str) -> tuple[str, BulkBacktestResult]:
        inst = config.INSTRUMENTS_BY_SYMBOL.get(ticker)
        lot = inst.lot_size if inst else 1
        try:
            return ticker, run_with_costs(
                ticker, start, end, initial_capital, mode,
                strategy_key=strategy_key, risk_reward=risk_reward,
                min_score=min_score, lot_size=lot,
                filters=filters, patterns=patterns,
                ignore_saved_patterns=ignore_saved_patterns,
                cost_model=cost_model, timeframe_minutes=timeframe_minutes,
                exit_style=exit_style, trail_atr_mult=trail_atr_mult,
                partial_exit_fraction=partial_exit_fraction,
                runner_rr_mult=runner_rr_mult,
                max_stop_pct=max_stop_pct, min_stop_pct=min_stop_pct,
            )
        except Exception as exc:
            print(f"[bulk_backtester] {ticker} failed ({exc}).")
            empty = pd.Series(dtype=float)
            empty_trades = pd.DataFrame()
            gross = backtester._metrics(empty, empty_trades, initial_capital,
                                         "15m", "error")
            return ticker, BulkBacktestResult(
                gross_metrics=gross,
                net_metrics=dict(gross, **{"Net Return %": 0.0,
                                           "Total Costs": 0.0}),
                equity_curve_gross=empty,
                equity_curve_net=empty,
                trades=empty_trades,
                data_source="error",
            )

    results: dict[str, BulkBacktestResult] = {}
    done = 0
    done_lock = threading.Lock()
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_run_one, t) for t in tickers]
        for future in as_completed(futures):
            ticker, result = future.result()
            results[ticker] = result
            with done_lock:
                done += 1
                done_snap = done
            if progress_cb:
                progress_cb(done_snap, total, ticker)

    return {t: results[t] for t in tickers if t in results}


def bulk_summary_frame(results: dict[str, BulkBacktestResult]) -> pd.DataFrame:
    """One-row-per-symbol comparison table, ranked by Net Return %."""
    rows = []
    for ticker, res in results.items():
        m = res.net_metrics
        rows.append({
            "Ticker": ticker,
            "Gross Return %": m.get("Gross Return %", 0.0),
            "Net Return %": m.get("Net Return %", 0.0),
            "Total Costs": m.get("Total Costs", 0.0),
            "Turnover": m.get("Turnover", 0.0),
            "Cost/Trade": m.get("Cost per Trade", 0.0),
            "Max Drawdown %": m.get("Max Drawdown %", 0.0),
            "Sharpe": m.get("Sharpe", 0.0),
            "Win Rate %": m.get("Win Rate %", 0.0),
            "Net Win Rate %": m.get("Net Win Rate %", 0.0),
            "Trades": m.get("Total Trades", 0),
            "Final Equity": m.get("Final Equity (Net)", 0.0),
            "Data Source": m.get("Data Source", ""),
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("Net Return %", ascending=False).reset_index(drop=True)
    return df
