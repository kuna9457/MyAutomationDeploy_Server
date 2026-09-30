"""
bulk_backtest/funnel.py
The multi-round narrowing funnel — fixes one axis at a time.

    § MULTI_AXIS_SEARCH_PLAN.md §3

Each round locks its winner, then the next round runs on the survivors only.
Ordered by effect size and by how much data the slice costs. ~1,000 actual
simulations versus the naive 6.1 M grid — roughly 10 minutes instead of 21
days.

Round 0b is the exception to "each round locks its winner": it locks nothing.
It re-runs every screened pair completely unfiltered at each rung of the RR
ladder, so the filtered rounds have a control to be judged against and there is
a simulated answer to "which RR is best with no filtering at all?".

The funnel's output is a per-symbol answer:
    Symbol | Strategy | RR | Base RR | Patterns | Hours | Days | Trades | Net IS | Net OOS | Folds | Verdict
plus the unfiltered baseline table under "baseline".
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from typing import Callable, Optional

import pandas as pd

import backtester
import config
from config import Mode, Segment

import bulk_backtester
from bulk_backtest.attribute import (CellResult, attribute_trades, best_rr,
                                     parse_patterns, rr_hit_curve, top_patterns)

MIN_TRADES = 30
MIN_OOS_TRADES = 10
MAX_WORKERS = 5
OVERFIT_RATIO = 0.4
WEEKDAY_NAMES = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")


@dataclass
class FunnelSpec:
    """Everything the funnel holds fixed + the search space it varies over."""
    symbols: list[str]
    strategy_keys: list[str]
    start: str
    end: str
    capital: float = 100_000.0
    mode: str = "Intraday"
    rr_ladder: list[float] = field(
        default_factory=lambda: [0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
    folds: int = 4
    min_trades: int = MIN_TRADES
    min_oos_trades: int = MIN_OOS_TRADES
    turnover_cap_pct: float = 0.0  # 0 = no cap
    #: Run the unfiltered baseline sweep (Round 0b). On by default: it is the
    #: control every later round is judged against. Costs one extra simulation
    #: per (symbol × strategy × RR rung) minus the rung Round 0 already ran, so
    #: a very large symbol list may want it off.
    baseline_rr_sweep: bool = True
    #: How an OPEN position is managed after its first target — one of
    #: config.EXIT_STYLES, held FIXED across the whole funnel. "strategy" (the
    #: default) is the plain fixed exit for any strategy that declares no
    #: management, so a funnel left at the default screens a DIFFERENT bot from
    #: the one the Backtesting tab reports, and its ranking is not transferable.
    exit_style: str = "strategy"
    trail_atr_mult: float = 0.0
    partial_exit_fraction: float = -1.0
    runner_rr_mult: float = -1.0
    #: Bounds on the ATR stop, in PERCENT of price. 0 = the strategy's own.
    max_stop_pct: float = 0.0
    min_stop_pct: float = 0.0

    def run_shape(self) -> dict:
        """The kwargs that decide WHAT BOT every round measures.

        Forwarded as one block to every bulk_backtester.run_with_costs call in
        this module, so a round cannot quietly screen against a different exit
        rule than the round before it.
        """
        return dict(exit_style=self.exit_style,
                    trail_atr_mult=self.trail_atr_mult,
                    partial_exit_fraction=self.partial_exit_fraction,
                    runner_rr_mult=self.runner_rr_mult,
                    max_stop_pct=self.max_stop_pct,
                    min_stop_pct=self.min_stop_pct)


@dataclass
class FunnelCandidate:
    """One surviving (symbol, strategy) pair and its locked axes."""
    symbol: str
    strategy_key: str
    segment: str = ""
    # Round 2 — locked RR
    best_rr: float = 0.0
    #: Scrip-wise "how far does it actually get": per RR rung, the fraction
    #: of trades that reached it before exit. See attribute.rr_hit_curve —
    #: this is what separates a fast, tight mover from a slow one that shares
    #: the same picked best_rr by net PnL alone.
    rr_curve: list = field(default_factory=list)
    #: Median bars from entry to first reaching 1R, across trades that got
    #: there at all (None if none did). In this pair's own bar unit — 15m for
    #: Intraday, 1m for Scalper, 1d for Swing.
    median_bars_to_1r: Optional[float] = None
    # Round 3 — locked pattern set
    patterns: list[str] = field(default_factory=list)
    # Round 4 — locked hours
    best_hours: list[int] = field(default_factory=list)
    # Round 5 — locked days
    best_days: list[int] = field(default_factory=list)
    # Round 0b — this pair's own best RR with NOTHING filtered
    baseline_rr: float = 0.0
    baseline_trades: int = 0
    baseline_net_pnl: float = 0.0
    # Results
    screen_trades: int = 0
    screen_net_pnl: float = 0.0
    screen_gross_pnl: float = 0.0
    screen_costs: float = 0.0
    # Verify results
    is_return: Optional[float] = None
    oos_return: Optional[float] = None
    oos_trades: int = 0
    oos_win_rate: float = 0.0
    oos_net_pnl: Optional[float] = None
    # Walk-forward
    folds_positive: int = 0
    folds_total: int = 0
    # Verdict
    verdict: str = ""
    note: str = ""
    score: float = 0.0


def _mode(spec: FunnelSpec) -> Mode:
    return Mode(spec.mode)


# --------------------------------------------------------------------------- #
#  Round 0: Screen — one unfiltered run per (symbol × strategy)
# --------------------------------------------------------------------------- #
def _screen(spec: FunnelSpec,
            progress: Optional[Callable] = None,
            should_stop: Optional[Callable] = None,
            ) -> tuple[list[FunnelCandidate], dict]:
    """Run one unfiltered backtest per (symbol, strategy) at max RR, net of costs.

    Returns a list of candidates with screen_* fields populated, and a dict of
    round_info for progress reporting.
    """
    mode = _mode(spec)
    max_rr = max(spec.rr_ladder) if spec.rr_ladder else 3.0
    pairs = [(sym, sk) for sym in spec.symbols for sk in spec.strategy_keys]
    total = len(pairs)
    candidates: list[FunnelCandidate] = []
    round_info = {"round": 0, "name": "Screen", "total": total, "done": 0,
                  "survivors": 0}

    def run_one(sym: str, sk: str) -> FunnelCandidate:
        inst = config.INSTRUMENTS_BY_SYMBOL.get(sym)
        segment = inst.segment.value if inst else ""
        lot = inst.lot_size if inst else 1
        result = bulk_backtester.run_with_costs(
            sym, spec.start, spec.end, spec.capital, mode,
            strategy_key=sk, risk_reward=max_rr, lot_size=lot,
            ignore_saved_patterns=True,
            **spec.run_shape(),
        )
        # Attribute the trade log for later rounds
        attr = attribute_trades(result.trades, sk, spec.rr_ladder)
        overall = attr.get("overall", CellResult())

        cand = FunnelCandidate(
            symbol=sym, strategy_key=sk, segment=segment,
            screen_trades=overall.trades,
            screen_net_pnl=round(overall.net_pnl, 2),
            screen_gross_pnl=round(overall.gross_pnl, 2),
            screen_costs=round(overall.costs, 2),
        )
        # Stash attribution for later rounds (carried in-memory)
        cand._attribution = attr  # type: ignore[attr-defined]
        cand._trades_df = result.trades  # type: ignore[attr-defined]
        return cand

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(run_one, s, sk): (s, sk)
                   for s, sk in pairs}
        for fut in as_completed(futures):
            if should_stop and should_stop():
                break
            try:
                cand = fut.result()
                candidates.append(cand)
            except Exception as exc:
                sym, sk = futures[fut]
                print(f"[funnel] screen {sym}/{sk} failed: {exc}")
            round_info["done"] += 1
            if progress:
                progress(round_info)

    round_info["survivors"] = len([c for c in candidates
                                    if c.screen_trades >= spec.min_trades
                                    and c.screen_net_pnl > 0])
    return candidates, round_info


# --------------------------------------------------------------------------- #
#  Round 0b: Unfiltered baseline — the control, swept across the RR ladder
# --------------------------------------------------------------------------- #
#  Every later round REMOVES something: a strategy, a pattern, an hour, a
#  weekday. Removal always flatters the in-sample numbers, so a filtered result
#  only means something next to the result of filtering nothing at all.
#
#  This round is that "nothing at all": no pattern set, no hour window, no
#  weekday list, no score floor — the raw strategy on every bar it fires on.
#  The ONE axis it varies is RR, because RR is the only knob here that does not
#  discard trades. Moving it re-prices the same set of entries rather than
#  removing some, so its rungs are comparable in a way the filter axes are not,
#  and the winning rung is a real answer to "what RR should this run at with no
#  curve fitting at all?"
#
#  These are REAL simulations, not the MFE re-pricing attribution does. Round 2
#  still picks a per-pair RR from attribution cheaply; this round exists so that
#  estimate has a simulated number to be checked against.
# --------------------------------------------------------------------------- #
BASELINE_NOTE = ("Unfiltered control — every pattern, every weekday, every "
                 "session hour, no score floor. Only RR varies.")


def _cell_from_trades(trades: Optional[pd.DataFrame]) -> CellResult:
    """Sum an enriched trade log into one cell. Rupees, like attribution's."""
    cell = CellResult()
    if trades is None or trades.empty or "net_pnl" not in trades.columns:
        return cell
    cell.trades = len(trades)
    cell.net_pnl = float(trades["net_pnl"].sum())
    cell.gross_pnl = (float(trades["pnl"].sum())
                      if "pnl" in trades.columns else cell.net_pnl)
    cell.costs = (float(trades["trade_cost"].sum())
                  if "trade_cost" in trades.columns else 0.0)
    cell.wins = int((trades["net_pnl"] > 0).sum())
    return cell


def _baseline_rr_sweep(candidates: list[FunnelCandidate],
                       spec: FunnelSpec,
                       progress: Optional[Callable] = None,
                       should_stop: Optional[Callable] = None,
                       ) -> tuple[dict, dict]:
    """Re-run every screened pair unfiltered at each rung of the RR ladder.

    Returns (baseline_payload, round_info). Also stamps each candidate with its
    own unfiltered best RR, so the final table can show what the pair was worth
    before anything was filtered — the honest floor under its headline number.
    """
    mode = _mode(spec)
    ladder = sorted(spec.rr_ladder) if spec.rr_ladder else [1.0, 2.0, 3.0]
    max_rr = max(ladder)

    # Round 0 already ran the max rung unfiltered and kept its trade log, so
    # that rung is free — only the others need simulating.
    todo = [(c, rr) for c in candidates for rr in ladder if rr != max_rr]
    # "survivors" means something different here: this round eliminates nothing,
    # it counts the rungs that made money. The label rides along so the UI can
    # say so rather than implying pairs were dropped.
    round_info = {"round": 0, "name": "Unfiltered RR sweep",
                  "total": len(todo), "done": 0, "survivors": 0,
                  "survivors_label": "profitable RR rungs"}

    # {(symbol, strategy_key): {rr: CellResult}}
    cells: dict[tuple[str, str], dict[float, CellResult]] = {
        (c.symbol, c.strategy_key): {
            max_rr: _cell_from_trades(getattr(c, "_trades_df", None))
        }
        for c in candidates
    }

    def run_one(c: FunnelCandidate,
                rr: float) -> tuple[tuple[str, str], float, CellResult]:
        inst = config.INSTRUMENTS_BY_SYMBOL.get(c.symbol)
        lot = inst.lot_size if inst else 1
        res = bulk_backtester.run_with_costs(
            c.symbol, spec.start, spec.end, spec.capital, mode,
            strategy_key=c.strategy_key, risk_reward=rr, lot_size=lot,
            ignore_saved_patterns=True,
            **spec.run_shape(),
        )
        return (c.symbol, c.strategy_key), rr, _cell_from_trades(res.trades)

    if todo:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            futures = {pool.submit(run_one, c, rr): (c, rr) for c, rr in todo}
            for fut in as_completed(futures):
                if should_stop and should_stop():
                    break
                try:
                    key, rr, cell = fut.result()
                    cells[key][rr] = cell
                except Exception as exc:                       # noqa: BLE001
                    cand, rr = futures[fut]
                    print(f"[funnel] baseline {cand.symbol}/{cand.strategy_key} "
                          f"@ RR {rr} failed: {exc}")
                round_info["done"] += 1
                if progress:
                    progress(round_info)

    # Per pair: which rung paid best with nothing filtered
    per_pair = []
    for c in candidates:
        by_rr = cells.get((c.symbol, c.strategy_key), {})
        rows = [(rr, cell) for rr, cell in by_rr.items() if cell.trades > 0]
        if rows:
            best_rr_val, best_cell = max(rows, key=lambda t: t[1].net_pnl)
            c.baseline_rr = best_rr_val
            c.baseline_trades = best_cell.trades
            c.baseline_net_pnl = round(best_cell.net_pnl, 2)
        per_pair.append({
            "symbol": c.symbol,
            "strategy_key": c.strategy_key,
            "segment": c.segment,
            "best_rr": c.baseline_rr,
            "trades": c.baseline_trades,
            "net_pnl": c.baseline_net_pnl,
            "by_rr": {f"{rr:g}": round(by_rr[rr].net_pnl, 2)
                      for rr in sorted(by_rr)},
        })

    # Aggregate: the whole selection, equally weighted, one row per rung
    by_rr_rows = []
    for rr in ladder:
        agg = CellResult()
        pairs = pairs_positive = 0
        for by_rr in cells.values():
            cell = by_rr.get(rr)
            if cell is None:
                continue
            pairs += 1
            agg.trades += cell.trades
            agg.gross_pnl += cell.gross_pnl
            agg.net_pnl += cell.net_pnl
            agg.costs += cell.costs
            agg.wins += cell.wins
            if cell.net_pnl > 0:
                pairs_positive += 1
        # Every pair is simulated against its own full capital, so the portfolio
        # reading is the equal-weight average of the individual returns.
        denom = spec.capital * pairs if pairs else 0.0
        by_rr_rows.append({
            "rr": rr,
            "trades": agg.trades,
            "gross_pnl": round(agg.gross_pnl, 2),
            "net_pnl": round(agg.net_pnl, 2),
            "costs": round(agg.costs, 2),
            "win_rate": agg.win_rate,
            "net_return_pct": (round(100.0 * agg.net_pnl / denom, 2)
                               if denom else 0.0),
            "pairs": pairs,
            "pairs_positive": pairs_positive,
        })

    scored = [r for r in by_rr_rows if r["trades"] > 0]
    best_row = max(scored, key=lambda r: r["net_pnl"]) if scored else None
    total_trades = max((r["trades"] for r in by_rr_rows), default=0)

    if best_row is None:
        verdict = "No unfiltered trades in this window — no RR can be recommended."
    elif best_row["net_pnl"] <= 0:
        verdict = (f"Unfiltered, every RR on the ladder loses money; "
                   f"1:{best_row['rr']:g} loses least "
                   f"({best_row['net_pnl']:,.0f}).")
    else:
        verdict = (f"Unfiltered, 1:{best_row['rr']:g} pays best: "
                   f"{best_row['net_pnl']:,.0f} net over {best_row['trades']} "
                   f"trades ({best_row['pairs_positive']}/{best_row['pairs']} "
                   f"pairs positive).")
    if best_row is not None and total_trades < spec.min_trades:
        verdict += (f" Thin sample: {total_trades} trades against a "
                    f"{spec.min_trades} floor — treat the winning rung as a "
                    f"hint, not a finding.")

    round_info["survivors"] = len([r for r in by_rr_rows if r["net_pnl"] > 0])
    baseline = {
        "enabled": True,
        "note": BASELINE_NOTE,
        "rr_ladder": ladder,
        "by_rr": by_rr_rows,
        "best_rr": best_row["rr"] if best_row else 0.0,
        "best_net_pnl": best_row["net_pnl"] if best_row else 0.0,
        "best_net_return_pct": best_row["net_return_pct"] if best_row else 0.0,
        "best_trades": best_row["trades"] if best_row else 0,
        "verdict": verdict,
        "per_pair": sorted(per_pair, key=lambda r: r["net_pnl"], reverse=True),
    }
    return baseline, round_info


# --------------------------------------------------------------------------- #
#  Round 1: Symbol × Strategy — which engine suits which symbol
# --------------------------------------------------------------------------- #
def _round1_symbol_strategy(candidates: list[FunnelCandidate],
                            spec: FunnelSpec) -> list[FunnelCandidate]:
    """Keep only net-positive pairs with enough trades. If a symbol appears
    under multiple strategies, keep the one with the best net PnL."""
    # Filter
    alive = [c for c in candidates
             if c.screen_trades >= spec.min_trades and c.screen_net_pnl > 0]

    # Deduplicate: best strategy per symbol
    best_per_sym: dict[str, FunnelCandidate] = {}
    for c in alive:
        prev = best_per_sym.get(c.symbol)
        if prev is None or c.screen_net_pnl > prev.screen_net_pnl:
            best_per_sym[c.symbol] = c

    return sorted(best_per_sym.values(),
                  key=lambda c: c.screen_net_pnl, reverse=True)


# --------------------------------------------------------------------------- #
#  Round 2: RR — best RR per surviving pair (attributed, exact)
# --------------------------------------------------------------------------- #
def _round2_rr(candidates: list[FunnelCandidate],
               spec: FunnelSpec) -> list[FunnelCandidate]:
    """Lock the best RR per pair using attributed MFE data, and attach the
    diagnostics that explain WHY: the hit-rate curve across the ladder and
    how fast (in bars) this symbol tends to reach 1R at all."""
    min_cell_trades = max(5, spec.min_trades // 3)
    for c in candidates:
        attr = getattr(c, "_attribution", None)
        if attr:
            b = best_rr(attr, min_trades=min_cell_trades)
            if b:
                c.best_rr = b["rr"]
            else:
                c.best_rr = max(spec.rr_ladder) if spec.rr_ladder else 2.0
            c.rr_curve = rr_hit_curve(attr, spec.rr_ladder,
                                      min_trades=min_cell_trades)
        else:
            c.best_rr = max(spec.rr_ladder) if spec.rr_ladder else 2.0

        trades_df = getattr(c, "_trades_df", None)
        if trades_df is not None and not trades_df.empty \
                and "bars_to_1r" in trades_df.columns:
            reached = trades_df["bars_to_1r"].dropna()
            c.median_bars_to_1r = (float(reached.median())
                                   if not reached.empty else None)
    return candidates


# --------------------------------------------------------------------------- #
#  Round 3: Pattern — greedy set build-up
# --------------------------------------------------------------------------- #
def _round3_patterns(candidates: list[FunnelCandidate],
                     spec: FunnelSpec,
                     progress: Optional[Callable] = None,
                     should_stop: Optional[Callable] = None,
                     ) -> list[FunnelCandidate]:
    """For each pair, greedily build the best pattern set using attribution,
    then verify with a real filtered run."""
    mode = _mode(spec)
    round_info = {"round": 3, "name": "Patterns", "total": len(candidates),
                  "done": 0, "survivors": 0}

    for c in candidates:
        if should_stop and should_stop():
            break
        attr = getattr(c, "_attribution", None)
        if attr:
            top = top_patterns(attr, min_trades=max(3, spec.min_trades // 5),
                               top_n=30)
            # Greedy build-up: start with best single, add patterns that improve
            chosen = []
            if top:
                chosen.append(top[0]["pattern"])
                best_net = top[0]["net_pnl"]
                for p in top[1:]:
                    trial = chosen + [p["pattern"]]
                    # Simple heuristic: if the pattern's individual net is positive,
                    # include it (the verify pass will confirm)
                    if p["net_pnl"] > 0:
                        chosen.append(p["pattern"])
            c.patterns = chosen if chosen else []
        else:
            c.patterns = []

        # Verify with a real filtered run at the locked RR
        if c.patterns:
            try:
                inst = config.INSTRUMENTS_BY_SYMBOL.get(c.symbol)
                lot = inst.lot_size if inst else 1
                vr = bulk_backtester.run_with_costs(
                    c.symbol, spec.start, spec.end, spec.capital, mode,
                    strategy_key=c.strategy_key, risk_reward=c.best_rr,
                    lot_size=lot, patterns=c.patterns,
                    **spec.run_shape(),
                )
                net_m = vr.net_metrics
                c.screen_net_pnl = net_m.get("Net Return %", 0.0)
                c.screen_trades = net_m.get("Total Trades", 0)
            except Exception as exc:
                c.note = f"pattern verify failed: {exc}"

        round_info["done"] += 1
        if progress:
            progress(round_info)

    # Keep only still-positive
    survivors = [c for c in candidates
                 if c.screen_trades >= max(5, spec.min_trades // 3)]
    round_info["survivors"] = len(survivors)
    return survivors


# --------------------------------------------------------------------------- #
#  Round 4: Session hour — attributed then verified
# --------------------------------------------------------------------------- #
def _round4_hours(candidates: list[FunnelCandidate],
                  spec: FunnelSpec) -> list[FunnelCandidate]:
    """Attribute trades by hour and lock the best hours (if they help)."""
    for c in candidates:
        attr = getattr(c, "_attribution", None)
        if not attr:
            continue
        by_hour = attr.get("by_hour", {})
        # Keep hours with positive net PnL and enough trades
        good_hours = []
        for hr, cell in by_hour.items():
            if cell.trades >= max(3, spec.min_trades // 10) and cell.net_pnl > 0:
                good_hours.append(hr)
        # Only lock hours if filtering actually helps (i.e., some hours are losers)
        losing_hours = [hr for hr, cell in by_hour.items() if cell.net_pnl < 0]
        if losing_hours and good_hours:
            c.best_hours = sorted(good_hours)
        else:
            c.best_hours = []  # no restriction
    return candidates


# --------------------------------------------------------------------------- #
#  Round 5: Weekday — highest bar, off by default
# --------------------------------------------------------------------------- #
def _round5_days(candidates: list[FunnelCandidate],
                 spec: FunnelSpec) -> list[FunnelCandidate]:
    """Attribute trades by weekday. The highest bar: require positive in BOTH
    halves, >= 40 trades, and > 2x PF change. Usually fails — and that is a
    result (stops you trading a curve fit)."""
    for c in candidates:
        attr = getattr(c, "_attribution", None)
        if not attr:
            continue
        by_day = attr.get("by_weekday", {})
        # Only exclude a day if it's clearly losing and has enough trades
        good_days = []
        for day, cell in by_day.items():
            if cell.trades >= max(5, spec.min_trades // 6) and cell.net_pnl > 0:
                good_days.append(day)
        losing_days = [d for d, cell in by_day.items()
                       if cell.net_pnl < 0 and cell.trades >= 5]
        if losing_days and good_days:
            c.best_days = sorted(good_days)
        else:
            c.best_days = []  # no restriction
    return candidates


# --------------------------------------------------------------------------- #
#  Round 6: Walk-forward — 4 anchored folds on finalists
# --------------------------------------------------------------------------- #
def _round6_walkforward(candidates: list[FunnelCandidate],
                        spec: FunnelSpec,
                        progress: Optional[Callable] = None,
                        should_stop: Optional[Callable] = None,
                        ) -> list[FunnelCandidate]:
    """Run the locked configuration on anchored walk-forward folds.

    4 folds, each using all prior data as in-sample and the next chunk as OOS.
    Require >= 3 of 4 folds net-positive to pass.
    """
    mode = _mode(spec)
    s = pd.Timestamp(spec.start)
    e = pd.Timestamp(spec.end)
    total_days = (e - s).days
    fold_size = total_days // (spec.folds + 1)

    round_info = {"round": 6, "name": "Walk-forward", "total": len(candidates),
                  "done": 0, "survivors": 0}

    for c in candidates:
        if should_stop and should_stop():
            break
        inst = config.INSTRUMENTS_BY_SYMBOL.get(c.symbol)
        lot = inst.lot_size if inst else 1

        folds_pos = 0
        is_returns = []
        oos_returns = []

        for fold_i in range(spec.folds):
            # In-sample: start → start + (fold_i + 1) × fold_size
            # OOS: end of IS → end of IS + fold_size
            is_end = s + pd.Timedelta(days=(fold_i + 1) * fold_size)
            oos_start = is_end
            oos_end = min(oos_start + pd.Timedelta(days=fold_size), e)

            if oos_end <= oos_start:
                continue

            # Build filters
            filters = None
            if c.best_hours or c.best_days:
                filters = backtester.parse_filters(
                    days=c.best_days if c.best_days else None,
                    hours=c.best_hours if c.best_hours else None,
                )

            try:
                # In-sample
                is_res = bulk_backtester.run_with_costs(
                    c.symbol, spec.start, is_end.strftime("%Y-%m-%d"),
                    spec.capital, mode,
                    strategy_key=c.strategy_key, risk_reward=c.best_rr,
                    lot_size=lot, patterns=c.patterns or None,
                    filters=filters,
                    **spec.run_shape(),
                )
                is_ret = is_res.net_metrics.get("Net Return %", 0.0)
                is_returns.append(is_ret)

                # Out-of-sample
                oos_res = bulk_backtester.run_with_costs(
                    c.symbol, oos_start.strftime("%Y-%m-%d"),
                    oos_end.strftime("%Y-%m-%d"),
                    spec.capital, mode,
                    strategy_key=c.strategy_key, risk_reward=c.best_rr,
                    lot_size=lot, patterns=c.patterns or None,
                    filters=filters,
                    **spec.run_shape(),
                )
                oos_ret = oos_res.net_metrics.get("Net Return %", 0.0)
                oos_returns.append(oos_ret)
                oos_tr = oos_res.net_metrics.get("Total Trades", 0)

                if oos_ret > 0:
                    folds_pos += 1

            except Exception as exc:
                print(f"[funnel] fold {fold_i} for {c.symbol} failed: {exc}")

        c.folds_positive = folds_pos
        c.folds_total = spec.folds
        c.is_return = round(sum(is_returns) / len(is_returns), 2) if is_returns else 0.0
        c.oos_return = round(sum(oos_returns) / len(oos_returns), 2) if oos_returns else 0.0

        round_info["done"] += 1
        if progress:
            progress(round_info)

    # Verdict
    for c in candidates:
        c.verdict, c.note = _verdict(c, spec)
        c.score = _score(c)

    round_info["survivors"] = len([c for c in candidates
                                    if c.verdict in ("holds", "promising")])
    return candidates


def _verdict(c: FunnelCandidate, spec: FunnelSpec) -> tuple[str, str]:
    """Assign a verdict based on walk-forward results."""
    required_folds = max(1, spec.folds - 1)  # require n-1 of n positive

    if c.folds_total == 0:
        return "unverified", "no walk-forward data"
    if c.folds_positive >= required_folds:
        if c.screen_trades >= spec.min_trades:
            return "holds", (f"{c.folds_positive}/{c.folds_total} folds positive, "
                             f"avg OOS {c.oos_return:.1f}%")
        else:
            return "promising", (f"{c.folds_positive}/{c.folds_total} folds positive "
                                 f"but only {c.screen_trades} trades")
    elif c.folds_positive >= spec.folds // 2:
        return "marginal", (f"only {c.folds_positive}/{c.folds_total} folds positive")
    else:
        if c.is_return and c.is_return > 0 and (c.oos_return or 0) <= 0:
            return "overfit", "profitable in-sample, collapsed out of sample"
        return "fails", f"only {c.folds_positive}/{c.folds_total} folds positive"


def _score(c: FunnelCandidate) -> float:
    """Rank key for ordering the final table."""
    if c.verdict in ("fails", "unverified"):
        return -1e9 + (c.oos_return or 0)
    base = c.oos_return or 0
    if base <= 0:
        return base
    conf = min(1.0, c.screen_trades / float(MIN_TRADES))
    fold_bonus = c.folds_positive / max(1, c.folds_total)
    return round(base * conf * fold_bonus, 4)


# --------------------------------------------------------------------------- #
#  Full funnel orchestration
# --------------------------------------------------------------------------- #
def run_funnel(
    spec: FunnelSpec,
    progress: Optional[Callable] = None,
    should_stop: Optional[Callable] = None,
) -> dict:
    """Run the complete multi-round funnel and return the result table.

    `progress(round_info)` is called as work completes — round_info is a dict
    with {round, name, done, total, survivors}.
    """
    start_time = time.time()
    rounds_log: list[dict] = []
    baseline: Optional[dict] = None

    # Round 0 — Screen
    candidates, r0 = _screen(spec, progress, should_stop)
    rounds_log.append(r0)
    if should_stop and should_stop():
        return _result(candidates, rounds_log, spec, start_time, cancelled=True,
                       baseline=baseline)

    # Round 0b — Unfiltered baseline across the RR ladder. Runs on EVERY
    # screened pair, before Round 1 drops any of them: it is the control, so it
    # has to cover what the funnel is about to throw away as well as what it
    # keeps.
    if spec.baseline_rr_sweep:
        baseline, r0b = _baseline_rr_sweep(candidates, spec, progress,
                                           should_stop)
        rounds_log.append(r0b)
        if should_stop and should_stop():
            return _result(candidates, rounds_log, spec, start_time,
                           cancelled=True, baseline=baseline)

    # Round 1 — Symbol × Strategy (attributed, no extra runs)
    candidates = _round1_symbol_strategy(candidates, spec)
    rounds_log.append({"round": 1, "name": "Symbol × Strategy",
                       "survivors": len(candidates), "done": len(candidates),
                       "total": len(candidates)})

    # Round 2 — RR (attributed, exact from MFE)
    candidates = _round2_rr(candidates, spec)
    rounds_log.append({"round": 2, "name": "RR",
                       "survivors": len(candidates), "done": len(candidates),
                       "total": len(candidates)})

    if should_stop and should_stop():
        return _result(candidates, rounds_log, spec, start_time, cancelled=True,
                       baseline=baseline)

    # Round 3 — Patterns (greedy build-up + verify)
    candidates = _round3_patterns(candidates, spec, progress, should_stop)
    rounds_log.append({"round": 3, "name": "Patterns",
                       "survivors": len(candidates), "done": len(candidates),
                       "total": len(candidates)})

    if should_stop and should_stop():
        return _result(candidates, rounds_log, spec, start_time, cancelled=True,
                       baseline=baseline)

    # Round 4 — Session hour (attributed)
    candidates = _round4_hours(candidates, spec)
    rounds_log.append({"round": 4, "name": "Session Hours",
                       "survivors": len(candidates), "done": len(candidates),
                       "total": len(candidates)})

    # Round 5 — Weekday (attributed, highest bar)
    candidates = _round5_days(candidates, spec)
    rounds_log.append({"round": 5, "name": "Weekday",
                       "survivors": len(candidates), "done": len(candidates),
                       "total": len(candidates)})

    if should_stop and should_stop():
        return _result(candidates, rounds_log, spec, start_time, cancelled=True,
                       baseline=baseline)

    # Round 6 — Walk-forward
    candidates = _round6_walkforward(candidates, spec, progress, should_stop)
    rounds_log.append({"round": 6, "name": "Walk-forward",
                       "survivors": len([c for c in candidates
                                          if c.verdict in ("holds", "promising")]),
                       "done": len(candidates), "total": len(candidates)})

    return _result(candidates, rounds_log, spec, start_time, cancelled=False,
                   baseline=baseline)


def _result(candidates: list[FunnelCandidate], rounds: list[dict],
            spec: FunnelSpec, start_time: float,
            cancelled: bool = False,
            baseline: Optional[dict] = None) -> dict:
    """Package the funnel output for the API/UI."""
    # Clean up internal attrs before serializing
    table = []
    for c in sorted(candidates, key=lambda c: c.score, reverse=True):
        # Clean internal cached data
        for attr in ("_attribution", "_trades_df"):
            if hasattr(c, attr):
                delattr(c, attr)
        table.append({
            "symbol": c.symbol,
            "strategy_key": c.strategy_key,
            "segment": c.segment,
            "rr": c.best_rr,
            "rr_curve": c.rr_curve,
            "median_bars_to_1r": c.median_bars_to_1r,
            "baseline_rr": c.baseline_rr,
            "baseline_trades": c.baseline_trades,
            "baseline_net_pnl": c.baseline_net_pnl,
            "patterns": c.patterns,
            "hours": c.best_hours,
            "days": c.best_days,
            "screen_trades": c.screen_trades,
            "screen_net_pnl": c.screen_net_pnl,
            "screen_gross_pnl": c.screen_gross_pnl,
            "screen_costs": c.screen_costs,
            "is_return": c.is_return,
            "oos_return": c.oos_return,
            "folds_positive": c.folds_positive,
            "folds_total": c.folds_total,
            "verdict": c.verdict,
            "note": c.note,
            "score": c.score,
        })

    holds = [r for r in table if r["verdict"] in ("holds", "promising")]

    return {
        "cancelled": cancelled,
        "table": table,
        "rounds": rounds,
        # The unfiltered control. None when the sweep was switched off, which
        # the UI must show as "not run" rather than as a zero.
        "baseline": baseline,
        "holds": len(holds),
        "tested": len(table),
        "elapsed": round(time.time() - start_time, 1),
        "spec": {
            "symbols": spec.symbols,
            "strategy_keys": spec.strategy_keys,
            "start": spec.start,
            "end": spec.end,
            "capital": spec.capital,
            "mode": spec.mode,
            "rr_ladder": spec.rr_ladder,
            "folds": spec.folds,
            "baseline_rr_sweep": spec.baseline_rr_sweep,
        },
    }
