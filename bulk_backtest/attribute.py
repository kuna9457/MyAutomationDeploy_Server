"""
bulk_backtest/attribute.py
Slice one enriched trade log across all search axes WITHOUT re-running.

One unfiltered simulation already records the conditions every trade was taken
under. Attribution exploits this: instead of running N × M × K simulations, we
run ONE and then read the cells from the trade log's columns. The funnel
(funnel.py) uses this to decide what is worth simulating, then simulation
produces the final numbers.

    § MULTI_AXIS_SEARCH_PLAN.md §2

IMPORTANT: attribution is an APPROXIMATION. Removing a pattern, a weekday or
an hour does not merely delete those trades: it frees the position slot, shifts
the re-entry cooldown, and changes the capital every later trade is sized
against. So attribution decides what is worth checking, and the verify pass's
numbers are the ones reported (same two-stage discipline as advanced_backtest).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd


@dataclass
class CellResult:
    """One cell of the multi-axis grid, attributed from a single trade log."""
    trades: int = 0
    gross_pnl: float = 0.0
    net_pnl: float = 0.0
    wins: int = 0
    costs: float = 0.0

    @property
    def win_rate(self) -> float:
        return round(100.0 * self.wins / self.trades, 2) if self.trades else 0.0

    @property
    def avg_net_pnl(self) -> float:
        return round(self.net_pnl / self.trades, 2) if self.trades else 0.0

    @property
    def profit_factor(self) -> float | None:
        """Gross win / gross loss. None if no losses."""
        # We don't have per-trade win/loss here; this is an aggregate.
        return None  # computed at verify time with full trade data


# --------------------------------------------------------------------------- #
#  Pattern parsing — per-strategy, fixing the Phase 4 bug from §2.2
# --------------------------------------------------------------------------- #
def _patterns_candlestick(reason: str) -> list[str]:
    """Phase 1 candlestick: 'Hammer, Doji (evidence 6.2)' -> ['Hammer', 'Doji']"""
    head = str(reason).split(" (")[0]
    return [p.strip() for p in head.split(",") if p.strip()]


def _patterns_chart(reason: str) -> list[str]:
    """Phase 3 chart: 'Double Top breakout' -> ['Double Top breakout']"""
    head = str(reason).split(" (")[0].strip()
    return [head] if head else []


def _patterns_context(reason: str) -> list[str]:
    """Phase 4 context: 'Context confluence 5.5 [uptrend, vol+]'
    -> ['uptrend', 'vol+']"""
    m = re.search(r"\[([^\]]+)\]", str(reason))
    if m:
        return [f.strip() for f in m.group(1).split(",") if f.strip()]
    # Fallback: treat the whole reason as one pattern
    head = str(reason).split(" (")[0].strip()
    return [head] if head else []


PATTERN_PARSERS = {
    "candlestick_engine": _patterns_candlestick,
    "chart_pattern_engine": _patterns_chart,
    "context_engine": _patterns_context,
}


def parse_patterns(reason: str, strategy_key: str = "") -> list[str]:
    """Extract pattern names from an entry_reason, using the correct parser
    for the strategy that produced it."""
    parser = PATTERN_PARSERS.get(strategy_key, _patterns_candlestick)
    return parser(reason)


# --------------------------------------------------------------------------- #
#  RR re-pricing from MFE — exact per trade (§2.1)
# --------------------------------------------------------------------------- #
def reprice_trade_at_rr(
    mfe_r: float,
    original_rr: float,
    target_rr: float,
    risk_amt: float,
    original_pnl: float,
    original_cost: float,
) -> tuple[float, float, bool]:
    """What would this trade's (gross_pnl, net_pnl, win) be at a different RR?

    Per §2.1: if mfe_r >= target_rr, the trade WINS at that RR and pnl =
    target_rr × risk_amt (a long's exact profit). If mfe_r < target_rr, the
    trade ends exactly as it actually ended (stop or time exit).

    Returns (gross_pnl, net_pnl, win).
    """
    if target_rr <= 0 or risk_amt <= 0:
        return original_pnl, original_pnl - original_cost, original_pnl > 0

    if mfe_r >= target_rr:
        # Trade would have hit the target at this RR
        gross = target_rr * risk_amt
        net = gross - original_cost  # costs are approximately the same
        return round(gross, 2), round(net, 2), True
    else:
        # Trade ends as it did originally (stop or time exit)
        return original_pnl, round(original_pnl - original_cost, 2), original_pnl > 0


# --------------------------------------------------------------------------- #
#  Full attribution: one trade log → the entire grid
# --------------------------------------------------------------------------- #
def attribute_trades(
    trades: pd.DataFrame,
    strategy_key: str = "",
    rr_ladder: Optional[list[float]] = None,
) -> dict[str, dict]:
    """Attribute an enriched trade log across all axes.

    Returns a dict with keys:
      "by_pattern"   -> {pattern_name: CellResult}
      "by_rr"        -> {rr_value: CellResult}
      "by_weekday"   -> {weekday_int: CellResult}
      "by_hour"      -> {hour_int: CellResult}
      "by_side"      -> {"BUY"|"SELL": CellResult}
      "by_pattern_rr" -> {(pattern, rr): CellResult}  # the big cross
      "overall"      -> CellResult  (the unfiltered total)
    """
    if rr_ladder is None:
        rr_ladder = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

    result = {
        "by_pattern": {},
        "by_rr": {},
        "by_weekday": {},
        "by_hour": {},
        "by_side": {},
        "by_pattern_rr": {},
        "overall": CellResult(),
    }

    if trades is None or trades.empty:
        return result

    # The overall bucket at the run's own RR
    overall = CellResult()

    for _, row in trades.iterrows():
        pnl = float(row.get("pnl", 0) or 0)
        cost = float(row.get("trade_cost", 0) or 0)
        net = float(row.get("net_pnl", pnl - cost))
        mfe_r = float(row.get("mfe_r", 0) or 0)
        risk_amt = float(row.get("risk_amt", 0) or 0)
        original_rr = float(row.get("rr", 2.0) or 2.0)
        reason = str(row.get("entry_reason", ""))
        weekday = int(row.get("weekday", 0) or 0)
        hour = int(row.get("hour", 0) or 0)
        side = str(row.get("side", "BUY"))
        win_net = net > 0

        # Overall
        overall.trades += 1
        overall.gross_pnl += pnl
        overall.net_pnl += net
        overall.costs += cost
        if win_net:
            overall.wins += 1

        # By pattern
        patterns = parse_patterns(reason, strategy_key)
        for pat in patterns:
            cell = result["by_pattern"].setdefault(pat, CellResult())
            cell.trades += 1
            cell.gross_pnl += pnl
            cell.net_pnl += net
            cell.costs += cost
            if win_net:
                cell.wins += 1

        # By weekday
        wd_cell = result["by_weekday"].setdefault(weekday, CellResult())
        wd_cell.trades += 1
        wd_cell.gross_pnl += pnl
        wd_cell.net_pnl += net
        wd_cell.costs += cost
        if win_net:
            wd_cell.wins += 1

        # By hour
        hr_cell = result["by_hour"].setdefault(hour, CellResult())
        hr_cell.trades += 1
        hr_cell.gross_pnl += pnl
        hr_cell.net_pnl += net
        hr_cell.costs += cost
        if win_net:
            hr_cell.wins += 1

        # By side
        sd_cell = result["by_side"].setdefault(side, CellResult())
        sd_cell.trades += 1
        sd_cell.gross_pnl += pnl
        sd_cell.net_pnl += net
        sd_cell.costs += cost
        if win_net:
            sd_cell.wins += 1

        # By RR — reprice the trade at every rung of the ladder
        for rr in rr_ladder:
            g, n, w = reprice_trade_at_rr(mfe_r, original_rr, rr,
                                           risk_amt, pnl, cost)
            rr_cell = result["by_rr"].setdefault(rr, CellResult())
            rr_cell.trades += 1
            rr_cell.gross_pnl += g
            rr_cell.net_pnl += n
            rr_cell.costs += cost
            if w:
                rr_cell.wins += 1

            # By (pattern, rr) cross
            for pat in patterns:
                pr_cell = result["by_pattern_rr"].setdefault(
                    (pat, rr), CellResult())
                pr_cell.trades += 1
                pr_cell.gross_pnl += g
                pr_cell.net_pnl += n
                pr_cell.costs += cost
                if w:
                    pr_cell.wins += 1

    result["overall"] = overall
    return result


def top_patterns(
    attributed: dict,
    min_trades: int = 10,
    top_n: int = 20,
) -> list[dict]:
    """Rank patterns by net PnL, filtering by minimum trades."""
    by_pat = attributed.get("by_pattern", {})
    rows = []
    for pat, cell in by_pat.items():
        if cell.trades >= min_trades:
            rows.append({
                "pattern": pat,
                "trades": cell.trades,
                "net_pnl": round(cell.net_pnl, 2),
                "gross_pnl": round(cell.gross_pnl, 2),
                "win_rate": cell.win_rate,
                "costs": round(cell.costs, 2),
                "avg_net": cell.avg_net_pnl,
            })
    rows.sort(key=lambda r: r["net_pnl"], reverse=True)
    return rows[:top_n]


def rr_hit_curve(attributed: dict, rr_ladder: list[float],
                 min_trades: int = 5) -> list[dict]:
    """The scrip-wise 'how far does it actually get' curve: for each rung of
    the RR ladder, what fraction of trades reached it before exit.

    This is exactly `by_rr[rr].win_rate` reshaped into a sorted list — the
    reprice in `reprice_trade_at_rr` sets win=True precisely when
    mfe_r >= rr, so the win rate at a rung IS the probability the trade got
    there. Two symbols can share a "best RR" from `best_rr()` (net PnL) and
    still mean opposite things operationally: one reaches 1R almost always
    and 2R rarely (tight, fast), the other reaches 2R about as often as 1R
    (it runs, given the time). The curve is what tells them apart; a single
    picked rung cannot.

    Rungs below `min_trades` are still returned (marked via `trades`) rather
    than dropped, so the UI can grey out — not hide — an unreliable point.
    """
    by_rr = attributed.get("by_rr", {})
    curve = []
    for rr in sorted(rr_ladder):
        cell = by_rr.get(rr)
        curve.append({
            "rr": rr,
            "trades": cell.trades if cell else 0,
            "hit_rate": cell.win_rate if cell else 0.0,
            "reliable": bool(cell and cell.trades >= min_trades),
        })
    return curve


def best_rr(attributed: dict, min_trades: int = 10) -> dict | None:
    """The RR rung with the highest net PnL, subject to minimum trades."""
    by_rr = attributed.get("by_rr", {})
    best = None
    for rr, cell in by_rr.items():
        if cell.trades >= min_trades:
            if best is None or cell.net_pnl > best["net_pnl"]:
                best = {
                    "rr": rr,
                    "trades": cell.trades,
                    "net_pnl": round(cell.net_pnl, 2),
                    "win_rate": cell.win_rate,
                }
    return best
