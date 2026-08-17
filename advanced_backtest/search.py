"""
advanced_backtest/search.py
The two-stage combination search.

    STAGE 1 — SCREEN. One UNFILTERED backtest per symbol. Every trade already
    records which patterns fired (`entry_reason`), so attributing its PnL back
    to those patterns yields the whole symbol x pattern grid from N runs
    instead of N x 49.

    STAGE 2 — VERIFY. Re-run only the shortlist, with the pattern actually
    applied via params.allowed_patterns.

Stage 1 is an APPROXIMATION and the split exists because of it. Removing a
pattern does not merely delete its trades: it frees the position slot (only one
per symbol at a time), shifts the re-entry cooldown, and changes how capital
compounds from that point on. So stage 1 decides what is worth checking, and
stage 2's numbers are the ones reported.

Every run is also split IN-SAMPLE / OUT-OF-SAMPLE. A search over hundreds of
candidates will always throw up a spectacular winner by luck; the only defence
that survives contact with reality is scoring on data the choice was not made
on.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Callable, Optional

import pandas as pd

import backtester
import config
from config import Mode

#: Candidates carried from the screen into the verify pass. Verification is a
#: full simulation each, so this bounds the expensive half of the search.
DEFAULT_VERIFY_TOP = 25

#: A combination below this is reported but never recommended. The improvement
#: doc uses the same rule of thumb, and it is the single most effective guard
#: against ranking noise.
MIN_TRADES = 30

#: Fraction of the window used to CHOOSE. The rest is held back to score on.
DEFAULT_SPLIT = 0.7

#: Parallelism. Matches run_bulk_backtest's default — comfortably inside broker
#: rate limits, and past this the simulations just contend for CPU.
MAX_WORKERS = 5


@dataclass
class SearchSpec:
    """Everything the search holds FIXED, plus the symbols it varies over.

    RR, signal score, session hours and direction are deliberately inputs
    rather than axes: the operator picks them once and the search answers
    "given this configuration, which symbol and which pattern".
    """
    symbols: list[str]
    start: str
    end: str
    capital: float = 100_000.0
    mode: Mode = Mode.INTRADAY
    strategy_key: str = "candlestick_engine"
    risk_reward: float = 0.0        # 0 = the strategy's own
    min_score: float = 0.0          # 0 = the strategy's own
    split: float = DEFAULT_SPLIT
    verify_top: int = DEFAULT_VERIFY_TOP

    def split_date(self) -> str:
        """The boundary between the choosing half and the scoring half."""
        s = pd.Timestamp(self.start)
        e = pd.Timestamp(self.end)
        return (s + (e - s) * self.split).strftime("%Y-%m-%d")


@dataclass
class Combo:
    """One (symbol, pattern) candidate and how it did."""
    symbol: str
    pattern: str
    # Stage 1, attributed — a screen, not a measurement.
    screen_trades: int = 0
    screen_pnl: float = 0.0
    screen_win_rate: float = 0.0
    # Stage 2, from a real filtered run.
    verified: bool = False
    is_return: Optional[float] = None      # in-sample  (the choosing half)
    oos_return: Optional[float] = None     # out-of-sample (the held-back half)
    oos_trades: int = 0
    oos_win_rate: float = 0.0
    oos_profit_factor: Optional[float] = None
    oos_max_dd: Optional[float] = None
    verdict: str = ""
    note: str = ""
    score: float = 0.0


def _patterns_of(reason: str) -> list[str]:
    """The pattern names inside an entry_reason, without the "(evidence …)"."""
    head = str(reason).split(" (")[0]
    return [p.strip() for p in head.split(",") if p.strip()]


# --------------------------------------------------------------------------- #
#  Stage 1 — screen
# --------------------------------------------------------------------------- #
def screen_symbol(spec: SearchSpec, symbol: str) -> tuple[list[Combo], dict]:
    """One unfiltered run; return a Combo per pattern that traded, plus the
    symbol's own unfiltered summary for comparison."""
    inst = config.INSTRUMENTS_BY_SYMBOL.get(symbol)
    lot = inst.lot_size if inst else 1
    res = backtester.run_backtest(
        symbol, spec.start, spec.end, spec.capital, spec.mode,
        lot_size=lot, strategy_key=spec.strategy_key,
        risk_reward=spec.risk_reward, min_score=spec.min_score)

    summary = {
        "symbol": symbol,
        "trades": res.metrics.get("Total Trades", 0),
        "return_pct": res.metrics.get("Total Return %", 0.0),
        "win_rate": res.metrics.get("Win Rate %", 0.0),
        "source": res.metrics.get("Data Source", ""),
    }
    if res.trades is None or res.trades.empty:
        return [], summary

    agg: dict[str, dict] = {}
    for _, t in res.trades.iterrows():
        pnl = float(t.get("pnl", 0.0) or 0.0)
        for name in _patterns_of(t.get("entry_reason", "")):
            a = agg.setdefault(name, {"n": 0, "pnl": 0.0, "wins": 0})
            a["n"] += 1
            a["pnl"] += pnl
            a["wins"] += 1 if pnl > 0 else 0

    combos = [
        Combo(symbol=symbol, pattern=name,
              screen_trades=a["n"],
              screen_pnl=round(a["pnl"], 2),
              screen_win_rate=round(100.0 * a["wins"] / a["n"], 1))
        for name, a in agg.items()
    ]
    return combos, summary


# --------------------------------------------------------------------------- #
#  Stage 2 — verify, in-sample and out-of-sample
# --------------------------------------------------------------------------- #
def verify_combo(spec: SearchSpec, combo: Combo) -> Combo:
    """Actually run this symbol with ONLY this pattern allowed, twice: once on
    the choosing half, once on the held-back half."""
    inst = config.INSTRUMENTS_BY_SYMBOL.get(combo.symbol)
    lot = inst.lot_size if inst else 1
    boundary = spec.split_date()

    def run(a: str, b: str):
        return backtester.run_backtest(
            combo.symbol, a, b, spec.capital, spec.mode, lot_size=lot,
            strategy_key=spec.strategy_key, risk_reward=spec.risk_reward,
            min_score=spec.min_score, patterns=[combo.pattern])

    try:
        ins = run(spec.start, boundary)
        oos = run(boundary, spec.end)
    except Exception as exc:
        combo.note = f"verify failed: {type(exc).__name__}: {exc}"[:160]
        return combo

    combo.verified = True
    combo.is_return = ins.metrics.get("Total Return %", 0.0)
    combo.oos_return = oos.metrics.get("Total Return %", 0.0)
    combo.oos_trades = oos.metrics.get("Total Trades", 0)
    combo.oos_win_rate = oos.metrics.get("Win Rate %", 0.0)
    combo.oos_max_dd = oos.metrics.get("Max Drawdown %", 0.0)

    if oos.trades is not None and not oos.trades.empty:
        pnl = pd.to_numeric(oos.trades["pnl"], errors="coerce").fillna(0.0)
        gross_win = float(pnl[pnl > 0].sum())
        gross_loss = float(-pnl[pnl < 0].sum())
        combo.oos_profit_factor = (round(gross_win / gross_loss, 3)
                                   if gross_loss > 0 else None)
    return combo


# --------------------------------------------------------------------------- #
#  Orchestration
# --------------------------------------------------------------------------- #
def run_search(spec: SearchSpec,
               progress: Optional[Callable[[int, int, str], None]] = None,
               should_stop: Optional[Callable[[], bool]] = None) -> dict:
    """Screen every symbol, verify the shortlist, and rank what survives.

    `progress(done, total, label)` is called from the calling thread as work
    completes, so it is safe to drive a UI with. `should_stop()` is polled
    between units so a cancelled job stops promptly instead of running to
    completion in the background.
    """
    symbols = [s for s in spec.symbols if s in config.INSTRUMENTS_BY_SYMBOL]
    if not symbols:
        raise ValueError("No valid symbols to search.")

    total = len(symbols) + spec.verify_top
    done = 0
    combos: list[Combo] = []
    summaries: list[dict] = []

    def tick(label: str) -> None:
        nonlocal done
        done += 1
        if progress:
            progress(done, total, label)

    # -- stage 1 ------------------------------------------------------------ #
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(screen_symbol, spec, s): s for s in symbols}
        for fut in as_completed(futures):
            sym = futures[fut]
            if should_stop and should_stop():
                break
            try:
                found, summary = fut.result()
                combos.extend(found)
                summaries.append(summary)
            except Exception as exc:
                # One bad symbol must not sink the search.
                summaries.append({"symbol": sym, "trades": 0, "return_pct": 0.0,
                                  "win_rate": 0.0, "source": "error",
                                  "error": str(exc)[:160]})
            tick(f"screened {sym}")

    if should_stop and should_stop():
        return {"cancelled": True, "combos": [], "symbols": summaries}

    # Shortlist by screened PnL, but only candidates with enough trades to be
    # worth a full verification run.
    shortlist = sorted(
        [c for c in combos if c.screen_trades >= max(5, MIN_TRADES // 3)],
        key=lambda c: c.screen_pnl, reverse=True)[:spec.verify_top]

    # -- stage 2 ------------------------------------------------------------ #
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(verify_combo, spec, c): c for c in shortlist}
        for fut in as_completed(futures):
            if should_stop and should_stop():
                break
            try:
                fut.result()
            except Exception:
                pass
            tick("verified a combination")

    # Any verify slots not used still count toward `total`; close the gap so a
    # finished job reads 100% rather than stalling at 90%.
    while done < total and progress:
        tick("done")

    from advanced_backtest.ranking import rank
    ranked = rank(combos, spec)
    return {
        "cancelled": bool(should_stop and should_stop()),
        "combos": ranked,
        "symbols": sorted(summaries,
                          key=lambda s: s.get("return_pct", 0), reverse=True),
        "split_date": spec.split_date(),
        "screened": len(combos),
        "verified": sum(1 for c in combos if c.verified),
    }
