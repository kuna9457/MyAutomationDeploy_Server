"""
advanced_backtest/ranking.py
Turn verified combinations into an ordered table and a recommended bucket.

The ranking deliberately does NOT sort on return. A search over hundreds of
candidates hands you a spectacular in-sample number for free, and sorting on it
would rank exactly the combinations that will disappoint. What is scored here
is what SURVIVED the held-back half of the window, discounted by how badly it
degraded from the half it was chosen on.
"""
from __future__ import annotations

from dataclasses import asdict

from advanced_backtest.search import MIN_TRADES, Combo, SearchSpec

#: A combination whose out-of-sample return falls below this fraction of its
#: in-sample return is treated as overfit, however good the raw numbers look.
#: 0.4 is deliberately lenient — the aim is to catch collapse, not to demand
#: that a strategy perform identically on two different market periods.
OVERFIT_RATIO = 0.4

#: Below this many out-of-sample trades nothing is claimed either way.
MIN_OOS_TRADES = 10


def _verdict(c: Combo) -> tuple[str, str]:
    """(verdict, note) — the honest reading of one verified combination."""
    if not c.verified:
        return "unverified", c.note or "not carried into the verify pass"
    if c.oos_trades < MIN_OOS_TRADES:
        return "thin", (f"only {c.oos_trades} out-of-sample trades — not enough "
                        f"to judge either way")
    if (c.oos_return or 0) <= 0:
        return "fails", "lost money on the held-back half"
    if (c.is_return or 0) > 0 and (c.oos_return or 0) < (c.is_return or 0) * OVERFIT_RATIO:
        return "overfit", (f"held up in-sample ({c.is_return:.1f}%) but mostly "
                           f"collapsed out of sample ({c.oos_return:.1f}%)")
    if c.oos_trades < MIN_TRADES:
        return "promising", (f"positive out of sample, but only {c.oos_trades} "
                             f"trades — treat as a lead, not a conclusion")
    return "holds", "profitable out of sample on a usable sample"


def _score(c: Combo) -> float:
    """Rank key. Out-of-sample return is the base; the rest are multipliers
    that punish thin samples and in/out collapse, so a lucky 6-trade winner
    cannot outrank a steady 60-trade one."""
    if not c.verified or c.oos_return is None:
        return -1e9
    base = c.oos_return
    if base <= 0:
        return base                       # losers rank by how badly they lost
    # Confidence from sample size, saturating at MIN_TRADES.
    conf = min(1.0, c.oos_trades / float(MIN_TRADES))
    # Consistency: how much of the in-sample edge survived.
    if c.is_return and c.is_return > 0:
        keep = max(0.0, min(1.5, c.oos_return / c.is_return))
    else:
        keep = 1.0
    pf = c.oos_profit_factor or 1.0
    return base * conf * min(keep, 1.0) * min(pf, 3.0) / 3.0


def rank(combos: list[Combo], spec: SearchSpec) -> list[dict]:
    """Every combination as a plain dict, best first, with its verdict."""
    for c in combos:
        c.verdict, c.note = _verdict(c)
        c.score = round(_score(c), 4)
    ordered = sorted(combos, key=lambda c: c.score, reverse=True)
    return [asdict(c) for c in ordered]


def recommended_bucket(ranked: list[dict], limit: int = 12) -> dict:
    """The symbols and patterns worth actually trading, from a finished search.

    Only "holds" qualifies — not "promising", and never "overfit" or "thin".
    The output is shaped to be pasted straight into the strategy board and the
    pattern allow-list, but it is NEVER applied automatically: a search result
    is evidence, and committing it to live money stays a human decision.
    """
    keep = [r for r in ranked if r["verdict"] == "holds"][:limit]
    return {
        "symbols": sorted({r["symbol"] for r in keep}),
        "patterns": sorted({r["pattern"] for r in keep}),
        "combinations": keep,
        "why": (f"{len(keep)} combination(s) stayed profitable on the held-back "
                f"half of the window with at least {MIN_TRADES} trades there."),
    }
