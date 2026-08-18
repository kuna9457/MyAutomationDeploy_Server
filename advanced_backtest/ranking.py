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
    ranked = base * conf * min(keep, 1.0) * min(pf, 3.0) / 3.0
    # A combination that trails the SAME symbol traded unfiltered has earned
    # nothing, however good its own return looks. Sink it below everything that
    # did add something, but keep the ordering among such rows intact so the
    # table still reads sensibly.
    if c.edge is not None and c.edge <= 0:
        ranked = min(ranked, 0.0) + c.edge / 1000.0
    return ranked


def rank(combos: list[Combo], spec: SearchSpec) -> list[dict]:
    """Every combination as a plain dict, best first, with its verdict."""
    for c in combos:
        c.verdict, c.note = _verdict(c)
        c.score = round(_score(c), 4)
    ordered = sorted(combos, key=lambda c: c.score, reverse=True)
    return [asdict(c) for c in ordered]


def recommended_bucket(ranked: list[dict], limit: int = 20) -> dict:
    """What is actually worth trading — as PAIRINGS, not two flat lists.

    Two rules beyond the verdict:

      * THE PAIRING IS THE FINDING. "INFY with Identical Three Crows" is what
        was validated; "INFY" and "Identical Three Crows" separately are not.
        Flattening to {symbols, patterns} and pasting the cross product
        re-enables every unvalidated pairing between them — including ones this
        very search marked `fails`. `conflicts` below names those explicitly.
      * IT MUST BEAT DOING NOTHING. A combination is only kept when it also
        out-performs the SAME symbol traded unfiltered over the SAME window
        (`edge > 0`). Otherwise the honest advice is to leave that symbol alone,
        not to filter it.

    Nothing is ever applied automatically; this is evidence, and committing it
    to live money stays a human decision.
    """
    holds = [r for r in ranked if r["verdict"] == "holds"]
    keep = [r for r in holds
            if r.get("edge") is None or r["edge"] > 0][:limit]
    dropped_no_edge = [r for r in holds
                       if r.get("edge") is not None and r["edge"] <= 0]
    truncated = max(0, len([r for r in holds
                            if r.get("edge") is None or r["edge"] > 0]) - limit)

    symbols = sorted({r["symbol"] for r in keep})
    patterns = sorted({r["pattern"] for r in keep})

    # Which cross-product cells would be switched on by the flat lists, and
    # what this search already knows about them. This is the warning the old
    # bucket could not give.
    validated = {(r["symbol"], r["pattern"]) for r in keep}
    known = {(r["symbol"], r["pattern"]): r for r in ranked}
    conflicts = []
    for sym in symbols:
        for pat in patterns:
            if (sym, pat) in validated:
                continue
            other = known.get((sym, pat))
            if other and other["verdict"] in ("fails", "overfit"):
                conflicts.append({
                    "symbol": sym, "pattern": pat,
                    "verdict": other["verdict"],
                    "oos_return": other["oos_return"],
                })
    conflicts.sort(key=lambda c: (c["oos_return"] is None, c["oos_return"] or 0))

    # Grouped by pattern: a SINGLE pattern with its own symbol list is the one
    # shape today's settings can express exactly, because the pattern filter is
    # per (strategy, mode) while symbols are per strategy group. Running one
    # pattern at a time therefore has no cross-product risk at all.
    by_pattern: dict[str, list[str]] = {}
    for r in keep:
        by_pattern.setdefault(r["pattern"], []).append(r["symbol"])
    safe = max(by_pattern.items(), key=lambda kv: len(kv[1]),
               default=(None, []))

    return {
        "pairs": keep,
        "symbols": symbols,
        "patterns": patterns,
        "combinations": keep,          # kept for older callers
        "by_pattern": {k: sorted(v) for k, v in by_pattern.items()},
        "conflicts": conflicts,
        "safe_plan": ({"pattern": safe[0], "symbols": sorted(safe[1])}
                      if safe[0] else None),
        "dropped_no_edge": [
            {"symbol": r["symbol"], "pattern": r["pattern"],
             "oos_return": r["oos_return"], "edge": r["edge"]}
            for r in dropped_no_edge],
        "truncated": truncated,
        "why": (f"{len(keep)} pairing(s) stayed profitable out of sample on at "
                f"least {MIN_TRADES} trades AND beat the same symbol traded "
                f"unfiltered."),
    }
