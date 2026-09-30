"""
analysis/walk_forward.py

WALK-FORWARD validation of the exit knobs — the discipline that keeps this
from becoming a curve fit.

Why not just grid-search the whole history
------------------------------------------
Two knobs (trail distance, partial fraction) over one fixed window will always
produce a "best" combination. On a few hundred trades that best value is
mostly noise, and trading it is trading the noise. So:

  * optimise on a ROLLING IN-SAMPLE window, then score the chosen setting on
    the OUT-OF-SAMPLE window immediately after it, then roll forward;
  * report the out-of-sample results only — the in-sample number is the search
    talking to itself;
  * check the in-sample surface for a PLATEAU. A robust trail multiple has
    neighbours that are nearly as good. A lone spike surrounded by bad values
    is an artefact and is reported as such;
  * refuse to say anything about a bucket with fewer than MIN_TRADES trades;
  * hold the most recent HOLDOUT_DAYS back entirely. Nothing in this script
    ever reads them — that window is the one honest test left after every
    other decision has been made.

Never more than two knobs at a time, on purpose: the number of ways to fool
yourself grows faster than the number of parameters.

Usage
-----
    python -m analysis.walk_forward RELIANCE --mode Intraday \\
        --start 2025-09-01 --end 2026-06-30 --style partial_trail

Run it from the `backend/` directory.
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import date, timedelta

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import backtester                                            # noqa: E402
from config import Mode                                      # noqa: E402


#: The two knobs, and nothing else. Adding a third multiplies the search space
#: and divides the trust you can place in the answer.
TRAIL_GRID = (1.0, 1.5, 2.0, 2.5, 3.0, 4.0)
PARTIAL_GRID = (0.0, 0.33, 0.5, 0.67)

IN_SAMPLE_DAYS = 90
OUT_SAMPLE_DAYS = 30
#: Never touched by this script. Test the final choice here, once.
HOLDOUT_DAYS = 60
MIN_TRADES = 30


def _run(ticker, start, end, mode, style, trail, partial, capital,
         strategy_key):
    res = backtester.run_backtest(
        ticker, start, end, capital, mode, strategy_key=strategy_key,
        exit_style=style, trail_atr_mult=trail,
        partial_exit_fraction=partial, include_costs=True)
    m = res.metrics
    return {
        "trail": trail, "partial": partial,
        "return_pct": m["Total Return %"],           # NET of costs
        "gross_pct": m.get("Gross Return %", 0.0),
        "max_dd_pct": m["Max Drawdown %"],
        "calmar": m["Calmar"],
        "positions": m.get("Total Positions", m["Total Trades"]),
    }


def _windows(start: date, end: date):
    """(in-sample, out-of-sample) date pairs, rolling forward."""
    cur = start
    while True:
        is_end = cur + timedelta(days=IN_SAMPLE_DAYS)
        oos_end = is_end + timedelta(days=OUT_SAMPLE_DAYS)
        if oos_end > end:
            return
        yield (cur, is_end, oos_end)
        cur = cur + timedelta(days=OUT_SAMPLE_DAYS)


def _plateau(rows: list[dict], best: dict) -> str:
    """Is the winner surrounded by near-winners, or standing alone?

    A trail multiple whose immediate neighbours in the grid keep most of its
    return is a real effect. One that collapses either side of it is the
    search having found a particular sequence of bars, and will not survive
    contact with the next quarter.
    """
    same_partial = sorted((r for r in rows if r["partial"] == best["partial"]),
                          key=lambda r: r["trail"])
    idx = [r["trail"] for r in same_partial].index(best["trail"])
    neighbours = [same_partial[i]["return_pct"]
                  for i in (idx - 1, idx + 1) if 0 <= i < len(same_partial)]
    if best["return_pct"] <= 0:
        # Nothing in the grid made money in sample. There is no surface to
        # judge and no setting worth carrying forward.
        return "no profitable setting"
    if not neighbours:
        return "edge of grid"
    keep = min(n / best["return_pct"] for n in neighbours)
    return "plateau" if keep >= 0.6 else "SPIKE (likely a curve fit)"


def walk_forward(ticker: str, start: str, end: str, mode: Mode, style: str,
                 capital: float = 200_000.0, strategy_key: str = "") -> None:
    s, e = date.fromisoformat(start), date.fromisoformat(end)
    holdout_start = e - timedelta(days=HOLDOUT_DAYS)
    if holdout_start <= s:
        print(f"Window too short: {HOLDOUT_DAYS} days are reserved as an "
              f"untouched holdout, leaving nothing to walk forward over.")
        return
    print(f"{ticker} / {mode.value} / {style}")
    print(f"  search window : {s} -> {holdout_start}")
    print(f"  HOLDOUT (never read here): {holdout_start} -> {e}\n")

    partial_grid = PARTIAL_GRID if style == "partial_trail" else (-1.0,)
    results = []
    for is_start, is_end, oos_end in _windows(s, holdout_start):
        rows = [
            _run(ticker, is_start.isoformat(), is_end.isoformat(), mode,
                 style, tr, pf, capital, strategy_key)
            for tr in TRAIL_GRID for pf in partial_grid
        ]
        rows = [r for r in rows if r["positions"] >= MIN_TRADES]
        if not rows:
            print(f"  {is_start} -> {is_end}: too few trades in sample, skipped")
            continue
        best = max(rows, key=lambda r: r["return_pct"])
        shape = _plateau(rows, best)
        # The only number that counts: the chosen setting, applied FORWARD to
        # bars the search never saw.
        oos = _run(ticker, is_end.isoformat(), oos_end.isoformat(), mode,
                   style, best["trail"], best["partial"], capital,
                   strategy_key)
        results.append({
            "is_start": is_start, "oos_end": oos_end,
            "chosen_trail": best["trail"], "chosen_partial": best["partial"],
            "in_sample_%": best["return_pct"], "surface": shape,
            "out_of_sample_%": oos["return_pct"],
            "oos_positions": oos["positions"],
        })

    if not results:
        print("No window had enough trades to optimise on.")
        return
    df = pd.DataFrame(results)
    print(df.to_string(index=False))
    oos_mean = df["out_of_sample_%"].mean()
    hit = 100.0 * float((df["out_of_sample_%"] > 0).mean())
    unstable = int((df["surface"] != "plateau").sum())
    thin = int((df["oos_positions"] < MIN_TRADES).sum())
    print(f"\n  mean OUT-OF-SAMPLE return : {oos_mean:.2f}%")
    print(f"  windows profitable OOS    : {hit:.0f}%")
    print(f"  windows whose in-sample best was NOT a plateau: "
          f"{unstable}/{len(df)}")
    if thin:
        # Said BEFORE the verdict, because it can invalidate it: a nine-trade
        # out-of-sample window is a coin flip wearing a percentage sign.
        print(f"  WARNING {thin}/{len(df)} out-of-sample windows hold fewer "
              f"than {MIN_TRADES} trades — those returns are noise whatever "
              f"they say. Lengthen the window or pool more symbols.")
    if unstable > len(df) / 2:
        print("  -> the chosen settings are not stable. Do not trade this.")
    elif oos_mean <= 0:
        print("  -> the optimisation does not survive out of sample. "
              "Stay on the baseline.")
    else:
        print("  -> promising. Confirm ONCE on the holdout window, then stop "
              "optimising.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[3])
    ap.add_argument("ticker")
    ap.add_argument("--mode", default="Intraday",
                    choices=[m.value for m in Mode])
    ap.add_argument("--strategy", default="")
    ap.add_argument("--style", default="partial_trail",
                    choices=["trail_full", "partial_trail"])
    ap.add_argument("--start", default="2025-09-01")
    ap.add_argument("--end", default="2026-06-30")
    ap.add_argument("--capital", type=float, default=200_000.0)
    args = ap.parse_args()
    walk_forward(args.ticker, args.start, args.end, Mode(args.mode),
                 args.style, args.capital, args.strategy)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
