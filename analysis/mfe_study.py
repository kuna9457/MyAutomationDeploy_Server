"""
analysis/mfe_study.py

MAXIMUM FAVORABLE EXCURSION — the study that decides Approach 1 vs Approach 2
objectively, instead of by preference.

The question
------------
Every trade currently closes at its first target (1R). Two ways to try to keep
more of the winners:

    Approach 1  trail the WHOLE position from entry, no partial
    Approach 2  book part at 1R, trail the runner

Which is better is not a matter of taste, it is a property of the instrument's
price path: how far do trades that reach +1R actually go afterwards?

    fat right tail past +2R    ->  Approach 1. There is a real trend to ride,
                                   and booking half at 1R gives most of it up.
    mass piled up near +1R     ->  Approach 2. The move mostly stops there, so
                                   bank it and let a small runner take the
                                   lottery ticket.

Expect the answer to DIFFER by instrument and by mode. That is the point of
measuring it per book rather than picking one style for everything.

What is measured
----------------
For every trade the BASELINE (fixed stop/target) run took, the excursion in R
from entry to the furthest favourable price the market printed before the trade
would have been closed by something other than its target — the time exit, or
the session square-off. In other words: how much was on the table, given the
holding horizon the live bot actually allows.

Deliberately NOT measured beyond that horizon. An intraday bot is flat by
15:09; excursion the following morning is not excursion this bot could have
captured, and counting it would argue for a trail that cannot exist.

Usage
-----
    python -m analysis.mfe_study RELIANCE TCS INFY --mode Intraday \\
        --start 2026-01-01 --end 2026-06-30

Run it from the `backend/` directory.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import backtester                                            # noqa: E402
import config                                                # noqa: E402
from config import Mode                                      # noqa: E402


#: R multiples the distribution is reported at. 2R is the decision boundary the
#: docstring above turns on; 3R and 5R say whether the tail is genuinely fat or
#: just one lucky trade.
BUCKETS = (1.0, 1.5, 2.0, 3.0, 5.0)

#: Below this many qualifying trades a per-instrument verdict is noise. Same
#: rule of thumb the walk-forward uses and the improvement doc states.
MIN_TRADES = 30


def _horizon_end(entry_ts, interval_minutes: int, params, cutoff):
    """The last moment the live bot could still have been holding this trade.

    Whichever comes first: its time exit (max_hold_minutes) or the session's
    square-off. Both are real constraints the bot enforces, so excursion after
    either of them was never capturable.
    """
    end = None
    if params.max_hold_minutes > 0:
        end = entry_ts + pd.Timedelta(minutes=params.max_hold_minutes)
    if cutoff is not None:
        session_end = pd.Timestamp(entry_ts.date()) + pd.Timedelta(
            hours=cutoff.hour, minutes=cutoff.minute)
        if session_end < entry_ts:                # already past it (shouldn't be)
            session_end = entry_ts
        end = session_end if end is None else min(end, session_end)
    return end


def study(ticker: str, start: str, end: str, mode: Mode,
          capital: float = 200_000.0, strategy_key: str = "") -> dict:
    """MFE distribution for one instrument. Returns a summary dict."""
    # The BASELINE run: fixed stop/target, nothing managed. Its trade list is
    # the set of entries every approach would have taken, because none of the
    # styles touches entry selection.
    res = backtester.run_backtest(ticker, start, end, capital, mode,
                                  strategy_key=strategy_key,
                                  exit_style="fixed", include_costs=True)
    trades = res.trades
    if trades.empty:
        return {"ticker": ticker, "trades": 0}

    inst = config.INSTRUMENTS_BY_SYMBOL.get(ticker)
    params = backtester.resolve_strategy(mode, strategy_key).params
    interval = {Mode.SWING: "1d", Mode.INTRADAY: "15m", Mode.SCALPER: "1m"}[mode]
    cutoff = (config.square_off_time_for(inst.segment, mode) if inst else None)
    token = config.UPSTOX_LIVE_ACCESS_TOKEN or config.UPSTOX_SANDBOX_TOKEN
    bars, _src = backtester.fetch_history(
        ticker, start, end, interval,
        instrument_key=(inst.instrument_key if inst else ""), token=token)
    if bars.empty:
        return {"ticker": ticker, "trades": 0}

    mfes, reached_1r = [], 0
    for _, t in trades.iterrows():
        risk = float(t.get("risk_dist") or 0.0)
        if risk <= 0:
            continue
        entry_ts = pd.Timestamp(t["entry_time"])
        stop_ts = _horizon_end(entry_ts, 0, params, cutoff)
        window = bars.loc[entry_ts:stop_ts] if stop_ts is not None \
            else bars.loc[entry_ts:]
        if window.empty:
            continue
        entry = float(t["entry"])
        if t["side"] == "BUY":
            best = float(window["high"].max())
            excursion = (best - entry) / risk
        else:
            best = float(window["low"].min())
            excursion = (entry - best) / risk
        mfes.append(excursion)
        if excursion >= 1.0:
            reached_1r += 1

    if not mfes:
        return {"ticker": ticker, "trades": 0}
    arr = np.array(mfes)
    winners = arr[arr >= 1.0]                # the population the choice is about
    out = {
        "ticker": ticker,
        "trades": len(arr),
        "reached_1R": reached_1r,
        "median_MFE_R": round(float(np.median(arr)), 2),
        "median_MFE_R_of_winners": (round(float(np.median(winners)), 2)
                                    if winners.size else 0.0),
    }
    for b in BUCKETS:
        # Of the trades that got to 1R, what share kept going to b R? This is
        # the number the decision turns on.
        out[f">={b:g}R"] = (round(100.0 * float((winners >= b).mean()), 1)
                            if winners.size else 0.0)
    out["verdict"] = _verdict(out, winners.size)
    return out


def _verdict(row: dict, n_winners: int) -> str:
    """One line, and an honest refusal when the sample is too small to say."""
    if n_winners < MIN_TRADES:
        return f"insufficient sample ({n_winners} winners, need {MIN_TRADES})"
    past_2r = row.get(">=2R", 0.0)
    past_3r = row.get(">=3R", 0.0)
    if past_2r >= 45.0 and past_3r >= 20.0:
        return "fat right tail -> favour Approach 1 (trail the full position)"
    if past_2r <= 25.0:
        return "mass near 1R -> favour Approach 2 (book the partial)"
    return "no clear edge either way -> baseline is hard to beat; test both net of costs"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[3])
    ap.add_argument("tickers", nargs="+")
    ap.add_argument("--mode", default="Intraday",
                    choices=[m.value for m in Mode])
    ap.add_argument("--strategy", default="")
    ap.add_argument("--start", default="2026-01-01")
    ap.add_argument("--end", default="2026-06-30")
    ap.add_argument("--capital", type=float, default=200_000.0)
    args = ap.parse_args()

    rows = [study(t, args.start, args.end, Mode(args.mode), args.capital,
                  args.strategy)
            for t in args.tickers]
    rows = [r for r in rows if r.get("trades")]
    if not rows:
        print("No trades in that window for any of those symbols.")
        return 1
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    print("\nRead it as: of the trades that reached +1R, what share kept going.")
    print("A verdict needs at least", MIN_TRADES, "winners to be worth acting on.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
