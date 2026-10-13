"""
tools/seed_commission_test.py
Give a TEST client some made-up closed trades so the weekly commission flow
(invoice -> QR -> mark paid -> received) can be clicked through end to end.

    python tools/seed_commission_test.py testing            # add mock profit
    python tools/seed_commission_test.py testing --remove   # take it out again
    python tools/seed_commission_test.py testing --remove --invoices
                                        # ...and delete that client's invoices

THESE ARE NOT TRADES. No order is placed and no engine is involved: the rows
are written straight into the trade ledger, already CLOSED (so a restart never
re-hydrates one as a position), dated in past sessions (so today's PnL and
risk limits are untouched), and marked broker "Mock" / strategy
"commission_test" so they can be found and removed exactly.

Refuses a client who has any genuine trade in that book — mock profit must
never be mixed into a real client's ledger.
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import date, datetime, time, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import commission  # noqa: E402
import user_manager  # noqa: E402
from config import Environment  # noqa: E402
from db_manager import DBManager, _collection_name  # noqa: E402

BROKER = "Mock"
STRATEGY = "commission_test"
REASON = "MOCK - commission test, not a real trade"

#: Net profit of each finished week, cycled: Rs1,000 then Rs1,500. Each has a
#: loser in it, so the invoice is visibly a NET figure.
FINISHED_WEEKS = ([900.0, -650.0, 750.0], [1200.0, -450.0, 750.0])
#: The week in progress — shows up as "accruing", not as an invoice.
CURRENT_WEEK = [400.0]

ENTRY, QTY = 1400.0, 10


def _is_mock(row: dict) -> bool:
    return row.get("broker") == BROKER and row.get("strategy") == STRATEGY


def _trade(db: DBManager, env: Environment, username: str, day: date,
           pnl: float) -> dict:
    opened = datetime.combine(day, time(5, 0))          # 10:30 IST, stored UTC
    doc = db.new_trade(
        mode="Intraday", environment=env.value, broker=BROKER,
        ticker="RELIANCE", side="BUY", entry_price=ENTRY,
        stop_loss=ENTRY - 100, target=ENTRY + 100, quantity=QTY,
        risk_amount=1000.0, segment="NSE_EQUITY", strategy=STRATEGY,
        entry_reason=REASON, user_id=username)
    doc.update(
        timestamp=opened.isoformat(),
        status="CLOSED",
        exit_price=round(ENTRY + pnl / QTY, 4),
        realized_pnl=round(pnl, 2),
        exit_timestamp=(opened + timedelta(hours=2)).isoformat(),
        exit_reason="TARGET" if pnl > 0 else "STOP-LOSS",
    )
    return doc


def _own_trades(db: DBManager, env: Environment, username: str) -> list[dict]:
    df = db.get_trades(env, user_id=username)
    return [] if df.empty else df.to_dict("records")


def seed(db: DBManager, env: Environment, username: str) -> None:
    rows = _own_trades(db, env, username)
    if any(not _is_mock(r) for r in rows):
        sys.exit(f"'{username}' has real {env.value} trades - refusing to mix "
                 "mock profit into them. Use a client that has never traded.")
    if rows:
        sys.exit(f"'{username}' already has {len(rows)} mock trades. "
                 "Run with --remove first to start again.")

    user = user_manager.get_user(username) or {}
    today = commission._today()
    this_week = commission.week_start(today)
    _, start = commission._rate(user)
    # No rate yet: seed the last two finished weeks, so setting a rate with
    # "First week to bill" two weeks back picks them up.
    week = (date.fromisoformat(start) if start
            else this_week - timedelta(days=14))

    n = 0
    while week < this_week:
        legs = FINISHED_WEEKS[n % len(FINISHED_WEEKS)]
        for offset, pnl in enumerate(legs, start=1):     # Tue, Wed, Thu
            db.insert_trade(_trade(db, env, username,
                                   week + timedelta(days=offset), pnl), env)
        print(f"  week of {week}: net Rs{sum(legs):,.2f}")
        n += 1
        week += timedelta(days=7)
    if this_week < today:                # never date a mock trade today
        for pnl in CURRENT_WEEK:
            db.insert_trade(_trade(db, env, username, this_week, pnl), env)
        print(f"  this week ({this_week}): Rs{sum(CURRENT_WEEK):,.2f} so far")


def remove(db: DBManager, env: Environment, username: str,
           invoices: bool) -> None:
    query = {"user_id": username, "broker": BROKER, "strategy": STRATEGY}
    if db.db is not None:
        gone = db.db[_collection_name(env)].delete_many(query).deleted_count
    else:
        rows = db._read_local(env)
        keep = [r for r in rows
                if not (r.get("user_id") == username and _is_mock(r))]
        gone = len(rows) - len(keep)
        with open(db._local_path(env), "w", encoding="utf-8") as fh:
            import json
            for r in keep:
                fh.write(json.dumps(r) + "\n")
    db.sync_excel_log(env)
    print(f"  removed {gone} mock {env.value} trades")

    if invoices:
        col = commission._col()
        if col is not None:
            n = col.delete_many({"username": username}).deleted_count
        else:
            with commission._lock:
                data = commission._read_local()
                keep = {k: v for k, v in data.items()
                        if v.get("username") != username}
                n = len(data) - len(keep)
                commission._write_local(keep)
        print(f"  removed {n} commission invoices")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("username")
    ap.add_argument("--env", choices=["Live", "Paper"],
                    help="book to write to (default: the one commission is "
                         "charged on)")
    ap.add_argument("--remove", action="store_true")
    ap.add_argument("--invoices", action="store_true",
                    help="with --remove: also delete this client's invoices")
    args = ap.parse_args()

    user = user_manager.get_user(args.username)
    if user is None or user.get("role") != "client":
        sys.exit(f"No client account named '{args.username}'.")
    env = Environment(args.env or commission.settings()["environment"])
    db = DBManager()
    print(f"{args.username} / {env.value} book / {db.backend}")
    if args.remove:
        remove(db, env, args.username, args.invoices)
    else:
        seed(db, env, args.username)
        for inv in commission.sync(args.username):
            print(f"  invoice {inv['week_start']}: {inv['status']} "
                  f"Rs{inv['amount']:,.2f}")


if __name__ == "__main__":
    main()
