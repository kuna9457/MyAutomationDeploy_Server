"""
db_manager.py
Persistence layer: trade logging + Excel analytics export.

Immutable Rule #2 is enforced here — Paper trades go to `paper_trades`, Live
trades go to `live_trades`, and the two never mix. The collection is chosen
purely from the Environment enum, so no caller can accidentally cross-write.

MongoDB is used when reachable; otherwise the manager transparently falls back
to newline-delimited JSON files under ./data so the bot never loses trades just
because Mongo isn't installed.
"""
from __future__ import annotations

import json
import os
import uuid
from datetime import datetime
from typing import Any, Optional

import pandas as pd

import config
import mongo_client
from config import Environment


PAPER_COLLECTION = "paper_trades"
LIVE_COLLECTION = "live_trades"

#: Indexes are created once per process, not once per DBManager — see
#: DBManager._ensure_indexes.
_indexes_ready = False


def _collection_name(env: Environment) -> str:
    return PAPER_COLLECTION if env == Environment.PAPER else LIVE_COLLECTION


class DBManager:
    def __init__(self):
        self.client = None
        self.db = None
        self._connect_mongo()

    # -- connection --------------------------------------------------------- #
    def _connect_mongo(self) -> None:
        """Attach to the process-wide MongoClient (see mongo_client.py).

        Every DBManager used to build its OWN client and pool. They now share
        one, but the semantics here are unchanged: we still ping at construction
        so `self.db` is only set when Mongo genuinely answers, and a failure
        still leaves this instance on the local-JSON path."""
        try:
            client = mongo_client.get_client()
            if client is None:
                raise RuntimeError("no Mongo client available")
            client.admin.command("ping")            # force a real connection test
            self.client = client
            self.db = client[config.MONGO_DB_NAME]
            self._ensure_indexes()
        except Exception as exc:
            print(f"[DBManager] MongoDB unavailable ({exc}); using local JSON.")
            self.client = None
            self.db = None

    def _ensure_indexes(self) -> None:
        """One-time index setup for query speed at cloud scale. Best-effort and
        idempotent — never raised past this method, since missing an index is a
        performance detail, never a reason to fail startup or drop to local JSON.

        Guarded by a process-wide flag: with a shared client the indexes only
        ever need creating once, not on every DBManager construction."""
        global _indexes_ready
        if _indexes_ready:
            return
        try:
            for coll_name in (PAPER_COLLECTION, LIVE_COLLECTION):
                coll = self.db[coll_name]
                coll.create_index("trade_id", unique=True)
                coll.create_index([("status", 1), ("timestamp", -1)])
            _indexes_ready = True
        except Exception as exc:
            print(f"[DBManager] Index setup skipped ({exc}).")

    def _demote_to_local(self, exc: Exception, where: str) -> None:
        """Circuit breaker: once a live Mongo call fails mid-session (cloud
        network blip, auth expiry, etc.), stop trusting Mongo for the rest of
        this process and route everything through local JSON instead. This is
        the same fallback the bot already uses when Mongo is unreachable at
        startup — just extended to a failure that shows up mid-session, so it
        can never raise up into the engine and abort a trade that's already
        live at the broker."""
        print(f"[DBManager] Mongo {where} failed ({exc}); switching to local "
              f"JSON for the remainder of this session.")
        self.client = None
        self.db = None

    @property
    def backend(self) -> str:
        return "MongoDB" if self.db is not None else "Local JSON"

    def _local_path(self, env: Environment) -> str:
        return os.path.join(config.LOCAL_DB_DIR, f"{_collection_name(env)}.jsonl")

    def _log_path(self, env: Environment) -> str:
        # A single, stable running workbook per environment — the "live bot
        # excel sheet". Distinct from export_excel's timestamped snapshots: this
        # one is rewritten after every insert/close so it always reflects every
        # trade the bot has taken. Environment-scoped, so live and paper never
        # share a file (Rule #2).
        return os.path.join(config.LOCAL_DB_DIR, f"{_collection_name(env)}_log.xlsx")

    def sync_excel_log(self, env: Environment) -> Optional[str]:
        """Refresh the running Excel log from the source of truth (Mongo/JSON).
        Rewritten in full so exits update the same rows as their entries. Failure
        (e.g. the file is open in Excel and locked) is logged, never raised — the
        bot must not lose a trade because a spreadsheet couldn't be written."""
        try:
            trades = self.get_trades(env)
            summary = self.analytics_summary(env)
            daily = self.daily_pnl(env)
            path = self._log_path(env)
            with pd.ExcelWriter(path, engine="openpyxl") as writer:
                (trades if not trades.empty else pd.DataFrame(
                    columns=["trade_id"])).to_excel(
                    writer, sheet_name="Trades", index=False)
                pd.DataFrame([summary]).T.rename(columns={0: "value"}).to_excel(
                    writer, sheet_name="Summary")
                daily.to_excel(writer, sheet_name="Daily PnL", index=False)
            return path
        except Exception as exc:
            print(f"[DBManager] Excel log update failed ({exc}).")
            return None

    # -- schema ------------------------------------------------------------- #
    @staticmethod
    def new_trade(
        mode: str, environment: str, broker: str, ticker: str, side: str,
        entry_price: float, stop_loss: float, target: float, quantity: int,
        risk_amount: float, segment: str = "", contract_multiplier: int = 1,
        strategy: str = "", entry_reason: str = "", broker_gtt_id: str = "",
        user_id: str = "admin",
    ) -> dict[str, Any]:
        """Builds a trade doc matching the Section-5 schema.

        `user_id` is additive (frontend_migration_plan.md §3): defaults to
        "admin" so every pre-multi-tenant call site (and existing stored
        trades, which simply lack the field) behaves exactly as before. Real
        client trades pass their own user_id so per-client PnL/open-position
        queries never see another user's trades.
        """
        return {
            "trade_id": str(uuid.uuid4()),
            "timestamp": datetime.utcnow().isoformat(),
            "user_id": user_id,
            "mode": mode,
            # Which registered strategy produced this trade. A mode can host
            # several, so without this you can't tell them apart in the logs.
            "strategy": strategy,
            # The pattern / price-action that triggered this entry — the SAME text
            # the dashboard shows in its "reason" column (e.g. "Price>VWAP + MACD
            # bullish cross"). Persisted so the Excel log and analytics record WHY
            # each trade was taken, not just its numbers.
            "entry_reason": entry_reason,
            "environment": environment,
            "broker": broker,
            "segment": segment,
            # Asset class (Equity / Commodity / Crypto). Derived from segment
            # but STORED, not computed on the fly, so a trade's bucket is fixed
            # at the moment it was taken — reclassifying an instrument later
            # must not silently rewrite history.
            "category": config.category_for_segment(segment),
            "ticker": ticker,
            "side": side,
            "entry_price": round(float(entry_price), 4),
            "stop_loss": round(float(stop_loss), 4),
            "target": round(float(target), 4),
            "quantity": int(quantity),
            # Stored per-trade rather than looked up at close time: contract specs
            # change at expiry, and a closed trade's PnL must stay reproducible
            # from the document itself.
            "contract_multiplier": int(contract_multiplier),
            "risk_amount": round(float(risk_amount), 2),
            # Broker-side protective order id (e.g. a Kite GTT trigger_id) that
            # mirrors this trade's SL/TP at the broker itself, so the position
            # is protected even if this bot is offline. "" when the broker
            # doesn't support it (place_oco_exit returned False) — never
            # treated as an error, since the engine's own polling loop is the
            # primary exit path regardless.
            "broker_gtt_id": broker_gtt_id or "",
            "status": "OPEN",
            "exit_price": None,
            "realized_pnl": None,
            "exit_timestamp": None,
            # Why the position was closed — "TARGET", "STOP-LOSS" or "TIME-EXIT".
            # Filled by close_trade so the log reads the full story: the entry
            # pattern that opened it and the exit condition that closed it.
            "exit_reason": None,
        }

    # -- create ------------------------------------------------------------- #
    def insert_trade(self, trade: dict, env: Environment) -> str:
        if self.db is not None:
            try:
                self.db[_collection_name(env)].insert_one(dict(trade))
            except Exception as exc:
                self._demote_to_local(exc, "insert")
        if self.db is None:
            with open(self._local_path(env), "a", encoding="utf-8") as fh:
                fh.write(json.dumps(trade) + "\n")
        self.sync_excel_log(env)
        return trade["trade_id"]

    # -- update (close a position) ----------------------------------------- #
    def close_trade(
        self, trade_id: str, exit_price: float, env: Environment,
        exit_reason: str = "",
    ) -> Optional[dict]:
        if self.db is not None:
            try:
                coll = self.db[_collection_name(env)]
                doc = coll.find_one({"trade_id": trade_id})
                if not doc:
                    return None
                pnl = self._pnl(doc, exit_price)
                coll.update_one(
                    {"trade_id": trade_id},
                    {"$set": {
                        "status": "CLOSED",
                        "exit_price": round(float(exit_price), 4),
                        "realized_pnl": round(pnl, 2),
                        "exit_timestamp": datetime.utcnow().isoformat(),
                        "exit_reason": exit_reason,
                    }},
                )
                doc.update(status="CLOSED", exit_price=exit_price, realized_pnl=pnl,
                           exit_reason=exit_reason)
                self.sync_excel_log(env)
                return doc
            except Exception as exc:
                self._demote_to_local(exc, "close")
        # local fallback: rewrite the file
        rows = self._read_local(env)
        updated = None
        for r in rows:
            if r["trade_id"] == trade_id and r["status"] == "OPEN":
                r["status"] = "CLOSED"
                r["exit_price"] = round(float(exit_price), 4)
                r["realized_pnl"] = round(self._pnl(r, exit_price), 2)
                r["exit_timestamp"] = datetime.utcnow().isoformat()
                r["exit_reason"] = exit_reason
                updated = r
        if updated is not None:
            with open(self._local_path(env), "w", encoding="utf-8") as fh:
                for r in rows:
                    fh.write(json.dumps(r) + "\n")
            self.sync_excel_log(env)
        return updated

    @staticmethod
    def _pnl(doc: dict, exit_price: float) -> float:
        """Realized PnL in rupees. Direction-aware (a SELL profits when price
        falls) and multiplier-aware for commodities. Trades written before
        contract_multiplier existed default to 1, which is what they assumed."""
        direction = 1 if doc["side"] == "BUY" else -1
        mult = int(doc.get("contract_multiplier", 1) or 1)
        return ((exit_price - doc["entry_price"]) * doc["quantity"]
                * direction * mult)

    # -- read --------------------------------------------------------------- #
    def _read_local(self, env: Environment) -> list[dict]:
        path = self._local_path(env)
        if not os.path.exists(path):
            return []
        with open(path, "r", encoding="utf-8") as fh:
            return [json.loads(line) for line in fh if line.strip()]

    def get_trades(self, env: Environment, user_id: Optional[str] = None,
                   category: Optional[str] = None) -> pd.DataFrame:
        """`user_id=None` (the default every pre-multi-tenant caller uses)
        returns every trade in the environment, unfiltered — identical to
        Phase 1. Pass a user_id to scope to one account's own trades; trades
        stored before the user_id field existed are treated as "admin" so
        old data isn't silently dropped from your own history."""
        if self.db is not None:
            try:
                docs = list(self.db[_collection_name(env)].find({}, {"_id": 0}))
            except Exception as exc:
                self._demote_to_local(exc, "read")
                docs = self._read_local(env)
        else:
            docs = self._read_local(env)
        if not docs:
            return pd.DataFrame()
        df = pd.DataFrame(docs).sort_values("timestamp", ascending=False)
        df = df.reset_index(drop=True)
        # An OPEN trade's exit_price/realized_pnl/exit_timestamp/exit_reason are
        # stored as null (see new_trade). Building one DataFrame out of OPEN and
        # CLOSED rows together coerces those column-wide to NaN (pandas' float
        # columns can't hold a bare None) — harmless for the internal
        # aggregations below (closed-only slices never touch these NaNs), but
        # NaN is not valid JSON, so it must become a real None again before any
        # caller (the API layer, in particular) can serialize a row that
        # includes an OPEN trade.
        # Cast to object dtype BEFORE replacing — on a float64 column, .where
        # silently reinserts NaN for None (a float64 array cannot hold a bare
        # None), so the substitution only sticks once the column can actually
        # hold a Python None.
        df = df.astype(object).where(df.notna(), None)
        if user_id is not None:
            owner = df["user_id"] if "user_id" in df.columns else pd.Series("admin", index=df.index)
            owner = owner.fillna("admin")
            df = df[owner == user_id].reset_index(drop=True)

        # Categorise EVERY row, including trades written before the field
        # existed. Derived here rather than only at write time so the split is
        # complete whether or not the one-off backfill has been run — an
        # uncategorised trade would silently drop out of every per-category
        # total, which is worse than the cost of computing it.
        if not df.empty:
            # Both columns may be absent entirely (a collection where no trade
            # has been categorised yet — a fresh deployment, or the moment
            # before the first backfill) or present-but-null for individual
            # rows. `str` is the ONLY value worth keeping: pandas represents a
            # missing cell as float NaN, and NaN is TRUTHY, so a plain
            # `c if c else derive` silently keeps the NaN and derives nothing —
            # which is how every category total came back zero.
            missing = [None] * len(df)
            existing = list(df["category"]) if "category" in df.columns else missing
            seg = list(df["segment"]) if "segment" in df.columns else missing
            df["category"] = [
                c if isinstance(c, str) and c
                else config.category_for_segment(s if isinstance(s, str) else "")
                for c, s in zip(existing, seg)
            ]
            if category:
                df = df[df["category"] == category].reset_index(drop=True)
        return df

    def get_open_trades(self, env: Environment, user_id: Optional[str] = None) -> list[dict]:
        df = self.get_trades(env, user_id=user_id)
        if df.empty:
            return []
        return df[df["status"] == "OPEN"].to_dict("records")

    # -- destructive: wipe an environment ---------------------------------- #
    # -- categories ---------------------------------------------------------- #
    def backfill_categories(self, env: Environment) -> dict[str, Any]:
        """Write `category` onto every stored trade that lacks one.

        Reads are already safe without this (get_trades derives the category
        for old rows), so this is about the STORED record: once backfilled,
        the raw collection and the Excel export carry the split too, and a
        category query can be pushed down to Mongo instead of filtered in
        pandas. Idempotent — running it twice touches nothing the second time.
        """
        updated = 0
        if self.db is not None:
            try:
                coll = self.db[_collection_name(env)]
                for doc in coll.find(
                        {"$or": [{"category": {"$exists": False}},
                                 {"category": None}, {"category": ""}]},
                        {"_id": 1, "segment": 1}):
                    coll.update_one(
                        {"_id": doc["_id"]},
                        {"$set": {"category": config.category_for_segment(
                            doc.get("segment", ""))}})
                    updated += 1
                return {"backend": "MongoDB", "updated": updated}
            except Exception as exc:
                self._demote_to_local(exc, "category backfill")

        path = self._local_path(env)
        if not os.path.exists(path):
            return {"backend": "Local JSON", "updated": 0}
        rows = self._read_local(env)
        for row in rows:
            if not row.get("category"):
                row["category"] = config.category_for_segment(row.get("segment", ""))
                updated += 1
        if updated:
            with open(path, "w", encoding="utf-8") as fh:
                for row in rows:
                    fh.write(json.dumps(row) + "\n")
        return {"backend": "Local JSON", "updated": updated}

    def category_summary(self, env: Environment, user_id: Optional[str] = None
                         ) -> list[dict[str, Any]]:
        """One analytics row PER CATEGORY, every category always present.

        Categories with no trades are returned as zeroes rather than omitted:
        a missing Crypto row reads as "something is broken", whereas an
        explicit zero reads as "nothing traded yet", which is the truth.
        """
        out = []
        for name in config.ALL_CATEGORIES:
            summary = self.analytics_summary(env, user_id=user_id, category=name)
            out.append({"category": name, **summary})
        return out

    # -- targeted deletion --------------------------------------------------- #
    def reset_range(self, env: Environment, start: str, end: str,
                    user_id: Optional[str] = None,
                    category: Optional[str] = None,
                    dry_run: bool = True) -> dict[str, Any]:
        """Delete trades whose TRADING DAY falls in [start, end] inclusive.

        Built for one specific job: removing trades that were punched against
        simulated or bad data, without discarding the real history around
        them. `reset_environment` is all-or-nothing; this is the scalpel.

        `dry_run=True` (the DEFAULT, deliberately) counts what WOULD go and
        deletes nothing. A destructive operation whose scope can only be
        discovered by performing it is not one anyone should be offered — the
        API exposes the preview first and requires an explicit confirm.

        Dates are `YYYY-MM-DD` trading-day keys, matching how trades are
        stamped (see `_trade_day`). Scoping is AND-ed: environment, then
        optionally user, optionally category, then the date range.
        """
        start, end = str(start or "")[:10], str(end or "")[:10]
        if not start or not end:
            raise ValueError("Both a start and an end date are required.")
        if start > end:
            raise ValueError(f"Start date {start} is after end date {end}.")

        df = self.get_trades(env, user_id=user_id, category=category)
        if df.empty:
            return {"matched": 0, "removed": 0, "dry_run": dry_run,
                    "start": start, "end": end, "by_day": {}, "open_matched": 0}

        days = df["timestamp"].map(self._trade_day)
        hit = df[(days >= start) & (days <= end)]
        if hit.empty:
            return {"matched": 0, "removed": 0, "dry_run": dry_run,
                    "start": start, "end": end, "by_day": {}, "open_matched": 0}

        ids = set(hit["trade_id"].tolist())
        by_day: dict[str, int] = {}
        for day in hit["timestamp"].map(self._trade_day):
            by_day[day] = by_day.get(day, 0) + 1
        # Surfaced separately: deleting a trade the bot still holds OPEN leaves
        # a real position with nothing tracking it. The caller decides, but it
        # must not be a surprise.
        open_matched = int((hit["status"] == "OPEN").sum())

        result = {"matched": len(ids), "removed": 0, "dry_run": dry_run,
                  "start": start, "end": end,
                  "by_day": dict(sorted(by_day.items())),
                  "open_matched": open_matched}
        if dry_run:
            return result

        removed = 0
        if self.db is not None:
            try:
                res = self.db[_collection_name(env)].delete_many(
                    {"trade_id": {"$in": list(ids)}})
                removed = int(getattr(res, "deleted_count", 0) or 0)
            except Exception as exc:
                self._demote_to_local(exc, "range reset")
        if self.db is None:
            path = self._local_path(env)
            if os.path.exists(path):
                kept = [r for r in self._read_local(env)
                        if r.get("trade_id") not in ids]
                removed = len(self._read_local(env)) - len(kept)
                with open(path, "w", encoding="utf-8") as fh:
                    for row in kept:
                        fh.write(json.dumps(row) + "\n")
        result["removed"] = removed
        # The running workbook is derived from the trades, so it has to be
        # rebuilt or it keeps showing rows that no longer exist.
        self.sync_excel_log(env)
        return result

    def reset_environment(self, env: Environment, user_id: Optional[str] = None) -> dict[str, Any]:
        """Permanently delete ALL trades for ONE environment and its running Excel
        log, returning the bot to a blank slate. IRREVERSIBLE.

        Scoped to a single environment on purpose — Immutable Rule #2 keeps paper
        and live separate, so resetting the paper book must never touch the live
        one (and vice-versa). Returns counts of what was removed for the UI to
        confirm. The whole-history day-wise record is derived from these trades, so
        clearing them clears every past day too — a true fresh start.

        `user_id=None` wipes the WHOLE environment (Phase 1 behavior, and still
        what the admin's own reset button does). A user_id scopes the delete to
        just that account's trades — used by a client resetting their own book —
        and re-syncs (rather than deletes) the shared Excel log so other users'
        trades still appear in it."""
        removed = 0
        files: list[str] = []
        if user_id is None:
            if self.db is not None:
                try:
                    res = self.db[_collection_name(env)].delete_many({})
                    removed = int(getattr(res, "deleted_count", 0) or 0)
                except Exception as exc:
                    self._demote_to_local(exc, "reset")
            if self.db is None:
                path = self._local_path(env)
                if os.path.exists(path):
                    with open(path, "r", encoding="utf-8") as fh:
                        removed = sum(1 for line in fh if line.strip())
                    os.remove(path)
                    files.append(path)
            # Drop the running Excel workbook too, so it doesn't resurrect old numbers.
            log = self._log_path(env)
            if os.path.exists(log):
                try:
                    os.remove(log)
                    files.append(log)
                except OSError as exc:            # e.g. open in Excel and locked
                    print(f"[DBManager] Could not remove {log} ({exc}).")
        else:
            if self.db is not None:
                try:
                    res = self.db[_collection_name(env)].delete_many({"user_id": user_id})
                    removed = int(getattr(res, "deleted_count", 0) or 0)
                except Exception as exc:
                    self._demote_to_local(exc, "reset")
            if self.db is None:
                path = self._local_path(env)
                rows = self._read_local(env)
                kept = [r for r in rows if r.get("user_id", "admin") != user_id]
                removed = len(rows) - len(kept)
                with open(path, "w", encoding="utf-8") as fh:
                    for r in kept:
                        fh.write(json.dumps(r) + "\n")
            self.sync_excel_log(env)   # re-sync (not delete) — other users' trades remain
        print(f"[DBManager] Reset {_collection_name(env)}"
              f"{f' for user {user_id}' if user_id else ''}: removed {removed} "
              f"trade(s).")
        return {"trades_removed": removed, "files_removed": files}

    # -- analytics + Excel export ------------------------------------------ #
    def analytics_summary(self, env: Environment, user_id: Optional[str] = None,
                          category: Optional[str] = None) -> dict[str, Any]:
        df = self.get_trades(env, user_id=user_id, category=category)
        closed = df[df["status"] == "CLOSED"] if not df.empty else pd.DataFrame()
        if closed.empty:
            return {"total_trades": len(df), "closed_trades": 0, "win_rate": 0.0,
                    "total_pnl": 0.0, "avg_pnl": 0.0, "best": 0.0, "worst": 0.0}
        pnl = closed["realized_pnl"].astype(float)
        wins = (pnl > 0).sum()
        return {
            "total_trades": len(df),
            "closed_trades": len(closed),
            "win_rate": round(100 * wins / len(closed), 2),
            "total_pnl": round(pnl.sum(), 2),
            "avg_pnl": round(pnl.mean(), 2),
            "best": round(pnl.max(), 2),
            "worst": round(pnl.min(), 2),
        }

    # -- day-wise tracking -------------------------------------------------- #
    # The "trading day" of a trade is the calendar date of its (UTC) timestamp.
    # Indian market hours (≈03:30–18:00 UTC for the equity + MCX evening session)
    # never cross a UTC-date boundary, so this date is also the IST trading date,
    # and it rolls over at 00:00 UTC = 05:30 IST — i.e. every pre-market morning,
    # which is exactly when the live day-PnL counter should reset to zero.
    @staticmethod
    def today_key() -> str:
        """The current trading-day key, consistent with how trades are stamped."""
        return datetime.utcnow().date().isoformat()

    @staticmethod
    def _trade_day(ts: Any) -> str:
        return str(ts)[:10] if ts else ""

    def today_realized(self, env: Environment, day: Optional[str] = None,
                       user_id: Optional[str] = None) -> float:
        """Realized PnL booked on ONE trading day (default: today).

        This is the source of truth for the live "today's PnL" figure. Because it
        is computed from stored trades, it survives a restart — the number the bot
        shows after you reopen it is rebuilt from disk, not lost. Grouping by the
        entry timestamp's date means each day owns the trades opened that day, so a
        new morning starts at ₹0 while yesterday stays on record."""
        df = self.get_trades(env, user_id=user_id)
        if df.empty:
            return 0.0
        day = day or self.today_key()
        closed = df[df["status"] == "CLOSED"].copy()
        if closed.empty:
            return 0.0
        closed["_day"] = closed["timestamp"].map(self._trade_day)
        today = closed[closed["_day"] == day]
        if today.empty:
            return 0.0
        return round(float(today["realized_pnl"].astype(float).sum()), 2)

    def trades_opened_today(self, env: Environment, day: Optional[str] = None,
                            user_id: Optional[str] = None) -> int:
        """Count of trades OPENED on one trading day (default: today), OPEN or
        CLOSED — i.e. every entry taken today. Derived from stored trades (not
        an in-memory counter) so a restart mid-day doesn't reset a
        max-trades-per-day kill switch back to zero. Mirrors today_realized's
        date-grouping exactly, so both agree on what "today" means."""
        df = self.get_trades(env, user_id=user_id)
        if df.empty:
            return 0
        day = day or self.today_key()
        return int((df["timestamp"].map(self._trade_day) == day).sum())

    def daily_pnl(self, env: Environment, user_id: Optional[str] = None,
                  category: Optional[str] = None) -> pd.DataFrame:
        """Day-wise history: one row per trading day, newest first. This is the
        permanent record the dashboard reads so past days are never lost on a
        restart — every day the bot has ever traded stays here, on disk."""
        df = self.get_trades(env, user_id=user_id, category=category)
        if df.empty:
            return pd.DataFrame(columns=[
                "Date", "Trades", "Closed", "Open", "Wins", "Win Rate %",
                "Realized PnL (₹)"])
        df = df.copy()
        df["_day"] = df["timestamp"].map(self._trade_day)
        rows = []
        for day, g in df.groupby("_day"):
            closed = g[g["status"] == "CLOSED"]
            pnl = closed["realized_pnl"].astype(float) if not closed.empty \
                else pd.Series(dtype=float)
            wins = int((pnl > 0).sum())
            rows.append({
                "Date": day,
                "Trades": int(len(g)),
                "Closed": int(len(closed)),
                "Open": int((g["status"] == "OPEN").sum()),
                "Wins": wins,
                "Win Rate %": round(100 * wins / len(closed), 2) if len(closed)
                else 0.0,
                "Realized PnL (₹)": round(float(pnl.sum()), 2),
            })
        out = pd.DataFrame(rows).sort_values("Date", ascending=False)
        return out.reset_index(drop=True)

    def export_excel(self, env: Environment, path: Optional[str] = None,
                     user_id: Optional[str] = None) -> str:
        """High-level analysis workbook: raw trades + a summary sheet."""
        if path is None:
            stamp = config.now_ist().strftime("%Y%m%d_%H%M%S")
            path = os.path.join(config.LOCAL_DB_DIR,
                                f"{_collection_name(env)}_analysis_{stamp}.xlsx")
        trades = self.get_trades(env, user_id=user_id)
        summary = self.analytics_summary(env, user_id=user_id)
        daily = self.daily_pnl(env, user_id=user_id)

        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            (trades if not trades.empty else pd.DataFrame(
                columns=["trade_id"])).to_excel(
                writer, sheet_name="Trades", index=False)
            pd.DataFrame([summary]).T.rename(columns={0: "value"}).to_excel(
                writer, sheet_name="Summary")
            daily.to_excel(writer, sheet_name="Daily PnL", index=False)
            if not trades.empty and "mode" in trades:
                by_mode = trades[trades["status"] == "CLOSED"].groupby("mode")[
                    "realized_pnl"].agg(["count", "sum", "mean"])
                if not by_mode.empty:
                    by_mode.to_excel(writer, sheet_name="By Mode")
        return path
