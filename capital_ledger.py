"""
capital_ledger.py
The coordination layer between one account's several strategy engines.

When a strategy board runs (strategy_groups.py), one TradingEngine is started
per group. Each engine is EXACTLY the engine that runs today — one strategy,
its own instruments, its own decision path, untouched. Everything that must be
true ACROSS those engines lives here instead of inside any of them:

  1. ONE WALLET. Engines size against `ceiling - committed`. Left alone, three
     engines would each see the full ceiling and could collectively commit
     three times the account. The ledger sums committed margin across ALL of
     an account's engines so the pool is shared, exactly as it is between two
     positions inside a single engine today.

  2. ONE POSITION PER SYMBOL. A stock may sit on several strategies (the whole
     point of the board), but only one trade in it may be open at a time; the
     others wait for it to close. `engine.holds()` already enforces this
     WITHIN an engine — this extends the same rule ACROSS them.

Deliberately not a registry of state: the ledger OWNS nothing and stores no
positions. It reads the engines' own `open_positions`, so a closed trade frees
both its margin and its symbol with no bookkeeping to leak or go stale. The
only state it keeps is a short-lived set of in-flight entry claims, which
exists solely to close the check-then-act race between two engines deciding on
the same symbol in the same instant.

A single-group run never constructs one, and every engine hook is a no-op when
`engine.ledger is None` — so the ordinary one-strategy path is byte-identical
to what it was before this module existed.
"""
from __future__ import annotations

import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:                                  # pragma: no cover
    from engine import TradingEngine


class CapitalLedger:
    """Shared by every engine belonging to ONE account. Not global: two
    accounts must never see each other's margin or block each other's
    symbols."""

    def __init__(self, owner: str = "") -> None:
        self.owner = owner
        self._lock = threading.RLock()
        self._engines: list["TradingEngine"] = []
        #: Symbols with an entry IN FLIGHT — claimed but not yet visible in
        #: any engine's open_positions. Held for the duration of one _enter.
        self._pending: set[str] = set()

    # -- membership --------------------------------------------------------- #
    def attach(self, engine: "TradingEngine") -> None:
        with self._lock:
            if engine not in self._engines:
                self._engines.append(engine)
                engine.ledger = self

    def detach(self, engine: "TradingEngine") -> None:
        with self._lock:
            if engine in self._engines:
                self._engines.remove(engine)
            engine.ledger = None

    def engines(self) -> list["TradingEngine"]:
        with self._lock:
            return list(self._engines)

    # -- one wallet --------------------------------------------------------- #
    def committed_margin(self) -> float:
        """Margin tied up by every open position across ALL of this account's
        engines. This is the only number engine._available_capital needs from
        the ledger; the ceiling stays the engine's own, so the LIVE
        capital_allocated cap keeps working exactly as before."""
        total = 0.0
        for eng in self.engines():
            # Each engine's own lock, taken one at a time and never nested
            # inside the ledger lock while another engine's is held — engines
            # never call back into each other, so there is no lock-order cycle.
            with eng.state.lock:
                total += sum(float(t.get("_margin", 0.0))
                             for t in eng.state.open_positions.values())
        return total

    # -- one position per symbol -------------------------------------------- #
    def symbol_busy(self, symbol: str) -> bool:
        """True when this stock already has a trade open (in ANY group) or one
        being opened right now."""
        with self._lock:
            if symbol in self._pending:
                return True
        return any(eng.holds(symbol) for eng in self.engines())

    def claim_symbol(self, symbol: str) -> bool:
        """Reserve `symbol` for an entry about to be attempted. False means
        another group got there first and this signal must be skipped.

        The claim closes the window between "nobody holds it" and "the
        position appears in open_positions" — without it, two groups signalling
        the same stock on the same tick would both pass the check and both
        open, which is precisely the double-position this feature must not
        create. ALWAYS pair with release_symbol() in a finally.
        """
        with self._lock:
            if symbol in self._pending:
                return False
            # Checked inside the ledger lock so two callers cannot both see a
            # free symbol; engine.holds takes only that engine's own lock.
            if any(eng.holds(symbol) for eng in self._engines):
                return False
            self._pending.add(symbol)
            return True

    def release_symbol(self, symbol: str) -> None:
        """Drop an in-flight claim. Safe to call when none is held, so the
        caller's `finally` never needs a condition. After a SUCCESSFUL entry
        the position itself keeps the symbol busy; after a failed one this is
        what puts the stock back in play."""
        with self._lock:
            self._pending.discard(symbol)
