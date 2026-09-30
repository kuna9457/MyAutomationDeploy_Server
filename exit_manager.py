"""
exit_manager.py

ONE exit state-machine, shared by the LIVE engine and the BACKTESTER.

Why this file exists
--------------------
Today engine.py manages exits (partial scale-out `_maybe_scale_out`, ATR
chandelier trail `_apply_trail`, break-even move, stop/target checks) but
backtester.py models NONE of it — it exits on the fixed entry stop/target/time
only. So the bot you backtest is not the bot you run. Any result you get for a
trailing / partial-booking approach is fiction, because the simulator never
trails and never scales out.

This module makes the exit LOGIC a pure function of (state, price, atr). The
live engine calls it per tick; the backtester calls it per bar. Same code, same
decisions, so a backtest of these approaches finally measures the real system.

Both of your approaches are just parameter settings on ONE machine:

    Approach 1  (trail the FULL position, no scaling)
        partial_exit_fraction = 0.0
        trail_from_entry      = True
        trail_atr_mult        = k         # chandelier distance in ATR

    Approach 2  (book part at target, then trail the runner)
        partial_exit_fraction = 0.5       # or whatever the backtest picks
        breakeven_buffer_ticks = 3        # SL to entry + friction after partial
        runner_rr_mult        = 3.0       # runner target, or 0 = pure trail
        trail_remainder       = True
        trail_atr_mult        = k

Design rules (kept identical to the engine's documented invariants)
-------------------------------------------------------------------
* RATCHET ONLY   — the stop moves toward price, never away. Quantity is fixed at
                   entry and never revisited, so risk-per-trade can only shrink.
* NEVER PAST PRICE / TARGET — a trailed stop is clamped one tick to the safe
                   side of the live price and of the target, so it can never
                   book a fill the market never printed.
* TICK-SNAPPED THE SAFE WAY — floor for longs, ceil for shorts (looser side).
* COSTS ARE REAL — every exit returns its fill so the caller can charge
                   round-trip costs. A partial is a SECOND exit and is charged
                   again; that double charge is the whole reason to test, not
                   assume, that scaling out helps on small equity size.

This module has NO broker, NO dataframe, NO IO — it only decides. The caller
turns decisions into real orders (engine) or PnL rows (backtester).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# --------------------------------------------------------------------------- #
#  Configuration — mirrors the StrategyParams fields engine.py already reads.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ExitParams:
    tick_size: float = 0.05
    contract_multiplier: int = 1

    # -- Partial scale-out (Approach 2) ------------------------------------- #
    #: Fraction of quantity booked when the FIRST target is reached. 0 disables
    #: scaling (Approach 1). Only positions with qty >= 2 can split; one
    #: indivisible lot always runs to the full target.
    partial_exit_fraction: float = 0.0
    #: After the partial, the stop moves to entry + this many ticks (long), so
    #: the runner is free AFTER round-trip costs, not merely level before them.
    #:
    #: A TICK COUNT ONLY DESCRIBES ONE INSTRUMENT. Real round-trip friction is
    #: mostly proportional to turnover (STT, exchange, GST, stamp, and above
    #: all slippage), so in ticks it swings enormously with the price and the
    #: tick grid — measured against cost_model, a ~Rs1L intraday equity round
    #: trip costs 7 ticks of a Rs250 share but 92 ticks of a Rs3,200 one, while
    #: CRUDEOIL's costs 9 ticks of its Rs1.0 grid. Use this field only where a
    #: FIXED tick count is genuinely what is meant; otherwise use the bps field
    #: below, which is the same number on every instrument.
    breakeven_buffer_ticks: float = 0.0
    #: The same buffer, in BASIS POINTS OF THE ENTRY PRICE — the unit that
    #: actually transfers between instruments. The two are ADDITIVE, and both
    #: default to 0, so a caller that sets neither gets a plain break-even stop
    #: and a caller that sets only ticks behaves exactly as before this field
    #: existed.
    #:
    #: 15 bps covers a round trip with room to spare on both books (measured:
    #: ~14.3 bps intraday equity, ~12.5 bps MCX crude — see cost_model.py).
    #: Stated as a constant rather than computed from cost_model so this module
    #: stays pure: no imports from the trading path, nothing to mock, and the
    #: same number whether it is the engine or the simulator asking.
    breakeven_buffer_bps: float = 0.0
    #: Runner target as a multiple of the ORIGINAL risk distance. 0 = no hard
    #: runner target: the trail alone defines the exit (usually captures more in
    #: a real trend — backtest both).
    runner_rr_mult: float = 0.0
    #: Trail the remaining runner after the partial. Default OFF, so a position
    #: with nothing configured stays a plain fixed-RR trade (current behaviour).
    trail_remainder: bool = False

    # -- Trailing (both approaches) ----------------------------------------- #
    #: Chandelier distance in ATR:  long stop = peak - mult*ATR.
    #: 0 disables trailing.
    trail_atr_mult: float = 0.0
    #: Trail the FULL position from entry (Approach 1). When False, trailing
    #: only switches on for a runner after a partial (Approach 2).
    trail_from_entry: bool = False

    # -- Locking the first target (Approach 2b) ----------------------------- #
    #: After the partial, the runner's stop is never allowed back below the
    #: FIRST TARGET — so the 1R the position already earned cannot be handed
    #: back. Price going above TP1 and then returning to it squares the runner
    #: off there; only a move on to TP2 (runner_rr_mult) pays more.
    #:
    #: Strictly tighter than the break-even buffer, and it REPLACES it rather
    #: than competing with it: the stop is placed at whichever of the two is
    #: further into profit, which is always TP1.
    #:
    #: The trade-off is real and is the whole reason this is a separate knob
    #: rather than the default: a runner floored at TP1 books ~1R reliably but
    #: is stopped out by any pullback to it, while one floored at break-even
    #: risks the 1R for the chance of 2R+. Which is better is a property of
    #: the instrument (see analysis/mfe_study.py), so measure, don't guess.
    lock_first_target: bool = False
    #: WHEN the stop is promoted to the first target, as a fraction of the way
    #: from TP1 to TP2.
    #:
    #:   0.0  promote IMMEDIATELY at TP1 (Approach 2b). The 1R is locked the
    #:        instant the partial books, but any pullback to TP1 ends the
    #:        runner — measured, that is 81% of them.
    #:   0.5  promote only once price reaches the MIDPOINT of TP1..TP2
    #:        (Approach 2c). Until then the runner sits on the break-even +
    #:        friction stop, so it is allowed to breathe: a dip that would have
    #:        ended it under 2b survives, and can still go on to TP2.
    #:
    #: The trade is a real one, not a preference. 2b keeps ~1R on the runner
    #: whenever it stalls; 2c gives that 1R back (down to break-even) in
    #: exchange for the runners that dip and then recover. Which wins depends
    #: on how often a stalled runner comes back, so it is a question for the
    #: backtest, not for taste.
    #:
    #: Ignored when the runner has no finite target (runner_rr_mult = 0) —
    #: there is no TP1..TP2 span to take a fraction of, so promotion falls
    #: back to immediate.
    lock_trigger_frac: float = 0.0
    #: The same trigger expressed in ATR instead of as a fraction of the
    #: TP1..TP2 span, and it WINS over `lock_trigger_frac` whenever both are
    #: set and a usable ATR is available.
    #:
    #: WHY IT IS THE BETTER UNIT. The fractional midpoint is a share of a
    #: distance the STRATEGY chose; it knows nothing about how much this
    #: instrument is actually moving today. On a quiet session half of a 1R
    #: span can sit inside the spread, so the promotion fires on noise; on a
    #: violent one it is far beyond anything the session will deliver. ATR is a
    #: direct reading of the day's noise, so "1.5 x ATR past TP1" means the
    #: same thing on every symbol and in every regime.
    #:
    #: CHECK IT AGAINST YOUR OWN 1R BEFORE CHOOSING A MULTIPLE. The stop is
    #: itself ATR-derived (`atr_sl_mult`), so with the usual 1.5x stop and a
    #: 1:1 target the whole TP1..TP2 span IS about 1.5 x ATR — a 1.5 multiple
    #: then puts the trigger on TP2 itself and the lock can never fire before
    #: the target does. Smaller multiples are the useful range.
    #:
    #: Falls back to `lock_trigger_frac` when the caller has no ATR to give
    #: (a short series, a NaN), so a missing ATR degrades to the old behaviour
    #: rather than silently disarming the promotion altogether.
    lock_trigger_atr_mult: float = 0.0


# --------------------------------------------------------------------------- #
#  StrategyParams -> ExitParams, in ONE place.
#
#  The live engine and the backtester must build the SAME ExitParams from the
#  same StrategyParams or the parity this module exists for is lost before a
#  single decision is made. So neither of them builds one by hand; both call
#  this, and only the two things that genuinely differ between them — the
#  per-symbol trail override (a LIVE admin control, deliberately not read by
#  the backtest) — are passed in as arguments.
# --------------------------------------------------------------------------- #
def params_from_strategy(sp, tick_size: float, contract_multiplier: int = 1,
                         trail_mult: Optional[float] = None,
                         trail_from_entry: Optional[bool] = None
                         ) -> "ExitParams":
    """Build ExitParams from a config.StrategyParams.

    `trail_mult` / `trail_from_entry` override what `sp` says; pass them only
    where a more specific rule exists (engine._exit_params passes the symbol's
    own). Left as None, the trail distance resolves `sp.trail_atr_mult` ->
    `sp.atr_sl_mult`, i.e. the multiple the entry stop itself was built from,
    which is what the trail used before `trail_atr_mult` existed.
    """
    base = float(getattr(sp, "trail_atr_mult", 0.0) or 0.0) or float(sp.atr_sl_mult)
    return ExitParams(
        tick_size=float(tick_size or 0.05),
        contract_multiplier=max(int(contract_multiplier or 1), 1),
        partial_exit_fraction=float(getattr(sp, "partial_exit_fraction", 0.0) or 0.0),
        breakeven_buffer_ticks=float(getattr(sp, "breakeven_buffer_ticks", 0.0) or 0.0),
        breakeven_buffer_bps=float(getattr(sp, "breakeven_buffer_bps", 0.0) or 0.0),
        runner_rr_mult=float(getattr(sp, "runner_rr_mult", 0.0) or 0.0),
        trail_remainder=bool(getattr(sp, "trail_remainder", False)),
        trail_atr_mult=float(base if trail_mult is None else trail_mult),
        trail_from_entry=bool(getattr(sp, "trail_from_entry", False)
                              if trail_from_entry is None else trail_from_entry),
        lock_first_target=bool(getattr(sp, "lock_first_target", False)),
        lock_trigger_frac=float(getattr(sp, "lock_trigger_frac", 0.0) or 0.0),
        lock_trigger_atr_mult=float(
            getattr(sp, "lock_trigger_atr_mult", 0.0) or 0.0),
    )


# --------------------------------------------------------------------------- #
#  Mutable per-position state. The caller owns one of these per open trade and
#  persists whatever it needs (the engine already persists stop/target/qty/
#  partial_done; the backtester keeps it in a local dict).
# --------------------------------------------------------------------------- #
@dataclass
class ExitState:
    side: str                     # "BUY" or "SELL"
    entry: float
    qty: int                      # remaining quantity
    stop: float                   # current (possibly trailed) stop
    target: float                 # current target (moves to runner after 1st target)
    risk_dist: float              # |entry - initial stop|, fixed at entry
    first_target_done: bool = False   # the initial target has been consumed
    partial_taken: bool = False       # a partial was actually booked (for reasons)
    trail_active: bool = False
    peak: float = field(default=0.0)   # best price SEEN (seeded to entry below)
    #: The FIRST target's price, kept after `target` has moved on to the
    #: runner target. 0.0 until the first-target event fires. This is the
    #: level `lock_first_target` floors the runner's stop at, so it has to
    #: outlive the field it was read from.
    first_target: float = 0.0

    def __post_init__(self):
        if not self.peak:
            self.peak = self.entry


class ActionType(str, Enum):
    PARTIAL = "PARTIAL"       # book part of the position at `price`
    MOVE_STOP = "MOVE_STOP"   # trail / break-even move; no fill
    EXIT = "EXIT"             # close the remaining position at `price`


@dataclass(frozen=True)
class Action:
    kind: ActionType
    price: float = 0.0        # fill price for PARTIAL / EXIT
    qty: int = 0              # quantity for PARTIAL / EXIT
    new_stop: float = 0.0     # for MOVE_STOP / after a PARTIAL
    reason: str = ""


# --------------------------------------------------------------------------- #
#  Rounding helpers — same "always to the looser side" rule the engine uses.
# --------------------------------------------------------------------------- #
#: Tolerance for "x is already exactly on the tick grid".
#:
#: x / tick is computed in binary floating point, where a price that IS on the
#: grid routinely lands a hair on the wrong side of the integer: 100.05 / 0.05
#: is 2000.9999999999998, so a bare math.floor() drops a WHOLE TICK. Measured
#: on a 0.05 grid, half of the sampled grid prices snapped to the wrong tick.
#:
#: The error was always to the looser side, so it never booked a fill the
#: market had not printed — it just silently gave back a tick on every
#: break-even move, TP1 lock and trail step. Quantising the ratio first fixes
#: it in one place for all of them. 1e-9 is ~7 orders of magnitude larger than
#: the double-precision error at these magnitudes and far smaller than any real
#: sub-tick amount.
_GRID_EPS = 1e-9


def _floor_tick(x: float, tick: float) -> float:
    return round(math.floor(round(x / tick, 9) + _GRID_EPS) * tick, 2)


def _ceil_tick(x: float, tick: float) -> float:
    return round(math.ceil(round(x / tick, 9) - _GRID_EPS) * tick, 2)


#: Public aliases. engine._enter snaps the ENTRY stop onto the same grid, and
#: it must use the same float-safe quantisation as every stop the exit manager
#: later moves it to — otherwise the level a position starts at and the levels
#: it is managed to are rounded by two different rules, one of them wrong.
floor_tick = _floor_tick
ceil_tick = _ceil_tick


# --------------------------------------------------------------------------- #
#  The one decision function — LIVE parity (one price per call).
# --------------------------------------------------------------------------- #
def step_tick(st: ExitState, price: float, atr: float,
              p: ExitParams) -> list[Action]:
    """Advance one position by one observed `price`. Returns the ordered list of
    actions the caller must apply, in this exact order:

        1. partial scale-out at the first target (books part, moves stop to
           break-even+buffer, sets the runner target, arms the trail),
        2. a trail move (chandelier),
        3. a stop or target exit of whatever remains.

    The order matters and mirrors engine.py: scaling and trailing both run
    BEFORE the stop/target check, so a tick that both improves and then breaches
    is resolved against the UPDATED stop, never a stale one.
    """
    actions: list[Action] = []
    if st.qty <= 0:                 # already fully closed — never act twice
        return actions
    is_long = st.side == "BUY"

    # -- 1. First-target event (once) --------------------------------------- #
    # The ONLY difference between your two approaches lives here: at the first
    # target, Approach 2 books a partial and Approach 1 books nothing. Everything
    # else — move the stop to break-even+buffer, set the next (runner) target,
    # switch the trail on — is identical, so it is written once.
    #
    # This branch fires only when a MANAGED style is configured (a partial, or a
    # trail on the remainder/from entry). With none configured the position is a
    # plain fixed-RR trade: first_target_done stays False, the target stays
    # finite, and the exit check below books the whole thing at the target,
    # exactly as before this module existed.
    managed = (p.partial_exit_fraction > 0 or p.trail_remainder
               or p.trail_from_entry)
    if (managed and not st.first_target_done and math.isfinite(st.target)
            and _reached(is_long, price, st.target)):
        st.first_target_done = True
        # Remember TP1 before `target` is rewritten to the runner target — it
        # is the level the whole lock rule below is defined against.
        st.first_target = st.target
        if p.partial_exit_fraction > 0 and st.qty >= 2:
            exit_qty = max(1, int(math.floor(st.qty * p.partial_exit_fraction)))
            exit_qty = min(exit_qty, st.qty - 1)      # always leave a runner
            actions.append(Action(ActionType.PARTIAL, price=st.target,
                                  qty=exit_qty, reason="PARTIAL-TARGET"))
            st.qty -= exit_qty
            st.partial_taken = True

        # Ticks AND bps, added: a fixed grid distance plus a share of the
        # notional. Either alone is a valid configuration; together they are
        # "at least this many ticks, plus this much of the trade's own size".
        buf = (p.breakeven_buffer_ticks * p.tick_size
               + (p.breakeven_buffer_bps / 10_000.0) * st.entry)
        be = (_floor_tick(st.entry + buf, p.tick_size) if is_long
              else _ceil_tick(st.entry - buf, p.tick_size))
        # CLAMPED ONE TICK TO THE SAFE SIDE OF THE PRICE, exactly as _trail is
        # and for exactly the same reason. The buffer is a cost estimate, not a
        # level the market has agreed to: on a tight stop it can be WIDER than
        # the whole 1R move it is being added to (a 15 bps buffer is Rs4.80 on
        # a Rs3,200 share, against a Rs1.50 risk distance), which would park
        # the stop ABOVE the price that just filled the partial and book an
        # exit at a price that never printed. Clamping leaves the runner
        # holding nearly the whole 1R instead, which is the honest outcome
        # when friction really does eat the trade.
        # With the lock on, the floor is TP1 rather than break-even. TP1 is
        # always the further of the two into profit, so this only ever
        # tightens.
        # Only promote to TP1 HERE when the trigger is immediate. With a
        # trigger fraction set, the runner starts on break-even + friction and
        # is promoted later, by block 1b, once it has proved itself.
        if p.lock_first_target and p.lock_trigger_frac <= 0:
            be = max(be, st.first_target) if is_long else min(be, st.first_target)
        st.stop = (min(be, _floor_tick(price - p.tick_size, p.tick_size))
                   if is_long
                   else max(be, _ceil_tick(price + p.tick_size, p.tick_size)))
        # The next target — your "new TP from volatility". As a fixed multiple of
        # the ORIGINAL risk here; swap this one line for an ATR-derived level if
        # the backtest prefers it. runner_rr_mult = 0 means NO hard target: the
        # trail alone decides the exit (captures more in a real trend).
        if p.runner_rr_mult > 0:
            st.target = round(
                st.entry + (1 if is_long else -1) * p.runner_rr_mult * st.risk_dist, 2)
        else:
            st.target = math.inf if is_long else -math.inf
        st.trail_active = bool(p.trail_remainder or p.trail_from_entry)
        st.peak = price
        actions.append(Action(ActionType.MOVE_STOP, new_stop=st.stop,
                              reason=("lock TP1"
                                      if p.lock_first_target
                                      and p.lock_trigger_frac <= 0
                                      else "break-even+buffer")))

    # -- 1b. Hold the stop at TP1 (lock_first_target) ------------------------ #
    # Runs EVERY tick, not just the event tick, and that is the point. On the
    # event tick the price IS TP1, so the clamp above had to leave the stop one
    # tick inside it — otherwise the exit check below would fire on that very
    # tick and the runner would never exist at all. As soon as price moves
    # away, this raises the stop the rest of the way to TP1 exactly.
    #
    # The result is the rule as stated: above TP1 and back to TP1 = out, with
    # the 1R kept. It cannot book a fill the market never printed, because
    # price has to have traded through TP1 on the way back down to trigger it.
    if p.lock_first_target and st.first_target_done and st.first_target:
        # WHERE price must get before the stop is promoted to TP1. With
        # lock_trigger_frac = 0 this is TP1 itself, i.e. promote at once. With
        # 0.5 it is the midpoint of TP1..TP2: the runner has to travel half way
        # to the second target before it gives up the right to retrace.
        trigger = st.first_target
        if p.lock_trigger_atr_mult > 0 and atr > 0:
            # Volatility-based: a fixed distance past TP1, measured in the
            # instrument's own noise rather than as a share of the target span.
            trigger = st.first_target + ((1 if is_long else -1)
                                         * p.lock_trigger_atr_mult * atr)
        elif p.lock_trigger_frac > 0 and math.isfinite(st.target):
            trigger = st.first_target + p.lock_trigger_frac * (st.target
                                                               - st.first_target)
        # Ratcheted, so once promoted the stop STAYS at TP1 even after price
        # falls back below the trigger — which is the whole point: the runner
        # earned the level, it does not lose it by retracing.
        if _reached(is_long, price, trigger):
            locked = _ratchet_to(st, st.first_target, price, p)
            if locked is not None:
                actions.append(Action(ActionType.MOVE_STOP, new_stop=locked,
                                      reason="lock TP1"))

    # -- 2. Trail (chandelier) ---------------------------------------------- #
    trailing = st.trail_active or (p.trail_from_entry and p.trail_atr_mult > 0)
    if trailing and p.trail_atr_mult > 0 and atr > 0:
        moved = _trail(st, price, atr, p)
        if moved is not None:
            # Once the trail has ACTUALLY moved the stop, this position's stop
            # is a trailed stop, and _exit_reason must say so rather than call
            # the eventual exit a plain "STOP-LOSS". Set on the move, not on
            # arming, so a trail-from-entry position that never got a
            # favourable tick still reports an honest STOP-LOSS.
            st.trail_active = True
            actions.append(Action(ActionType.MOVE_STOP, new_stop=moved,
                                  reason=f"trail {p.trail_atr_mult:g}xATR"))

    # -- 3. Stop / target exit of the remainder ----------------------------- #
    if is_long:
        if price <= st.stop:
            actions.append(Action(ActionType.EXIT, price=st.stop, qty=st.qty,
                                  reason=_exit_reason(st, "STOP", p)))
        elif price >= st.target:
            actions.append(Action(ActionType.EXIT, price=st.target, qty=st.qty,
                                  reason="TARGET"))
    else:
        if price >= st.stop:
            actions.append(Action(ActionType.EXIT, price=st.stop, qty=st.qty,
                                  reason=_exit_reason(st, "STOP", p)))
        elif price <= st.target:
            actions.append(Action(ActionType.EXIT, price=st.target, qty=st.qty,
                                  reason="TARGET"))
    # Mark the position closed so a caller that keeps calling cannot re-book it.
    if actions and actions[-1].kind == ActionType.EXIT:
        st.qty = 0
    return actions


# --------------------------------------------------------------------------- #
#  BACKTEST helper — one OHLC bar, PESSIMISTIC intrabar ordering.
# --------------------------------------------------------------------------- #
def step_bar(st: ExitState, o: float, h: float, l: float, c: float,
             atr: float, p: ExitParams) -> list[Action]:
    """Advance one position across one OHLC bar WITHOUT look-ahead.

    OHLC hides the path within the bar. Trailing/partial backtests are the most
    prone to optimistic self-deception here: assume the favourable order (target
    before stop, new high lifts the trail before the low tests it) and every
    result flatters itself. So we assume the ADVERSE order:

        long : open -> LOW (tests the current stop) -> HIGH (partial/trail) -> close
        short: open -> HIGH -> LOW -> close

    i.e. the stop is tested at the WORST price the bar reached before any
    favourable move is credited. This under-states trailing systems slightly;
    that is the correct direction to be wrong in. For a scalper the honest fix is
    to feed 1-minute bars (or ticks) through step_tick instead — call this only
    when a finer series isn't available.
    """
    is_long = st.side == "BUY"
    adverse = l if is_long else h        # worst price the bar reached first
    favourable = h if is_long else l     # best price, credited only afterwards

    out: list[Action] = []
    # Adverse extreme first: can the CURRENT stop be hit before anything good?
    out += step_tick(st, adverse, atr, p)
    if any(a.kind == ActionType.EXIT for a in out):
        return out
    # Then the favourable extreme: partial / trail / target.
    out += step_tick(st, favourable, atr, p)
    if any(a.kind == ActionType.EXIT for a in out):
        return out
    # Finally the close, so a trail that tightened mid-bar can still stop the
    # position out on the close.
    out += step_tick(st, c, atr, p)
    return out


# --------------------------------------------------------------------------- #
#  Internals
# --------------------------------------------------------------------------- #
def _reached(is_long: bool, price: float, level: float) -> bool:
    return price >= level if is_long else price <= level


def _exit_reason(st: ExitState, base: str, p: ExitParams) -> str:
    """Name the stop honestly so analytics can separate a real stop-loss from a
    trailed-out winner from a break-even scratch.

    Evaluated at EXIT time and gated on the lock being configured, not on
    where the stop happens to sit: a trailed stop that has ratcheted past TP1
    on its own is a trailed exit, and calling it a lock would credit the wrong
    rule for the money.
    """
    if not st.first_target_done and not st.trail_active:
        return "STOP-LOSS"
    # Stopped out AT the locked first target = the lock did its job. Named
    # apart from a trailed exit because it is a different decision with a
    # different expectation, and mixing them would hide which one is paying.
    # ONE TICK of tolerance, not half: a position that reversed straight off
    # TP1 without ever trading above it is stopped at TP1 - one tick, because
    # that is where the same-tick clamp had to leave the stop. That is still
    # the lock doing its job, and calling it a trailed exit (which is what the
    # `locked > 0` branch below would do) would credit a rule that is not even
    # switched on in this style.
    if (p.lock_first_target and st.first_target
            and abs(st.stop - round(st.first_target, 2)) <= p.tick_size * 1.5):
        return "TP1-LOCK"
    # Stopped on the break-even + friction stop: the runner never earned its
    # promotion. Named against the level that was actually SET rather than by
    # a percentage-of-risk guess, because with a bps buffer on a tight stop
    # "near enough to entry" can be a fifth of R and would read as a trail.
    be_level = st.entry + (1 if st.side == "BUY" else -1) * (
        p.breakeven_buffer_ticks * p.tick_size
        + (p.breakeven_buffer_bps / 10_000.0) * st.entry)
    if (p.partial_exit_fraction > 0
            and abs(st.stop - be_level) <= p.tick_size * 1.5):
        return "BREAK-EVEN"
    locked = (st.stop - st.entry) * (1 if st.side == "BUY" else -1)
    if locked > 0:
        return "TRAIL-PROFIT"
    if abs(locked) < st.risk_dist * 0.05:
        return "BREAK-EVEN"
    return "TRAIL-STOP"


def _ratchet_to(st: ExitState, level: float, price: float,
                p: ExitParams) -> Optional[float]:
    """Move the stop TOWARD `level`, obeying every invariant the trail obeys.

    Ratchet only, never past the live price, never past the target, snapped to
    the tick grid. Returns the new stop if it actually improved, else None.
    Shared by the TP1 lock so the lock cannot become a second, subtly
    different set of rules.
    """
    tick = p.tick_size
    if st.side == "BUY":
        new_stop = min(level, _floor_tick(price - tick, tick), st.target - tick)
        if new_stop > st.stop + tick / 2:
            st.stop = round(new_stop, 2)
            return st.stop
    else:
        new_stop = max(level, _ceil_tick(price + tick, tick), st.target + tick)
        if new_stop < st.stop - tick / 2:
            st.stop = round(new_stop, 2)
            return st.stop
    return None


def _trail(st: ExitState, price: float, atr: float,
           p: ExitParams) -> Optional[float]:
    """Chandelier ratchet. Returns the new stop if it actually improved, else
    None. Identical maths to engine._apply_trail."""
    tick = p.tick_size
    mult = p.trail_atr_mult
    if st.side == "BUY":
        st.peak = max(st.peak, price)
        new_stop = _floor_tick(st.peak - mult * atr, tick)
        new_stop = min(new_stop, price - tick, st.target - tick)
        if new_stop > st.stop + tick / 2:
            st.stop = new_stop
            return new_stop
    else:
        st.peak = min(st.peak, price)
        new_stop = _ceil_tick(st.peak + mult * atr, tick)
        new_stop = max(new_stop, price + tick, st.target + tick)
        if new_stop < st.stop - tick / 2:
            st.stop = new_stop
            return new_stop
    return None
