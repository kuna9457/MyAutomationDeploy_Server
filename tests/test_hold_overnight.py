"""`hold_overnight` and `ignore_entry_cutoff` — the two research flags.

Both answer questions about the SIGNAL rather than configure the bot:

    hold_overnight       does this setup have edge when it is given room,
                         instead of being flattened at 15:09?
    ignore_entry_cutoff  is the late-entry gate still earning its keep once
                         a position is allowed to run?

They are SEPARATE flags on purpose and the separation is the point of this
file. Tied together they move two variables per run — one changes how trades
CLOSE, the other how many are TAKEN — and the difference is attributable to
neither. Measured on RELIANCE (Sep 2025 - Aug 2026, Rs2L, Candlestick Phase 1,
net of intraday costs) they pull in OPPOSITE directions, which is exactly why
a single combined flag read as a wash:

    hold  cutoff    RR 1.0   RR 2.0   RR 3.0
    off   11:59      +2.33    +3.66    +4.24     <- shipped behaviour
    off   dropped    -3.25    +0.27    +0.67     <- cutoff pays for itself
    on    11:59      +3.56    +7.46    +6.18     <- holding pays
    on    dropped    -1.69    +0.03    -2.31     <- and the two cancel out

Neither flag may leak into live trading, and both must be inert by default.
"""
from __future__ import annotations

import inspect
from datetime import time

import pytest

import backtester
import bulk_backtester
import config
from config import Mode, Segment

P = config.CANDLE_INTRADAY_PARAMS


# --------------------------------------------------------------------------- #
#  Inert at their defaults.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("flag", ("hold_overnight", "ignore_entry_cutoff"))
def test_flag_defaults_to_off_on_every_backtest_entry_point(flag):
    """A caller that has never heard of these flags measures what it always
    measured. Checked on every runner, because a flag that defaults True on
    one of them would silently change that surface's numbers."""
    for fn in (backtester.run_backtest, backtester.run_bulk_backtest,
               backtester.run_rr_sweep, bulk_backtester.run_with_costs):
        sig = inspect.signature(fn)
        assert flag in sig.parameters, f"{fn.__name__} cannot forward {flag}"
        assert sig.parameters[flag].default is False, (
            f"{fn.__name__}.{flag} must default to False")


# --------------------------------------------------------------------------- #
#  The two flags are independent.
# --------------------------------------------------------------------------- #
def test_the_flags_are_separate_parameters_not_one_combined_switch():
    """The regression this guards: an earlier version derived the entry cutoff
    from the flat-out, so removing the square-off removed the cutoff too. That
    is defensible in principle and useless in practice — it made the holding
    question unanswerable, and the two effects happen to have opposite signs
    and similar size, so the combined flag read as roughly neutral while
    hiding a real +3.8pp gain from holding."""
    sig = inspect.signature(backtester.run_backtest)
    assert sig.parameters["hold_overnight"] is not sig.parameters[
        "ignore_entry_cutoff"]


# --------------------------------------------------------------------------- #
#  Live trading cannot reach them.
# --------------------------------------------------------------------------- #
def test_live_square_off_is_untouched_by_the_research_flags():
    """These are backtest arguments. Nothing in config's live derivation takes
    them, so no combination of them can leave a real intraday position open
    past the flat-out."""
    assert Mode.INTRADAY in config.SQUARE_OFF_MODES
    assert config.square_off_time_for(Segment.EQUITY, Mode.INTRADAY) == time(15, 9)
    # Candlestick Intraday's cutoff was removed 2026-10-07 — none to derive.
    assert config.entry_cutoff_for(Segment.EQUITY, Mode.INTRADAY, P) is None
    for fn in (config.square_off_time_for, config.entry_cutoff_for):
        params = set(inspect.signature(fn).parameters)
        assert not params & {"hold_overnight", "ignore_entry_cutoff"}, (
            f"config.{fn.__name__} must not know about the research flags")


def test_admin_config_cannot_arm_them():
    """AdminConfigRequest is what the live bot is configured from. Both flags
    are deliberately absent from it: a held-overnight cash position is
    DELIVERY, which changes what can be traded (no shorts), what it costs
    (delivery STT/stamp) and how it must be funded. That is a trading decision,
    not a checkbox."""
    from api import schemas
    fields = set(schemas.AdminConfigRequest.model_fields)
    assert not fields & {"hold_overnight", "ignore_entry_cutoff"}


def test_every_backtest_request_shape_carries_both_flags():
    """...and every backtest surface DOES have them, so a bulk screen and the
    single-symbol tab beside it cannot measure differently shaped runs — the
    same rule test_bulk_run_shape.py enforces for the exit style."""
    from api import schemas
    for cls in (schemas.BacktestRequest, schemas.BulkBacktestRequest,
                schemas.RRSweepRequest):
        fields = set(cls.model_fields)
        for flag in ("hold_overnight", "ignore_entry_cutoff"):
            assert flag in fields, f"{cls.__name__} is missing {flag}"
            assert cls.model_fields[flag].default is False
