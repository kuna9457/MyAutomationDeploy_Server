"""
cost_model.py
Indian intraday equity transaction-cost calculator.

Every number here is a STATUTORY or BROKER-SCHEDULE constant, not a tuning knob.
If a statutory rate changes (e.g. STT goes from 0.025% to 0.02%), update the
default here and re-run; the backtest output moves by the exact amount the real
book would.

This module is STANDALONE — it imports nothing from the trading path, engine,
strategy, or backtester. It is called by bulk_backtester.py to deduct costs
from each trade, producing the net P&L the improvement plan demands (§0).

All percentages are stored as PERCENT (0.025 means 0.025%), converted to
fractions internally, so the numbers in the constructor match what a broker's
tariff card prints.

The slippage field is a FLAT estimate, not a model. It exists because the
backtest fills at the candle's printed price, and a real order would not.
3 bps per leg (6 bps round trip) is the conservative end for liquid Nifty-100
names on a 15-minute bar; a 1-minute scalper in a less liquid name would be
higher. The honest answer is that slippage is unknowable from OHLCV data, and
this number is a haircut that keeps the ranking from rewarding strategies
whose edge is smaller than the market-impact they would cause.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CostModel:
    """All-in round-trip cost for one Indian intraday equity trade.

    Defaults match a flat-fee discount broker (Zerodha / Upstox / Dhan) as of
    2025-26. MCX commodity trades use the same statutory heads at different
    rates; pass those rates explicitly when constructing.
    """
    #: Flat brokerage, charged per ORDER (entry and exit are two orders).
    brokerage_per_order: float = 20.0

    #: Securities Transaction Tax — charged on the SELL side only for intraday
    #: equity (0.025% of sell turnover as of Finance Act 2023-24).
    stt_sell_pct: float = 0.025

    #: Exchange transaction charge (NSE equity, ~0.00325% of turnover).
    exchange_txn_pct: float = 0.00325

    #: GST — 18% of (brokerage + exchange transaction charges).
    gst_pct: float = 18.0

    #: Stamp duty — charged on the BUY side (0.003% of buy turnover, capped
    #: per state but effectively unlimited for intraday turnover).
    stamp_pct: float = 0.003

    #: SEBI turnover fee (0.0001% of turnover).
    sebi_pct: float = 0.0001

    #: Estimated slippage, in basis points PER LEG.  3 bps × 2 legs = 6 bps
    #: round trip. This is not a fee — it is the gap between the candle price
    #: the backtest fills at and the price a real order would get.
    slippage_bps: float = 3.0

    def round_trip_cost(self, entry_price: float, exit_price: float,
                        qty: float, contract_multiplier: float = 1.0) -> float:
        """Total cost of one round-trip trade, in ₹.

        `qty` is the number of shares/lots. For MCX, pass `contract_multiplier`
        so turnover is in the correct unit (qty × price × multiplier).

        Returns a POSITIVE number — the caller subtracts it from gross P&L.
        """
        buy_turnover = entry_price * qty * contract_multiplier
        sell_turnover = exit_price * qty * contract_multiplier
        total_turnover = buy_turnover + sell_turnover

        # Brokerage: 2 orders (entry + exit), capped at 20 per order for
        # flat-fee brokers. For percentage brokers, override this class.
        brokerage = self.brokerage_per_order * 2

        # STT: sell side only for intraday equity.
        stt = sell_turnover * (self.stt_sell_pct / 100.0)

        # Exchange transaction: both legs.
        exchange_txn = total_turnover * (self.exchange_txn_pct / 100.0)

        # GST: 18% of (brokerage + exchange txn charges).
        gst = (brokerage + exchange_txn) * (self.gst_pct / 100.0)

        # Stamp duty: buy side only.
        stamp = buy_turnover * (self.stamp_pct / 100.0)

        # SEBI turnover fee: both legs.
        sebi = total_turnover * (self.sebi_pct / 100.0)

        # Slippage: estimated per-leg cost.
        slippage = total_turnover * (self.slippage_bps / 10_000.0)

        return brokerage + stt + exchange_txn + gst + stamp + sebi + slippage

    def cost_per_crore(self) -> float:
        """Approximate all-in cost per ₹1 crore of round-trip turnover.

        Useful as a quick sanity check: intraday equity is typically
        ₹2,500–₹4,000 per crore depending on the broker.
        """
        # Use a reference trade at ₹1,000 × 100 shares = ₹1L per leg
        ref_cost = self.round_trip_cost(1000.0, 1000.0, 100)
        ref_turnover = 1000.0 * 100 * 2  # both legs
        return ref_cost / ref_turnover * 1e7  # scale to 1 crore


#: Default cost model for Indian intraday equity on a flat-fee broker.
INTRADAY_EQUITY = CostModel()

#: MCX commodity intraday — same statutory heads, different STT rate (0.01%
#: on sell side for non-agri commodity futures) and no stamp duty on futures.
MCX_COMMODITY = CostModel(
    stt_sell_pct=0.01,       # lower for commodity futures
    exchange_txn_pct=0.0026, # MCX exchange txn charge
    stamp_pct=0.002,         # futures stamp duty (lower than equity)
    slippage_bps=5.0,        # MCX is less liquid; wider estimate
)
