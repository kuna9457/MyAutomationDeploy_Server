"""
broker_api.py
Execution layer. A thin, uniform wrapper over each broker so the engine can
place/track orders without knowing which broker is behind it.

    BaseBroker (interface)
      ├── SimulatedBroker   -> paper fills, no network, always available
      ├── UpstoxBroker      -> sandbox (paper) OR live, via upstox-python-sdk
      ├── DhanBroker        -> live, via dhanhq
      ├── ZerodhaBroker     -> live, via kiteconnect (equity only, see class doc)
      └── KotakNeoBroker    -> live, via neo-api-client

Every broker returns the same OrderResult shape, so strategy/engine code is
broker-agnostic (Immutable Rule #3). Real SDK calls are wrapped in try/except
and degrade to a clear error rather than crashing the bot.

LIMIT ORDERS: implemented for Simulated, Upstox and Zerodha (equity). Dhan and
Kotak inherit the BaseBroker default, which degrades a limit request to a
MARKET order — correct but subject to slippage, which matters for the
Scalper's 1:1 RR. They are left un-implemented rather than written blind: an
untested order-placement path is a worse failure than a market fill, since it
risks a malformed LIVE order.

available_funds(): real broker-reported free cash/margin, used by engine.py as
a LIVE-only safety gate before sizing a trade — independent of, and stricter
than, the user's configured capital allocation (risk_manager.py). Returns None
when a broker doesn't expose it (the caller then skips the extra check rather
than blocking trading on a missing feature).
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import config
from config import Broker, Environment, Instrument, Segment

# Upstox's real order-margin endpoint. Returns SPAN + Exposure + peak margin for a
# basket of instruments — the ACTUAL cash the broker blocks, which for MCX futures
# is nothing like notional ÷ leverage. Read-only, so it is safe to call with the
# live token even from Paper mode.
UPSTOX_MARGIN_URL = "https://api.upstox.com/v2/charges/margin"


def fetch_upstox_margin(
    token: str, instrument_key: str, quantity: int,
    side: str = "BUY", product: str = "D", timeout: float = 10.0,
) -> Optional[float]:
    """Real margin (₹) required to trade `quantity` units of one instrument, from
    Upstox's /charges/margin API. Returns None if it can't be determined — no
    token, network/auth error, or an unexpected response shape — so callers can
    fall back to an estimate. NEVER raises.

    `quantity` is in the SAME units an order uses (lots × lot_size). `price` is
    sent as 0 so Upstox margins against the live LTP; no live price is needed here.
    `product` is "D" (delivery/NRML) for F&O/commodity, "I" (intraday/MIS) for
    equity — margin differs between them.
    """
    if not token or quantity <= 0:
        return None
    try:
        import requests
        body = {"instruments": [{
            "instrument_key": instrument_key,
            "quantity": int(quantity),
            "transaction_type": side,
            "product": product,
            "price": 0,
        }]}
        resp = requests.post(
            UPSTOX_MARGIN_URL,
            headers={"Authorization": f"Bearer {token}",
                     "accept": "application/json",
                     "Content-Type": "application/json"},
            json=body, timeout=timeout)
        if resp.status_code != 200:
            return None
        data = resp.json().get("data", {}) or {}
        # Upstox reports the basket total under final_margin/required_margin; fall
        # back to summing the per-instrument legs if those aren't present.
        for key in ("final_margin", "required_margin"):
            val = data.get(key)
            if isinstance(val, (int, float)) and val > 0:
                return float(val)
        margins = data.get("margins") or []
        total = sum(float(m.get("total_margin", 0) or 0) for m in margins)
        return total if total > 0 else None
    except Exception:
        return None


@dataclass
class OrderResult:
    ok: bool
    order_id: str
    broker: str
    filled_price: float = 0.0
    quantity: int = 0
    message: str = ""


class BaseBroker:
    name: str = "Base"

    def connect(self) -> bool:
        raise NotImplementedError

    def place_market_order(
        self, instrument: Instrument, side: str, quantity: int,
        ref_price: float = 0.0,
    ) -> OrderResult:
        raise NotImplementedError

    def place_limit_order(
        self, instrument: Instrument, side: str, quantity: int,
        limit_price: float, ref_price: float = 0.0,
    ) -> OrderResult:
        """Entry at a chosen price rather than whatever the book offers — the
        Scalper uses this to sit just inside the spread (scalping.md §4), because
        at a 1:1 RR on 1-minute bars, market-order slippage eats the edge.

        Default: brokers that don't implement limits degrade to a market order so
        no broker silently drops the order.
        """
        return self.place_market_order(instrument, side, quantity, ref_price)

    def required_margin(
        self, instrument: Instrument, quantity: int, side: str = "BUY",
        ref_price: float = 0.0,
    ) -> Optional[float]:
        """Real margin (₹) the broker blocks to hold `quantity` of `instrument`, or
        None if this broker can't tell us (the caller then falls back to an
        estimate). `ref_price` is the price to margin against — brokers whose
        margin API needs one (Dhan) use it; brokers that margin off their own
        live LTP (Upstox) ignore it. Only Upstox, Dhan and Zerodha implement
        this today; the rest inherit None."""
        return None

    def available_funds(self) -> Optional[float]:
        """Real free cash/margin (₹) currently available in the broker account
        right now, or None if this broker can't tell us. Used as an independent
        LIVE safety check — the account's configured capital allocation
        (risk_manager.py) can be stale or simply wrong; this asks the broker
        directly. NEVER raises."""
        return None

    def place_oco_exit(
        self, instrument: Instrument, side: str, quantity: int,
        stop_loss: float, target: float, ref_price: float,
    ) -> tuple[bool, str, str]:
        """Place the SL+TP as a REAL protective order at the broker (e.g. Kite's
        GTT OCO), so the position is protected even if this bot goes offline —
        not just watched by our own polling loop. `side` is the position's own
        side (BUY/SELL); the protective order is on the OPPOSITE side.

        Returns (ok, broker_order_id, message). Only Zerodha implements this
        today; every other broker inherits this no-op, which is silent and
        harmless — the position is still fully protected by the engine's own
        SL/TP/time-exit polling exactly as before this existed."""
        return False, "", "not supported by this broker"

    def cancel_oco_exit(self, broker_order_id: str) -> bool:
        """Cancel a protective order placed by place_oco_exit — called right
        before the engine's own software exit fires, so the broker-side order
        doesn't linger and later fire on a position that's already closed.
        Best-effort: returns False rather than raising if it fails (the
        engine logs but does not block the close on this)."""
        return False

    def get_position(self, instrument: Instrument) -> Optional[dict]:
        """Real broker-reported position snapshot for one instrument, or None
        if unavailable/not held. Shape: {"quantity": int (signed: + long,
        - short, 0 flat), "average_price": float, "last_price": float,
        "pnl": float (realized+unrealized for the day)}. Used two ways in
        engine.py: (1) as the source of truth for LIVE unrealized PnL/avg
        price instead of our own bookkeeping, and (2) as a race-condition
        check before a software-triggered square-off — if the broker already
        shows the position flat (its own GTT fired first), the engine skips
        re-closing it. Only Zerodha implements this today."""
        return None

    def get_all_positions(self) -> list[dict]:
        """EVERY non-flat position the broker currently reports, regardless of
        whether this bot's own DB is tracking it — the ground truth for "what
        do I actually hold right now". Shape per entry: {"symbol": str,
        "side": "BUY"|"SELL" (the position's own side, derived from the sign
        of the broker's net quantity), "quantity": int (always positive —
        the SIZE, not the signed net), "average_price": float,
        "last_price": float, "pnl": float}. Empty list if unavailable or
        nothing open. Only Zerodha implements this today."""
        return []

    def square_off(
        self, instrument: Instrument, side: str, quantity: int,
        ref_price: float = 0.0,
    ) -> OrderResult:
        # Exit is just an opposite market order; default impl flips the side.
        # Exits stay MARKET on purpose: a stop or a time-exit must actually get
        # out, and an unfilled limit would leave the position open past its stop.
        opposite = "SELL" if side == "BUY" else "BUY"
        return self.place_market_order(instrument, opposite, quantity, ref_price)


# --------------------------------------------------------------------------- #
#  Simulated broker — paper trading with no credentials. Always works.
# --------------------------------------------------------------------------- #
class SimulatedBroker(BaseBroker):
    name = "Simulated"

    def connect(self) -> bool:
        return True

    def place_market_order(self, instrument, side, quantity, ref_price=0.0):
        return OrderResult(
            ok=True,
            order_id=f"SIM-{uuid.uuid4().hex[:10]}",
            broker=self.name,
            filled_price=ref_price,
            quantity=quantity,
            message="Simulated fill",
        )

    def place_limit_order(self, instrument, side, quantity, limit_price,
                          ref_price=0.0):
        # Optimistic: assumes the limit fills at its price. Real limits inside the
        # spread sometimes don't fill at all, so paper results here are slightly
        # kinder than live would be.
        return OrderResult(
            ok=True,
            order_id=f"SIM-{uuid.uuid4().hex[:10]}",
            broker=self.name,
            filled_price=limit_price or ref_price,
            quantity=quantity,
            message=f"Simulated LIMIT fill @ {limit_price:.2f}",
        )


# --------------------------------------------------------------------------- #
#  Upstox — handles BOTH sandbox (paper) and live via the same SDK.
# --------------------------------------------------------------------------- #
class UpstoxBroker(BaseBroker):
    name = "Upstox"

    def __init__(self, sandbox: bool, access_token: str = ""):
        self.sandbox = sandbox
        # A caller-supplied token (a client's own OAuth token — see
        # engine.TradingEngine.broker_access_token) takes priority over the
        # shared .env credentials in every method below. Empty = today's
        # admin-only behavior, unchanged.
        self.access_token = access_token
        self._client = None
        self._order_api = None

    def connect(self) -> bool:
        token = self.access_token or (
            config.UPSTOX_SANDBOX_TOKEN if self.sandbox
            else config.UPSTOX_LIVE_ACCESS_TOKEN)
        if not token:
            return False
        try:
            import upstox_client  # type: ignore
            cfg = upstox_client.Configuration()
            # Sandbox flag per the plan: Upstox sandbox environment for paper.
            if hasattr(cfg, "sandbox"):
                cfg.sandbox = self.sandbox
            cfg.access_token = token
            self._client = upstox_client.ApiClient(cfg)
            self._order_api = upstox_client.OrderApi(self._client)
            # Validate the token up front so a bad/expired token is caught here
            # rather than at the moment we try to place a real order.
            if not self.sandbox:
                profile = upstox_client.UserApi(self._client).get_profile(
                    api_version="v2")
                print(f"[UpstoxBroker] live token OK — "
                      f"{getattr(profile.data, 'user_name', 'user')}")
            return True
        except Exception as exc:  # SDK missing or auth failed
            print(f"[UpstoxBroker] connect failed: {exc}")
            return False

    def _place(self, instrument, side, quantity, order_type: str,
               price: float, ref_price: float) -> OrderResult:
        """Shared order path. `price` is ignored by the API for MARKET orders and
        is the limit price for LIMIT orders."""
        if self._order_api is None:
            return OrderResult(False, "", self.name, message="Not connected")
        try:
            import upstox_client  # type: ignore
            body = upstox_client.PlaceOrderRequest(
                quantity=quantity,
                product="I" if instrument.segment == Segment.EQUITY else "D",
                validity="DAY",
                price=round(float(price), 2) if order_type == "LIMIT" else 0,
                instrument_token=instrument.instrument_key,
                order_type=order_type,
                transaction_type=side,
                disclosed_quantity=0,
                trigger_price=0,
                is_amo=False,
            )
            resp = self._order_api.place_order(body, api_version="v2")
            oid = getattr(getattr(resp, "data", None), "order_id", "") or "UPX"
            # NOTE: this reports the order as filled at the requested price. Upstox
            # returns an order_id, not a fill — a LIMIT resting inside the spread
            # may fill later, partially, or not at all. Polling get_order_details
            # for the true average price is the correct next step before trusting
            # live PnL to the rupee.
            fill = price if order_type == "LIMIT" else ref_price
            return OrderResult(True, oid, self.name, fill, quantity,
                               f"{'sandbox' if self.sandbox else 'live'} {order_type}")
        except Exception as exc:
            return OrderResult(False, "", self.name, message=str(exc))

    def place_market_order(self, instrument, side, quantity, ref_price=0.0):
        return self._place(instrument, side, quantity, "MARKET", 0.0, ref_price)

    def place_limit_order(self, instrument, side, quantity, limit_price,
                          ref_price=0.0):
        return self._place(instrument, side, quantity, "LIMIT", limit_price,
                           ref_price)

    def required_margin(self, instrument, quantity, side="BUY", ref_price=0.0):
        # ref_price is ignored: Upstox's margin API prices against its own live
        # LTP when price=0 (see fetch_upstox_margin), so a caller-supplied price
        # would only add staleness risk here.
        # Prefer the LIVE token — the margin endpoint is read-only and the live
        # token is the one that stays valid, so this works even in Paper mode.
        token = self.access_token or (
            config.UPSTOX_LIVE_ACCESS_TOKEN
            or (config.UPSTOX_SANDBOX_TOKEN if self.sandbox else ""))
        product = "I" if instrument.segment == Segment.EQUITY else "D"
        return fetch_upstox_margin(token, instrument.instrument_key, quantity,
                                   side, product)

    def available_funds(self) -> Optional[float]:
        token = self.access_token or (
            config.UPSTOX_LIVE_ACCESS_TOKEN if not self.sandbox
            else config.UPSTOX_SANDBOX_TOKEN)
        if not token:
            return None
        try:
            import requests
            resp = requests.get(
                "https://api.upstox.com/v2/user/get-funds-and-margin",
                headers={"Authorization": f"Bearer {token}",
                         "accept": "application/json"},
                params={"segment": "SEC"}, timeout=10)
            if resp.status_code != 200:
                return None
            data = resp.json().get("data", {}) or {}
            # "equity" holds cash segment funds; commodity funds live under
            # "commodity" but the SEC-segment query above targets equity — an
            # MCX-only account should widen this if it starts trading commodities
            # live on Upstox specifically.
            avail = (data.get("equity", {}) or {}).get("available_margin")
            return float(avail) if avail is not None else None
        except Exception:
            return None


# --------------------------------------------------------------------------- #
#  Dhan — live, via dhanhq
# --------------------------------------------------------------------------- #
class DhanBroker(BaseBroker):
    name = "Dhan"

    def __init__(self):
        self._client = None

    def connect(self) -> bool:
        if not config.has_dhan():
            return False
        try:
            from dhanhq import dhanhq  # type: ignore
            client = dhanhq(config.DHAN_CLIENT_ID, config.DHAN_ACCESS_TOKEN)
            # Validate the token up front (same reasoning as UpstoxBroker): a
            # dead/expired DHAN_ACCESS_TOKEN must be caught here, not at the
            # moment we try to place a real order. fund_limits is read-only.
            funds = self._call_fund_limits(client)
            if funds is None:
                print("[DhanBroker] connect failed: token rejected fetching "
                      "fund limits (expired/invalid DHAN_ACCESS_TOKEN?).")
                return False
            self._client = client
            print(f"[DhanBroker] live token OK — available balance "
                  f"₹{funds:,.2f}")
            return True
        except Exception as exc:
            print(f"[DhanBroker] connect failed: {exc}")
            return False

    @staticmethod
    def _call_fund_limits(client) -> Optional[float]:
        """dhanhq has renamed this method across SDK versions
        (get_fund_limits / fund_limits) — try both rather than pin one and
        silently stop working on an upgrade. Returns available balance (₹) or
        None if neither call succeeds / the response shape is unexpected."""
        for name in ("get_fund_limits", "fund_limits"):
            fn = getattr(client, name, None)
            if fn is None:
                continue
            try:
                resp = fn()
                data = resp.get("data", resp) if isinstance(resp, dict) else {}
                for key in ("availabelBalance", "availableBalance",
                           "available_balance", "sodLimit"):
                    val = data.get(key)
                    if isinstance(val, (int, float)):
                        return float(val)
            except Exception:
                continue
        return None

    def place_market_order(self, instrument, side, quantity, ref_price=0.0):
        if self._client is None:
            return OrderResult(False, "", self.name, message="Not connected")
        try:
            exchange = ("MCX" if instrument.segment == Segment.MCX
                        else self._client.NSE)
            resp = self._client.place_order(
                security_id=instrument.instrument_key.split("|")[-1],
                exchange_segment=exchange,
                transaction_type=(self._client.BUY if side == "BUY"
                                  else self._client.SELL),
                quantity=quantity,
                order_type=self._client.MARKET,
                product_type=self._client.INTRA,
                price=0,
            )
            oid = str(resp.get("data", {}).get("orderId", "DHAN"))
            return OrderResult(True, oid, self.name, ref_price, quantity, "live")
        except Exception as exc:
            return OrderResult(False, "", self.name, message=str(exc))

    def required_margin(self, instrument, quantity, side="BUY", ref_price=0.0):
        if self._client is None:
            return None
        try:
            exchange = ("MCX" if instrument.segment == Segment.MCX
                        else self._client.NSE)
            # Dhan's margin_calculator requires an actual price to margin
            # against; fall back to the instrument's seed reference price only
            # if the caller has no live quote yet (better than nothing, worse
            # than a real tick — never trust it further than that).
            price = float(ref_price) if ref_price > 0 else float(
                instrument.reference_price)
            resp = self._client.margin_calculator(
                security_id=instrument.instrument_key.split("|")[-1],
                exchange_segment=exchange,
                transaction_type=(self._client.BUY if side == "BUY"
                                  else self._client.SELL),
                quantity=int(quantity),
                product_type=self._client.INTRA,
                price=price,
            )
            data = resp.get("data", resp) if isinstance(resp, dict) else {}
            val = data.get("totalMargin")
            return float(val) if isinstance(val, (int, float)) and val > 0 else None
        except Exception:
            return None

    def available_funds(self) -> Optional[float]:
        if self._client is None:
            return None
        try:
            return self._call_fund_limits(self._client)
        except Exception:
            return None


# --------------------------------------------------------------------------- #
#  Zerodha (Kite Connect) — live, via kiteconnect.
#
#  EQUITY ONLY. Kite's tradingsymbol for NSE cash equity is the plain symbol
#  (RELIANCE, TCS, ...), which matches Instrument.symbol here — so equity maps
#  cleanly. MCX futures tradingsymbols carry an expiry suffix Kite assigns
#  itself (e.g. CRUDEOIL26JUNFUT) that this codebase's Instrument.symbol
#  (CRUDEOIL) does NOT match, and guessing the expiry format risks sending a
#  malformed real order. So MCX orders are explicitly rejected with a clear
#  message rather than attempted — same principle as the un-implemented Dhan/
#  Kotak limit orders above: an untested path is worse than a clean refusal.
# --------------------------------------------------------------------------- #
class ZerodhaBroker(BaseBroker):
    name = "Zerodha"

    def __init__(self, access_token: str = "", api_key: str = ""):
        self._kite = None
        # A caller-supplied token (a client's own Kite login — see
        # engine.TradingEngine.broker_access_token) takes priority over the
        # shared .env ZERODHA_ACCESS_TOKEN.
        #
        # `api_key` is supplied alongside it when the client owns their own
        # Kite Connect app: a token is only valid for the api_key that minted
        # it, so pairing a client's token with the shared .env key would fail
        # authentication. Empty = the shared .env app, today's behaviour.
        self.access_token = access_token
        self.api_key = api_key

    def connect(self) -> bool:
        if not self.access_token and not config.has_zerodha():
            return False
        api_key = self.api_key or config.ZERODHA_API_KEY
        if not api_key:
            return False
        try:
            from kiteconnect import KiteConnect  # type: ignore
            kite = KiteConnect(api_key=api_key)
            kite.set_access_token(self.access_token or config.ZERODHA_ACCESS_TOKEN)
            # Validate the token up front — same reasoning as Upstox/Dhan: a
            # dead/expired ZERODHA_ACCESS_TOKEN (Kite sessions expire daily) must
            # be caught here, not at the moment we try to place a real order.
            profile = kite.profile()
            self._kite = kite
            print(f"[ZerodhaBroker] live token OK — "
                  f"{profile.get('user_name', 'user')}")
            return True
        except Exception as exc:
            print(f"[ZerodhaBroker] connect failed: {exc}")
            return False

    @staticmethod
    def _reject_mcx(instrument: Instrument) -> Optional[OrderResult]:
        if instrument.segment == Segment.MCX:
            return OrderResult(
                False, "", "Zerodha", message=(
                    f"{instrument.symbol}: Zerodha execution is equity-only in "
                    "this bot — the MCX Kite tradingsymbol (with its exchange-"
                    "assigned expiry suffix) isn't mapped, so the order was "
                    "refused rather than guessed."))
        return None

    def _place(self, instrument, side, quantity, order_type, price,
              ref_price) -> OrderResult:
        rejected = self._reject_mcx(instrument)
        if rejected is not None:
            return rejected
        if self._kite is None:
            return OrderResult(False, "", self.name, message="Not connected")
        try:
            oid = self._kite.place_order(
                variety=self._kite.VARIETY_REGULAR,
                exchange=self._kite.EXCHANGE_NSE,
                tradingsymbol=instrument.symbol,
                transaction_type=(self._kite.TRANSACTION_TYPE_BUY if side == "BUY"
                                  else self._kite.TRANSACTION_TYPE_SELL),
                quantity=int(quantity),
                product=self._kite.PRODUCT_MIS,
                order_type=(self._kite.ORDER_TYPE_LIMIT if order_type == "LIMIT"
                           else self._kite.ORDER_TYPE_MARKET),
                price=round(float(price), 2) if order_type == "LIMIT" else None,
                # Kite now REJECTS API market orders that don't declare market
                # protection ("Market orders without market protection are not
                # allowed via API"). -1 = automatic protection per Kite's own
                # exchange-guideline band (their recommended default — see the
                # place_order docstring). Only meaningful for MARKET orders — a
                # LIMIT already has an explicit price, so this stays unset
                # there and is correctly dropped by the SDK (it strips None
                # params before sending).
                market_protection=-1 if order_type == "MARKET" else None,
            )
            # NOTE: like Upstox, this reports the order as filled at the
            # requested price — Kite returns an order_id, not a fill. A LIMIT
            # resting inside the spread may fill later, partially, or not at
            # all; polling order_history for the true average price is the
            # correct next step before trusting live PnL to the rupee.
            fill = price if order_type == "LIMIT" else ref_price
            return OrderResult(True, str(oid), self.name, fill, quantity,
                               f"live {order_type}")
        except Exception as exc:
            return OrderResult(False, "", self.name, message=str(exc))

    def place_market_order(self, instrument, side, quantity, ref_price=0.0):
        return self._place(instrument, side, quantity, "MARKET", 0.0, ref_price)

    def place_limit_order(self, instrument, side, quantity, limit_price,
                          ref_price=0.0):
        return self._place(instrument, side, quantity, "LIMIT", limit_price,
                           ref_price)

    def required_margin(self, instrument, quantity, side="BUY", ref_price=0.0):
        if self._kite is None or instrument.segment == Segment.MCX:
            return None
        try:
            orders = [{
                "exchange": "NSE",
                "tradingsymbol": instrument.symbol,
                "transaction_type": side,
                "variety": "regular",
                "product": "MIS",
                "order_type": "MARKET",
                "quantity": int(quantity),
                "price": 0,
                "trigger_price": 0,
            }]
            resp = self._kite.order_margins(orders)
            total = sum(float(m.get("total", 0) or 0) for m in resp)
            return total if total > 0 else None
        except Exception:
            return None

    def available_funds(self) -> Optional[float]:
        if self._kite is None:
            return None
        try:
            margins = self._kite.margins(segment="equity")
            avail = (margins.get("available", {}) or {}).get("live_balance")
            return float(avail) if avail is not None else None
        except Exception:
            return None

    def place_oco_exit(self, instrument, side, quantity, stop_loss, target,
                       ref_price) -> tuple[bool, str, str]:
        """A Kite GTT OCO order: two LIMIT legs (SL and target), either of
        which cancels the other once triggered. This is what protects the
        position at ZERODHA'S OWN SERVERS if this bot goes offline — the
        engine's own polling loop is the primary exit path (usually faster);
        this is the safety net for when it isn't running.

        The SL leg's LIMIT price sits a couple of ticks BEYOND its trigger (in
        the exit direction) rather than exactly at it — GTT legs are LIMIT
        orders, and a stop-loss limit priced exactly at the trigger can fail
        to fill in a fast-moving market; a small buffer makes the fill close
        to certain once triggered, at the cost of a few ticks of extra
        slippage versus a true stop-market.
        """
        if instrument.segment == Segment.MCX:
            return False, "", ("MCX not supported for Zerodha GTT protection "
                              "in this bot (see place_market_order's MCX "
                              "refusal for why).")
        if self._kite is None:
            return False, "", "Not connected"
        try:
            exit_side = "SELL" if side == "BUY" else "BUY"
            tick = instrument.tick_size or 0.05
            buf = 2 * tick
            sl_limit = (round(stop_loss - buf, 2) if side == "BUY"
                       else round(stop_loss + buf, 2))
            tp_limit = round(target, 2)
            sl_leg = {"transaction_type": exit_side, "quantity": int(quantity),
                      "order_type": "LIMIT", "product": self._kite.PRODUCT_MIS,
                      "price": sl_limit}
            tp_leg = {"transaction_type": exit_side, "quantity": int(quantity),
                      "order_type": "LIMIT", "product": self._kite.PRODUCT_MIS,
                      "price": tp_limit}
            # Kite's OCO pairs trigger_values[i] with orders[i] BY INDEX — it
            # does not infer which leg is the stop and which is the target, so
            # the two must be sorted together (ascending trigger) regardless
            # of whether this is a long or a short. Sorting anything else here
            # (e.g. assuming stop-then-target) would silently swap the legs
            # for shorts, arming the SL at the target price and vice versa.
            legs = sorted([(stop_loss, sl_leg), (target, tp_leg)],
                         key=lambda pair: pair[0])
            trigger_values = [legs[0][0], legs[1][0]]
            orders = [legs[0][1], legs[1][1]]
            resp = self._kite.place_gtt(
                trigger_type=self._kite.GTT_TYPE_OCO,
                tradingsymbol=instrument.symbol,
                exchange=self._kite.EXCHANGE_NSE,
                trigger_values=trigger_values,
                last_price=round(float(ref_price), 2),
                orders=orders,
            )
            gtt_id = str(resp.get("trigger_id", ""))
            if not gtt_id:
                return False, "", f"place_gtt returned no trigger_id: {resp}"
            return True, gtt_id, ""
        except Exception as exc:
            return False, "", str(exc)

    def cancel_oco_exit(self, broker_order_id: str) -> bool:
        if self._kite is None or not broker_order_id:
            return False
        try:
            self._kite.delete_gtt(int(broker_order_id))
            return True
        except Exception:
            # Already triggered/deleted/expired is the common, harmless case
            # (Kite errors on deleting a GTT that no longer exists) — treat any
            # failure here as "nothing left to cancel", never a reason to
            # block the close.
            return False

    def get_position(self, instrument: Instrument) -> Optional[dict]:
        if self._kite is None or instrument.segment == Segment.MCX:
            return None
        try:
            positions = self._kite.positions()
            # "day" positions cover intraday (MIS) — a position opened and
            # closed today lives here even after it's flat, which is exactly
            # what the race-condition check in engine.py needs to see.
            for p in positions.get("day", []) or []:
                if (p.get("tradingsymbol") == instrument.symbol
                        and p.get("product") == self._kite.PRODUCT_MIS):
                    return {
                        "quantity": int(p.get("quantity", 0) or 0),
                        "average_price": float(p.get("average_price", 0) or 0),
                        "last_price": float(p.get("last_price", 0) or 0),
                        "pnl": float(p.get("pnl", 0) or 0),
                    }
            return None
        except Exception:
            return None

    def get_all_positions(self) -> list[dict]:
        if self._kite is None:
            return []
        try:
            positions = self._kite.positions()
            out: list[dict] = []
            for p in positions.get("day", []) or []:
                qty = int(p.get("quantity", 0) or 0)
                if qty == 0 or p.get("product") != self._kite.PRODUCT_MIS:
                    continue
                out.append({
                    "symbol": p.get("tradingsymbol", ""),
                    "side": "BUY" if qty > 0 else "SELL",
                    "quantity": abs(qty),
                    "average_price": float(p.get("average_price", 0) or 0),
                    "last_price": float(p.get("last_price", 0) or 0),
                    "pnl": float(p.get("pnl", 0) or 0),
                })
            return out
        except Exception:
            return []


# --------------------------------------------------------------------------- #
#  Kotak Neo — live, via neo-api-client
# --------------------------------------------------------------------------- #
class KotakNeoBroker(BaseBroker):
    name = "Kotak Neo"

    def __init__(self):
        self._client = None

    def connect(self) -> bool:
        if not config.has_kotak():
            return False
        try:
            from neo_api_client import NeoAPI  # type: ignore
            self._client = NeoAPI(
                access_token=config.KOTAK_NEO_ACCESS_TOKEN,
                environment="prod",
            )
            return True
        except Exception as exc:
            print(f"[KotakNeoBroker] connect failed: {exc}")
            return False

    def place_market_order(self, instrument, side, quantity, ref_price=0.0):
        if self._client is None:
            return OrderResult(False, "", self.name, message="Not connected")
        try:
            exchange = "mcx" if instrument.segment == Segment.MCX else "nse_cm"
            resp = self._client.place_order(
                exchange_segment=exchange,
                product="MIS",
                price="0",
                order_type="MKT",
                quantity=str(quantity),
                validity="DAY",
                trading_symbol=instrument.symbol,
                transaction_type="B" if side == "BUY" else "S",
            )
            oid = str(resp.get("nOrdNo", "KOTAK"))
            return OrderResult(True, oid, self.name, ref_price, quantity, "live")
        except Exception as exc:
            return OrderResult(False, "", self.name, message=str(exc))


# --------------------------------------------------------------------------- #
#  Factory — pick the right broker for the chosen environment, and fall back to
#  the Simulator whenever credentials are missing so the bot always runs.
# --------------------------------------------------------------------------- #
def make_broker(environment: Environment, broker_choice: Broker,
                access_token: str = "", api_key: str = "") -> BaseBroker:
    """`access_token` is additive (Phase 2 multi-tenancy — see
    frontend_migration_plan.md §3): a client's own OAuth token, which takes
    priority over the shared .env credentials for Upstox/Zerodha. Empty
    string (every pre-Phase-2 call site) reproduces the original
    admin-only, .env-driven behavior exactly."""
    if environment == Environment.PAPER:
        # Paper execution is simulated (guaranteed fills, zero risk of a real
        # order, no dependency on a possibly-expired sandbox token). Real MARKET
        # DATA still comes from the live feed, so this is true paper trading:
        # real prices in, simulated fills out — logged to `paper_trades`.
        # (To route paper orders to the actual Upstox Sandbox instead, set a
        # valid UPSTOX_SANDBOX_TOKEN and swap in UpstoxBroker(sandbox=True).)
        sim = SimulatedBroker()
        sim.connect()
        return sim

    # LIVE
    mapping = {
        Broker.UPSTOX: lambda: UpstoxBroker(sandbox=False, access_token=access_token),
        Broker.DHAN: DhanBroker,
        Broker.ZERODHA: lambda: ZerodhaBroker(access_token=access_token,
                                              api_key=api_key),
        Broker.KOTAK: KotakNeoBroker,
    }
    factory = mapping.get(broker_choice)
    if factory:
        b = factory()
        if b.connect():
            return b
        print(f"[make_broker] {broker_choice} live connect failed; "
              f"falling back to Simulated to avoid unintended orders.")
    sim = SimulatedBroker()
    sim.connect()
    return sim


# --------------------------------------------------------------------------- #
#  Broker status lights — a lightweight, READ-ONLY authentication check per
#  broker, for the sidebar's 🟢/🔴 indicators. Deliberately separate from
#  make_broker(): this never returns a broker object or gets called by the
#  engine — it exists purely so the UI can show "is this credential actually
#  live right now" WITHOUT starting the bot. Each check is the cheapest
#  read-only call that proves a real session (profile / fund-limits), never an
#  order-placement path.
#
#  `ok` is one of:
#    True  — credentials present AND the broker accepted them just now.
#    False — credentials present but the broker rejected them (expired/invalid
#            token, or — for Kotak — the login flow isn't implemented yet).
#    None  — no credentials configured at all; neutral, not a failure.
# --------------------------------------------------------------------------- #
def check_broker_status() -> dict[str, dict]:
    import kite_auth
    import upstox_auth

    status: dict[str, dict] = {}

    # Upstox — profile endpoint (works for the live token; sandbox has no
    # equivalent cheap check, so sandbox-only setups show "no live token").
    if config.has_upstox_live():
        r = upstox_auth.check_token(config.UPSTOX_LIVE_ACCESS_TOKEN)
        status["Upstox"] = {
            "ok": bool(r["ok"]),
            "detail": (f"live — {r.get('user_name', 'user')}" if r["ok"]
                      else r.get("error", "invalid token")),
        }
    else:
        status["Upstox"] = {"ok": None, "detail": "No live token set."}

    # Dhan — fund_limits (already the same call DhanBroker.connect() uses to
    # validate). Constructs its own short-lived client; no state is kept.
    if config.has_dhan():
        try:
            b = DhanBroker()
            if b.connect():
                funds = b.available_funds()
                status["Dhan"] = {
                    "ok": True,
                    "detail": (f"live — ₹{funds:,.0f} available"
                              if funds is not None else "live"),
                }
            else:
                status["Dhan"] = {
                    "ok": False,
                    "detail": "token rejected (expired/invalid, or dhanhq "
                             "not installed).",
                }
        except Exception as exc:
            status["Dhan"] = {"ok": False, "detail": str(exc)}
    else:
        status["Dhan"] = {"ok": None, "detail": "Client ID/token not set."}

    # Zerodha — profile endpoint via plain requests (kite_auth.check_token),
    # so this works even before the kiteconnect package is installed —
    # exactly the gap that caused the earlier silent Simulated fallback.
    if config.has_zerodha():
        r = kite_auth.check_token(config.ZERODHA_ACCESS_TOKEN,
                                  config.ZERODHA_API_KEY)
        status["Zerodha"] = {
            "ok": bool(r["ok"]),
            "detail": (f"live — {r.get('user_name', 'user')}" if r["ok"]
                      else r.get("error", "invalid token")),
        }
    else:
        status["Zerodha"] = {"ok": None, "detail": "API key/token not set."}

    # Kotak Neo — connect() here is a bare access-token wrapper, not a real
    # login/session flow (see README §5), so it cannot be trusted to prove
    # anything either way. Reported as a distinct, explicit "not ready" rather
    # than attempting a call that would misrepresent readiness.
    if config.has_kotak():
        status["Kotak Neo"] = {
            "ok": False,
            "detail": "Not wired up yet — connect() needs a real login/OTP "
                     "session, not just an access token. See README §5.",
        }
    else:
        status["Kotak Neo"] = {"ok": None, "detail": "Not configured."}

    return status
