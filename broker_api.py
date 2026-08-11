"""
broker_api.py
Execution layer. A thin, uniform wrapper over each broker so the engine can
place/track orders without knowing which broker is behind it.

    BaseBroker (interface)
      ├── SimulatedBroker   -> paper fills, no network, always available
      ├── UpstoxBroker      -> sandbox (paper) OR live, via upstox-python-sdk
      ├── DhanBroker        -> live, via dhanhq
      ├── ZerodhaBroker     -> live, via kiteconnect (equity + MCX futures)
      └── KotakNeoBroker    -> live, via neo-api-client

Every broker returns the same OrderResult shape, so strategy/engine code is
broker-agnostic (Immutable Rule #3). Real SDK calls are wrapped in try/except
and degrade to a clear error rather than crashing the bot.

LIMIT ORDERS: implemented for Simulated, Upstox and Zerodha. Dhan and
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
import kite_symbols
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


# --------------------------------------------------------------------------- #
#  Protective-order handles.
#
#  A broker-side SL/TP is not one thing: a Kite GTT OCO and a plain resting
#  SL-M order are cancelled and amended through completely different APIs. The
#  engine stores exactly ONE string per trade (`broker_gtt_id`), so the KIND is
#  namespaced into that string rather than added as a new schema field — which
#  keeps the stored document exactly the Section-5 shape it already was.
#
#      "gtt:1234"    Kite GTT OCO   -> delete_gtt / modify_gtt
#      "order:5678"  resting SL-M   -> cancel_order / modify_order
#
#  A BARE id with no prefix is a trade opened before this namespacing existed;
#  those could only ever have been Kite GTTs, so that is what they decode to.
# --------------------------------------------------------------------------- #
PROT_GTT = "gtt"
PROT_ORDER = "order"


def _prot_encode(kind: str, raw_id: str) -> str:
    return f"{kind}:{raw_id}"


def _prot_decode(protection_id: str) -> tuple[str, str]:
    """(kind, raw_id) from a stored handle. Unprefixed legacy ids decode as
    Kite GTTs — see the note above."""
    pid = (protection_id or "").strip()
    if not pid:
        return "", ""
    kind, sep, raw = pid.partition(":")
    if not sep:
        return PROT_GTT, pid
    return kind, raw


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
        """Place the SL (and, where the broker supports it, the TP) as a REAL
        resting order at the broker, so the position is protected even if this
        bot goes offline — not just watched by our own polling loop. `side` is
        the position's own side (BUY/SELL); the protective order is on the
        OPPOSITE side.

        Returns (ok, protection_id, message), where `protection_id` is a
        namespaced handle (see PROT_* below) that cancel/modify parse to know
        which broker API the handle belongs to.

        Zerodha and Upstox implement this; every other broker inherits this
        no-op, which is silent and harmless — the position is still fully
        protected by the engine's own SL/TP/time-exit polling exactly as
        before this existed."""
        return False, "", "not supported by this broker"

    def modify_oco_exit(
        self, instrument: Instrument, side: str, quantity: int,
        stop_loss: float, target: float, ref_price: float,
        protection_id: str,
    ) -> tuple[bool, str, str]:
        """Re-arm an existing protective order at a NEW stop/target — what a
        trailing stop needs so the broker-side order follows the software one
        instead of staying pinned at the original stop.

        Returns (ok, protection_id, message). The id is returned because a
        broker that cannot amend in place is free to cancel-and-replace, which
        yields a NEW id the engine must persist. On failure the caller keeps
        the old id and the OLD broker-side stop stays armed — a stale-but-live
        stop is strictly safer than no stop, so a failure here never cancels
        what is already resting."""
        return False, protection_id, "not supported by this broker"

    def cancel_oco_exit(self, protection_id: str) -> bool:
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

    def get_protective_orders(self) -> list[dict]:
        """EVERY protective SL/TP the broker is currently HOLDING — read back
        from the broker itself, not from what this bot believes it armed. The
        companion to get_all_positions: that answers "what do I hold", this
        answers "what is actually guarding it".

        Deliberately a fresh read rather than a mirror of `broker_gtt_id`,
        because the two can disagree in exactly the cases that matter — a GTT
        cancelled by hand in Kite, one that fired while this bot was offline,
        or a stray leg left resting after the "could not be cancelled" warning
        in engine._close_position. Reading the broker is the only way to see
        those.

        Shape per entry:
          {"symbol": str,     # the BROKER's own tradingsymbol, so this joins
                              # to get_all_positions()["symbol"] directly
           "side": str,       # the PROTECTIVE order's side (the exit side —
                              # opposite the position it guards)
           "quantity": int,
           "kind": "GTT"|"SLM",
           "stop": float,     # 0.0 when this mechanism carries no stop leg
           "target": float,   # 0.0 when it carries no target leg (SL-M never
                              # does — see place_oco_exit)
           "status": str,     # broker's own word ("active", "TRIGGER PENDING")
           "id": str}         # namespaced handle, matches trade.broker_gtt_id

        Empty list if unavailable. Only Zerodha implements this today; every
        other broker inherits this no-op, and the UI simply shows no
        broker-side protection for them."""
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

    def _product_code(self, instrument) -> str:
        """"I" (intraday/MIS) for equity, "D" (delivery/NRML) for commodity —
        the same split the margin call uses, kept in one place so an order and
        the margin checked for it can never disagree."""
        return "I" if instrument.segment == Segment.EQUITY else "D"

    def _place(self, instrument, side, quantity, order_type: str,
               price: float, ref_price: float,
               trigger_price: float = 0.0) -> OrderResult:
        """Shared order path. `price` is ignored by the API for MARKET orders and
        is the limit price for LIMIT orders. `trigger_price` is the stop trigger
        for SL/SL-M orders and is ignored otherwise."""
        if self._order_api is None:
            return OrderResult(False, "", self.name, message="Not connected")
        try:
            import upstox_client  # type: ignore
            body = upstox_client.PlaceOrderRequest(
                quantity=quantity,
                product=self._product_code(instrument),
                validity="DAY",
                price=round(float(price), 2) if order_type in ("LIMIT", "SL") else 0,
                instrument_token=instrument.instrument_key,
                order_type=order_type,
                transaction_type=side,
                disclosed_quantity=0,
                trigger_price=round(float(trigger_price), 2),
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
        return fetch_upstox_margin(token, instrument.instrument_key, quantity,
                                   side, self._product_code(instrument))

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

    # -- broker-side protection ---------------------------------------------- #
    def place_oco_exit(self, instrument, side, quantity, stop_loss, target,
                       ref_price) -> tuple[bool, str, str]:
        """A real resting SL-M on the exit side, so the stop lives at UPSTOX
        rather than only in this bot's polling loop.

        The STOP only — the target stays with the engine, for both segments.
        Upstox does have a multi-leg GTT that could hold both, but resting a
        TP order next to the SL creates the orphan-leg hazard: if this bot is
        down when one leg fills, the other is still live at the broker, and an
        exit order with no position behind it OPENS a fresh naked one. A lone
        SL-M cannot do that — the worst case is a missed take-profit, which
        costs profit rather than principal.
        """
        if self._order_api is None:
            return False, "", "Not connected"
        exit_side = "SELL" if side == "BUY" else "BUY"
        res = self._place(instrument, exit_side, int(quantity), "SL-M",
                          0.0, ref_price, trigger_price=stop_loss)
        if not res.ok or not res.order_id:
            return False, "", res.message or "SL-M rejected"
        return True, _prot_encode(PROT_ORDER, res.order_id), ""

    def modify_oco_exit(self, instrument, side, quantity, stop_loss, target,
                        ref_price, protection_id) -> tuple[bool, str, str]:
        """Amend the resting SL-M's trigger in place — the handle is unchanged,
        so the position is never momentarily unprotected."""
        if self._order_api is None:
            return False, protection_id, "Not connected"
        _kind, raw_id = _prot_decode(protection_id)
        if not raw_id:
            return False, protection_id, "no protective order to modify"
        try:
            import upstox_client  # type: ignore
            body = upstox_client.ModifyOrderRequest(
                order_id=raw_id,
                order_type="SL-M",
                quantity=int(quantity),
                validity="DAY",
                price=0,
                trigger_price=round(float(stop_loss), 2),
                disclosed_quantity=0,
            )
            self._order_api.modify_order(body, api_version="v2")
            return True, protection_id, ""
        except Exception as exc:
            return False, protection_id, str(exc)

    def cancel_oco_exit(self, protection_id: str) -> bool:
        if self._order_api is None:
            return False
        _kind, raw_id = _prot_decode(protection_id)
        if not raw_id:
            return False
        try:
            self._order_api.cancel_order(raw_id, api_version="v2")
            return True
        except Exception:
            # Already triggered/cancelled is the common, harmless case. The
            # engine verifies against the broker's real position before it
            # sends its own square-off, so this never blocks a close.
            return False


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
#  EQUITY + MCX FUTURES.
#
#  Equity maps for free: Kite's tradingsymbol for NSE cash is the plain symbol
#  (RELIANCE), which is exactly Instrument.symbol. MCX futures do NOT — Kite
#  names them with an exchange-assigned expiry suffix (CRUDEOILM26AUGFUT).
#
#  Commodity orders used to be refused outright for that reason, because
#  guessing the suffix risks a malformed REAL order. They are no longer
#  guessed OR refused: kite_symbols.py reads Kite's own published instrument
#  list and resolves the exact contract by (root, expiry). A symbol that does
#  not resolve is still refused with an actionable message — the original
#  principle holds, it just almost never fires now.
#
#  Two consequences worth knowing:
#    * Commodity orders use PRODUCT_NRML, matching the full SPAN+exposure
#      margin the engine sizes against. Equity keeps PRODUCT_MIS.
#    * Kite reports lot_size=1 on every MCX future, i.e. it counts quantity in
#      LOTS — the same convention Upstox and the engine already use.
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
    def _kite_symbol(instrument: Instrument) -> Optional[str]:
        """Kite's tradingsymbol for this instrument, or None if unresolvable.

        Equity is its own symbol. MCX futures carry Kite's exchange-assigned
        expiry suffix (CRUDEOILM26AUGFUT) and are looked up EXACTLY, by
        (root, expiry), against Kite's own published instrument list — never
        guessed from the format. See kite_symbols.py."""
        if instrument.segment != Segment.MCX:
            return instrument.symbol
        return kite_symbols.resolve(instrument)

    def _exchange(self, instrument: Instrument):
        return ("MCX" if instrument.segment == Segment.MCX
                else self._kite.EXCHANGE_NSE)

    def _product(self, instrument: Instrument):
        """NRML for commodities, MIS for equity.

        This is not cosmetic: the engine sizes commodity trades against the
        FULL SPAN+exposure margin (Upstox product "D"), which is the NRML
        requirement. Sending MIS would have the broker block a different
        (smaller, intraday) margin than the one the position was sized
        against, so the bot's funding maths and the broker's would disagree."""
        return (self._kite.PRODUCT_NRML if instrument.segment == Segment.MCX
                else self._kite.PRODUCT_MIS)

    def _unresolved(self, instrument: Instrument) -> Optional[OrderResult]:
        """A refusal OrderResult when the tradingsymbol can't be resolved.
        Refusing beats guessing: a wrong suffix is a real order on the wrong
        contract."""
        if self._kite_symbol(instrument) is None:
            return OrderResult(False, "", "Zerodha",
                               message=kite_symbols.describe_failure(instrument))
        return None

    def _place(self, instrument, side, quantity, order_type, price,
              ref_price) -> OrderResult:
        if self._kite is None:
            return OrderResult(False, "", self.name, message="Not connected")
        rejected = self._unresolved(instrument)
        if rejected is not None:
            return rejected
        try:
            oid = self._kite.place_order(
                variety=self._kite.VARIETY_REGULAR,
                exchange=self._exchange(instrument),
                tradingsymbol=self._kite_symbol(instrument),
                transaction_type=(self._kite.TRANSACTION_TYPE_BUY if side == "BUY"
                                  else self._kite.TRANSACTION_TYPE_SELL),
                # Quantity is a LOT COUNT for MCX. Kite reports lot_size=1 on
                # every MCX future, i.e. it counts contracts — the same
                # convention Upstox and the engine use, so no conversion.
                quantity=int(quantity),
                product=self._product(instrument),
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
        if self._kite is None:
            return None
        tradingsymbol = self._kite_symbol(instrument)
        if tradingsymbol is None:
            return None
        is_mcx = instrument.segment == Segment.MCX
        try:
            orders = [{
                "exchange": "MCX" if is_mcx else "NSE",
                "tradingsymbol": tradingsymbol,
                "transaction_type": side,
                "variety": "regular",
                # Must match what _place actually sends, or the margin quoted
                # is for a product the order won't use.
                "product": "NRML" if is_mcx else "MIS",
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
        """Arm this position's protection at ZERODHA'S OWN SERVERS, so it
        survives this bot going offline. The engine's polling loop remains the
        PRIMARY exit path (usually faster, and it alone honours the time-exit);
        this is the safety net for when it isn't running.

        Which mechanism gets used is forced by the product, not by preference:

        * MCX / NRML -> GTT OCO. Both legs (SL and target) rest at Zerodha and
          either one cancels the other.
        * EQUITY / MIS -> a plain resting SL-M. Kite's GTT accepts CNC and
          NRML only, so an MIS GTT is rejected outright; and Zerodha withdrew
          bracket orders in 2020, so there is no OCO for intraday equity at
          all. An SL-M is therefore the only real exchange-resting protection
          available here, and it covers the STOP only — the target stays with
          the engine's polling. That asymmetry is deliberate: if this bot dies
          you are still stopped out by the exchange, and the worst case is a
          missed take-profit rather than an unbounded loss. Resting a second
          TP order alongside it would create the opposite hazard — one leg
          filling while the bot is down leaves the other live, and a resting
          exit order with no position behind it OPENS a fresh naked one.
        """
        if self._kite is None:
            return False, "", "Not connected"
        tradingsymbol = self._kite_symbol(instrument)
        if tradingsymbol is None:
            return False, "", kite_symbols.describe_failure(instrument)
        if self._product(instrument) == self._kite.PRODUCT_MIS:
            return self._place_stop_order(instrument, side, quantity, stop_loss)
        return self._place_gtt_oco(instrument, side, quantity, stop_loss,
                                   target, ref_price)

    def _place_gtt_oco(self, instrument, side, quantity, stop_loss, target,
                       ref_price) -> tuple[bool, str, str]:
        """The GTT OCO path (NRML only — see place_oco_exit).

        The SL leg's LIMIT price sits a couple of ticks BEYOND its trigger (in
        the exit direction) rather than exactly at it — GTT legs are LIMIT
        orders, and a stop-loss limit priced exactly at the trigger can fail
        to fill in a fast-moving market; a small buffer makes the fill close
        to certain once triggered, at the cost of a few ticks of extra
        slippage versus a true stop-market.
        """
        try:
            trigger_values, orders = self._gtt_legs(
                instrument, side, quantity, stop_loss, target)
            resp = self._kite.place_gtt(
                trigger_type=self._kite.GTT_TYPE_OCO,
                tradingsymbol=self._kite_symbol(instrument),
                exchange=self._exchange(instrument),
                trigger_values=trigger_values,
                last_price=round(float(ref_price), 2),
                orders=orders,
            )
            gtt_id = str(resp.get("trigger_id", ""))
            if not gtt_id:
                return False, "", f"place_gtt returned no trigger_id: {resp}"
            return True, _prot_encode(PROT_GTT, gtt_id), ""
        except Exception as exc:
            return False, "", str(exc)

    def _gtt_legs(self, instrument, side, quantity, stop_loss, target):
        """(trigger_values, orders) for a GTT OCO. Shared by place and modify
        so a trailed GTT is rebuilt exactly the way it was first armed."""
        exit_side = "SELL" if side == "BUY" else "BUY"
        tick = instrument.tick_size or 0.05
        buf = 2 * tick
        sl_limit = (round(stop_loss - buf, 2) if side == "BUY"
                   else round(stop_loss + buf, 2))
        product = self._product(instrument)
        sl_leg = {"transaction_type": exit_side, "quantity": int(quantity),
                  "order_type": "LIMIT", "product": product,
                  "price": sl_limit}
        tp_leg = {"transaction_type": exit_side, "quantity": int(quantity),
                  "order_type": "LIMIT", "product": product,
                  "price": round(target, 2)}
        # Kite's OCO pairs trigger_values[i] with orders[i] BY INDEX — it
        # does not infer which leg is the stop and which is the target, so
        # the two must be sorted together (ascending trigger) regardless
        # of whether this is a long or a short. Sorting anything else here
        # (e.g. assuming stop-then-target) would silently swap the legs
        # for shorts, arming the SL at the target price and vice versa.
        legs = sorted([(stop_loss, sl_leg), (target, tp_leg)],
                     key=lambda pair: pair[0])
        return [legs[0][0], legs[1][0]], [legs[0][1], legs[1][1]]

    def _place_stop_order(self, instrument, side, quantity,
                          stop_loss) -> tuple[bool, str, str]:
        """A real resting SL-M on the exit side (MIS equity — see
        place_oco_exit). SL-M rather than SL: once the trigger is hit this
        must actually get out, and a stop-LIMIT can be jumped in a fast market
        and leave the position running past its stop.

        (SL-M is rejected by Zerodha on F&O, but this path is only ever taken
        for equity — MCX goes through the GTT above.)"""
        try:
            order_id = self._kite.place_order(
                variety=self._kite.VARIETY_REGULAR,
                exchange=self._exchange(instrument),
                tradingsymbol=self._kite_symbol(instrument),
                transaction_type=("SELL" if side == "BUY" else "BUY"),
                quantity=int(quantity),
                product=self._product(instrument),
                order_type=self._kite.ORDER_TYPE_SLM,
                trigger_price=round(float(stop_loss), 2),
                validity=self._kite.VALIDITY_DAY,
            )
            if not order_id:
                return False, "", "place_order returned no order_id"
            return True, _prot_encode(PROT_ORDER, str(order_id)), ""
        except Exception as exc:
            return False, "", str(exc)

    def modify_oco_exit(self, instrument, side, quantity, stop_loss, target,
                        ref_price, protection_id) -> tuple[bool, str, str]:
        """Re-arm at a new stop/target. Both mechanisms amend IN PLACE, so the
        handle is unchanged and there is never a window where the position
        sits unprotected (which a cancel-then-replace would open)."""
        if self._kite is None:
            return False, protection_id, "Not connected"
        kind, raw_id = _prot_decode(protection_id)
        if not raw_id:
            return False, protection_id, "no protective order to modify"
        try:
            if kind == PROT_ORDER:
                self._kite.modify_order(
                    variety=self._kite.VARIETY_REGULAR,
                    order_id=raw_id,
                    trigger_price=round(float(stop_loss), 2),
                )
            else:
                trigger_values, orders = self._gtt_legs(
                    instrument, side, quantity, stop_loss, target)
                self._kite.modify_gtt(
                    trigger_id=int(raw_id),
                    trigger_type=self._kite.GTT_TYPE_OCO,
                    tradingsymbol=self._kite_symbol(instrument),
                    exchange=self._exchange(instrument),
                    trigger_values=trigger_values,
                    last_price=round(float(ref_price), 2),
                    orders=orders,
                )
            return True, protection_id, ""
        except Exception as exc:
            return False, protection_id, str(exc)

    def cancel_oco_exit(self, protection_id: str) -> bool:
        if self._kite is None:
            return False
        kind, raw_id = _prot_decode(protection_id)
        if not raw_id:
            return False
        try:
            if kind == PROT_ORDER:
                self._kite.cancel_order(variety=self._kite.VARIETY_REGULAR,
                                        order_id=raw_id)
            else:
                self._kite.delete_gtt(int(raw_id))
            return True
        except Exception:
            # Already triggered/cancelled/expired is the common, harmless case
            # (Kite errors on removing something that no longer exists) — treat
            # any failure here as "nothing left to cancel", never a reason to
            # block the close. The engine verifies the outcome against the
            # broker's real position before it sends its own square-off.
            return False

    def get_position(self, instrument: Instrument) -> Optional[dict]:
        if self._kite is None:
            return None
        # MCX positions come back under Kite's expiry-suffixed tradingsymbol,
        # so match on the resolved name, not ours.
        want_symbol = self._kite_symbol(instrument)
        if want_symbol is None:
            return None
        want_product = self._product(instrument)
        try:
            positions = self._kite.positions()
            # "day" positions cover intraday — a position opened and closed
            # today lives here even after it's flat, which is exactly what the
            # race-condition check in engine.py needs to see.
            for p in positions.get("day", []) or []:
                if (p.get("tradingsymbol") == want_symbol
                        and p.get("product") == want_product):
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
            # Both products: equity trades MIS, commodity trades NRML, and
            # this is the broker-truth panel — filtering to one would hide
            # half the real book.
            wanted = {self._kite.PRODUCT_MIS, self._kite.PRODUCT_NRML}
            for p in positions.get("day", []) or []:
                qty = int(p.get("quantity", 0) or 0)
                if qty == 0 or p.get("product") not in wanted:
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

    @staticmethod
    def _classify_leg(trigger: float, ref: float, exit_side: str) -> str:
        """"stop" or "target" for one GTT leg.

        Decided by which SIDE of the reference price the trigger sits on, not
        by leg order — that way a GTT created by hand in Kite (whose legs we
        never sorted) reads correctly too. For a long the exit is a SELL, so a
        trigger BELOW the reference is the stop and one above is the target;
        for a short both tests invert.
        """
        below = trigger < ref
        if exit_side == "SELL":               # guarding a LONG
            return "stop" if below else "target"
        return "target" if below else "stop"  # guarding a SHORT

    def _gtt_to_row(self, g: dict) -> Optional[dict]:
        """One get_gtts() entry -> one get_protective_orders() row, or None if
        it isn't a live protective trigger we can make sense of."""
        cond = g.get("condition") or {}
        legs = g.get("orders") or []
        triggers = [float(t) for t in (cond.get("trigger_values") or [])]
        if not legs or not triggers:
            return None
        exit_side = str(legs[0].get("transaction_type", "") or "").upper()
        # The price the trigger was set against — the only neutral reference
        # for telling a stop from a target (see _classify_leg).
        ref = float(cond.get("last_price", 0) or 0)
        stop = target = 0.0
        if ref > 0:
            for trig in triggers:
                if self._classify_leg(trig, ref, exit_side) == "stop":
                    stop = trig
                else:
                    target = trig
        # No usable reference, or both legs classified the same way (possible
        # if the market gapped clean through one side before we read it): fall
        # back to the ordering _gtt_legs guarantees — ascending trigger, so the
        # lower one is the stop for a long and the target for a short.
        if len(triggers) == 2 and (stop == 0.0 or target == 0.0):
            lo, hi = min(triggers), max(triggers)
            stop, target = (lo, hi) if exit_side == "SELL" else (hi, lo)
        return {
            "symbol": cond.get("tradingsymbol", ""),
            "side": exit_side,
            "quantity": int(legs[0].get("quantity", 0) or 0),
            "kind": "GTT",
            "stop": stop,
            "target": target,
            "status": str(g.get("status", "") or ""),
            "id": _prot_encode(PROT_GTT, str(g.get("id", ""))),
        }

    def get_protective_orders(self) -> list[dict]:
        """Both mechanisms in one list — GTT OCOs (commodity/NRML) and resting
        SL-M orders (equity/MIS). See place_oco_exit for why the two differ.

        Each source is fetched in its own try/except so one failing API (or an
        older kiteconnect without get_gtts) still returns the other rather
        than an empty panel that would wrongly read as "nothing is guarded".
        """
        if self._kite is None:
            return []
        out: list[dict] = []

        try:
            for g in self._kite.get_gtts() or []:
                # Only triggers still capable of firing. A "triggered" or
                # "cancelled" GTT is history and would be alarming noise in a
                # panel whose whole job is "what is guarding me RIGHT NOW".
                if str(g.get("status", "")).lower() != "active":
                    continue
                row = self._gtt_to_row(g)
                if row is not None and row["symbol"]:
                    out.append(row)
        except Exception:
            pass

        try:
            # A resting SL/SL-M sits at "TRIGGER PENDING" until its trigger is
            # hit; anything else is filled, cancelled or rejected and is no
            # longer protection.
            for o in self._kite.orders() or []:
                if str(o.get("status", "")).upper() != "TRIGGER PENDING":
                    continue
                if str(o.get("order_type", "")).upper() not in ("SL", "SL-M"):
                    continue
                out.append({
                    "symbol": o.get("tradingsymbol", ""),
                    "side": str(o.get("transaction_type", "") or "").upper(),
                    "quantity": int(o.get("quantity", 0) or 0),
                    "kind": "SLM",
                    "stop": float(o.get("trigger_price", 0) or 0),
                    # An SL-M carries no target leg at all — the engine's
                    # polling loop owns the take-profit for equity.
                    "target": 0.0,
                    "status": str(o.get("status", "") or ""),
                    "id": _prot_encode(PROT_ORDER, str(o.get("order_id", ""))),
                })
        except Exception:
            pass

        return out


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
