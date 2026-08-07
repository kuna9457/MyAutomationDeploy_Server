"""
api/routers/broker.py
Broker status lights + the Upstox/Zerodha daily OAuth token refresh flow —
the same functions app.py's sidebar panels call
(upstox_auth.py / kite_auth.py / broker_api.check_broker_status), reached
over HTTP. The OAuth "redirect lands back on us" step becomes the
frontend's job (it owns the redirect URI now); this router does the
code/request_token -> access_token exchange.

WHOSE credentials are used, and where the resulting token goes, depends on
who is calling:

  • admin  — the shared .env app credentials, and the token is written back
             to .env exactly as before. Unchanged behaviour.
  • client — THEIR OWN broker app's api_key/api_secret, which they pasted
             into the UI and which live encrypted on their user record
             (credentials_vault.py). The resulting token is stored encrypted
             on that same record. Nothing about a client ever touches .env,
             so onboarding a new client needs no server config at all.

The one broker thing .env still owns is the admin Upstox token used for
market data (engine.py) — market data is identical for everyone, so there is
no reason to make each client supply it.
"""
from __future__ import annotations

import os

import broker_api
import config
import credentials_vault
import kite_auth
import upstox_auth
import user_manager
from api.auth import CurrentUser, get_current_user, require_admin
from api.schemas import (BrokerCredentialsRequest, UpstoxExchangeRequest,
                         ZerodhaExchangeRequest)
from config import Broker, Environment
from fastapi import APIRouter, Depends, HTTPException

router = APIRouter(prefix="/broker", tags=["broker"])

_CLIENT_CREDENTIAL_BROKERS = ("Upstox", "Zerodha")


def _client_redirect_uri() -> str:
    """The single Redirect URI every client registers in their own broker app.
    Not a secret — it's displayed in the UI for them to copy."""
    return (os.getenv("CLIENT_REDIRECT_URI", "") or "").strip() \
        or upstox_auth.get_credentials()[2]


def _require_broker(broker: str) -> str:
    if broker not in _CLIENT_CREDENTIAL_BROKERS:
        raise HTTPException(
            400, f"Only {'/'.join(_CLIENT_CREDENTIAL_BROKERS)} support "
                 "self-service credentials.")
    return broker


def _resolve_credentials(user: CurrentUser, broker: str) -> tuple[str, str, str]:
    """(api_key, api_secret, redirect_uri) for whoever is calling.

    A client's come from their own encrypted record; the admin's from .env.
    This is the single place that decision is made — every OAuth endpoint
    below goes through it, so the two paths can't drift apart."""
    if user.role == "client":
        api_key, api_secret = user_manager.get_broker_credentials(user.user_id, broker)
        if not (api_key and api_secret):
            raise HTTPException(
                400, f"Save your {broker} API key and secret first — see the "
                     "Broker Credentials panel.")
        return api_key, api_secret, _client_redirect_uri()

    if broker == "Upstox":
        api_key, api_secret, redirect_uri = upstox_auth.get_credentials()
    else:
        api_key, api_secret = kite_auth.get_credentials()
        redirect_uri = ""
    if not (api_key and api_secret):
        raise HTTPException(
            400, f"{broker} API key/secret missing in .env.")
    return api_key, api_secret, redirect_uri


# -- a client's own broker app credentials ------------------------------------- #
@router.get("/egress-ip")
def egress_ip(_: CurrentUser = Depends(get_current_user)):
    """The public IP this server presents on outbound calls — i.e. exactly what
    a broker's static-IP allowlist compares against.

    Exists because that failure is otherwise invisible from the outside: the
    broker rejects with UDAPI1154 naming an origin IP, and there is no way to
    confirm what the server would send without asking it. If `ip` here is an
    IPv6 address while the allowlist holds IPv4, that mismatch IS the failure —
    see net_config.FORCE_IPV4."""
    import net_config
    ip = net_config.egress_ip()
    return {
        "ip": ip,
        "family": ("IPv6" if ":" in ip else "IPv4" if ip else "unknown"),
        "force_ipv4": net_config.status(),
        "note": ("Add this exact address to your broker app's static-IP "
                 "allowlist, or clear the restriction there."),
    }


@router.get("/onboarding-info")
def onboarding_info(_: CurrentUser = Depends(get_current_user)):
    """Everything a client needs to create their own broker app. No secrets."""
    return {
        "redirect_uri": _client_redirect_uri(),
        "vault_ready": credentials_vault.is_configured(),
        "brokers": [
            {"key": "Upstox", "console_url": "https://account.upstox.com/developer/apps",
             "note": "Create an app, set the Redirect URI below, then copy its "
                     "API Key and API Secret."},
            {"key": "Zerodha", "console_url": "https://developers.kite.trade/apps",
             "note": "Create a Kite Connect app (billed per user by Zerodha), "
                     "set the Redirect URL below, then copy the API Key and Secret."},
        ],
    }


@router.get("/{broker}/credentials")
def get_credentials_summary(broker: str, user: CurrentUser = Depends(get_current_user)):
    """Masked key + status only. The secret has no read path anywhere."""
    _require_broker(broker)
    return user_manager.credential_summary(user.user_id, broker)


@router.put("/{broker}/credentials")
def save_credentials(broker: str, req: BrokerCredentialsRequest,
                     user: CurrentUser = Depends(get_current_user)):
    _require_broker(broker)
    api_key = req.api_key.strip()
    api_secret = req.api_secret.strip()
    if not api_key or not api_secret:
        raise HTTPException(400, "Both API key and API secret are required.")
    try:
        user_manager.set_broker_credentials(user.user_id, broker, api_key, api_secret)
    except credentials_vault.VaultUnavailable:
        raise HTTPException(
            503, "Server is not configured to store credentials securely "
                 "(CREDENTIALS_KEY missing). Contact your admin.")
    return user_manager.credential_summary(user.user_id, broker)


@router.delete("/{broker}/credentials")
def delete_credentials(broker: str, user: CurrentUser = Depends(get_current_user)):
    """Removes key, secret and token. Stops a running bot first — it would
    otherwise keep trading on a token the client just revoked."""
    _require_broker(broker)
    from api import engine_registry
    engine_registry.stop_engine(user.username)
    user_manager.clear_broker_credentials(user.user_id, broker)
    return user_manager.credential_summary(user.user_id, broker)


@router.get("/status")
def status(_: CurrentUser = Depends(require_admin)):
    """The shared .env credentials' status — meaningful for the admin
    account only; a client's own connection status comes from the
    token-status endpoints below, scoped to their own saved token."""
    return broker_api.check_broker_status()


# -- Upstox ------------------------------------------------------------------ #
@router.get("/upstox/login-url")
def upstox_login_url(user: CurrentUser = Depends(get_current_user)):
    api_key, _secret, redirect_uri = _resolve_credentials(user, "Upstox")
    return {"login_url": upstox_auth.build_login_url(api_key, redirect_uri),
            "redirect_uri": redirect_uri}


@router.post("/upstox/exchange")
def upstox_exchange(req: UpstoxExchangeRequest,
                    user: CurrentUser = Depends(get_current_user)):
    api_key, api_secret, redirect_uri = _resolve_credentials(user, "Upstox")
    code = upstox_auth.extract_code(req.code) or req.code
    res = upstox_auth.exchange_code(code, api_key, api_secret, redirect_uri)
    if not res["ok"]:
        raise HTTPException(400, res["error"])
    who = res.get("user_name") or res.get("email") or "user"
    if user.role == "client":
        user_manager.set_broker_token(user.user_id, "Upstox", res["token"])
        return {"ok": True, "message": f"Upstox connected for {who}."}
    upstox_auth.save_token(res["token"])
    config.reload_tokens()
    return {"ok": True, "message": f"Token refreshed for {who}."}


@router.get("/upstox/token-status")
def upstox_token_status(user: CurrentUser = Depends(get_current_user)):
    token = (user_manager.get_broker_token(user.user_id, "Upstox")
            if user.role == "client" else config.UPSTOX_LIVE_ACCESS_TOKEN)
    if not token:
        return {"has_token": False}
    r = upstox_auth.check_token(token)
    return {"has_token": True, "valid": r["ok"],
            "detail": r.get("user_name") or r.get("error")}


# -- Zerodha ------------------------------------------------------------------ #
@router.get("/zerodha/login-url")
def zerodha_login_url(user: CurrentUser = Depends(get_current_user)):
    api_key, _secret, _redirect = _resolve_credentials(user, "Zerodha")
    return {"login_url": kite_auth.build_login_url(api_key)}


@router.post("/zerodha/exchange")
def zerodha_exchange(req: ZerodhaExchangeRequest,
                     user: CurrentUser = Depends(get_current_user)):
    api_key, api_secret, _redirect = _resolve_credentials(user, "Zerodha")
    rtok = kite_auth.extract_request_token(req.request_token) or req.request_token
    res = kite_auth.exchange_request_token(rtok, api_key, api_secret)
    if not res["ok"]:
        raise HTTPException(400, res["error"])
    who = res.get("user_name") or res.get("email") or "user"
    if user.role == "client":
        user_manager.set_broker_token(user.user_id, "Zerodha", res["token"])
        return {"ok": True, "message": f"Zerodha connected for {who}."}
    kite_auth.save_token(res["token"])
    config.reload_tokens()
    return {"ok": True, "message": f"Zerodha token refreshed for {who}."}


@router.get("/zerodha/token-status")
def zerodha_token_status(user: CurrentUser = Depends(get_current_user)):
    # Status is polled by the UI, so missing credentials must read as "not
    # connected" rather than raising the way the OAuth endpoints do.
    if user.role == "client":
        api_key, _secret = user_manager.get_broker_credentials(user.user_id, "Zerodha")
        token = user_manager.get_broker_token(user.user_id, "Zerodha")
    else:
        api_key, _ = kite_auth.get_credentials()
        token = config.ZERODHA_ACCESS_TOKEN
    if not token or not api_key:
        return {"has_token": False}
    r = kite_auth.check_token(token, api_key)
    return {"has_token": True, "valid": r["ok"],
            "detail": r.get("user_name") or r.get("error")}


# -- Dhan (manual token, no OAuth; admin-only until a per-client flow exists) - #
@router.get("/dhan/status")
def dhan_status(_: CurrentUser = Depends(require_admin)):
    if not config.has_dhan():
        return {"configured": False}
    b = broker_api.DhanBroker()
    if b.connect():
        funds = b.available_funds()
        return {"configured": True, "valid": True, "available_funds": funds}
    return {"configured": True, "valid": False}


# -- A client's own fetched capital ------------------------------------------- #
@router.get("/my-funds")
def my_funds(broker: str, user: CurrentUser = Depends(get_current_user)):
    """Real available funds from the CALLER's own connected broker account —
    what the client sidebar shows as "capital you can allocate". Uses the
    exact same broker_api.available_funds() every engine relies on; this
    just calls it once, outside of a running engine, for display."""
    _require_broker(broker)
    token = user_manager.get_broker_token(user.user_id, broker)
    if not token:
        raise HTTPException(400, f"Connect your {broker} account first.")
    # Zerodha validates the token against the api_key that minted it, so a
    # client's own app key has to travel with it.
    api_key, _secret = user_manager.get_broker_credentials(user.user_id, broker)
    choice = Broker.UPSTOX if broker == "Upstox" else Broker.ZERODHA
    b = broker_api.make_broker(Environment.LIVE, choice, access_token=token,
                               api_key=api_key)
    if b.name == "Simulated":
        raise HTTPException(400, f"Could not connect to {broker} — token may be expired.")
    funds = b.available_funds()
    return {"broker": broker, "available_funds": funds}
