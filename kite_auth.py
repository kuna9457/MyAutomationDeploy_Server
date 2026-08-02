"""
kite_auth.py
Shared Zerodha (Kite Connect) login helpers used by the Streamlit UI (app.py),
mirroring upstox_auth.py so the two brokers refresh the same way.

Kite access tokens expire daily (the session resets ~06:00 IST every trading
day), so — exactly like Upstox — this refresh flow is a normal part of each
trading day, not a one-time setup step. The login round trip is a request
token exchanged for an access token via generate_session(); it is deliberately
implemented with plain `requests` rather than the `kiteconnect` SDK so the
token panel works even before that optional dependency is installed.
"""
from __future__ import annotations

import hashlib
import os
import re
import urllib.parse

import requests

try:
    from dotenv import load_dotenv, set_key, find_dotenv
except Exception:  # dotenv is optional
    load_dotenv = None
    set_key = None
    find_dotenv = None


LOGIN_URL = "https://kite.zerodha.com/connect/login"
SESSION_URL = "https://api.kite.trade/session/token"
PROFILE_URL = "https://api.kite.trade/user/profile"
KITE_VERSION = "3"

TOKEN_ENV_KEY = "ZERODHA_ACCESS_TOKEN"


def _env(key: str, default: str = "") -> str:
    return (os.getenv(key, default) or "").strip()


def get_credentials() -> tuple[str, str]:
    """(api_key, api_secret) from the environment."""
    return _env("ZERODHA_API_KEY"), _env("ZERODHA_API_SECRET")


def build_login_url(api_key: str) -> str:
    """The Kite Connect login dialog URL the user authenticates through.
    Zerodha redirects back to the app's registered Redirect URL (set in the
    Kite Connect developer console, NOT passed here) with ?request_token=..."""
    params = {"v": KITE_VERSION, "api_key": api_key}
    return f"{LOGIN_URL}?{urllib.parse.urlencode(params)}"


def extract_request_token(user_input: str) -> str | None:
    """Accept either the full redirected URL or a bare request_token."""
    user_input = (user_input or "").strip()
    if not user_input:
        return None
    if "request_token=" in user_input:
        qs = urllib.parse.urlparse(user_input).query or user_input
        params = urllib.parse.parse_qs(qs)
        if "request_token" in params:
            return params["request_token"][0]
        m = re.search(r"request_token=([^&\s]+)", user_input)
        return m.group(1) if m else None
    return user_input  # assume they pasted just the token


def exchange_request_token(request_token: str, api_key: str, api_secret: str) -> dict:
    """
    Exchange a single-use request_token for a Kite access_token via
    generate_session (POST /session/token). checksum is the Kite-mandated
    SHA-256 of api_key + request_token + api_secret.

    Returns {ok, token, user_name, email, error}. Never raises.
    """
    checksum = hashlib.sha256(
        (api_key + request_token + api_secret).encode("utf-8")).hexdigest()
    data = {
        "api_key": api_key,
        "request_token": request_token,
        "checksum": checksum,
    }
    try:
        resp = requests.post(
            SESSION_URL, data=data,
            headers={"X-Kite-Version": KITE_VERSION}, timeout=15)
    except Exception as exc:
        return {"ok": False, "error": f"Network error: {exc}"}

    if resp.status_code != 200:
        return {"ok": False, "error": (
            f"Session request failed (HTTP {resp.status_code}): {resp.text}\n"
            "Common causes: request_token already used (single-use — get a "
            "fresh one by logging in again), or wrong API key/secret.")}

    payload = resp.json().get("data", {}) or {}
    token = payload.get("access_token")
    if not token:
        return {"ok": False, "error": f"No access_token in response: {payload}"}
    return {
        "ok": True,
        "token": token,
        "user_name": payload.get("user_name", ""),
        "email": payload.get("email", ""),
        "error": "",
    }


def check_token(token: str, api_key: str) -> dict:
    """
    Cheap read-only validity check: hit the profile endpoint with the token.
    Returns {ok, user_name, error}. Never raises.
    """
    if not token:
        return {"ok": False, "error": "No token set."}
    try:
        resp = requests.get(
            PROFILE_URL,
            headers={"Authorization": f"token {api_key}:{token}",
                     "X-Kite-Version": KITE_VERSION},
            timeout=10)
    except Exception as exc:
        return {"ok": False, "error": f"Network error: {exc}"}
    if resp.status_code == 200:
        data = resp.json().get("data", {})
        return {"ok": True, "user_name": data.get("user_name", "user"), "error": ""}
    return {"ok": False, "error": f"Token invalid/expired (HTTP {resp.status_code})."}


def save_token(token: str) -> str:
    """
    Persist the token to .env as ZERODHA_ACCESS_TOKEN and return the path.
    Raises RuntimeError if python-dotenv isn't available so the caller can
    fall back to telling the user to paste it manually.
    """
    if not (set_key and find_dotenv):
        raise RuntimeError("python-dotenv not available")
    dotenv_path = find_dotenv() or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), ".env")
    if not os.path.exists(dotenv_path):
        open(dotenv_path, "a").close()
    set_key(dotenv_path, TOKEN_ENV_KEY, token)
    os.environ[TOKEN_ENV_KEY] = token
    return dotenv_path
