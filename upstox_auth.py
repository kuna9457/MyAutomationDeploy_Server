"""
upstox_auth.py
Shared Upstox OAuth helpers used by BOTH the CLI (get_token.py) and the Streamlit
UI (app.py). Keeping the login-URL build, code exchange and .env save in one place
means the two entry points can never drift apart.

Upstox LIVE access tokens expire daily (~03:30 IST), so this refresh flow is a
normal part of each trading day. Sandbox/paper needs no OAuth.
"""
from __future__ import annotations

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


AUTH_URL = "https://api.upstox.com/v2/login/authorization/dialog"
TOKEN_URL = "https://api.upstox.com/v2/login/authorization/token"
PROFILE_URL = "https://api.upstox.com/v2/user/profile"

# Must EXACTLY match the Redirect URI registered in your Upstox app.
DEFAULT_REDIRECT = "https://127.0.0.1:5000/"

TOKEN_ENV_KEY = "UPSTOX_LIVE_ACCESS_TOKEN"


def _env(key: str, default: str = "") -> str:
    return (os.getenv(key, default) or "").strip()


def get_credentials() -> tuple[str, str, str]:
    """(api_key, api_secret, redirect_uri) from the environment."""
    return (
        _env("UPSTOX_LIVE_API_KEY"),
        _env("UPSTOX_LIVE_SECRET"),
        _env("UPSTOX_REDIRECT_URI", DEFAULT_REDIRECT),
    )


def build_login_url(api_key: str, redirect_uri: str) -> str:
    """The Upstox authorization dialog URL the user logs in through."""
    params = {
        "response_type": "code",
        "client_id": api_key,
        "redirect_uri": redirect_uri,
    }
    return f"{AUTH_URL}?{urllib.parse.urlencode(params)}"


def extract_code(user_input: str) -> str | None:
    """Accept either the full redirected URL or a bare authorization code."""
    user_input = (user_input or "").strip()
    if not user_input:
        return None
    if "code=" in user_input:
        qs = urllib.parse.urlparse(user_input).query or user_input
        params = urllib.parse.parse_qs(qs)
        if "code" in params:
            return params["code"][0]
        m = re.search(r"code=([^&\s]+)", user_input)
        return m.group(1) if m else None
    return user_input  # assume they pasted just the code


def exchange_code(
    code: str, api_key: str, api_secret: str, redirect_uri: str
) -> dict:
    """
    Exchange a single-use authorization code for a LIVE access token.
    Returns {ok, token, user_name, email, error}. Never raises — network and HTTP
    errors are captured in `error` so callers (CLI or UI) can display them.
    """
    headers = {
        "accept": "application/json",
        "Content-Type": "application/x-www-form-urlencoded",
    }
    data = {
        "code": code,
        "client_id": api_key,
        "client_secret": api_secret,
        "redirect_uri": redirect_uri,
        "grant_type": "authorization_code",
    }
    try:
        resp = requests.post(TOKEN_URL, headers=headers, data=data, timeout=15)
    except Exception as exc:
        return {"ok": False, "error": f"Network error: {exc}"}

    if resp.status_code != 200:
        return {"ok": False, "error": (
            f"Token request failed (HTTP {resp.status_code}): {resp.text}\n"
            "Common causes: redirect_uri mismatch, expired/reused code "
            "(codes are single-use — get a fresh one), or wrong API key/secret.")}

    payload = resp.json()
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


def check_token(token: str) -> dict:
    """
    Cheap read-only validity check: hit the profile endpoint with the token.
    Returns {ok, user_name, error}. A valid live token returns HTTP 200 + name;
    an expired one returns 401. Never raises.
    """
    if not token:
        return {"ok": False, "error": "No token set."}
    try:
        resp = requests.get(
            PROFILE_URL, headers={"Authorization": f"Bearer {token}",
                                  "accept": "application/json"}, timeout=10)
    except Exception as exc:
        return {"ok": False, "error": f"Network error: {exc}"}
    if resp.status_code == 200:
        data = resp.json().get("data", {})
        return {"ok": True, "user_name": data.get("user_name", "user"),
                "error": ""}
    return {"ok": False, "error": f"Token invalid/expired (HTTP "
            f"{resp.status_code})."}


def save_token(token: str) -> str:
    """
    Persist the token to .env as UPSTOX_LIVE_ACCESS_TOKEN and return the path.
    Raises RuntimeError if python-dotenv isn't available so the caller can fall
    back to telling the user to paste it manually.
    """
    if not (set_key and find_dotenv):
        raise RuntimeError("python-dotenv not available")
    dotenv_path = find_dotenv() or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), ".env")
    if not os.path.exists(dotenv_path):
        open(dotenv_path, "a").close()
    set_key(dotenv_path, TOKEN_ENV_KEY, token)
    # Reflect it in the current process immediately.
    os.environ[TOKEN_ENV_KEY] = token
    return dotenv_path
