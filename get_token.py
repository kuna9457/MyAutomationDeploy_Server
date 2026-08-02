"""
get_token.py
One-shot CLI helper to generate a fresh Upstox LIVE access token via OAuth and
write it into your .env (UPSTOX_LIVE_ACCESS_TOKEN).

Run it each trading day (Upstox live tokens expire ~03:30 IST daily):

    python get_token.py

It reads UPSTOX_LIVE_API_KEY / UPSTOX_LIVE_SECRET / UPSTOX_REDIRECT_URI from .env,
opens the Upstox login page in your browser, you log in, then paste back the URL
you land on (it contains ?code=...). The script exchanges that code for a token
and offers to save it.

The same flow is also available as a button in the Streamlit UI (sidebar →
"Upstox Token") — this CLI and that button share upstox_auth.py, so they behave
identically.

NOTE: This is only for LIVE. Sandbox/paper needs no OAuth — just click "Generate"
in the Upstox Sandbox app and paste the token into UPSTOX_SANDBOX_TOKEN.
"""
from __future__ import annotations

import sys
import webbrowser

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

import upstox_auth as auth


def main() -> int:
    api_key, api_secret, redirect_uri = auth.get_credentials()

    if not api_key or not api_secret:
        print("❌ UPSTOX_LIVE_API_KEY / UPSTOX_LIVE_SECRET missing in .env.")
        print("   Add them (from your Upstox app) and run again.")
        return 1

    # 1) Build + open the login URL
    login_url = auth.build_login_url(api_key, redirect_uri)
    print("\nRedirect URI in use:", redirect_uri)
    print("(This must EXACTLY match the one registered in your Upstox app.)\n")
    print("Opening the Upstox login page in your browser...")
    print("If it doesn't open, paste this URL manually:\n")
    print(login_url, "\n")
    try:
        webbrowser.open(login_url)
    except Exception:
        pass

    print("After logging in, your browser will redirect to a page that likely")
    print("says 'can't connect' — that's expected. Copy the FULL URL from the")
    print("address bar (it contains ?code=...) and paste it below.\n")

    # 2) Read the code
    user_input = input("Paste redirected URL (or just the code): ").strip()
    code = auth.extract_code(user_input)
    if not code:
        print("❌ Could not find an authorization code in that input.")
        return 1
    print(f"\n✅ Got authorization code: {code[:6]}...")

    # 3) Exchange for an access token
    print("Exchanging code for access token...")
    result = auth.exchange_code(code, api_key, api_secret, redirect_uri)
    if not result["ok"]:
        print(f"❌ {result['error']}")
        return 1

    token = result["token"]
    print("\n🎉 Access token generated successfully!")
    print("   User:", result.get("user_name") or result.get("email") or "-")
    print(f"   Token (first 12 chars): {token[:12]}...")

    # 4) Offer to save into .env
    save = input("\nSave this token to .env as UPSTOX_LIVE_ACCESS_TOKEN? [Y/n] ")
    if save.strip().lower() in ("", "y", "yes"):
        try:
            path = auth.save_token(token)
            print(f"✅ Saved to {path}")
        except Exception:
            print("python-dotenv not available — copy this line into your .env:")
            print(f'UPSTOX_LIVE_ACCESS_TOKEN="{token}"')
    else:
        print("Not saved. Copy this into your .env when ready:")
        print(f'UPSTOX_LIVE_ACCESS_TOKEN="{token}"')

    print("\nDone. Restart the bot (streamlit run app.py) to use the new token.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
