"""
net_config.py
Process-wide outbound network preference: resolve hostnames to IPv4 first.

Why this exists
---------------
Upstox lets you lock an API app to a list of static IPs. That list is IPv4:

    configured static IP : 3.111.230.103, 122.161.241.233
    request origin IP    : 2406:da1a:1f80:e600:5f13:3100:c79f:a397
    -> UDAPI1154 "Access to this API is blocked due to static IP restrictions"

On a dual-stack host Python may open the connection over IPv6, and an IPv6
source address can NEVER match an IPv4 allowlist entry — the comparison fails
on address family alone, no matter whether the machine is the right one.
Allowlisting the v6 address doesn't help either: the suffix pattern above is a
temporary/privacy address, which rotates.

So the fix is to make outbound connections deterministic. With IPv4 preferred,
the server presents the address the allowlist was written for.

What it actually does
---------------------
Patches `socket.getaddrinfo` so that a lookup which did not specify an address
family (AF_UNSPEC — what `requests`, `urllib3`, the Upstox SDK, kiteconnect and
pymongo all use) is answered with A records only.

Two deliberate safety properties:

  * A caller that EXPLICITLY asks for AF_INET6 still gets IPv6. Code that
    means to use v6 is not overridden.
  * If the IPv4 lookup fails (no A record — an IPv6-only host), the original
    unrestricted lookup is retried. A host reachable only over IPv6 keeps
    working; the pin is a preference, not a prohibition.

Together those mean the worst case is "behaves exactly as before".

Configuration
-------------
    FORCE_IPV4=true   (default) prefer IPv4 for outbound connections
    FORCE_IPV4=false  leave resolution entirely alone

This affects DNS resolution only. It touches no trading logic, no strategy, no
sizing — a connection's address family cannot change what a signal decides.
"""
from __future__ import annotations

import os
import socket

try:
    from dotenv import load_dotenv
    load_dotenv()          # same pattern as config.py — don't rely on import order
except Exception:
    pass

_original_getaddrinfo = socket.getaddrinfo
_applied = False


def _enabled() -> bool:
    return (os.getenv("FORCE_IPV4", "true") or "").strip().lower() \
        not in ("0", "false", "no", "off")


def _ipv4_first_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
    """A records only for unspecified-family lookups; everything else untouched."""
    if family == socket.AF_UNSPEC:
        try:
            return _original_getaddrinfo(host, port, socket.AF_INET,
                                         type, proto, flags)
        except socket.gaierror:
            # No IPv4 for this host. Fall through rather than fail — an
            # IPv6-only endpoint must still resolve.
            pass
    return _original_getaddrinfo(host, port, family, type, proto, flags)


def apply() -> bool:
    """Install the preference. Idempotent; returns whether it is now active.

    Call this ONCE, as early as possible — before anything opens a socket.
    Patching after a connection pool has been built only affects connections
    made from then on.
    """
    global _applied
    if not _enabled():
        return False
    if not _applied:
        socket.getaddrinfo = _ipv4_first_getaddrinfo
        _applied = True
        print("[net_config] Outbound connections prefer IPv4 "
              "(FORCE_IPV4). Set FORCE_IPV4=false to disable.")
    return True


def revert() -> None:
    """Restore stock resolution. For tests."""
    global _applied
    if _applied:
        socket.getaddrinfo = _original_getaddrinfo
        _applied = False


def status() -> dict:
    return {"enabled": _enabled(), "applied": _applied}


def egress_ip(timeout: float = 15.0) -> str:
    """The public IP this process actually presents right now — i.e. what an
    allowlist will see. Never raises; returns "" if it can't be determined."""
    try:
        import urllib.request
        return urllib.request.urlopen(
            "https://api64.ipify.org", timeout=timeout).read().decode().strip()
    except Exception:
        return ""
