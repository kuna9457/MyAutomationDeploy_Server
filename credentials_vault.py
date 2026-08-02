"""
credentials_vault.py
Encryption at rest for per-client broker credentials (client_broker_onboarding_plan.md §4).

A client pastes their own broker API key + secret into the UI; those are
stored on their user record as Fernet ciphertext, never in plaintext and
never returned by any endpoint. Only the server, holding CREDENTIALS_KEY,
can turn them back into something usable — and it does so only inside the
request that needs them, never into a log or an error message.

Fernet (AES-128-CBC + HMAC-SHA256, from `cryptography`) is used because
`cryptography` is already an installed dependency, so this adds no new
native-build risk — the same constraint that made security.py stick to
hashlib rather than bcrypt.

CREDENTIALS_KEY lives in .env. Generate one with:

    python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"

Losing the key means every stored credential becomes undecryptable; clients
then simply re-enter theirs. Rotating it has the same effect. Neither is
catastrophic, but back it up alongside JWT_SECRET.
"""
from __future__ import annotations

import os

_KEY_ENV = "CREDENTIALS_KEY"

_fernet = None
_init_error = ""

try:
    from cryptography.fernet import Fernet, InvalidToken  # type: ignore

    _raw_key = (os.getenv(_KEY_ENV, "") or "").strip()
    if _raw_key:
        try:
            _fernet = Fernet(_raw_key.encode())
        except Exception as exc:
            _init_error = f"{_KEY_ENV} is set but not a valid Fernet key ({exc})."
    else:
        _init_error = (
            f"{_KEY_ENV} is not set — clients cannot save broker credentials. "
            f"Generate one with: python -c \"from cryptography.fernet import "
            f"Fernet; print(Fernet.generate_key().decode())\"")
except ImportError:  # pragma: no cover - cryptography ships with pymongo
    InvalidToken = Exception  # type: ignore
    _init_error = ("`cryptography` is not installed — run "
                   "`pip install -r requirements.txt`.")

if _init_error:
    print(f"[credentials_vault] WARNING: {_init_error}")


class VaultUnavailable(RuntimeError):
    """Raised when a write is attempted with no usable key. Callers turn this
    into a 503 rather than silently storing plaintext."""


def is_configured() -> bool:
    return _fernet is not None


def status_detail() -> str:
    """Why the vault is unusable, for an operator-facing error. Never contains
    the key itself."""
    return _init_error


def encrypt(plaintext: str) -> str:
    if _fernet is None:
        raise VaultUnavailable(_init_error)
    return _fernet.encrypt(plaintext.encode("utf-8")).decode("utf-8")


def decrypt(ciphertext: str) -> str:
    """Plaintext, or "" if the value can't be decrypted (wrong/rotated key, or
    a value stored before encryption existed). Returning "" rather than raising
    lets callers treat it as "not connected", which is the correct recovery:
    the client re-enters their credentials."""
    if _fernet is None or not ciphertext:
        return ""
    try:
        return _fernet.decrypt(ciphertext.encode("utf-8")).decode("utf-8")
    except (InvalidToken, Exception):
        return ""


def mask(value: str) -> str:
    """A recognisable but unusable echo of an API KEY — enough for a client to
    confirm which key is saved. Never call this on a secret or a token: those
    have no read path at all."""
    if not value:
        return ""
    if len(value) <= 8:
        return "•" * len(value)
    return f"{value[:4]}{'•' * 6}{value[-4:]}"
