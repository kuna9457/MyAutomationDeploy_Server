"""
ai_auditor/providers.py
One interface, two LLM back-ends. Plain HTTPS — no vendor SDK to install.

Every failure surfaces as a clean AuditProviderError. Nothing here retries: a
retry loop on a paid endpoint is how a stuck run turns into a bill, and the
operator pressed the button once on purpose.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Optional

import requests

import config
from ai_auditor.prompts import REPORT_SCHEMA


class AuditProviderError(RuntimeError):
    """Anything that stopped us getting a usable report."""


class AuditFatalError(AuditProviderError):
    """A failure that trying another MODEL cannot fix — a bad key, no credit,
    a rate limit. Distinguished so the model chain stops immediately instead of
    replaying the same rejection down the whole list."""


@dataclass
class AuditResponse:
    report: dict
    provider: str
    model: str
    raw_text: str
    usage: dict


def _extract_json(text: str) -> dict:
    """Parse the model's reply into the report dict.

    Tolerates the two things models do even when told not to: wrapping the JSON
    in a ```json fence, and adding a sentence before it.
    """
    t = (text or "").strip()
    if t.startswith("```"):
        t = t.split("```", 2)[1] if t.count("```") >= 2 else t.strip("`")
        if t.lstrip().lower().startswith("json"):
            t = t.lstrip()[4:]
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        pass
    start, end = t.find("{"), t.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(t[start:end + 1])
        except json.JSONDecodeError as exc:
            raise AuditProviderError(
                f"The model's reply was not valid JSON ({exc}).")
    raise AuditProviderError("The model returned no JSON at all.")


# --------------------------------------------------------------------------- #
#  OpenRouter — OpenAI-compatible chat completions
# --------------------------------------------------------------------------- #
class OpenRouterProvider:
    name = "openrouter"
    URL = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(self, model: str = ""):
        self.api_key = config.OPENROUTER_API_KEY
        # An explicit model (from the UI or .env) is used alone. With none, walk
        # the configured chain strongest-first so a retired model id degrades to
        # the next best rather than failing the audit.
        chosen = model or config.OPENROUTER_MODEL
        self.models = [chosen] if chosen else list(config.OPENROUTER_MODEL_CHAIN)
        self.model = self.models[0] if self.models else ""
        self.tried: list[str] = []

    @staticmethod
    def available() -> bool:
        return bool(config.OPENROUTER_API_KEY)

    def complete(self, system: str, user: str) -> AuditResponse:
        """Try each model in turn; the first that answers wins."""
        if not self.models:
            raise AuditProviderError(
                "No OpenRouter model configured. Set OPENROUTER_MODEL or "
                "OPENROUTER_MODEL_CHAIN in .env.")
        errors: list[str] = []
        for name in self.models:
            self.tried.append(name)
            try:
                return self._complete_one(name, system, user)
            except AuditFatalError:
                # Key/credit/rate problems repeat identically on every model.
                raise
            except AuditProviderError as exc:
                errors.append(f"{name}: {exc}")
        raise AuditProviderError(
            "Every OpenRouter model failed.\n" + "\n".join(errors)
            + "\nModel ids change — check https://openrouter.ai/models and set "
              "OPENROUTER_MODEL_CHAIN in .env.")

    def _complete_one(self, model: str, system: str, user: str) -> AuditResponse:
        if not self.api_key:
            raise AuditFatalError(
                "OPENROUTER_API_KEY is not set — add it to .env and restart, "
                "or reconnect a broker (which reloads .env).")
        body = {
            "model": model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "max_tokens": config.AI_AUDITOR_MAX_TOKENS,
            # Low but not zero: the report is analysis, not sampling.
            "temperature": 0.2,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "audit_report", "strict": False,
                                "schema": REPORT_SCHEMA},
            },
        }
        try:
            r = requests.post(
                self.URL, json=body, timeout=config.AI_AUDITOR_TIMEOUT_SECONDS,
                headers={"Authorization": f"Bearer {self.api_key}",
                         "Content-Type": "application/json",
                         # OpenRouter asks callers to identify themselves.
                         "X-Title": "Trading Bot AI Auditor"})
        except requests.Timeout:
            raise AuditProviderError(
                f"OpenRouter timed out after "
                f"{config.AI_AUDITOR_TIMEOUT_SECONDS}s. Try a smaller window "
                f"or a faster model.")
        except requests.RequestException as exc:
            raise AuditProviderError(f"Could not reach OpenRouter: {exc}")
        if r.status_code != 200:
            # 401/403 = key; 402 = no credit; 429 = rate limit. None of those
            # change if we ask for a different model, so stop the chain.
            if r.status_code in (401, 402, 403, 429):
                raise AuditFatalError(
                    f"OpenRouter rejected the request (HTTP {r.status_code}): "
                    f"{r.text[:200]}")
            raise AuditProviderError(
                f"HTTP {r.status_code}: {r.text[:200]}")
        data = r.json()
        # OpenRouter reports some upstream failures as 200 with an error body.
        if isinstance(data, dict) and data.get("error") and not data.get("choices"):
            raise AuditProviderError(str(data["error"])[:200])
        try:
            text = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError):
            raise AuditProviderError(
                f"Unexpected response shape: {str(data)[:200]}")
        # The model that actually answered — not necessarily the one asked for,
        # since OpenRouter may route, and the report must record the truth.
        used = data.get("model") or model
        self.model = used
        return AuditResponse(_extract_json(text), self.name, used, text,
                             data.get("usage") or {})


# --------------------------------------------------------------------------- #
#  Google Gemini — generateContent with a response schema
# --------------------------------------------------------------------------- #
class GeminiProvider:
    name = "gemini"
    URL = ("https://generativelanguage.googleapis.com/v1beta/models/"
           "{model}:generateContent")

    def __init__(self, model: str = ""):
        self.api_key = config.GEMINI_API_KEY
        self.model = model or config.GEMINI_MODEL

    @staticmethod
    def available() -> bool:
        return bool(config.GEMINI_API_KEY)

    def complete(self, system: str, user: str) -> AuditResponse:
        if not self.api_key:
            raise AuditProviderError(
                "GEMINI_API_KEY is not set — add it to .env and restart.")
        body = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": {
                "temperature": 0.2,
                "maxOutputTokens": config.AI_AUDITOR_MAX_TOKENS,
                "responseMimeType": "application/json",
            },
        }
        try:
            r = requests.post(
                self.URL.format(model=self.model), json=body,
                timeout=config.AI_AUDITOR_TIMEOUT_SECONDS,
                headers={"Content-Type": "application/json",
                         "x-goog-api-key": self.api_key})
        except requests.Timeout:
            raise AuditProviderError(
                f"Gemini timed out after {config.AI_AUDITOR_TIMEOUT_SECONDS}s.")
        except requests.RequestException as exc:
            raise AuditProviderError(f"Could not reach Gemini: {exc}")
        if r.status_code != 200:
            raise AuditProviderError(
                f"Gemini returned HTTP {r.status_code}: {r.text[:300]}")
        data = r.json()
        try:
            text = data["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError):
            # A blocked or truncated response lands here; surface why.
            reason = (data.get("promptFeedback", {}).get("blockReason")
                      or str(data)[:300])
            raise AuditProviderError(f"Gemini returned no usable text: {reason}")
        return AuditResponse(_extract_json(text), self.name, self.model, text,
                             data.get("usageMetadata") or {})


_PROVIDERS = {"openrouter": OpenRouterProvider, "gemini": GeminiProvider}


def available_providers() -> list[dict]:
    """What the UI may offer. A provider without a key is listed but disabled,
    so the reason it cannot be used is visible rather than mysterious."""
    return [
        {"name": "openrouter", "available": OpenRouterProvider.available(),
         "model": config.OPENROUTER_MODEL},
        {"name": "gemini", "available": GeminiProvider.available(),
         "model": config.GEMINI_MODEL},
    ]


def get_provider(name: str = "", model: str = ""):
    key = (name or config.AI_AUDITOR_PROVIDER or "openrouter").lower()
    cls = _PROVIDERS.get(key)
    if cls is None:
        raise AuditProviderError(
            f"Unknown provider {name!r}. Use one of: "
            f"{', '.join(_PROVIDERS)}.")
    return cls(model=model)


def provider_chain(name: str = "", model: str = "") -> list:
    """Providers to try, in order: the requested one first, then the other if
    fallback is enabled and it has a key.

    A per-request `model` is only ever applied to the FIRST provider — it was
    chosen for that provider and would be meaningless (or wrong) on the other,
    which then uses its own default.
    """
    first_name = (name or config.AI_AUDITOR_PROVIDER or "openrouter").lower()
    chain = [get_provider(first_name, model)]
    if not config.AI_AUDITOR_FALLBACK:
        return chain
    for other in _PROVIDERS:
        if other == first_name:
            continue
        cls = _PROVIDERS[other]
        if cls.available():
            chain.append(cls())
    return chain
