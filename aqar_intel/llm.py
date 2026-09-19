"""LLM access layer.

All model calls go through a tiny interface (`LLM.chat`) so that:

* the provider can be swapped (OpenRouter today; Azure/Bedrock/on-prem tomorrow),
* every module can be unit-tested with a scripted `FakeLLM`,
* the whole platform can run offline in `mock` mode for demos without a key.

OpenRouter exposes an OpenAI-compatible API, so we use the official `openai`
SDK with a custom `base_url`.
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, Protocol

from .config import Settings, get_settings

log = logging.getLogger(__name__)

Message = dict[str, str]


class LLM(Protocol):
    name: str

    def chat(
        self,
        messages: list[Message],
        *,
        temperature: float = 0.0,
        json_mode: bool = False,
        max_tokens: int = 1024,
    ) -> str: ...


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def extract_json(text: str) -> Any:
    """Robustly pull a JSON object/array out of an LLM reply.

    Handles code fences, leading prose and trailing commentary. Raises
    ``ValueError`` if nothing parseable is found.
    """
    if text is None:
        raise ValueError("empty response")
    candidate = text.strip()
    m = _FENCE_RE.search(candidate)
    if m:
        candidate = m.group(1).strip()
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass
    # Fall back to the outermost {...} or [...]
    for open_ch, close_ch in (("{", "}"), ("[", "]")):
        start = candidate.find(open_ch)
        end = candidate.rfind(close_ch)
        if start != -1 and end > start:
            try:
                return json.loads(candidate[start : end + 1])
            except json.JSONDecodeError:
                continue
    raise ValueError(f"no JSON found in response: {text[:200]!r}")


# --------------------------------------------------------------------------- #
# OpenRouter
# --------------------------------------------------------------------------- #
class OpenRouterLLM:
    """Chat completions via OpenRouter (OpenAI-compatible)."""

    def __init__(self, settings: Settings | None = None, model: str | None = None):
        from openai import OpenAI  # local import keeps mock mode dependency-free

        self.settings = settings or get_settings()
        if not self.settings.openrouter_api_key:
            raise RuntimeError("OPENROUTER_API_KEY is not set")
        self.model = model or self.settings.chat_model
        self.name = f"openrouter:{self.model}"
        self._client = OpenAI(
            api_key=self.settings.openrouter_api_key,
            base_url=self.settings.openrouter_base_url,
            timeout=self.settings.request_timeout,
            default_headers={
                # Optional OpenRouter attribution headers
                "HTTP-Referer": self.settings.app_url,
                "X-Title": self.settings.app_name,
            },
        )

    def chat(
        self,
        messages: list[Message],
        *,
        temperature: float = 0.0,
        json_mode: bool = False,
        max_tokens: int = 1024,
        retries: int = 2,
    ) -> str:
        kwargs: dict[str, Any] = dict(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        last_err: Exception | None = None
        for attempt in range(retries + 1):
            try:
                resp = self._client.chat.completions.create(**kwargs)
                content = resp.choices[0].message.content or ""
                return content
            except Exception as e:  # noqa: BLE001 - we want to retry on any transport/provider error
                last_err = e
                msg = str(e).lower()
                # Some providers reject response_format; retry once without it.
                if json_mode and ("response_format" in msg or "json_object" in msg):
                    kwargs.pop("response_format", None)
                    json_mode = False
                    continue
                if attempt < retries:
                    sleep = 1.5 * (attempt + 1)
                    log.warning("LLM call failed (%s); retrying in %.1fs", e, sleep)
                    time.sleep(sleep)
        raise RuntimeError(f"LLM call failed after retries: {last_err}")


# --------------------------------------------------------------------------- #
# Mock / Fake
# --------------------------------------------------------------------------- #
class MockLLM:
    """Offline stand-in used when no API key is configured.

    It never pretends to reason; it returns clearly-labelled placeholder text so
    the rest of the pipeline (retrieval, SQL guardrails, validation, analytics)
    can still be exercised end-to-end.
    """

    name = "mock"

    def chat(
        self,
        messages: list[Message],
        *,
        temperature: float = 0.0,
        json_mode: bool = False,
        max_tokens: int = 1024,
    ) -> str:
        user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        if json_mode:
            return "{}"
        snippet = user.strip().replace("\n", " ")[:300]
        return (
            "[mock LLM – set OPENROUTER_API_KEY for real answers]\n"
            f"Received prompt: {snippet}..."
        )


class FakeLLM:
    """Scripted LLM for tests: returns queued responses in order."""

    name = "fake"

    def __init__(self, responses: list[str] | None = None):
        self.responses = list(responses or [])
        self.calls: list[list[Message]] = []

    def chat(self, messages: list[Message], **_: Any) -> str:
        self.calls.append(messages)
        if not self.responses:
            raise AssertionError("FakeLLM ran out of scripted responses")
        return self.responses.pop(0)


def get_llm(settings: Settings | None = None) -> LLM:
    settings = settings or get_settings()
    if settings.use_openrouter:
        return OpenRouterLLM(settings)
    log.warning("No OPENROUTER_API_KEY found – running with MockLLM (offline mode).")
    return MockLLM()
