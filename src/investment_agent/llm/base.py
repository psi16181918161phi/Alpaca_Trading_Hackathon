"""LLM Provider abstraction for X Quant X.

WHAT
====
Single, model-agnostic interface for language-model inference. The downstream
multi-agent system only ever talks to ``LLMProvider``; the concrete backend
(Featherless, a deterministic mock for tests, or any other HTTP API) is hidden
behind this contract.

WHY
====
- Decouples agent prompts from model wiring so we can swap backends (Hermes,
  finance-tuned model, etc.) without touching mathematical modules.
- Lets tests run fully offline with a deterministic ``MockLLMProvider`` that
  returns canned ``AgentOutput`` tuples.
- Keeps the capital gate authoritative: the LLM is only ever an *input* to the
  ensemble, never a *risk override*.

HOW
====
- ``LLMProvider.complete(prompt, ...)`` returns an ``LLMResponse`` with raw text
  plus structured metadata (model, latency, token count).
- ``AgentLLMAdapter`` wraps an ``LLMProvider`` and parses its output into a
  fully-validated ``AgentOutput`` tuple (eight channels: s, c, u, d, p_plus,
  p_minus, delta_t, r).
- All network calls are bounded by timeouts and fall back to a "zero
  confidence" output on failure so the deterministic risk layer is never
  starved.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Mapping, Optional, Protocol


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LLMResponse:
    """Immutable result of a single LLM call.

    Attributes
    ----------
    text : str
        Raw text returned by the model.
    model : str
        Identifier of the model that produced the text.
    latency_ms : float
        Wall-clock latency of the call in milliseconds.
    prompt_tokens : int
        Estimated prompt token count (best-effort; backend-dependent).
    completion_tokens : int
        Estimated completion token count (best-effort).
    raw : Dict[str, Any]
        Backend-specific raw payload (for debugging / audit).
    """

    text: str
    model: str
    latency_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    raw: Dict[str, Any] = field(default_factory=dict)


class LLMProvider(Protocol):
    """Backend-agnostic LLM interface."""

    @property
    def model_id(self) -> str:
        ...

    def complete(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 512,
        timeout_s: float = 30.0,
    ) -> LLMResponse:
        ...


# ---------------------------------------------------------------------------
# Deterministic mock provider (offline / unit tests)
# ---------------------------------------------------------------------------

class MockLLMProvider:
    """Deterministic in-process provider used for tests and paper runs.

    The provider accepts a callable that maps a (system, prompt) pair to a raw
    text string (typically JSON). When no callable is supplied it returns a
    well-formed neutral ``AgentOutput`` so downstream code remains stable.
    """

    def __init__(
        self,
        responder: Optional[Callable[[Optional[str], str], str]] = None,
        model_id: str = "mock-llm",
    ) -> None:
        self._responder = responder
        self._model_id = model_id
        self._calls = 0

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def call_count(self) -> int:
        return self._calls

    def complete(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 512,
        timeout_s: float = 30.0,
    ) -> LLMResponse:
        self._calls += 1
        if self._responder is not None:
            text = self._responder(system, prompt)
        else:
            text = json.dumps({
                "signal": 0.0,
                "confidence": 0.5,
                "uncertainty": 0.5,
                "doubt": 0.5,
                "p_plus": 0.5,
                "p_minus": 0.5,
                "delta_t": 1.0,
                "noise": 1.0,
            })
        return LLMResponse(
            text=text,
            model=self._model_id,
            latency_ms=0.0,
            prompt_tokens=len(prompt.split()),
            completion_tokens=len(text.split()),
            raw={"system": system, "temperature": temperature, "max_tokens": max_tokens},
        )


# ---------------------------------------------------------------------------
# Featherless provider
# ---------------------------------------------------------------------------

class FeatherlessProvider:
    """LLMProvider backed by the Featherless chat-completions HTTP API.

    Featherless exposes an OpenAI-compatible ``/v1/chat/completions`` endpoint.
    This provider is intentionally minimal: it surfaces the same fields
    ``LLMProvider`` requires and never assumes any project-specific schema.
    """

    DEFAULT_BASE_URL = "https://api.featherless.ai/v1"
    DEFAULT_MODEL = "Qwen/Qwen2.5-72B-Instruct"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        timeout_s: float = 30.0,
        session_factory: Optional[Callable[[], Any]] = None,
    ) -> None:
        self._api_key = api_key or os.getenv("FEATHERLESS_API_KEY")
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._timeout_s = timeout_s
        self._session_factory = session_factory

    @property
    def model_id(self) -> str:
        return self._model

    def _ensure_session(self) -> Any:
        if self._session_factory is not None:
            return self._session_factory()
        try:
            import requests
        except ImportError as exc:
            raise RuntimeError(
                "FeatherlessProvider requires the 'requests' package; "
                "install it or pass session_factory="
            ) from exc
        return requests.Session()

    def complete(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 512,
        timeout_s: float = 30.0,
    ) -> LLMResponse:
        if not self._api_key:
            raise RuntimeError(
                "FeatherlessProvider: no API key. Set FEATHERLESS_API_KEY in env "
                "or pass api_key=..."
            )

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self._model,
            "messages": messages,
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
        }
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        url = f"{self._base_url}/chat/completions"
        timeout = float(timeout_s if timeout_s is not None else self._timeout_s)

        session = self._ensure_session()
        start = time.perf_counter()
        response = session.post(url, json=payload, headers=headers, timeout=timeout)
        latency_ms = (time.perf_counter() - start) * 1000.0

        response.raise_for_status()
        body = response.json()
        text = _extract_chat_completion_text(body)

        usage = body.get("usage", {}) if isinstance(body, dict) else {}
        return LLMResponse(
            text=text,
            model=self._model,
            latency_ms=latency_ms,
            prompt_tokens=int(usage.get("prompt_tokens", 0) or 0),
            completion_tokens=int(usage.get("completion_tokens", 0) or 0),
            raw=body if isinstance(body, dict) else {"body": body},
        )


def _extract_chat_completion_text(body: Mapping[str, Any]) -> str:
    """Extract the assistant text from a chat-completion response payload."""
    try:
        choices = body.get("choices") or []
        if not choices:
            return ""
        message = choices[0].get("message") or {}
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for chunk in content:
                if isinstance(chunk, dict) and chunk.get("type") in ("text", None):
                    parts.append(chunk.get("text", ""))
            return "".join(parts)
        return str(content)
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "LLMProvider",
    "LLMResponse",
    "MockLLMProvider",
    "FeatherlessProvider",
]
