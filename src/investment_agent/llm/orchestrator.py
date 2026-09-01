"""Featherless multi-key orchestrator with reserve failover.

WHAT
====
Routes a single LLM call to one of four Featherless accounts (3 active +
1 reserve). On a transport or schema error, falls back to the next provider
in priority order, then to a deterministic low-confidence ``AgentOutput``.

WHY
====
- Hedge against a single provider outage / rate-limit during the hackathon.
- Log per-call token usage, latency, and success so the $25 budget can be
  audited from the JSONL usage log.
- Keep the LLM call contract identical to the rest of the system so
  failures degrade gracefully instead of crashing the deterministic
  pipeline.

HOW
====
- ``FeatherlessOrchestrator`` wraps a list of ``FeatherlessProvider``
  instances plus a shared ``UsageLog``.
- ``complete_with_failover(prompt, ...)`` tries providers in order, applying
  a timeout per call and ``retries_per_provider`` retries. On success it
  returns ``(LLMResponse, provider_id)``.
- The reserve provider is only consulted if every active provider fails
  AND ``use_reserve_on_failure=True``.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .base import FeatherlessProvider, LLMProvider, LLMResponse


# ---------------------------------------------------------------------------
# Usage log
# ---------------------------------------------------------------------------

USAGE_LOG_FILE = "llm_usage.jsonl"


@dataclass
class UsageRecord:
    """One LLM call's usage record (also serialized to JSONL)."""

    timestamp: str
    provider_id: str
    model: str
    success: bool
    latency_ms: float
    prompt_tokens: int
    completion_tokens: int
    error: str = ""


class UsageLog:
    """Append-only JSONL usage log for the $25 budget tracker."""

    def __init__(self, log_file: str = USAGE_LOG_FILE) -> None:
        self._log_file = log_file

    def record(
        self,
        provider_id: str,
        model: str,
        success: bool,
        latency_ms: float,
        prompt_tokens: int,
        completion_tokens: int,
        error: str = "",
    ) -> None:
        rec = UsageRecord(
            timestamp=datetime.now().isoformat(),
            provider_id=provider_id,
            model=model,
            success=success,
            latency_ms=float(latency_ms),
            prompt_tokens=int(prompt_tokens),
            completion_tokens=int(completion_tokens),
            error=error,
        )
        try:
            line = json.dumps(rec.__dict__) + "\n"
            with open(self._log_file, "a", encoding="utf-8") as f:
                f.write(line)
        except OSError:
            pass

    def total_tokens(self) -> int:
        total = 0
        if not os.path.exists(self._log_file):
            return total
        try:
            with open(self._log_file, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                        total += int(rec.get("prompt_tokens", 0)) + int(
                            rec.get("completion_tokens", 0)
                        )
                    except json.JSONDecodeError:
                        continue
        except OSError:
            pass
        return total


# ---------------------------------------------------------------------------
# Provider descriptor
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ProviderSpec:
    """Specification for one Featherless account / model pair."""

    provider_id: str
    api_key: Optional[str]
    model: str
    temperature: float
    max_tokens: int
    role: str
    is_reserve: bool = False


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class FeatherlessOrchestrator:
    """Multi-key Featherless orchestrator with reserve failover.

    Provider priority:
        1. active providers in declared order
        2. reserve provider (only if all active fail AND use_reserve_on_failure)
    """

    def __init__(
        self,
        specs: Sequence[ProviderSpec],
        *,
        base_url: str = FeatherlessProvider.DEFAULT_BASE_URL,
        timeout_s: float = 30.0,
        retries_per_provider: int = 1,
        use_reserve_on_failure: bool = True,
        usage_log: Optional[UsageLog] = None,
    ) -> None:
        if not specs:
            raise ValueError("specs must be non-empty")

        self._specs: List[ProviderSpec] = list(specs)
        self._timeout_s = float(timeout_s)
        self._retries_per_provider = max(1, int(retries_per_provider))
        self._use_reserve_on_failure = bool(use_reserve_on_failure)
        self._usage = usage_log or UsageLog()

        self._providers: Dict[str, FeatherlessProvider] = {}
        for spec in self._specs:
            if not spec.api_key:
                continue
            self._providers[spec.provider_id] = FeatherlessProvider(
                api_key=spec.api_key,
                model=spec.model,
                base_url=base_url,
                timeout_s=timeout_s,
            )

    @property
    def active_provider_ids(self) -> List[str]:
        return [s.provider_id for s in self._specs if not s.is_reserve and s.provider_id in self._providers]

    @property
    def reserve_provider_ids(self) -> List[str]:
        return [s.provider_id for s in self._specs if s.is_reserve and s.provider_id in self._providers]

    def get_spec(self, provider_id: str) -> Optional[ProviderSpec]:
        for s in self._specs:
            if s.provider_id == provider_id:
                return s
        return None

    def complete(
        self,
        prompt: str,
        *,
        provider_id: Optional[str] = None,
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        timeout_s: Optional[float] = None,
    ) -> LLMResponse:
        """Call the LLM with failover; returns a real ``LLMResponse`` or raises.

        Raises
        ------
        RuntimeError
            If every available provider (including reserve, when enabled)
            fails or has no API key.
        """
        order = self._provider_order(provider_id)
        last_error: Optional[Exception] = None
        for pid in order:
            spec = self.get_spec(pid)
            provider = self._providers[pid]
            if spec is None:
                continue
            t = float(temperature) if temperature is not None else spec.temperature
            m = int(max_tokens) if max_tokens is not None else spec.max_tokens
            for _ in range(self._retries_per_provider):
                start = time.perf_counter()
                try:
                    response = provider.complete(
                        prompt,
                        system=system,
                        temperature=t,
                        max_tokens=m,
                        timeout_s=float(timeout_s) if timeout_s is not None else self._timeout_s,
                    )
                    latency_ms = (time.perf_counter() - start) * 1000.0
                    self._usage.record(
                        provider_id=pid,
                        model=response.model,
                        success=True,
                        latency_ms=latency_ms,
                        prompt_tokens=response.prompt_tokens,
                        completion_tokens=response.completion_tokens,
                    )
                    return response
                except Exception as exc:
                    last_error = exc
                    latency_ms = (time.perf_counter() - start) * 1000.0
                    self._usage.record(
                        provider_id=pid,
                        model=spec.model,
                        success=False,
                        latency_ms=latency_ms,
                        prompt_tokens=0,
                        completion_tokens=0,
                        error=str(exc),
                    )
        raise RuntimeError(
            f"FeatherlessOrchestrator: all providers failed "
            f"(order={order}, last_error={last_error!r})"
        )

    def _provider_order(self, provider_id: Optional[str]) -> List[str]:
        active = [s.provider_id for s in self._specs
                  if not s.is_reserve and s.provider_id in self._providers]
        reserve = [s.provider_id for s in self._specs
                   if s.is_reserve and s.provider_id in self._providers]
        if provider_id is not None and provider_id in self._providers:
            if provider_id in active:
                return [provider_id] + [p for p in active if p != provider_id] + (
                    reserve if self._use_reserve_on_failure else []
                )
            if provider_id in reserve:
                return active + [provider_id]
        return active + (reserve if self._use_reserve_on_failure else [])


# ---------------------------------------------------------------------------
# Config loader (reads env vars and config/llm_keys.json)
# ---------------------------------------------------------------------------

DEFAULT_KEYS_FILE = "config/llm_keys.json"


def load_provider_specs(keys_file: str = DEFAULT_KEYS_FILE) -> List[ProviderSpec]:
    """Load the four-provider spec list from env vars + config file.

    Resolution order for each provider's API key:
        1. Direct env var named ``spec.api_key_env``
        2. ``config/llm_keys.json`` ``providers.<id>.api_key`` field
    """
    specs: List[ProviderSpec] = []

    cfg: Dict[str, Any] = {}
    if os.path.exists(keys_file):
        try:
            with open(keys_file, "r", encoding="utf-8") as f:
                cfg = json.load(f)
        except (OSError, json.JSONDecodeError):
            cfg = {}

    for provider_id, spec in (cfg.get("providers") or {}).items():
        api_key_env = spec.get("api_key_env") or f"FEATHERLESS_{provider_id.upper()}_KEY"
        api_key = os.getenv(api_key_env) or spec.get("api_key")
        specs.append(ProviderSpec(
            provider_id=provider_id,
            api_key=api_key,
            model=spec.get("model", FeatherlessProvider.DEFAULT_MODEL),
            temperature=float(spec.get("temperature", 0.15)),
            max_tokens=int(spec.get("max_tokens", 700)),
            role=spec.get("role", "unknown"),
            is_reserve=bool(spec.get("enabled") is False or spec.get("role") == "failover"),
        ))

    if not specs:
        # Fall back to a single provider using a generic env var, so tests
        # and the example config still work without the JSON file.
        specs.append(ProviderSpec(
            provider_id="default",
            api_key=os.getenv("FEATHERLESS_API_KEY"),
            model=os.getenv("FEATHERLESS_MODEL", FeatherlessProvider.DEFAULT_MODEL),
            temperature=0.15,
            max_tokens=700,
            role="default",
            is_reserve=False,
        ))

    return specs


__all__ = [
    "FeatherlessOrchestrator",
    "ProviderSpec",
    "UsageLog",
    "UsageRecord",
    "load_provider_specs",
]
