"""LLM → AgentOutput adapter.

WHAT
====
Parses the raw text returned by an ``LLMProvider`` into the canonical
``AgentOutput`` tuple consumed by the deterministic pipeline
(``ensemble_signal`` → ``investment_kalman_gain`` → ``capital_gate``).

WHY
====
Specialist agents are required to produce *structured* output, not free-form
prose. The adapter is the single place that:

- extracts the first JSON object found in the response;
- validates the eight channels ``(s, c, u, d, p_plus, p_minus, delta_t, r)``;
- clamps invalid values to safe bounds;
- falls back to a neutral output on any parsing failure so the risk layer
  is never starved.

HOW
====
- Prompt templates ask the model to emit a single JSON object delimited by
  a fenced ``json`` block; ``parse_agent_output`` finds the first ``{...}``
  region and ``json.loads`` it.
- Validation reuses the bounds enforced by ``AgentOutput.__post_init__``.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional

from .base import LLMProvider, LLMResponse
from ..signals.ensemble_signal import AgentOutput


# ---------------------------------------------------------------------------
# JSON extraction
# ---------------------------------------------------------------------------

_FENCED_JSON_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
_BARE_JSON_RE = re.compile(r"(\{[\s\S]*\})")


def extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    """Return the first JSON object found in ``text``, or None.

    Tries (in order):
    1. A fenced ```json ... ``` block.
    2. The first brace-balanced JSON object in the raw text.
    """
    if not text:
        return None
    m = _FENCED_JSON_RE.search(text)
    if m:
        try:
            obj = json.loads(m.group(1))
            return obj if isinstance(obj, dict) else None
        except json.JSONDecodeError:
            pass
    # Walk the string to find the first brace-balanced JSON object.
    for start in range(len(text)):
        if text[start] != "{":
            continue
        depth = 0
        for end in range(start, len(text)):
            ch = text[end]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start:end + 1]
                    try:
                        obj = json.loads(candidate)
                    except json.JSONDecodeError:
                        break
                    if isinstance(obj, dict):
                        return obj
                    break
    return None


# ---------------------------------------------------------------------------
# Field coercion
# ---------------------------------------------------------------------------

def _coerce_float(value: Any, default: float) -> float:
    if value is None:
        return default
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        f = float(value)
        if math.isnan(f) or math.isinf(f):
            return default
        return f
    if isinstance(value, str):
        try:
            f = float(value.strip())
        except ValueError:
            return default
        if math.isnan(f) or math.isinf(f):
            return default
        return f
    return default


def _clip(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AgentLLMAdapter:
    """Bridge between an LLM provider and the deterministic pipeline.

    Use ``adapter.call(prompt, system=...)`` to obtain a fully-validated
    ``AgentOutput`` plus the raw ``LLMResponse`` (for audit / provenance).
    """

    provider: LLMProvider
    agent_id: str
    fallback_signal: float = 0.0
    fallback_confidence: float = 0.25
    fallback_uncertainty: float = 0.75
    fallback_doubt: float = 0.5

    def call(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 256,
    ) -> tuple[AgentOutput, LLMResponse]:
        response = self.provider.complete(
            prompt,
            system=system,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        parsed = extract_json_object(response.text)
        if parsed is None:
            return self._fallback(response), response
        try:
            return self._build_output(parsed), response
        except (TypeError, ValueError):
            return self._fallback(response), response

    def _build_output(self, obj: Dict[str, Any]) -> AgentOutput:
        s = _clip(_coerce_float(obj.get("signal"), 0.0), -1.0, 1.0)
        c = _clip(_coerce_float(obj.get("confidence"), self.fallback_confidence), 0.01, 1.0)
        u = _clip(_coerce_float(obj.get("uncertainty"), self.fallback_uncertainty), 0.0, 1.0)
        d = _clip(_coerce_float(obj.get("doubt"), self.fallback_doubt), 0.0, 1.0)
        p_plus = _clip(_coerce_float(obj.get("p_plus"), 0.5), 0.0, 1.0)
        p_minus = _clip(_coerce_float(obj.get("p_minus"), 0.5), 0.0, 1.0)
        delta_t = _clip(_coerce_float(obj.get("delta_t"), 1.0), 0.01, 100.0)
        r = _clip(_coerce_float(obj.get("noise"), 0.1), 0.0, 100.0)
        return AgentOutput(
            s=s, c=c, u=u, d=d,
            p_plus=p_plus, p_minus=p_minus,
            delta_t=delta_t, r=r,
            agent_id=self.agent_id,
        )

    def _fallback(self, response: LLMResponse) -> AgentOutput:
        return AgentOutput(
            s=self.fallback_signal,
            c=self.fallback_confidence,
            u=self.fallback_uncertainty,
            d=self.fallback_doubt,
            p_plus=0.5,
            p_minus=0.5,
            delta_t=1.0,
            r=0.5,
            agent_id=self.agent_id,
        )


__all__ = [
    "AgentLLMAdapter",
    "extract_json_object",
]
