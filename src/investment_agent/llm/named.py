"""Three named Featherless specialists + the orchestrator-side adapter.

WHAT
====
Implements the three production LLM specialists called out in the hackathon
spec:

* ``DeepHermesAgent``      — reasoning / market synthesis
* ``FinanceLlamaAgent``    — financial / fundamental
* ``QwenTradingAgent``     — execution-context

Each agent receives the same compact state snapshot but uses a different
system prompt, temperature, and token budget. The deterministic risk
layer (capital gate) is the hard authority for every output.

WHY
====
- Different model families contribute genuinely different information
  (reasoning vs fundamentals vs execution). One model can't substitute.
- LLM output is forced through ``AgentLLMAdapter`` into the canonical
  eight-channel ``AgentOutput`` so the ensemble downstream is
  model-agnostic.

HOW
====
- Each ``NamedSpecialist`` wraps a ``FeatherlessOrchestrator`` (or any
  ``LLMProvider``) and a role-specific ``AgentRole`` (system prompt +
  user template).
- The user template renders the compact snapshot, not the raw data.
- The role's system prompt contains the explicit cap on absolute signal
  magnitude, mirroring the per-agent caps in
  ``agents/specialist.py``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .adapter import AgentLLMAdapter
from .base import LLMProvider, LLMResponse
from .snapshot import build_snapshot
from ..agents.specialist import AgentRole, AgentContext
from ..signals.ensemble_signal import AgentOutput


# ---------------------------------------------------------------------------
# Roles for the three named specialists
# ---------------------------------------------------------------------------

DEEPHERMES_ROLE = AgentRole(
    agent_id="agent_deephermes_reasoning",
    domain="water",
    description="Reasoning / market-synthesis specialist. Identifies relationships and contradictions across the supplied evidence.",
    system_prompt=(
        "You are the *Reasoning / Market-Synthesis Specialist* in a multi-agent "
        "trading system. You reason across macro, fundamental, and microstructure "
        "evidence and surface relationships, contradictions, or regime shifts. "
        "You do NOT pick position sizes; the deterministic risk layer does that. "
        "Cap absolute signal at 0.5. Be conservative; if evidence is mixed, signal "
        "should be near zero. "
        "Return valid JSON only. Keep rationale under 80 words. Do not repeat input data."
    ),
    user_template=(
        "Role: Reasoning / market-synthesis specialist (DeepHermes).\n"
        "Snapshot (compact state):\n```json\n{snapshot}\n```\n"
        "Most relevant prior trades:\n```json\n{memory}\n```\n"
        "Peer agent signals in this cycle:\n```json\n{peer_agents}\n```\n"
        "Return the eight-channel JSON with a one-sentence rationale embedded in the "
        "'noise' field as a short string when uncertain."
    ),
)

FINANCE_LLAMA_ROLE = AgentRole(
    agent_id="agent_finance_llama",
    domain="earth",
    description="Financial / fundamental specialist. Conservative, valuation-aware.",
    system_prompt=(
        "You are the *Financial / Fundamental Specialist*. You reason about "
        "valuation, earnings, credit, and intrinsic value. Your horizon is long. "
        "Cap absolute signal at 0.4. Be the most conservative of the three specialists. "
        "Return valid JSON only. Keep rationale under 60 words. Do not repeat input data."
    ),
    user_template=(
        "Role: Financial / fundamental specialist (Finance-Llama).\n"
        "Snapshot:\n```json\n{snapshot}\n```\n"
        "Most relevant prior trades:\n```json\n{memory}\n```\n"
        "Return the eight-channel JSON. Emphasize p_plus / p_minus and prefer "
        "high confidence only when valuation clearly supports it."
    ),
)

QWEN_TRADING_ROLE = AgentRole(
    agent_id="agent_qwen_trading",
    domain="fire",
    description="Trading opportunity / execution-context specialist.",
    system_prompt=(
        "You are the *Trading Opportunity / Execution-Context Specialist*. You "
        "evaluate whether the current evidence represents a meaningful, executable "
        "opportunity. You do NOT override the risk gate. Cap absolute signal at 0.6. "
        "If risk_flags is non-empty you SHOULD return low confidence. "
        "Return valid JSON only. Keep rationale under 80 words. Do not repeat input data."
    ),
    user_template=(
        "Role: Trading opportunity / execution-context specialist (Qwen).\n"
        "Snapshot:\n```json\n{snapshot}\n```\n"
        "Most relevant prior trades:\n```json\n{memory}\n```\n"
        "Peer agent signals in this cycle:\n```json\n{peer_agents}\n```\n"
        "Return the eight-channel JSON. If risk_flags non-empty, drop confidence below 0.4."
    ),
)


NAMED_ROLES: Tuple[AgentRole, ...] = (
    DEEPHERMES_ROLE,
    FINANCE_LLAMA_ROLE,
    QWEN_TRADING_ROLE,
)


# ---------------------------------------------------------------------------
# Adapter wrapper
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class NamedSpecialist:
    """Single named Featherless specialist."""

    role: AgentRole
    provider: LLMProvider
    adapter: AgentLLMAdapter = field(init=False)
    temperature: float = 0.15
    max_tokens: int = 700

    def __post_init__(self) -> None:
        # ``frozen=True`` dataclasses use object.__setattr__ in __post_init__.
        object.__setattr__(
            self,
            "adapter",
            AgentLLMAdapter(
                provider=self.provider,
                agent_id=self.role.agent_id,
                fallback_signal=0.0,
                fallback_confidence=0.25,
            ),
        )

    def run(
        self,
        snapshot: Dict[str, Any],
        memory: Optional[List[Any]] = None,
        peer_agents: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> Tuple[AgentOutput, LLMResponse]:
        """Run the specialist and return ``(AgentOutput, LLMResponse)``."""
        prompt = self._build_prompt(snapshot, memory or [], peer_agents or {})
        output, response = self.adapter.call(
            prompt,
            system=self.role.system_prompt,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return output, response

    def _build_prompt(
        self,
        snapshot: Dict[str, Any],
        memory: List[Any],
        peer_agents: Dict[str, Dict[str, float]],
    ) -> str:
        memory_block = json.dumps(
            [
                {
                    "decision_id": getattr(s, "experience", s).decision_id,
                    "regime": getattr(s, "experience", s).regime,
                    "action": getattr(s, "experience", s).position_action,
                    "pnl": getattr(s, "experience", s).pnl,
                    "similarity": getattr(s, "similarity_score", 0.0),
                }
                for s in memory[:3]
            ],
            default=str,
        )
        peer_block = json.dumps(peer_agents, default=str)
        return self.role.user_template.format(
            snapshot=json.dumps(snapshot, default=str),
            memory=memory_block,
            peer_agents=peer_block,
        )


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

def build_named_specialists(
    provider: LLMProvider,
    *,
    roles: Tuple[AgentRole, ...] = NAMED_ROLES,
    temperature_overrides: Optional[Dict[str, float]] = None,
    max_tokens_overrides: Optional[Dict[str, int]] = None,
) -> Dict[str, NamedSpecialist]:
    """Build the three named Featherless specialists."""
    temperature_overrides = temperature_overrides or {}
    max_tokens_overrides = max_tokens_overrides or {}
    specialists: Dict[str, NamedSpecialist] = {}
    for role in roles:
        spec_temp = 0.15
        spec_max = 700
        if role.agent_id == "agent_deephermes_reasoning":
            spec_temp, spec_max = 0.22, 850
        elif role.agent_id == "agent_finance_llama":
            spec_temp, spec_max = 0.12, 600
        elif role.agent_id == "agent_qwen_trading":
            spec_temp, spec_max = 0.15, 750
        t = temperature_overrides.get(role.agent_id, spec_temp)
        m = max_tokens_overrides.get(role.agent_id, spec_max)
        specialists[role.agent_id] = NamedSpecialist(
            role=role,
            provider=provider,
            temperature=t,
            max_tokens=m,
        )
    return specialists


def run_named_specialists(
    specialists: Dict[str, NamedSpecialist],
    snapshot: Dict[str, Any],
    *,
    memory: Optional[List[Any]] = None,
    peer_agents: Optional[Dict[str, Dict[str, float]]] = None,
) -> Dict[str, AgentOutput]:
    """Run every named specialist and return ``{agent_id: AgentOutput}``."""
    out: Dict[str, AgentOutput] = {}
    for agent_id, sp in specialists.items():
        try:
            output, _ = sp.run(snapshot, memory=memory, peer_agents=peer_agents)
        except Exception:
            output = AgentOutput(
                s=0.0, c=0.25, u=0.75, d=0.5,
                p_plus=0.5, p_minus=0.5, delta_t=1.0, r=0.5,
                agent_id=agent_id,
            )
        out[agent_id] = output
    return out


__all__ = [
    "DEEPHERMES_ROLE",
    "FINANCE_LLAMA_ROLE",
    "QWEN_TRADING_ROLE",
    "NAMED_ROLES",
    "NamedSpecialist",
    "build_named_specialists",
    "run_named_specialists",
]
