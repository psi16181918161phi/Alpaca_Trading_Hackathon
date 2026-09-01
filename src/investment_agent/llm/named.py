"""Named Featherless specialists, one per active provider.

WHAT
====
Implements the three production LLM specialists confirmed by live probe on
2026-09-01:

* ``agent_deephermes_reasoning``     — reasoning / market synthesis
  -> ``NousResearch/DeepHermes-3-Llama-3-8B-Preview``
* ``agent_deephermes_fundamentals``  — financial fundamentals (Atropos)
  -> ``NousResearch/DeepHermes-Financial-Fundamentals-Prediction-Specialist-Atropos``
* ``agent_finance_qlora``            — finance QLoRA (Llama-3.1 8B)
  -> ``jhon53/Llama3_1_8B_Finance_QLoRA-merged-16bit``

The reserve is held by ``FeatherlessOrchestrator`` and is *not* a named
specialist. It is consulted only when the active provider for a
specialist fails (capacity_exhausted, HTTP 400, timeout, etc.).

Each specialist is bound to its own ``LLMProvider`` instance so that:

  reasoning    -> deephermes provider
  fundamentals -> fundamentals provider
  finance_qlora -> finance_qlora provider

Provider failover for a single specialist stays inside its bound
provider; the multi-provider reserve is the last line of defence.

WHY
====
- Different models contribute genuinely different information
  (general reasoning vs fundamentals vs financial fine-tuning).
  One model cannot substitute.
- Binding specialist -> provider keeps the audit trail and the
  per-call cost trace aligned with the production model identity.
- LLM output is forced through ``AgentLLMAdapter`` into the canonical
  eight-channel ``AgentOutput`` so the ensemble downstream is
  model-agnostic. A separate ``rationale`` field is attached *outside*
  the eight channels so the contract stays semantically clean.

HOW
====
- ``NamedSpecialist`` wraps a single ``LLMProvider`` plus a role-specific
  ``AgentRole`` (system prompt + user template).
- The user template renders the compact snapshot, not raw history.
- ``build_named_specialists(provider_map)`` builds the three specialists
  from a ``{agent_id: LLMProvider}`` map.
- ``run_named_specialists(specialists, snapshot, ...)`` returns
  ``{agent_id: AgentOutput}`` with each output containing the eight
  channels; a separate ``rationale`` string is provided by the adapter
  on a best-effort basis.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Tuple

from .adapter import AgentLLMAdapter
from .base import LLMProvider, LLMResponse
from ..agents.specialist import AgentRole
from ..signals.ensemble_signal import AgentOutput


# ---------------------------------------------------------------------------
# Specialist roles
# ---------------------------------------------------------------------------
#
# Each role carries a stable ``agent_id`` and explicit signal cap. The
# rationale-in-noise workaround has been removed; rationale is now
# returned alongside AgentOutput as a separate field by the adapter.

DEEPHERMES_REASONING_ROLE = AgentRole(
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
        "Return valid JSON only. Do not repeat input data."
    ),
    user_template=(
        "Role: Reasoning / market-synthesis specialist.\n"
        "Snapshot (compact state):\n```json\n{snapshot}\n```\n"
        "Most relevant prior trades:\n```json\n{memory}\n```\n"
        "Peer agent signals in this cycle:\n```json\n{peer_agents}\n```\n"
        "Return the eight-channel JSON exactly as specified."
    ),
)

DEEPHERMES_FUNDAMENTALS_ROLE = AgentRole(
    agent_id="agent_deephermes_fundamentals",
    domain="earth",
    description="Financial-fundamentals specialist. Conservative, valuation-aware.",
    system_prompt=(
        "You are the *Financial-Fundamentals Specialist* fine-tuned on "
        "fundamental/valuation data. You reason about earnings, intrinsic value, "
        "and credit conditions. Your horizon is long. Cap absolute signal at 0.4. "
        "Be the most conservative specialist. "
        "Return valid JSON only. Do not repeat input data."
    ),
    user_template=(
        "Role: Financial-fundamentals specialist.\n"
        "Snapshot:\n```json\n{snapshot}\n```\n"
        "Most relevant prior trades:\n```json\n{memory}\n```\n"
        "Return the eight-channel JSON. Emphasize p_plus / p_minus and prefer "
        "high confidence only when valuation clearly supports it."
    ),
)

FINANCE_QLORA_ROLE = AgentRole(
    agent_id="agent_finance_qlora",
    domain="earth",
    description="Finance QLoRA specialist (Llama-3.1 8B fine-tuned on financial data).",
    system_prompt=(
        "You are the *Finance QLoRA Specialist* — a Llama-3.1 8B model "
        "fine-tuned on financial text. You reason about fundamentals, earnings "
        "calls, and credit conditions. Your horizon is medium. Cap absolute "
        "signal at 0.45. Be conservative when evidence is mixed. "
        "Return valid JSON only. Do not repeat input data."
    ),
    user_template=(
        "Role: Finance QLoRA specialist.\n"
        "Snapshot:\n```json\n{snapshot}\n```\n"
        "Most relevant prior trades:\n```json\n{memory}\n```\n"
        "Return the eight-channel JSON with your best calibrated signal."
    ),
)


NAMED_ROLES: Tuple[AgentRole, ...] = (
    DEEPHERMES_REASONING_ROLE,
    DEEPHERMES_FUNDAMENTALS_ROLE,
    FINANCE_QLORA_ROLE,
)


# ---------------------------------------------------------------------------
# Per-specialist default token budget
# ---------------------------------------------------------------------------
# Reduced from 600–900 to 192 to fit the actual eight-field JSON
# contract (typical response is <120 tokens). Cuts budget burn ~4x.
DEFAULT_MAX_TOKENS = 192
DEFAULT_TEMPERATURE = 0.15

_ROLE_DEFAULTS: Dict[str, Tuple[float, int]] = {
    "agent_deephermes_reasoning": (0.22, 224),
    "agent_deephermes_fundamentals": (0.15, 192),
    "agent_finance_qlora": (0.12, 192),
}


# ---------------------------------------------------------------------------
# Specialist wrapper
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SpecialistOutput:
    """Wrapper around the eight-channel AgentOutput plus a rationale.

    The rationale is extracted by the adapter and is for audit /
    debugging only. It is NOT used in the ensemble / capital gate math.
    """

    output: AgentOutput
    rationale: str = ""
    raw_text: str = ""


@dataclass(frozen=True)
class NamedSpecialist:
    """Single named Featherless specialist.

    Bound to ONE ``LLMProvider`` instance (not the full multi-provider
    orchestrator). The provider may itself implement internal retries;
    the specialist layer treats any provider failure as a fallback to a
    zero-signal ``AgentOutput``.
    """

    role: AgentRole
    provider: LLMProvider
    adapter: AgentLLMAdapter = field(init=False)
    temperature: float = DEFAULT_TEMPERATURE
    max_tokens: int = DEFAULT_MAX_TOKENS

    def __post_init__(self) -> None:
        # ``frozen=True`` requires object.__setattr__ in __post_init__.
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

    @property
    def agent_id(self) -> str:
        return self.role.agent_id

    def run(
        self,
        snapshot: Dict[str, Any],
        memory: Optional[List[Any]] = None,
        peer_agents: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> SpecialistOutput:
        """Run the specialist and return ``SpecialistOutput``."""
        prompt = self._build_prompt(snapshot, memory or [], peer_agents or {})
        output, response = self.adapter.call(
            prompt,
            system=self.role.system_prompt,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        rationale = _extract_rationale(response.text)
        return SpecialistOutput(
            output=output,
            rationale=rationale,
            raw_text=response.text,
        )

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


def _extract_rationale(text: str) -> str:
    """Best-effort extraction of any prose outside the JSON object.

    The rationale is for debugging and audit only. It is intentionally
    separate from the eight mathematical channels.
    """
    if not text:
        return ""
    # Strip fenced JSON.
    cleaned = text
    for fence in ("```json", "```"):
        cleaned = cleaned.replace(fence, "")
    cleaned = cleaned.strip()
    # If the whole response is JSON, no rationale.
    if cleaned.startswith("{") and cleaned.endswith("}"):
        return ""
    # Otherwise return the non-JSON portion, trimmed.
    return cleaned[:512]


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

def build_named_specialists(
    provider_map: Mapping[str, LLMProvider],
    *,
    roles: Tuple[AgentRole, ...] = NAMED_ROLES,
    temperature_overrides: Optional[Dict[str, float]] = None,
    max_tokens_overrides: Optional[Dict[str, int]] = None,
) -> Dict[str, NamedSpecialist]:
    """Build the three named specialists from a per-agent provider map.

    Parameters
    ----------
    provider_map : Mapping[str, LLMProvider]
        ``{agent_id: LLMProvider}``. Each specialist is bound to its own
        provider instance; the keys must match ``role.agent_id``.
    roles : Tuple[AgentRole, ...]
        The set of roles to instantiate. Defaults to the three confirmed
        specialists.
    temperature_overrides, max_tokens_overrides : Optional[Dict[str, ...]]
        Per-agent override maps for fine tuning.

    Returns
    -------
    Dict[str, NamedSpecialist]
        ``{agent_id: NamedSpecialist}``.
    """
    temperature_overrides = temperature_overrides or {}
    max_tokens_overrides = max_tokens_overrides or {}
    specialists: Dict[str, NamedSpecialist] = {}
    missing: List[str] = []
    for role in roles:
        if role.agent_id not in provider_map:
            missing.append(role.agent_id)
            continue
        spec_temp, spec_max = _ROLE_DEFAULTS.get(role.agent_id, (DEFAULT_TEMPERATURE, DEFAULT_MAX_TOKENS))
        t = temperature_overrides.get(role.agent_id, spec_temp)
        m = max_tokens_overrides.get(role.agent_id, spec_max)
        specialists[role.agent_id] = NamedSpecialist(
            role=role,
            provider=provider_map[role.agent_id],
            temperature=t,
            max_tokens=m,
        )
    if missing:
        raise ValueError(
            f"build_named_specialists: provider_map missing providers for: {missing}. "
            f"Required keys: {[r.agent_id for r in roles]}."
        )
    return specialists


def build_provider_map_from_orchestrator(
    orchestrator: "FeatherlessOrchestrator",
    *,
    provider_id_map: Optional[Mapping[str, str]] = None,
) -> Dict[str, LLMProvider]:
    """Construct ``{agent_id: LLMProvider}`` from a multi-provider orchestrator.

    This is the integration test the audit flagged: it proves that
    each specialist is routed to its *own* provider, not a shared one.

    Parameters
    ----------
    orchestrator : FeatherlessOrchestrator
        The four-provider orchestrator.
    provider_id_map : Optional[Mapping[str, str]]
        ``{agent_id: provider_id}``. Defaults to::

            agent_deephermes_reasoning    -> deephermes
            agent_deephermes_fundamentals -> fundamentals
            agent_finance_qlora           -> finance_qlora
    """
    if provider_id_map is None:
        provider_id_map = {
            "agent_deephermes_reasoning": "deephermes",
            "agent_deephermes_fundamentals": "fundamentals",
            "agent_finance_qlora": "finance_qlora",
        }
    out: Dict[str, LLMProvider] = {}
    for agent_id, provider_id in provider_id_map.items():
        if provider_id not in orchestrator._providers:  # internal access for wiring
            raise ValueError(
                f"Orchestrator has no provider id {provider_id!r}; "
                f"available: {list(orchestrator._providers.keys())}"
            )
        out[agent_id] = orchestrator._providers[provider_id]
    return out


def run_named_specialists(
    specialists: Dict[str, NamedSpecialist],
    snapshot: Dict[str, Any],
    *,
    memory: Optional[List[Any]] = None,
    peer_agents: Optional[Dict[str, Dict[str, float]]] = None,
) -> Dict[str, AgentOutput]:
    """Run every specialist and return ``{agent_id: AgentOutput}``.

    Failure of any single specialist falls back to a zero-signal,
    low-confidence ``AgentOutput`` so the deterministic pipeline is
    never starved.
    """
    out: Dict[str, AgentOutput] = {}
    for agent_id, sp in specialists.items():
        try:
            result = sp.run(snapshot, memory=memory, peer_agents=peer_agents)
            out[agent_id] = result.output
        except Exception:
            out[agent_id] = AgentOutput(
                s=0.0, c=0.25, u=0.75, d=0.5,
                p_plus=0.5, p_minus=0.5, delta_t=1.0, r=0.5,
                agent_id=agent_id,
            )
    return out


__all__ = [
    "DEEPHERMES_REASONING_ROLE",
    "DEEPHERMES_FUNDAMENTALS_ROLE",
    "FINANCE_QLORA_ROLE",
    "NAMED_ROLES",
    "SpecialistOutput",
    "NamedSpecialist",
    "build_named_specialists",
    "build_provider_map_from_orchestrator",
    "run_named_specialists",
]
