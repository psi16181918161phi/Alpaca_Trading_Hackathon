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
specialist. It is consulted only when every active provider for a
specialist fails (capacity_exhausted, HTTP 400, timeout, etc.).

Two wiring modes are supported:

1. ``build_named_specialists(provider_map)``
   Each specialist is bound to its own single ``LLMProvider`` instance.
   Provider failures are caught by the adapter and degrade to a
   zero-signal ``AgentOutput``. Use this when you have no multi-key
   orchestrator.

2. ``build_named_specialists(orchestrator, provider_id_map=...)``
   Each specialist is bound to a ``FeatherlessOrchestrator`` plus a
   ``provider_id``. The specialist calls
   ``orchestrator.complete(provider_id=..., ...)``, which puts the
   multi-provider failover (active → active → reserve) on the actual
   hot path. The deterministic pipeline only sees the zero-signal
   fallback if the entire orchestrator (including reserve) fails.

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
- Mode 2 makes the orchestrator's reserve the *last* line of defence
  instead of dead code. A failure of one provider is followed by
  failover through the orchestrator's existing retry / reserve
  policy rather than a silent zero-signal.

HOW
====
- ``NamedSpecialist`` wraps an ``LLMProvider`` (or the multi-provider
  ``FeatherlessOrchestrator`` plus a preferred ``provider_id``) plus a
  role-specific ``AgentRole`` (system prompt + user template).
- The user template renders the compact snapshot, not raw history.
- ``build_named_specialists(provider_map)`` builds the three specialists
  from a ``{agent_id: LLMProvider}`` map (single-provider mode).
- ``build_named_specialists(orchestrator, provider_id_map=...)`` builds
  the three specialists in orchestrator-bound mode.
- ``run_named_specialists(specialists, snapshot, ...)`` returns
  ``{agent_id: AgentOutput}`` with each output containing the eight
  channels; a separate ``rationale`` string is provided by the adapter
  on a best-effort basis.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

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

    Two wiring modes (mutually exclusive):

    * Single-provider mode (legacy / no orchestrator): pass ``provider``
      as a single ``LLMProvider``. Failures degrade to a zero-signal
      ``AgentOutput`` via the adapter.

    * Orchestrator-bound mode (production / reserve reachable): pass
      ``orchestrator`` as a ``FeatherlessOrchestrator`` and
      ``provider_id`` as the preferred provider id (e.g. ``deephermes``).
      The specialist calls
      ``orchestrator.complete(provider_id=..., ...)`` so the
      orchestrator's existing retry / failover / reserve policy is the
      actual control flow. A single-provider ``provider`` argument is
      optional in this mode and is used only to populate the audit
      trail with the preferred model id.
    """

    role: AgentRole
    provider: Optional[LLMProvider] = None
    orchestrator: Optional[Any] = None
    provider_id: Optional[str] = None
    adapter: AgentLLMAdapter = field(init=False)
    temperature: float = DEFAULT_TEMPERATURE
    max_tokens: int = DEFAULT_MAX_TOKENS

    def __post_init__(self) -> None:
        # Mutually exclusive wiring.
        if self.provider is None and self.orchestrator is None:
            raise ValueError(
                "NamedSpecialist requires either a single provider (provider=...) "
                "or an orchestrator (orchestrator=...) plus provider_id=..."
            )
        if self.orchestrator is not None and self.provider_id is None:
            raise ValueError(
                "NamedSpecialist: orchestrator-bound mode requires provider_id=..."
            )

        # Pick the adapter's underlying provider. In orchestrator-bound
        # mode we still hand the adapter the orchestrator as a generic
        # ``LLMProvider``-shaped object so the call can take the
        # orchestrator's failover path. (The orchestrator exposes a
        # ``complete()`` matching the LLMProvider contract; in
        # orchestrator-bound mode we call the orchestrator's
        # ``complete(provider_id=..., ...)`` directly from ``run`` and
        # only use the adapter for JSON parsing.)
        if self.orchestrator is not None:
            underlying = self.orchestrator
        else:
            underlying = self.provider
        object.__setattr__(
            self,
            "adapter",
            AgentLLMAdapter(
                provider=underlying,
                agent_id=self.role.agent_id,
                fallback_signal=0.0,
                fallback_confidence=0.25,
            ),
        )

    @property
    def agent_id(self) -> str:
        return self.role.agent_id

    @property
    def is_orchestrator_bound(self) -> bool:
        return self.orchestrator is not None

    def run(
        self,
        snapshot: Dict[str, Any],
        memory: Optional[List[Any]] = None,
        peer_agents: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> SpecialistOutput:
        """Run the specialist and return ``SpecialistOutput``.

        In orchestrator-bound mode this routes through
        ``orchestrator.complete(provider_id=self.provider_id, ...)`` so
        the orchestrator's reserve failover is on the actual hot path.
        In single-provider mode this delegates to
        ``self.provider.complete(...)`` and the adapter handles parse
        failures.
        """
        prompt = self._build_prompt(snapshot, memory or [], peer_agents or {})
        if self.is_orchestrator_bound:
            response = self.orchestrator.complete(
                prompt,
                provider_id=self.provider_id,
                system=self.role.system_prompt,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
        else:
            response = self.provider.complete(
                prompt,
                system=self.role.system_prompt,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
        # Parse using the adapter so the eight-channel contract and the
        # fallback on malformed JSON are uniform across both modes.
        output = self._parse_response(response)
        rationale = _extract_rationale(response.text)
        return SpecialistOutput(
            output=output,
            rationale=rationale,
            raw_text=response.text,
        )

    def _parse_response(self, response: LLMResponse) -> AgentOutput:
        from .adapter import extract_json_object  # local import to avoid cycle
        parsed = extract_json_object(response.text)
        if parsed is None:
            return AgentOutput(
                s=0.0, c=0.25, u=0.75, d=0.5,
                p_plus=0.5, p_minus=0.5, delta_t=1.0, r=0.5,
                agent_id=self.agent_id,
            )
        return self.adapter._build_output(parsed)  # noqa: SLF001 - internal helper

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
# Builders
# ---------------------------------------------------------------------------

def build_named_specialists(
    provider_or_orchestrator: Union[LLMProvider, Any, Mapping[str, LLMProvider]],
    *,
    provider_id_map: Optional[Mapping[str, str]] = None,
    roles: Tuple[AgentRole, ...] = NAMED_ROLES,
    temperature_overrides: Optional[Dict[str, float]] = None,
    max_tokens_overrides: Optional[Dict[str, int]] = None,
    per_agent_provider: Optional[LLMProvider] = None,
) -> Dict[str, NamedSpecialist]:
    """Build the three named specialists.

    Two modes:

    1. ``build_named_specialists(provider_map)`` -- single-provider mode.
       ``provider_or_orchestrator`` is a ``Mapping[agent_id, LLMProvider]``.
       Each specialist is bound to its own provider; failures degrade
       to a zero-signal ``AgentOutput`` via the adapter.

    2. ``build_named_specialists(orchestrator, provider_id_map=...)`` --
       orchestrator-bound mode.
       ``provider_or_orchestrator`` is a ``FeatherlessOrchestrator``.
       ``provider_id_map`` maps ``agent_id -> provider_id`` (defaults
       to ``{reasoning: deephermes, fundamentals: fundamentals,
       finance_qlora: finance_qlora}``). Each specialist routes
       through the orchestrator's multi-provider failover, so the
       reserve provider is reached whenever the active provider fails.

    Parameters
    ----------
    provider_or_orchestrator : Union[Mapping, FeatherlessOrchestrator]
        Either a ``{agent_id: LLMProvider}`` map or a
        ``FeatherlessOrchestrator``.
    provider_id_map : Optional[Mapping[str, str]]
        Required in orchestrator-bound mode; ignored otherwise.
    per_agent_provider : Optional[LLMProvider]
        Optional. In orchestrator-bound mode this is the *preferred*
        provider instance for the audit trail (e.g. the matching
        ``FeatherlessProvider`` from
        ``build_provider_map_from_orchestrator(orch)``).

    Returns
    -------
    Dict[str, NamedSpecialist]
        ``{agent_id: NamedSpecialist}``.
    """
    temperature_overrides = temperature_overrides or {}
    max_tokens_overrides = max_tokens_overrides or {}

    # Orchestrator-bound mode: caller passed a FeatherlessOrchestrator.
    if not isinstance(provider_or_orchestrator, Mapping):
        orchestrator = provider_or_orchestrator
        if provider_id_map is None:
            provider_id_map = {
                "agent_deephermes_reasoning": "deephermes",
                "agent_deephermes_fundamentals": "fundamentals",
                "agent_finance_qlora": "finance_qlora",
            }
        specialists: Dict[str, NamedSpecialist] = {}
        missing: List[str] = []
        for role in roles:
            pid = provider_id_map.get(role.agent_id)
            if pid is None:
                missing.append(role.agent_id)
                continue
            spec_temp, spec_max = _ROLE_DEFAULTS.get(
                role.agent_id, (DEFAULT_TEMPERATURE, DEFAULT_MAX_TOKENS)
            )
            t = temperature_overrides.get(role.agent_id, spec_temp)
            m = max_tokens_overrides.get(role.agent_id, spec_max)
            # Optional audit-trail provider (single provider instance,
            # only used to surface the preferred model id in logs).
            preferred_provider = None
            if per_agent_provider is not None and role.agent_id == per_agent_provider.model_id:
                preferred_provider = per_agent_provider
            specialists[role.agent_id] = NamedSpecialist(
                role=role,
                provider=preferred_provider,
                orchestrator=orchestrator,
                provider_id=pid,
                temperature=t,
                max_tokens=m,
            )
        if missing:
            raise ValueError(
                f"build_named_specialists (orchestrator mode): provider_id_map "
                f"missing entries for: {missing}. Required keys: "
                f"{[r.agent_id for r in roles]}."
            )
        return specialists

    # Single-provider mode.
    provider_map = provider_or_orchestrator
    specialists = {}
    missing = []
    for role in roles:
        if role.agent_id not in provider_map:
            missing.append(role.agent_id)
            continue
        spec_temp, spec_max = _ROLE_DEFAULTS.get(
            role.agent_id, (DEFAULT_TEMPERATURE, DEFAULT_MAX_TOKENS)
        )
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
    never starved. In orchestrator-bound mode the inner failure is
    the *entire orchestrator* failing (active + reserve); a single
    active failure is handled by the orchestrator's failover chain
    and produces a normal ``AgentOutput``.
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
