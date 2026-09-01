"""Seven specialist LLM agents for the X Quant X pipeline.

WHAT
====
Each agent owns a clearly-bounded analytical role and emits a fully-validated
``AgentOutput`` via an ``AgentLLMAdapter``. The seven roles correspond to the
four financial domains of the whitepaper (Earth, Air, Fire, Water) plus three
cross-domain specialists (regime, sentiment, risk-sentiment) so the ensemble
sees genuinely diverse signals.

WHY
====
- Specialization is the only way an LLM ensemble can beat a single model.
- The deterministic pipeline (ensemble → Kalman → capital gate) must remain
  the risk authority; the LLM layer only contributes to ``s`` and ``c`` for
  each agent.
- Memory and reputation are injected into the prompt so the LLM is
  *informed* by prior outcomes without ever being allowed to override the
  capital gate.

HOW
====
- ``build_specialist_agents(provider, *, memory=None, reputation=None)`` is
  the single entry point. It returns a dict ``{agent_id: SpecialistAgent}``.
- Each ``SpecialistAgent`` carries: role, model_hint, system prompt, and a
  structured user prompt template. Calling ``agent.run(context)`` returns
  ``(AgentOutput, LLMResponse)``.
- All agents share the same ``AgentOutput`` contract; the only difference
  is the prompt they receive.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..signals.ensemble_signal import AgentOutput
from ..memory.trade_memory import TradeMemory, SimilarExperience  # noqa: F401

# AgentLLMAdapter and LLMProvider are imported lazily inside
# ``build_specialist_agents`` to break the historical circular import:
# ``llm/__init__.py`` re-exports ``named.py`` which in turn imports
# ``AgentRole`` from this module. Eagerly importing ``from ..llm...``
# at module load triggers the cycle.


# ---------------------------------------------------------------------------
# Role definitions
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AgentRole:
    """Static role description for a specialist agent.

    Attributes
    ----------
    agent_id : str
        Stable identifier (e.g., "agent_economic").
    domain : str
        One of {"earth", "air", "fire", "water", "cross"}.
    description : str
        One-paragraph role description.
    system_prompt : str
        Behavioural system prompt sent to the LLM.
    user_template : str
        Jinja-style template for the user prompt. Variables: symbol, regime,
        features, agent_signals, memory.
    """

    agent_id: str
    domain: str
    description: str
    system_prompt: str
    user_template: str


# ---------------------------------------------------------------------------
# Prompt template helpers
# ---------------------------------------------------------------------------

_JSON_INSTRUCTION = (
    "Respond with EXACTLY one JSON object inside a ```json fenced block. "
    "The object must contain these eight keys with numeric values: "
    '"signal" in [-1, 1] (directional conviction), '
    '"confidence" in (0, 1] (how sure you are), '
    '"uncertainty" in [0, 1] (1.0 = maximally uncertain), '
    '"doubt" in [0, 1] (calibration quality, 0.0 = excellent), '
    '"p_plus" in [0, 1] (probability of favourable outcome), '
    '"p_minus" in [0, 1] (probability of unfavourable outcome), '
    '"delta_t" > 0 (your intended time horizon in bars), '
    '"noise" > 0 (your self-estimated measurement noise). '
    "Do not include any other text outside the fenced JSON."
)


def _user_template(role_line: str) -> str:
    return (
        "You are a specialist trading agent.\n"
        f"Role: {role_line}\n\n"
        "Symbol: {symbol}\n"
        "Current regime: {regime} (posterior: {regime_probs})\n"
        "Recent regime features: {features}\n"
        "Most recent ensemble signal: {ensemble_signal} "
        "(disagreement: {disagreement})\n"
        "Memory (similar past trades, top {memory_count}):\n{memory}\n"
        "Other agents in this cycle: {peer_agents}\n\n"
        "Return your eight-channel output as specified.\n"
        + _JSON_INSTRUCTION
    )


# ---------------------------------------------------------------------------
# The seven role definitions
# ---------------------------------------------------------------------------

ECONOMIC_ROLE = AgentRole(
    agent_id="agent_economic",
    domain="earth",
    description=(
        "Reads macro / fiscal indicators and produces a directional view on "
        "the fundamental economic regime. Bias toward mean-reversion when "
        "macro features are extreme."
    ),
    system_prompt=(
        "You are the *Economic State Specialist* in a multi-agent trading "
        "system. You reason about GDP, fiscal, and macro features. You are "
        "pessimistic by default; you only assign a non-zero signal when the "
        "macro picture clearly supports a directional view. Never exceed "
        "absolute signal 0.7 in your domain. " + _JSON_INSTRUCTION
    ),
    user_template=_user_template("economic state specialist (Earth)"),
)

FINANCIAL_ROLE = AgentRole(
    agent_id="agent_financial",
    domain="earth",
    description=(
        "Reads credit, rates, and money-market features; produces a view on "
        "financial plumbing stress."
    ),
    system_prompt=(
        "You are the *Financial State Specialist*. You reason about credit "
        "spreads, interbank rates, and money-market stress. Stress regimes "
        "should push your signal negative. Cap absolute signal at 0.6. "
        + _JSON_INSTRUCTION
    ),
    user_template=_user_template("financial plumbing specialist (Earth)"),
)

FISCAL_ROLE = AgentRole(
    agent_id="agent_fiscal",
    domain="earth",
    description=(
        "Reads fiscal / policy / stimulus indicators. Models when policy "
        "support is present or absent."
    ),
    system_prompt=(
        "You are the *Fiscal State Specialist*. You reason about policy and "
        "stimulus capacity. Be conservative — most of the time your signal "
        "should be near zero. Cap absolute signal at 0.5. " + _JSON_INSTRUCTION
    ),
    user_template=_user_template("fiscal policy specialist (Earth)"),
)

PORTFOLIO_ROLE = AgentRole(
    agent_id="agent_portfolio",
    domain="earth",
    description=(
        "Reads the existing portfolio state (drawdown, liquidity, exposure) "
        "and produces a defensive/risk-on tilt."
    ),
    system_prompt=(
        "You are the *Portfolio State Specialist*. You reason about "
        "drawdowns, exposure, and liquidity. When the portfolio state is "
        "stressed you SHOULD push the ensemble toward HOLD by signalling "
        "low confidence. Cap absolute signal at 0.4. " + _JSON_INSTRUCTION
    ),
    user_template=_user_template("portfolio risk specialist (Earth)"),
)

FUNDAMENTAL_ROLE = AgentRole(
    agent_id="agent_fundamental",
    domain="earth",
    description=(
        "Reads valuation / earnings features. Produces a slow-moving "
        "fundamental view."
    ),
    system_prompt=(
        "You are the *Fundamental State Specialist*. You reason about "
        "earnings, valuation, and intrinsic value. Your horizon is long "
        "(delta_t >= 5). Cap absolute signal at 0.5. " + _JSON_INSTRUCTION
    ),
    user_template=_user_template("fundamental specialist (Earth)"),
)

MARKET_ROLE = AgentRole(
    agent_id="agent_market",
    domain="water",
    description=(
        "Reads microstructure / liquidity features. Produces a short-horizon "
        "view on execution conditions."
    ),
    system_prompt=(
        "You are the *Market Microstructure Specialist*. You reason about "
        "liquidity, order flow, and execution conditions. When liquidity is "
        "poor you SHOULD lower confidence. Cap absolute signal at 0.7. "
        + _JSON_INSTRUCTION
    ),
    user_template=_user_template("market microstructure specialist (Water)"),
)

SECTOR_ROLE = AgentRole(
    agent_id="agent_sector",
    domain="water",
    description=(
        "Reads sector-relative and cross-market features; produces a "
        "rotation / dispersion view."
    ),
    system_prompt=(
        "You are the *Sector / Cross-Market Specialist*. You reason about "
        "sector dispersion and cross-market rotation. Cap absolute signal at "
        "0.5. " + _JSON_INSTRUCTION
    ),
    user_template=_user_template("sector rotation specialist (Water)"),
)


# Volatility / Options specialist from the Fire domain
VOLATILITY_ROLE = AgentRole(
    agent_id="agent_volatility",
    domain="fire",
    description=(
        "Reads implied volatility, skew, and term-structure features; "
        "produces an options-aware view on direction and magnitude."
    ),
    system_prompt=(
        "You are the *Volatility / Options Specialist*. You reason about "
        "implied vol surface, put-call skew, and term structure. You are "
        "responsible for signalling whether options are a sensible vehicle "
        "right now. Cap absolute signal at 0.8. " + _JSON_INSTRUCTION
    ),
    user_template=_user_template("volatility / options specialist (Fire)"),
)


# Default set of seven agents the ensemble expects
DEFAULT_ROLES: Tuple[AgentRole, ...] = (
    ECONOMIC_ROLE,
    FINANCIAL_ROLE,
    FISCAL_ROLE,
    PORTFOLIO_ROLE,
    FUNDAMENTAL_ROLE,
    MARKET_ROLE,
    SECTOR_ROLE,
)


# ---------------------------------------------------------------------------
# Specialist agent wrapper
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AgentContext:
    """Inputs handed to a specialist agent for one cycle."""

    symbol: str
    regime: str
    regime_probabilities: Dict[str, float]
    features: Dict[str, float]
    ensemble_signal: float = 0.0
    disagreement: float = 0.0
    peer_agents: Dict[str, Dict[str, float]] = field(default_factory=dict)
    memory: List[SimilarExperience] = field(default_factory=list)


@dataclass(frozen=True)
class SpecialistAgent:
    """Single LLM-backed specialist agent."""

    role: AgentRole
    adapter: AgentLLMAdapter

    @property
    def agent_id(self) -> str:
        return self.role.agent_id

    def run(self, ctx: AgentContext) -> Tuple[AgentOutput, Any]:
        """Run the agent and return ``(output, raw_response)``."""
        prompt = self._build_prompt(ctx)
        output, response = self.adapter.call(
            prompt,
            system=self.role.system_prompt,
            temperature=0.0,
            max_tokens=256,
        )
        return output, response

    def _build_prompt(self, ctx: AgentContext) -> str:
        # Cap memory for prompt budget
        memory_lines = []
        for i, s in enumerate(ctx.memory[:5], 1):
            memory_lines.append(
                f"  {i}. symbol={s.experience.symbol} "
                f"action={s.experience.position_action} "
                f"pnl={s.experience.pnl:.2f} "
                f"regime={s.experience.regime} "
                f"sim={s.similarity_score:.2f}"
            )
        memory_text = "\n".join(memory_lines) if memory_lines else "  (no similar memory)"

        regime_probs_str = ", ".join(
            f"{k}={v:.2f}" for k, v in sorted(ctx.regime_probabilities.items())
        ) or "(unknown)"

        features_str = ", ".join(
            f"{k}={v:.4f}" for k, v in ctx.features.items()
        ) or "(none)"

        peer_str = ", ".join(
            f"{aid}: {vals.get('signal', 0.0):+.2f}"
            for aid, vals in ctx.peer_agents.items()
        ) or "(none)"

        return self.role.user_template.format(
            symbol=ctx.symbol,
            regime=ctx.regime,
            regime_probs=regime_probs_str,
            features=features_str,
            ensemble_signal=f"{ctx.ensemble_signal:+.4f}",
            disagreement=f"{ctx.disagreement:.4f}",
            memory=memory_text,
            memory_count=len(ctx.memory[:5]),
            peer_agents=peer_str,
        )


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

def build_specialist_agents(
    provider: "LLMProvider",
    *,
    roles: Sequence[AgentRole] = DEFAULT_ROLES,
    fallback_signal: float = 0.0,
    fallback_confidence: float = 0.25,
) -> Dict[str, SpecialistAgent]:
    """Build the seven (or more) specialist agents for one provider."""
    from ..llm.adapter import AgentLLMAdapter  # lazy: breaks circular import
    agents: Dict[str, SpecialistAgent] = {}
    for role in roles:
        adapter = AgentLLMAdapter(
            provider=provider,
            agent_id=role.agent_id,
            fallback_signal=fallback_signal,
            fallback_confidence=fallback_confidence,
        )
        agents[role.agent_id] = SpecialistAgent(role=role, adapter=adapter)
    return agents


def run_agents(
    agents: Dict[str, SpecialistAgent],
    ctx: AgentContext,
) -> Dict[str, AgentOutput]:
    """Run every agent and return ``{agent_id: AgentOutput}``."""
    outputs: Dict[str, AgentOutput] = {}
    for agent_id, agent in agents.items():
        try:
            output, _response = agent.run(ctx)
        except Exception:
            output = AgentOutput(
                s=0.0, c=0.25, u=0.75, d=0.5,
                p_plus=0.5, p_minus=0.5, delta_t=1.0, r=0.5,
                agent_id=agent_id,
            )
        outputs[agent_id] = output
    return outputs


__all__ = [
    "AgentRole",
    "AgentContext",
    "SpecialistAgent",
    "DEFAULT_ROLES",
    "VOLATILITY_ROLE",
    "build_specialist_agents",
    "run_agents",
]
