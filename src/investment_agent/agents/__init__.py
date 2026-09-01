"""Agent Reputation Subpackage — Bayesian Beta-Prior Reputation Layer.

WHAT
====
Provides per-agent, per-regime Bayesian Beta-prior reputation tracking.
Tracks α and β parameters for each (agent_id, regime) pair and computes
posterior expectation weights w_i = E[θ_i] = α_i / (α_i + β_i).

WHY
===
In a multi-agent quantitative architecture, agents demonstrate varying
accuracy across different market regimes (R01-R12). Dynamic Bayesian
reputation tracking enables the system to continuously update agent weights
based on historical accuracy, downweighting poor performers and upweighting
consistent performers.

HOW
===
Uses lazy attribute resolution to expose the public API without triggering
circular imports during package initialization.

Architectural Role
==================
Analytical reputation tracking layer. Feeds weight dictionaries into
ensemble signal aggregation (signals/ensemble_signal.py).
"""

from __future__ import annotations

_public_api: dict = {}


def __getattr__(name: str):
    """Lazy import resolver for agent subpackage namespace."""
    if name in _public_api:
        module_path, attr = _public_api[name]
        import importlib
        module = importlib.import_module(module_path)
        return getattr(module, attr)
    raise AttributeError(f"module 'investment_agent.agents' has no attribute {name!r}")


def __dir__() -> list:
    """Return sorted list of public API names for IDE autocompletion."""
    return sorted(set(dir(__builtins__)) | set(_public_api.keys()))


_public_api["AgentReputationTracker"] = (
    "investment_agent.agents.agent_reputation",
    "AgentReputationTracker",
)
_public_api["SpecialistAgent"] = (
    "investment_agent.agents.specialist",
    "SpecialistAgent",
)
_public_api["AgentRole"] = (
    "investment_agent.agents.specialist",
    "AgentRole",
)
_public_api["AgentContext"] = (
    "investment_agent.agents.specialist",
    "AgentContext",
)
_public_api["DEFAULT_ROLES"] = (
    "investment_agent.agents.specialist",
    "DEFAULT_ROLES",
)
_public_api["build_specialist_agents"] = (
    "investment_agent.agents.specialist",
    "build_specialist_agents",
)
_public_api["run_agents"] = (
    "investment_agent.agents.specialist",
    "run_agents",
)

__all__ = list(_public_api.keys())
