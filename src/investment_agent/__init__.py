"""X Quant X — Investment Agent Package.

WHAT
====
Top-level package for the X Quant X quantitative investment architecture.
Coordinates Bayesian agent reputation, multi-agent ensemble signal aggregation,
Kalman-filter-based capital allocation, seven-state circuit-breaker gating,
and Alpaca execution into a unified survival-first trading system.

WHY
===
Encapsulates the complete quantitative pipeline under a single importable
namespace while preserving strict module boundaries and testability.

HOW
===
Uses lazy attribute resolution to expose public APIs from submodules without
triggering circular imports during package initialization. Tracked runtime
modules (execution, memory, hedge_signal, run_agent) are imported from the
repository root, not from the src/ package tree.

Architectural Role
==================
Package namespace. Contains no implementation logic; defers to submodules.
"""

from __future__ import annotations

_public_api: dict = {}


def __getattr__(name: str):
    """Lazy import resolver for top-level package namespace."""
    if name in _public_api:
        module_path, attr = _public_api[name]
        import importlib
        module = importlib.import_module(module_path)
        return getattr(module, attr)
    raise AttributeError(f"module 'investment_agent' has no attribute {name!r}")


def __dir__() -> list:
    """Return sorted list of public API names for IDE autocompletion."""
    return sorted(set(dir(__builtins__)) | set(_public_api.keys()))


# agents
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

# llm
_public_api["LLMProvider"] = (
    "investment_agent.llm",
    "LLMProvider",
)
_public_api["LLMResponse"] = (
    "investment_agent.llm",
    "LLMResponse",
)
_public_api["MockLLMProvider"] = (
    "investment_agent.llm",
    "MockLLMProvider",
)
_public_api["FeatherlessProvider"] = (
    "investment_agent.llm",
    "FeatherlessProvider",
)
_public_api["AgentLLMAdapter"] = (
    "investment_agent.llm",
    "AgentLLMAdapter",
)

# capital
_public_api["CapitalGateResult"] = (
    "investment_agent.capital.capital_gate",
    "CapitalGateResult",
)
_public_api["RiskVerdict"] = (
    "investment_agent.capital.capital_gate",
    "RiskVerdict",
)
_public_api["SevenStateVector"] = (
    "investment_agent.capital.capital_gate",
    "SevenStateVector",
)
_public_api["compute_gating_factor"] = (
    "investment_agent.capital.capital_gate",
    "compute_gating_factor",
)
_public_api["evaluate"] = (
    "investment_agent.capital.capital_gate",
    "evaluate",
)

# execution
_public_api["MAX_POSITION_PCT"] = (
    "investment_agent.execution.execution",
    "MAX_POSITION_PCT",
)
_public_api["get_account_summary"] = (
    "investment_agent.execution.execution",
    "get_account_summary",
)
_public_api["get_option_contract"] = (
    "investment_agent.execution.execution",
    "get_option_contract",
)
_public_api["is_trade_safe"] = (
    "investment_agent.execution.execution",
    "is_trade_safe",
)
_public_api["place_order"] = (
    "investment_agent.execution.execution",
    "place_order",
)

# filters
_public_api["compute_effective_measurement_noise"] = (
    "investment_agent.filters.investment_kalman_gain",
    "compute_effective_measurement_noise",
)
_public_api["compute_investment_kalman_gain"] = (
    "investment_agent.filters.investment_kalman_gain",
    "compute_investment_kalman_gain",
)
_public_api["KalmanFilter"] = (
    "investment_agent.filters.kalman_filter",
    "KalmanFilter",
)
_public_api["KalmanState"] = (
    "investment_agent.filters.kalman_filter",
    "KalmanState",
)

# memory
_public_api["MEMORY_FILE"] = (
    "investment_agent.memory.memory",
    "MEMORY_FILE",
)
_public_api["already_hedged_recently"] = (
    "investment_agent.memory.memory",
    "already_hedged_recently",
)
_public_api["log_decision"] = (
    "investment_agent.memory.memory",
    "log_decision",
)
_public_api["reflect"] = (
    "investment_agent.memory.memory",
    "reflect",
)

# regimes
_public_api["VALID_REGIMES"] = (
    "investment_agent.regimes.regimes",
    "VALID_REGIMES",
)

# signals
_public_api["AgentOutput"] = (
    "investment_agent.signals.ensemble_signal",
    "AgentOutput",
)
_public_api["EnsembleAggregate"] = (
    "investment_agent.signals.ensemble_signal",
    "EnsembleAggregate",
)
# hedge_signal
_public_api["DROP_THRESHOLD_PCT"] = (
    "investment_agent.signals.hedge_signal",
    "DROP_THRESHOLD_PCT",
)
_public_api["check_for_drop"] = (
    "investment_agent.signals.hedge_signal",
    "check_for_drop",
)
_public_api["get_recent_prices"] = (
    "investment_agent.signals.hedge_signal",
    "get_recent_prices",
)
_public_api["run_hedge_check"] = (
    "investment_agent.signals.hedge_signal",
    "run_hedge_check",
)

# ensemble_signal (continued)
_public_api["compute_dampened_signal"] = (
    "investment_agent.signals.ensemble_signal",
    "compute_dampened_signal",
)
_public_api["compute_disagreement"] = (
    "investment_agent.signals.ensemble_signal",
    "compute_disagreement",
)
_public_api["compute_effective_confidence"] = (
    "investment_agent.signals.ensemble_signal",
    "compute_effective_confidence",
)
_public_api["compute_ensemble_aggregate"] = (
    "investment_agent.signals.ensemble_signal",
    "compute_ensemble_aggregate",
)
_public_api["compute_ensemble_signal"] = (
    "investment_agent.signals.ensemble_signal",
    "compute_ensemble_signal",
)

__all__ = list(_public_api.keys())
