"""Public API surface smoke test for every lazy-import package `__init__.py`.

Each `investment_agent` subpackage (and the top-level package) resolves its
public API lazily via `__getattr__` + a `_public_api` dict, per each
package's own docstring ("Uses lazy attribute resolution to expose the
public API without triggering circular imports"). This test imports every
name a caller is expected to use from each of these modules, so a typo'd
target path or a broken re-export is caught immediately rather than only
when some far-away caller happens to import that one name.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

import investment_agent
from investment_agent import agents, capital, execution, filters, memory, regimes, signals

TOP_LEVEL_NAMES = [
    "AgentReputationTracker", "SpecialistAgent", "AgentRole", "AgentContext",
    "DEFAULT_ROLES", "build_specialist_agents", "run_agents",
    "LLMProvider", "LLMResponse", "MockLLMProvider", "FeatherlessProvider",
    "AgentLLMAdapter", "CapitalGateResult", "RiskVerdict", "SevenStateVector",
    "compute_gating_factor", "evaluate", "MAX_POSITION_PCT",
    "get_account_summary", "get_option_contract", "is_trade_safe",
    "place_order", "compute_effective_measurement_noise",
    "compute_investment_kalman_gain", "KalmanFilter", "KalmanState",
    "MEMORY_FILE", "already_hedged_recently", "log_decision", "reflect",
    "VALID_REGIMES", "AgentOutput", "EnsembleAggregate", "DROP_THRESHOLD_PCT",
    "check_for_drop", "get_recent_prices", "run_hedge_check",
    "compute_dampened_signal", "compute_disagreement",
    "compute_effective_confidence", "compute_ensemble_aggregate",
    "compute_ensemble_signal",
]

SUBPACKAGE_NAMES = {
    agents: ["AgentReputationTracker", "SpecialistAgent", "AgentRole",
             "AgentContext", "DEFAULT_ROLES", "build_specialist_agents", "run_agents"],
    capital: ["CapitalGateResult", "RiskVerdict", "SevenStateVector",
              "compute_gating_factor", "evaluate"],
    execution: ["MAX_POSITION_PCT", "get_account_summary", "get_account_snapshot",
                "load_account_baseline", "save_account_baseline", "get_option_contract",
                "is_trade_safe", "place_order", "HedgeRiskAssessment",
                "evaluate_hedge_risk", "record_hedge_placement",
                "cleanup_hedge_history", "get_recent_hedge_symbols"],
    filters: ["compute_effective_measurement_noise", "compute_investment_kalman_gain",
              "KalmanFilter", "KalmanState"],
    memory: ["MEMORY_FILE", "already_hedged_recently", "log_decision", "reflect"],
    regimes: ["VALID_REGIMES", "RegimeClassification", "MarketFeatures",
              "RegimeDetector", "detect_regime"],
    signals: ["AgentOutput", "EnsembleAggregate", "compute_dampened_signal",
              "compute_disagreement", "compute_effective_confidence",
              "compute_ensemble_aggregate", "compute_ensemble_signal",
              "DROP_THRESHOLD_PCT", "check_for_drop", "get_recent_prices",
              "run_hedge_check"],
}


@pytest.mark.parametrize("name", TOP_LEVEL_NAMES)
def test_top_level_public_api_resolves(name):
    assert getattr(investment_agent, name) is not None


@pytest.mark.parametrize("module,name", [
    (module, name) for module, names in SUBPACKAGE_NAMES.items() for name in names
])
def test_subpackage_public_api_resolves(module, name):
    assert getattr(module, name) is not None


def test_top_level_unknown_attribute_raises():
    with pytest.raises(AttributeError):
        investment_agent.__getattr__("NotARealExportedName")


def test_top_level_dir_includes_public_api_names():
    exported = investment_agent.__dir__()
    assert "AgentReputationTracker" in exported
    assert "KalmanFilter" in exported


@pytest.mark.parametrize("module", [agents, capital, execution, filters, memory, regimes, signals])
def test_subpackage_unknown_attribute_raises(module):
    with pytest.raises(AttributeError):
        module.__getattr__("NotARealExportedName")


@pytest.mark.parametrize("module", [agents, capital, execution, filters, memory, regimes, signals])
def test_subpackage_dir_is_sorted_and_nonempty(module):
    exported = module.__dir__()
    assert exported == sorted(exported)
    assert len(exported) > 0
