"""Performance budget tests for the pure-computation hot paths.

Each test asserts a concrete wall-clock budget for a single call rather
than merely "runs without error", per
``alpaca_paper_trading_specifications_x_quant_x/012_xquantx_performance_tests.txt``.
Budgets are generous (order-of-magnitude headroom) since these run in
CI on shared/virtualized hardware; the goal is to catch an accidental
O(n^2) regression, not to micro-benchmark.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from investment_agent.capital.capital_gate import SevenStateVector, compute_gating_factor
from investment_agent.regimes.regime_detector import RegimeDetector
from investment_agent.signals.ensemble_signal import AgentOutput, compute_ensemble_aggregate

CAPITAL_GATE_BUDGET_S = 0.01
REGIME_DETECTOR_BUDGET_S = 0.05
ENSEMBLE_BUDGET_S = 0.01


def test_capital_gate_gating_factor_within_budget():
    states = SevenStateVector(
        economic=0.5, financial=0.5, fiscal=0.5, portfolio=0.5,
        fundamental=0.5, market=0.5, sector=0.5,
    )
    start = time.perf_counter()
    for _ in range(100):
        compute_gating_factor(states)
    elapsed_per_call = (time.perf_counter() - start) / 100
    assert elapsed_per_call < CAPITAL_GATE_BUDGET_S, (
        f"compute_gating_factor took {elapsed_per_call:.6f}s/call, "
        f"budget is {CAPITAL_GATE_BUDGET_S}s"
    )


def test_regime_detector_classify_within_budget():
    prices = [100.0 + (i % 7) * 0.3 for i in range(60)]
    detector = RegimeDetector()
    start = time.perf_counter()
    detector.classify(prices)
    elapsed = time.perf_counter() - start
    assert elapsed < REGIME_DETECTOR_BUDGET_S, (
        f"RegimeDetector.classify took {elapsed:.6f}s, budget is {REGIME_DETECTOR_BUDGET_S}s"
    )


def test_ensemble_aggregate_within_budget():
    agents = [
        AgentOutput(s=0.3, c=0.8, u=0.2, d=0.1, p_plus=0.6, p_minus=0.3,
                    delta_t=1.0, r=0.2, agent_id=f"agent{i}")
        for i in range(7)
    ]
    weights = {f"agent{i}": 1.0 for i in range(7)}
    start = time.perf_counter()
    for _ in range(100):
        compute_ensemble_aggregate(agents, weights)
    elapsed_per_call = (time.perf_counter() - start) / 100
    assert elapsed_per_call < ENSEMBLE_BUDGET_S, (
        f"compute_ensemble_aggregate took {elapsed_per_call:.6f}s/call, "
        f"budget is {ENSEMBLE_BUDGET_S}s"
    )
