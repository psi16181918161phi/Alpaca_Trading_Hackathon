"""Property-based correctness tests for the seven-state capital gate.

Uses ``hypothesis`` to generate randomized state vectors (rather than the
fixed examples already covered in ``tests/unit/capital/test_capital_gate.py``)
and asserts the mathematical invariants that must hold for *every* input,
not just the hand-picked boundary cases.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from hypothesis import given, settings, strategies as st

from investment_agent.capital.capital_gate import SevenStateVector, compute_gating_factor

_unit_float = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)


@given(
    economic=_unit_float, financial=_unit_float, fiscal=_unit_float,
    portfolio=_unit_float, fundamental=_unit_float, market=_unit_float,
    sector=_unit_float,
)
@settings(max_examples=200)
def test_gating_factor_always_in_unit_interval(
    economic, financial, fiscal, portfolio, fundamental, market, sector
):
    states = SevenStateVector(
        economic=economic, financial=financial, fiscal=fiscal,
        portfolio=portfolio, fundamental=fundamental, market=market,
        sector=sector,
    )
    gating_factor, breakdown = compute_gating_factor(states)
    assert 0.0 <= gating_factor <= 1.0
    for name, charge in breakdown.items():
        assert 0.0 <= charge <= 1.0, f"{name} charge {charge} out of [0, 1]"


@given(economic=_unit_float, financial=_unit_float, fiscal=_unit_float,
       portfolio=_unit_float, fundamental=_unit_float, market=_unit_float,
       sector=_unit_float)
@settings(max_examples=100)
def test_gating_factor_is_deterministic_given_same_input(
    economic, financial, fiscal, portfolio, fundamental, market, sector
):
    states = SevenStateVector(
        economic=economic, financial=financial, fiscal=fiscal,
        portfolio=portfolio, fundamental=fundamental, market=market,
        sector=sector,
    )
    first, _ = compute_gating_factor(states)
    second, _ = compute_gating_factor(states)
    assert first == second
