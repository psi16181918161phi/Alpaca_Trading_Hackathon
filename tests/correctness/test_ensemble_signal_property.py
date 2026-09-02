"""Property-based correctness tests for the seven-agent signal ensemble.

Generates randomized agent outputs and weights via ``hypothesis`` and
asserts the invariants the ensemble math must satisfy for any valid input:
signal stays in [-1, 1], disagreement stays in [0, 1], and effective
confidence stays in [0, 1].
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from hypothesis import given, settings, strategies as st

from investment_agent.signals.ensemble_signal import (
    AgentOutput,
    compute_disagreement,
    compute_effective_confidence,
    compute_ensemble_signal,
)

_signed_unit = st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False)
_unit_open = st.floats(min_value=1e-6, max_value=1.0, allow_nan=False, allow_infinity=False)
_unit_closed = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)

AGENT_IDS = [f"agent-{i}" for i in range(7)]


def _p_plus_p_minus_strategy():
    """p_plus, p_minus pair that always satisfies p_plus + p_minus <= 1.0."""
    return _unit_closed.flatmap(
        lambda p_plus: st.tuples(
            st.just(p_plus),
            st.floats(min_value=0.0, max_value=1.0 - p_plus, allow_nan=False, allow_infinity=False),
        )
    )


def _agent_fields_strategy():
    return st.tuples(_signed_unit, _unit_open, _unit_closed, _unit_closed).flatmap(
        lambda base: _p_plus_p_minus_strategy().map(lambda pp_pm: base + pp_pm)
    )


def _agents_and_weights_strategy():
    """Build a fixed-cardinality (7-agent) list of AgentOutput + a matching weight dict."""
    return st.tuples(*[_agent_fields_strategy() for _ in AGENT_IDS])


@given(values=_agents_and_weights_strategy(), weight_values=st.lists(
    st.floats(min_value=0.01, max_value=10.0, allow_nan=False, allow_infinity=False),
    min_size=7, max_size=7,
))
@settings(max_examples=150)
def test_ensemble_signal_and_disagreement_stay_bounded(values, weight_values):
    agents = [
        AgentOutput(s=s, c=c, u=u, d=d, p_plus=pp, p_minus=pm,
                    delta_t=1.0, r=0.1, agent_id=agent_id)
        for agent_id, (s, c, u, d, pp, pm) in zip(AGENT_IDS, values)
    ]
    weights = dict(zip(AGENT_IDS, weight_values))

    ensemble_signal = compute_ensemble_signal(agents, weights)
    assert -1.0 <= ensemble_signal <= 1.0

    disagreement = compute_disagreement(agents, weights, ensemble_signal)
    assert 0.0 <= disagreement <= 1.0

    effective_confidence = compute_effective_confidence(agents, weights)
    assert 0.0 <= effective_confidence <= 1.0
