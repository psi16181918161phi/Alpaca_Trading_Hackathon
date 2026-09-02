"""Property-based correctness tests for market regime classification.

Generates randomized (but numerically valid) price/volume series via
``hypothesis`` and asserts the invariants ``RegimeDetector.classify`` must
satisfy for any input: affinity scores sum to 1.0, confidence stays in
[0, 1], and the returned regime is always one of the 12 canonical regimes.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from hypothesis import given, settings, strategies as st

from investment_agent.regimes.regime_detector import RegimeDetector
from investment_agent.regimes.regimes import VALID_REGIMES

_price = st.floats(min_value=1.0, max_value=10_000.0, allow_nan=False, allow_infinity=False)
_volume = st.floats(min_value=1.0, max_value=1_000_000.0, allow_nan=False, allow_infinity=False)


@given(prices=st.lists(_price, min_size=25, max_size=60))
@settings(max_examples=100)
def test_affinity_scores_sum_to_one_for_random_price_series(prices):
    detector = RegimeDetector()
    result = detector.classify(prices)
    assert result.regime in VALID_REGIMES
    assert 0.0 <= result.confidence <= 1.0
    total = sum(result.regime_affinity.values())
    assert abs(total - 1.0) < 1e-6


@given(prices=st.lists(_price, min_size=25, max_size=60),
       volumes=st.lists(_volume, min_size=25, max_size=60))
@settings(max_examples=75)
def test_classification_with_volume_stays_within_bounds(prices, volumes):
    n = min(len(prices), len(volumes))
    detector = RegimeDetector()
    result = detector.classify(prices[:n], volumes[:n])
    assert result.regime in VALID_REGIMES
    assert 0.0 <= result.confidence <= 1.0
    for score in result.regime_affinity.values():
        assert 0.0 <= score <= 1.0
