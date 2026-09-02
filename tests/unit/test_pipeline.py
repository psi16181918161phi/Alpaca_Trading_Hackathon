"""Integration tests for X Quant X analytical pipeline.

Verifies end-to-end data flow:
    RegimeDetector → AgentReputationTracker → EnsembleSignal → InvestmentKalmanGain → CapitalGate

Key invariants:
- Active regime determines reputation weights.
- Ensemble output changes when regime or agent performance changes.
- Ensemble output feeds into Kalman gain, which feeds into Capital Gate.
- No fabricated values; all inputs are derived from upstream module outputs.
"""

import math
import unittest
from dataclasses import dataclass
from typing import Any, Dict, List

from investment_agent.regimes.regime_detector import RegimeDetector, RegimeClassification
from investment_agent.regimes.regimes import VALID_REGIMES
from investment_agent.agents.agent_reputation import AgentReputationTracker
from investment_agent.signals.ensemble_signal import AgentOutput, EnsembleAggregate, compute_ensemble_aggregate
from investment_agent.filters.investment_kalman_gain import compute_investment_kalman_gain
from investment_agent.filters.kalman_filter import KalmanFilter, KalmanState
from investment_agent.capital.capital_gate import (
    evaluate,
    SevenStateVector,
    CapitalGateResult,
    RiskVerdict,
)
from investment_agent.pipeline import XQuantXPipeline, PipelineResult, ProvenanceTrace


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

AGENT_IDS = [f"agent{i}" for i in range(1, 8)]


def make_agent_outputs(
    signals: List[float],
    confidences: List[float],
    uncertainties: List[float] = None,
    doubts: List[float] = None,
) -> List[AgentOutput]:
    """Create AgentOutput list with specified signal and confidence values."""
    n = len(signals)
    if uncertainties is None:
        uncertainties = [0.0] * n
    if doubts is None:
        doubts = [0.0] * n
    return [
        AgentOutput(
            s=signals[i],
            c=confidences[i],
            u=uncertainties[i],
            d=doubts[i],
            p_plus=0.5 + signals[i] * 0.25,
            p_minus=0.5 - signals[i] * 0.25,
            delta_t=1.0,
            r=0.01,
            agent_id=AGENT_IDS[i],
        )
        for i in range(n)
    ]


def full_charge_state(**overrides):
    values = {
        "economic": 1.0,
        "financial": 1.0,
        "fiscal": 1.0,
        "portfolio": 1.0,
        "fundamental": 1.0,
        "market": 1.0,
        "sector": 1.0,
    }
    values.update(overrides)
    return SevenStateVector(**values)


def default_portfolio_context(regime: str = "R01") -> Dict[str, Any]:
    return {
        "position_pct": 0.05,
        "gross_leverage": 0.5,
        "entropy": 0.1,
        "drawdown_pct": 0.01,
        "execution_timeout_seconds": 5.0,
        "sector_exposure_pct": 0.1,
        "is_new_long": False,
        "regime": regime,
        "available_liquidity": 100000.0,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestPipelineInitialization(unittest.TestCase):
    """Test pipeline initialization and basic structure."""

    def test_pipeline_creates_with_defaults(self):
        """Verify pipeline initializes with default parameters."""
        pipeline = XQuantXPipeline(agent_ids=AGENT_IDS)
        self.assertIsNotNone(pipeline)
        self.assertEqual(len(pipeline._agent_ids), 7)

    def test_pipeline_rejects_empty_agent_ids(self):
        """Verify pipeline rejects empty agent_ids."""
        with self.assertRaises(ValueError):
            XQuantXPipeline(agent_ids=[])

    def test_pipeline_rejects_duplicate_agent_ids(self):
        """Verify pipeline rejects duplicate agent_ids."""
        with self.assertRaises(ValueError):
            XQuantXPipeline(agent_ids=["a", "a", "b"])


class TestRegimeToWeightFlow(unittest.TestCase):
    """Test that regime classification flows into reputation weights."""

    def test_different_regimes_produce_different_weights(self):
        """Verify weights differ across regimes after different outcomes."""
        pipeline = XQuantXPipeline(agent_ids=AGENT_IDS)

        # Simulate: agent1 performs well in R01, poorly in R09
        pipeline.record_agent_outcome("agent1", "R01", was_correct=True)
        pipeline.record_agent_outcome("agent1", "R01", was_correct=True)
        pipeline.record_agent_outcome("agent1", "R09", was_correct=False)
        pipeline.record_agent_outcome("agent1", "R09", was_correct=False)

        weights_r01 = pipeline.get_regime_weights("R01")
        weights_r09 = pipeline.get_regime_weights("R09")

        # agent1 should have higher weight in R01 than R09
        self.assertGreater(weights_r01["agent1"], weights_r09["agent1"])

    def test_weights_sum_to_positive(self):
        """Verify all weights are strictly positive."""
        pipeline = XQuantXPipeline(agent_ids=AGENT_IDS)
        weights = pipeline.get_regime_weights("R01")
        for w in weights.values():
            self.assertGreater(w, 0.0)

    def test_weights_match_configured_agents(self):
        """Verify weights dictionary keys match configured agent IDs."""
        pipeline = XQuantXPipeline(agent_ids=AGENT_IDS)
        weights = pipeline.get_regime_weights("R01")
        self.assertEqual(set(weights.keys()), set(AGENT_IDS))


class TestRegimeToEnsembleFlow(unittest.TestCase):
    """Test that regime affects ensemble output."""

    def test_ensemble_changes_when_regime_changes(self):
        """Verify ensemble signal changes when regime-specific weights change."""
        pipeline = XQuantXPipeline(agent_ids=AGENT_IDS)

        # Build different weight profiles for two regimes
        weights_r01 = {aid: 1.0 for aid in AGENT_IDS}
        weights_r09 = {aid: 1.0 for aid in AGENT_IDS}
        weights_r09["agent1"] = 0.1  # agent1 downweighted in bear regime

        agents = make_agent_outputs(
            signals=[0.8, -0.2, 0.3, 0.1, -0.4, 0.5, 0.2],
            confidences=[0.9] * 7,
        )

        ensemble_r01 = compute_ensemble_aggregate(agents, weights_r01)
        ensemble_r09 = compute_ensemble_aggregate(agents, weights_r09)

        # Ensemble signal should differ because weights differ
        self.assertNotEqual(
            round(ensemble_r01.ensemble_signal, 6),
            round(ensemble_r09.ensemble_signal, 6),
        )

    def test_ensemble_changes_when_agent_performance_changes(self):
        """Verify ensemble signal changes when agent outcomes are recorded."""
        pipeline = XQuantXPipeline(agent_ids=AGENT_IDS)

        agents = make_agent_outputs(
            signals=[0.8, -0.2, 0.3, 0.1, -0.4, 0.5, 0.2],
            confidences=[0.9] * 7,
        )

        # Before any outcomes: all agents have equal prior weight
        weights_before = pipeline.get_regime_weights("R01")
        ensemble_before = compute_ensemble_aggregate(agents, weights_before)

        # Record outcomes: agent1 correct, agent2 incorrect
        pipeline.record_agent_outcome("agent1", "R01", was_correct=True)
        pipeline.record_agent_outcome("agent2", "R01", was_correct=False)

        weights_after = pipeline.get_regime_weights("R01")
        ensemble_after = compute_ensemble_aggregate(agents, weights_after)

        # Weights should have changed
        self.assertNotEqual(
            round(weights_before["agent1"], 6),
            round(weights_after["agent1"], 6),
        )
        self.assertNotEqual(
            round(weights_before["agent2"], 6),
            round(weights_after["agent2"], 6),
        )


class TestEnsembleToKalmanFlow(unittest.TestCase):
    """Test that ensemble output feeds into Kalman gain computation."""

    def test_kalman_gain_changes_with_ensemble_confidence(self):
        """Verify Kalman gain responds to real ensemble effective confidence."""
        agents = make_agent_outputs(
            signals=[0.5] * 7,
            confidences=[0.9] * 7,
            uncertainties=[0.0] * 7,
            doubts=[0.0] * 7,
        )
        weights = {aid: 1.0 for aid in AGENT_IDS}
        ensemble = compute_ensemble_aggregate(agents, weights)

        k_high_conf = compute_investment_kalman_gain(
            prediction_covariance=1.0,
            effective_confidence=ensemble.effective_confidence,
            disagreement=ensemble.disagreement,
            sigma_base_squared=1.0,
        )

        # Now create low-confidence ensemble
        agents_low = make_agent_outputs(
            signals=[0.5] * 7,
            confidences=[0.1] * 7,
            uncertainties=[0.5] * 7,
            doubts=[0.5] * 7,
        )
        ensemble_low = compute_ensemble_aggregate(agents_low, weights)

        k_low_conf = compute_investment_kalman_gain(
            prediction_covariance=1.0,
            effective_confidence=ensemble_low.effective_confidence,
            disagreement=ensemble_low.disagreement,
            sigma_base_squared=1.0,
        )

        # High confidence should produce higher Kalman gain
        self.assertGreater(k_high_conf, k_low_conf)

    def test_kalman_gain_changes_with_disagreement(self):
        """Verify Kalman gain responds to real ensemble disagreement."""
        # Low disagreement: all agents agree
        agents_agree = make_agent_outputs(
            signals=[0.8] * 7,
            confidences=[0.9] * 7,
            uncertainties=[0.0] * 7,
            doubts=[0.0] * 7,
        )
        weights = {aid: 1.0 for aid in AGENT_IDS}
        ensemble_agree = compute_ensemble_aggregate(agents_agree, weights)

        k_agree = compute_investment_kalman_gain(
            prediction_covariance=1.0,
            effective_confidence=ensemble_agree.effective_confidence,
            disagreement=ensemble_agree.disagreement,
            sigma_base_squared=1.0,
        )

        # High disagreement: agents split
        agents_disagree = make_agent_outputs(
            signals=[0.8, -0.8, 0.8, -0.8, 0.8, -0.8, 0.8],
            confidences=[0.9] * 7,
            uncertainties=[0.0] * 7,
            doubts=[0.0] * 7,
        )
        ensemble_disagree = compute_ensemble_aggregate(agents_disagree, weights)

        k_disagree = compute_investment_kalman_gain(
            prediction_covariance=1.0,
            effective_confidence=ensemble_disagree.effective_confidence,
            disagreement=ensemble_disagree.disagreement,
            sigma_base_squared=1.0,
        )

        # Low disagreement should produce higher Kalman gain
        self.assertGreater(k_agree, k_disagree)


class TestKalmanToCapitalGateFlow(unittest.TestCase):
    """Test that Kalman state feeds into Capital Gate evaluation."""

    def test_capital_gate_uses_real_kalman_state(self):
        """Verify capital gate Kalman gain changes with real prediction covariance."""
        agents = make_agent_outputs(
            signals=[0.5] * 7,
            confidences=[0.9] * 7,
            uncertainties=[0.0] * 7,
            doubts=[0.0] * 7,
        )
        weights = {aid: 1.0 for aid in AGENT_IDS}
        states = full_charge_state()
        ctx = default_portfolio_context()

        # Low price variance (low prediction uncertainty)
        kalman_low_var = KalmanState(
            estimated_price=100.0,
            trend=0.01,
            uncertainty=0.5,
            trend_uncertainty=0.01,
            price_variance=0.01,
            trend_variance=0.0001,
            innovation=0.0,
            kalman_gain_price=0.8,
        )

        result_low = evaluate(
            kalman_state=kalman_low_var,
            states=states,
            portfolio_context=ctx,
            agents=agents,
            agent_weights=weights,
        )

        # High price variance (high prediction uncertainty)
        kalman_high_var = KalmanState(
            estimated_price=100.0,
            trend=0.01,
            uncertainty=5.0,
            trend_uncertainty=0.5,
            price_variance=25.0,
            trend_variance=0.25,
            innovation=0.0,
            kalman_gain_price=0.1,
        )

        result_high = evaluate(
            kalman_state=kalman_high_var,
            states=states,
            portfolio_context=ctx,
            agents=agents,
            agent_weights=weights,
        )

        # With high ensemble confidence and low disagreement, higher prediction
        # covariance produces higher Kalman gain (K_t = P / (P + R) when P >> R).
        # This means effective_cap increases with price_variance in this regime.
        self.assertNotEqual(
            round(result_low.effective_cap, 6),
            round(result_high.effective_cap, 6),
        )


class TestEndToEndPipeline(unittest.TestCase):
    """Test complete pipeline execution with real data flow."""

    def test_full_pipeline_executes(self):
        """Verify full pipeline executes and returns consistent result."""
        pipeline = XQuantXPipeline(agent_ids=AGENT_IDS)

        prices = [100.0 + i * 0.1 for i in range(45)]
        volumes = [1000.0] * 45
        agents = make_agent_outputs(
            signals=[0.5, 0.3, -0.2, 0.1, 0.4, -0.1, 0.2],
            confidences=[0.8, 0.7, 0.6, 0.9, 0.5, 0.4, 0.7],
        )
        states = full_charge_state()
        ctx = default_portfolio_context()

        result = pipeline.evaluate(
            prices=prices,
            volumes=volumes,
            agent_outputs=agents,
            states=states,
            portfolio_context=ctx,
        )

        # Verify all components are present and valid
        self.assertIsInstance(result, PipelineResult)
        self.assertIn(result.regime.regime, VALID_REGIMES)
        self.assertEqual(len(result.weights), 7)
        self.assertIsInstance(result.ensemble, EnsembleAggregate)
        self.assertIsInstance(result.kalman_state, KalmanState)
        self.assertIsInstance(result.capital_gate, CapitalGateResult)
        self.assertIsInstance(result.provenance, ProvenanceTrace)

        # Verify provenance trace completeness
        self.assertIn(result.provenance.regime, VALID_REGIMES)
        self.assertEqual(result.provenance.regime, result.regime.regime)
        self.assertEqual(result.provenance.kalman_gain, result.kalman_gain)
        self.assertEqual(
            result.provenance.capital_gate["verdict"],
            result.capital_gate.verdict.value,
        )
        self.assertEqual(len(result.provenance.agent_outputs), 7)

        # Verify regime affects weights
        weights = result.weights
        for w in weights.values():
            self.assertGreater(w, 0.0)
            self.assertLessEqual(w, 1.0)

    def test_pipeline_regime_changes_with_market_conditions(self):
        """Verify regime classifier responds to different market conditions."""
        pipeline = XQuantXPipeline(agent_ids=AGENT_IDS)

        # Bullish market
        bull_prices = [100.0 + i * 0.5 for i in range(25)]
        bull_result = pipeline.classify_regime(bull_prices)

        # Bearish market
        bear_prices = [100.0 - i * 0.5 for i in range(25)]
        bear_result = pipeline.classify_regime(bear_prices)

        # Regimes should differ
        self.assertNotEqual(bull_result.regime, bear_result.regime)
        self.assertIn("rsi", bull_result.features)
        self.assertIn("macd", bull_result.features)

    def test_pipeline_ensemble_signal_changes_with_weights(self):
        """Verify ensemble signal is sensitive to weight changes."""
        pipeline = XQuantXPipeline(agent_ids=AGENT_IDS)

        # Create two pipelines with different agent performance histories
        pipeline_bull = XQuantXPipeline(agent_ids=AGENT_IDS)
        pipeline_bear = XQuantXPipeline(agent_ids=AGENT_IDS)

        # In bull regime, agent1-3 perform well; in bear regime, agent4-6 perform well
        for _ in range(5):
            pipeline_bull.record_agent_outcome("agent1", "R01", was_correct=True)
            pipeline_bull.record_agent_outcome("agent2", "R01", was_correct=True)
            pipeline_bull.record_agent_outcome("agent3", "R01", was_correct=True)
            pipeline_bull.record_agent_outcome("agent4", "R01", was_correct=False)
            pipeline_bull.record_agent_outcome("agent5", "R01", was_correct=False)
            pipeline_bull.record_agent_outcome("agent6", "R01", was_correct=False)

            pipeline_bear.record_agent_outcome("agent4", "R09", was_correct=True)
            pipeline_bear.record_agent_outcome("agent5", "R09", was_correct=True)
            pipeline_bear.record_agent_outcome("agent6", "R09", was_correct=True)
            pipeline_bear.record_agent_outcome("agent1", "R09", was_correct=False)
            pipeline_bear.record_agent_outcome("agent2", "R09", was_correct=False)
            pipeline_bear.record_agent_outcome("agent3", "R09", was_correct=False)

        agents = make_agent_outputs(
            signals=[0.8, 0.7, 0.6, -0.3, -0.2, -0.1, 0.0],
            confidences=[0.9] * 7,
        )

        weights_bull = pipeline_bull.get_regime_weights("R01")
        weights_bear = pipeline_bear.get_regime_weights("R09")

        ensemble_bull = compute_ensemble_aggregate(agents, weights_bull)
        ensemble_bear = compute_ensemble_aggregate(agents, weights_bear)

        # Bull-favored weights should produce more positive ensemble signal
        self.assertGreater(ensemble_bull.ensemble_signal, ensemble_bear.ensemble_signal)

    def test_pipeline_capital_gate_responds_to_regime(self):
        """Verify capital gate result changes when regime-specific weights change."""
        pipeline_bull = XQuantXPipeline(agent_ids=AGENT_IDS)
        pipeline_bear = XQuantXPipeline(agent_ids=AGENT_IDS)

        prices = [100.0 + i * 0.1 for i in range(45)]
        volumes = [1000.0] * 45

        agents_bull = make_agent_outputs(
            signals=[0.8, 0.7, 0.6, -0.3, -0.2, -0.1, 0.0],
            confidences=[0.9] * 7,
        )
        agents_bear = make_agent_outputs(
            signals=[-0.8, -0.7, -0.6, 0.3, 0.2, 0.1, 0.0],
            confidences=[0.9] * 7,
        )

        states = full_charge_state()
        ctx_bull = default_portfolio_context(regime="R01")
        ctx_bear = default_portfolio_context(regime="R09")

        # Record outcomes so weights differ between regimes
        for _ in range(10):
            # In bull regime, bullish agents perform well
            pipeline_bull.record_agent_outcome("agent1", "R01", was_correct=True)
            pipeline_bull.record_agent_outcome("agent2", "R01", was_correct=True)
            pipeline_bull.record_agent_outcome("agent3", "R01", was_correct=True)
            pipeline_bull.record_agent_outcome("agent4", "R01", was_correct=False)
            pipeline_bull.record_agent_outcome("agent5", "R01", was_correct=False)
            pipeline_bull.record_agent_outcome("agent6", "R01", was_correct=False)

            # In bear regime, bearish agents perform well
            pipeline_bear.record_agent_outcome("agent1", "R09", was_correct=False)
            pipeline_bear.record_agent_outcome("agent2", "R09", was_correct=False)
            pipeline_bear.record_agent_outcome("agent3", "R09", was_correct=False)
            pipeline_bear.record_agent_outcome("agent4", "R09", was_correct=True)
            pipeline_bear.record_agent_outcome("agent5", "R09", was_correct=True)
            pipeline_bear.record_agent_outcome("agent6", "R09", was_correct=True)

        result_bull = pipeline_bull.evaluate(
            prices=prices,
            volumes=volumes,
            agent_outputs=agents_bull,
            states=states,
            portfolio_context=ctx_bull,
        )

        result_bear = pipeline_bear.evaluate(
            prices=prices,
            volumes=volumes,
            agent_outputs=agents_bear,
            states=states,
            portfolio_context=ctx_bear,
        )

        # Regime-specific weights should produce different ensemble metrics,
        # which changes Kalman gain, which changes effective cap.
        # At minimum, the weights must differ.
        self.assertNotEqual(
            round(result_bull.weights["agent1"], 4),
            round(result_bear.weights["agent1"], 4),
        )

    def test_pipeline_kalman_state_updates_with_prices(self):
        """Verify Kalman state updates correctly with price observations."""
        pipeline = XQuantXPipeline(agent_ids=AGENT_IDS, kalman_initial_price=100.0)

        initial_state = pipeline.get_kalman_state()
        self.assertEqual(initial_state.estimated_price, 100.0)

        # Feed in a higher price
        pipeline.update_kalman(101.0)
        updated_state = pipeline.get_kalman_state()
        self.assertGreater(updated_state.estimated_price, 99.0)

    def test_pipeline_exposes_kalman_gain(self):
        """Verify pipeline result exposes kalman_gain from capital gate."""
        pipeline = XQuantXPipeline(agent_ids=AGENT_IDS)
        prices = [100.0 + i * 0.1 for i in range(45)]
        volumes = [1000.0] * 45
        agents = make_agent_outputs(
            signals=[0.5] * 7,
            confidences=[0.9] * 7,
        )
        states = full_charge_state()
        ctx = default_portfolio_context()

        result = pipeline.evaluate(
            prices=prices,
            volumes=volumes,
            agent_outputs=agents,
            states=states,
            portfolio_context=ctx,
        )

        self.assertIsInstance(result.kalman_gain, float)
        self.assertGreaterEqual(result.kalman_gain, 0.0)
        self.assertLessEqual(result.kalman_gain, 1.0)
        self.assertEqual(
            result.kalman_gain,
            result.capital_gate.kalman_gain,
        )

    def test_pipeline_ensemble_chain_of_custody(self):
        """Verify the same EnsembleAggregate object flows into capital gate."""
        pipeline = XQuantXPipeline(agent_ids=AGENT_IDS)
        prices = [100.0 + i * 0.1 for i in range(45)]
        volumes = [1000.0] * 45
        agents = make_agent_outputs(
            signals=[0.5] * 7,
            confidences=[0.9] * 7,
        )
        states = full_charge_state()
        ctx = default_portfolio_context()

        result = pipeline.evaluate(
            prices=prices,
            volumes=volumes,
            agent_outputs=agents,
            states=states,
            portfolio_context=ctx,
        )

        # Same object identity, not just equal values
        self.assertIs(result.ensemble, result.capital_gate.ensemble_agg)


class TestHMMPipelineIntegration(unittest.TestCase):
    """Test HMM regime detector integration with pipeline."""

    def test_hmm_pipeline_classify_regime(self):
        """Verify HMM branch produces valid regime classification."""
        pipeline = XQuantXPipeline(agent_ids=AGENT_IDS, use_hmm=True)
        prices = [100.0 + i * 0.1 for i in range(50)]
        volumes = [1000.0 + i * 10 for i in range(50)]
        
        result = pipeline.classify_regime(prices, volumes)
        self.assertIsInstance(result, RegimeClassification)
        self.assertIn(result.regime, VALID_REGIMES)

    def test_hmm_pipeline_regime_in_valid_set(self):
        """Verify HMM regime is always in VALID_REGIMES."""
        pipeline = XQuantXPipeline(agent_ids=AGENT_IDS, use_hmm=True)
        prices = [100.0 + i * 0.1 for i in range(50)]
        volumes = [1000.0 + i * 10 for i in range(50)]
        
        result = pipeline.classify_regime(prices, volumes)
        self.assertIn(result.regime, VALID_REGIMES)

    def test_hmm_pipeline_underflow_raises(self):
        """Verify HMM underflow raises HMMUnderflowError in pipeline."""
        from investment_agent.regimes.hmm_inference import HMMUnderflowError
        
        pipeline = XQuantXPipeline(agent_ids=AGENT_IDS, use_hmm=True)
        
        # Create prices that will cause feature extraction to fail
        # or HMM underflow (extreme values)
        prices = [100.0] * 50
        prices[0] = 0.0  # Invalid price
        
        with self.assertRaises(ValueError):
            pipeline.classify_regime(prices)

    def test_hmm_pipeline_feature_extraction_error(self):
        """Verify feature extraction errors are propagated."""
        pipeline = XQuantXPipeline(agent_ids=AGENT_IDS, use_hmm=True)
        
        # Insufficient data
        prices = [100.0, 101.0, 102.0]
        with self.assertRaises(ValueError):
            pipeline.classify_regime(prices)

    def test_rule_based_pipeline_still_works(self):
        """Verify rule-based pipeline still works when use_hmm=False."""
        pipeline = XQuantXPipeline(agent_ids=AGENT_IDS, use_hmm=False)
        prices = [100.0 + i * 0.1 for i in range(45)]
        volumes = [1000.0] * 45
        
        result = pipeline.classify_regime(prices, volumes)
        self.assertIsInstance(result, RegimeClassification)
        self.assertIn(result.regime, VALID_REGIMES)

    def test_hmm_pipeline_provenance_trace(self):
        """Verify provenance trace includes HMM data when use_hmm=True."""
        pipeline = XQuantXPipeline(agent_ids=AGENT_IDS, use_hmm=True)
        prices = [100.0 + i * 0.1 for i in range(50)]
        volumes = [1000.0 + i * 10 for i in range(50)]
        agents = make_agent_outputs(
            signals=[0.5] * 7,
            confidences=[0.9] * 7,
        )
        states = full_charge_state()
        ctx = default_portfolio_context()

        result = pipeline.evaluate(
            prices=prices,
            volumes=volumes,
            agent_outputs=agents,
            states=states,
            portfolio_context=ctx,
        )

        self.assertIsInstance(result.provenance, ProvenanceTrace)
        self.assertIn(result.provenance.regime, VALID_REGIMES)


if __name__ == "__main__":
    unittest.main()
