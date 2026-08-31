"""Tests for trade memory and orchestrator."""
import math
import os
import tempfile
import unittest
from datetime import datetime, timedelta
from typing import Any, Dict, List
import numpy as np
from investment_agent.regimes.regime_detector import RegimeClassification
from investment_agent.regimes.regimes import VALID_REGIMES
from investment_agent.signals.ensemble_signal import AgentOutput, EnsembleAggregate, compute_ensemble_aggregate
from investment_agent.filters.kalman_filter import KalmanState
from investment_agent.capital.capital_gate import CapitalGateResult, RiskVerdict
from investment_agent.memory.trade_memory import TradeExperience, SimilarExperience, TradeMemory, DEFAULT_MEMORY_FILE, MAX_MEMORY_PER_SYMBOL
from investment_agent.orchestrator import XQuantXOrchestrator, TradingDecision, CycleResult, AuditLog
from investment_agent.memory.trade_memory import TradeLifecycle
AGENT_IDS = [f'agent{i}' for i in range(1, 8)]

def make_agent_outputs(signals: List[float], confidences: List[float]) -> List[AgentOutput]:
    """Create AgentOutput list."""
    return [AgentOutput(s=signals[i], c=confidences[i], u=0.0, d=0.0, p_plus=0.5 + signals[i] * 0.25, p_minus=0.5 - signals[i] * 0.25, delta_t=1.0, r=0.01, agent_id=AGENT_IDS[i]) for i in range(len(signals))]

def make_kalman_state(**overrides) -> KalmanState:
    """Create KalmanState with defaults."""
    defaults = {'estimated_price': 100.0, 'trend': 0.01, 'uncertainty': 1.0, 'trend_uncertainty': 0.1, 'price_variance': 1.0, 'trend_variance': 0.01, 'innovation': 0.0, 'kalman_gain_price': 0.5}
    defaults.update(overrides)
    return KalmanState(**defaults)

def make_capital_gate_result(**overrides) -> CapitalGateResult:
    """Create CapitalGateResult with defaults."""
    defaults = {'verdict': RiskVerdict.ALLOW, 'gating_factor': 0.8, 'effective_cap': 0.5, 'reduce_factor': 1.0, 'state_charges': {}, 'state_gatings': {}, 'triggered_rules': (), 'reason': 'Test gate', 'kalman_gain': 0.5}
    defaults.update(overrides)
    return CapitalGateResult(**defaults)

def make_regime_classification(**overrides) -> RegimeClassification:
    """Create RegimeClassification with defaults."""
    defaults = {'regime': 'R01', 'confidence': 0.8, 'timestamp': datetime.now(), 'features': {'annualized_return': 0.1, 'annualized_volatility': 0.2}, 'regime_affinity': {f'R{i:02d}': 1.0 / 12 for i in range(1, 13)}}
    defaults.update(overrides)
    return RegimeClassification(**defaults)

class TestTradeMemory(unittest.TestCase):
    """Test trade memory persistence and retrieval."""

    def setUp(self):
        """Create fresh memory for each test."""
        self.memory_file = f'test_memory_{id(self)}.json'
        self.memory = TradeMemory(memory_file=self.memory_file)

    def tearDown(self):
        """Clean up test memory file."""
        import os
        for f in [self.memory_file, f'{self.memory_file}.tmp']:
            if os.path.exists(f):
                try:
                    os.remove(f)
                except PermissionError:
                    pass

    def test_log_and_retrieve_experience(self):
        """Verify experience can be logged and retrieved."""
        exp = TradeExperience(decision_id='test-decision', timestamp=datetime.now(), symbol='AAPL', regime='R01', regime_probabilities={'R01': 0.8}, agent_signals={'agent1': 0.5}, ensemble_signal=0.5, disagreement=0.2, effective_confidence=0.8, kalman_gain=0.5, kalman_price=100.0, kalman_trend=0.01, capital_gate_verdict='ALLOW', effective_cap=0.5, state_charges={'economic': 1.0}, position_action='BUY', quantity=1.0, confidence=0.8, expected_outcome='Price up', realized_outcome='PENDING', pnl=0.0, lesson='')
        self.memory.log_experience(exp)
        history = self.memory.get_history()
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0].symbol, 'AAPL')

    def test_find_similar_experiences(self):
        """Verify similar experience retrieval."""
        exp1 = TradeExperience(decision_id='test-decision', timestamp=datetime.now(), symbol='AAPL', regime='R01', regime_probabilities={'R01': 0.8}, agent_signals={'agent1': 0.5}, ensemble_signal=0.5, disagreement=0.2, effective_confidence=0.8, kalman_gain=0.5, kalman_price=100.0, kalman_trend=0.01, capital_gate_verdict='ALLOW', effective_cap=0.5, state_charges={'economic': 1.0}, position_action='BUY', quantity=1.0, confidence=0.8, expected_outcome='Price up', realized_outcome='PENDING', pnl=0.0, lesson='')
        self.memory.log_experience(exp1)
        current = TradeExperience(decision_id='test-decision', timestamp=datetime.now(), symbol='AAPL', regime='R01', regime_probabilities={'R01': 0.8}, agent_signals={'agent1': 0.5}, ensemble_signal=0.5, disagreement=0.2, effective_confidence=0.8, kalman_gain=0.5, kalman_price=100.0, kalman_trend=0.01, capital_gate_verdict='ALLOW', effective_cap=0.5, state_charges={'economic': 1.0}, position_action='BUY', quantity=1.0, confidence=0.8, expected_outcome='Price up', realized_outcome='PENDING', pnl=0.0, lesson='')
        similar = self.memory.find_similar(current, top_k=5)
        self.assertEqual(len(similar), 1)
        self.assertIsInstance(similar[0], SimilarExperience)
        self.assertGreater(similar[0].similarity_score, 0.5)

    def test_performance_summary(self):
        """Verify performance summary computation."""
        exp1 = TradeExperience(decision_id='test-decision', timestamp=datetime.now(), symbol='AAPL', regime='R01', regime_probabilities={'R01': 0.8}, agent_signals={'agent1': 0.5}, ensemble_signal=0.5, disagreement=0.2, effective_confidence=0.8, kalman_gain=0.5, kalman_price=100.0, kalman_trend=0.01, capital_gate_verdict='ALLOW', effective_cap=0.5, state_charges={'economic': 1.0}, position_action='BUY', quantity=1.0, confidence=0.8, expected_outcome='Price up', realized_outcome='PENDING', pnl=100.0, lesson='')
        exp2 = TradeExperience(decision_id='test-decision', timestamp=datetime.now(), symbol='AAPL', regime='R01', regime_probabilities={'R01': 0.8}, agent_signals={'agent1': 0.5}, ensemble_signal=0.5, disagreement=0.2, effective_confidence=0.8, kalman_gain=0.5, kalman_price=100.0, kalman_trend=0.01, capital_gate_verdict='ALLOW', effective_cap=0.5, state_charges={'economic': 1.0}, position_action='BUY', quantity=1.0, confidence=0.8, expected_outcome='Price up', realized_outcome='PENDING', pnl=-50.0, lesson='')
        self.memory.log_experience(exp1)
        self.memory.log_experience(exp2)
        summary = self.memory.get_performance_summary()
        self.assertEqual(summary['count'], 2)
        self.assertEqual(summary['wins'], 1)
        self.assertEqual(summary['losses'], 1)
        self.assertEqual(summary['total_pnl'], 50.0)

    def test_memory_limits_enforced(self):
        """Verify per-symbol memory limits are enforced."""
        for i in range(MAX_MEMORY_PER_SYMBOL + 10):
            exp = TradeExperience(decision_id='test-decision', timestamp=datetime.now(), symbol='AAPL', regime='R01', regime_probabilities={'R01': 0.8}, agent_signals={'agent1': 0.5}, ensemble_signal=0.5, disagreement=0.2, effective_confidence=0.8, kalman_gain=0.5, kalman_price=100.0, kalman_trend=0.01, capital_gate_verdict='ALLOW', effective_cap=0.5, state_charges={'economic': 1.0}, position_action='BUY', quantity=1.0, confidence=0.8, expected_outcome='Price up', realized_outcome='PENDING', pnl=0.0, lesson='')
            self.memory.log_experience(exp)
        history = self.memory.get_history('AAPL')
        self.assertLessEqual(len(history), MAX_MEMORY_PER_SYMBOL)

class TestOrchestrator(unittest.TestCase):
    """Test X Quant X orchestrator."""

    def setUp(self):
        """Create fresh orchestrator with unique memory file for each test."""
        self.memory_file = f'test_orchestrator_memory_{id(self)}.json'
        self.orchestrator = XQuantXOrchestrator(agent_ids=AGENT_IDS, symbol='AAPL', use_hmm=False, enable_trading=False, memory_file=self.memory_file)

    def tearDown(self):
        """Clean up test memory file."""
        import os
        for f in [self.memory_file, f'{self.memory_file}.tmp']:
            if os.path.exists(f):
                try:
                    os.remove(f)
                except PermissionError:
                    pass

    def test_orchestrator_initialization(self):
        """Verify orchestrator initializes correctly."""
        self.assertIsNotNone(self.orchestrator)

    def test_orchestrator_rejects_empty_agent_ids(self):
        """Verify orchestrator rejects empty agent_ids."""
        with self.assertRaises(ValueError):
            XQuantXOrchestrator(agent_ids=[], symbol='AAPL')

    def test_orchestrator_rejects_empty_symbol(self):
        """Verify orchestrator rejects empty symbol."""
        with self.assertRaises(ValueError):
            XQuantXOrchestrator(agent_ids=AGENT_IDS, symbol='')

    def test_run_cycle_returns_result(self):
        """Verify run_cycle returns CycleResult."""
        prices = [100.0 + i * 0.1 for i in range(45)]
        volumes = [1000.0] * 45
        agents = make_agent_outputs(signals=[0.5] * 7, confidences=[0.9] * 7)
        from investment_agent.capital.capital_gate import SevenStateVector
        states = SevenStateVector(economic=1.0, financial=1.0, fiscal=1.0, portfolio=1.0, fundamental=1.0, market=1.0, sector=1.0)
        result = self.orchestrator.run_cycle(prices=prices, volumes=volumes, agent_outputs=agents, states=states, portfolio_context={'position_pct': 0.05, 'gross_leverage': 0.5, 'entropy': 0.1, 'drawdown_pct': 0.01, 'execution_timeout_seconds': 5.0, 'sector_exposure_pct': 0.1, 'is_new_long': False, 'regime': 'R01', 'available_liquidity': 100000.0})
        self.assertIsInstance(result, CycleResult)
        self.assertIsInstance(result.decision, TradingDecision)
        self.assertIn(result.regime.regime, VALID_REGIMES)
        self.assertEqual(len(result.weights), 7)

    def test_run_cycle_records_experience(self):
        """Verify run_cycle records trade experience."""
        prices = [100.0 + i * 0.1 for i in range(45)]
        volumes = [1000.0] * 45
        agents = make_agent_outputs(signals=[0.5] * 7, confidences=[0.9] * 7)
        from investment_agent.capital.capital_gate import SevenStateVector
        states = SevenStateVector(economic=1.0, financial=1.0, fiscal=1.0, portfolio=1.0, fundamental=1.0, market=1.0, sector=1.0)
        result = self.orchestrator.run_cycle(prices=prices, volumes=volumes, agent_outputs=agents, states=states, portfolio_context={'position_pct': 0.05, 'gross_leverage': 0.5, 'entropy': 0.1, 'drawdown_pct': 0.01, 'execution_timeout_seconds': 5.0, 'sector_exposure_pct': 0.1, 'is_new_long': False, 'regime': 'R01', 'available_liquidity': 100000.0})
        self.assertIsInstance(result.experience, TradeExperience)
        self.assertEqual(result.experience.symbol, 'AAPL')
        self.assertEqual(result.experience.position_action, result.decision.action)

    def test_run_cycle_provenance_trace(self):
        """Verify run_cycle produces complete provenance."""
        prices = [100.0 + i * 0.1 for i in range(45)]
        volumes = [1000.0] * 45
        agents = make_agent_outputs(signals=[0.5] * 7, confidences=[0.9] * 7)
        from investment_agent.capital.capital_gate import SevenStateVector
        states = SevenStateVector(economic=1.0, financial=1.0, fiscal=1.0, portfolio=1.0, fundamental=1.0, market=1.0, sector=1.0)
        result = self.orchestrator.run_cycle(prices=prices, volumes=volumes, agent_outputs=agents, states=states, portfolio_context={'position_pct': 0.05, 'gross_leverage': 0.5, 'entropy': 0.1, 'drawdown_pct': 0.01, 'execution_timeout_seconds': 5.0, 'sector_exposure_pct': 0.1, 'is_new_long': False, 'regime': 'R01', 'available_liquidity': 100000.0})
        provenance = result.decision.provenance
        self.assertIn('regime', provenance)
        self.assertIn('ensemble_signal', provenance)
        self.assertIn('kalman_gain', provenance)
        self.assertIn('effective_cap', provenance)
        self.assertIn('verdict', provenance)
        self.assertIn('weights', provenance)

    def test_learn_from_outcome_updates_memory(self):
        """Verify learn_from_outcome updates experience and reputation."""
        experience = TradeExperience(decision_id='test-decision', timestamp=datetime.now(), symbol='AAPL', regime='R01', regime_probabilities={'R01': 0.8}, agent_signals={'agent1': 0.5}, ensemble_signal=0.5, disagreement=0.2, effective_confidence=0.8, kalman_gain=0.5, kalman_price=100.0, kalman_trend=0.01, capital_gate_verdict='ALLOW', effective_cap=0.5, state_charges={'economic': 1.0}, position_action='BUY', quantity=1.0, confidence=0.8, expected_outcome='Price up', realized_outcome='PENDING', pnl=0.0, lesson='')
        # P0-1: must first log the PENDING experience, then close it.
        self.orchestrator._trade_memory.log_experience(experience)
        self.orchestrator.learn_from_outcome(experience=experience, realized_pnl=100.0, realized_outcome='Profit target hit', lesson='Strong trend continuation in R01')
        history = self.orchestrator.get_trade_history('AAPL')
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0].pnl, 100.0)
        self.assertEqual(history[0].realized_outcome, 'Profit target hit')

    def test_learn_from_outcome_negative_pnl(self):
        """Verify learn_from_outcome handles negative P&L correctly."""
        experience = TradeExperience(decision_id='test-decision', timestamp=datetime.now(), symbol='AAPL', regime='R01', regime_probabilities={'R01': 0.8}, agent_signals={'agent1': 0.5}, ensemble_signal=0.5, disagreement=0.2, effective_confidence=0.8, kalman_gain=0.5, kalman_price=100.0, kalman_trend=0.01, capital_gate_verdict='ALLOW', effective_cap=0.5, state_charges={'economic': 1.0}, position_action='BUY', quantity=1.0, confidence=0.8, expected_outcome='Price up', realized_outcome='PENDING', pnl=0.0, lesson='')
        # P0-1: log then close
        self.orchestrator._trade_memory.log_experience(experience)
        self.orchestrator.learn_from_outcome(experience=experience, realized_pnl=-50.0, realized_outcome='Stop loss hit', lesson='Overestimated trend strength in R01')
        history = self.orchestrator.get_trade_history('AAPL')
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0].pnl, -50.0)

    def test_historical_context_returns_stats(self):
        """Verify get_historical_context returns aggregated statistics."""
        for i in range(10):
            exp = TradeExperience(decision_id='test-decision', timestamp=datetime.now(), symbol='AAPL', regime='R01', regime_probabilities={'R01': 0.8}, agent_signals={'agent1': 0.5}, ensemble_signal=0.5, disagreement=0.2, effective_confidence=0.8, kalman_gain=0.5, kalman_price=100.0, kalman_trend=0.01, capital_gate_verdict='ALLOW', effective_cap=0.5, state_charges={'economic': 1.0}, position_action='BUY', quantity=1.0, confidence=0.8, expected_outcome='Price up', realized_outcome='PENDING', pnl=100.0 if i % 2 == 0 else -50.0, lesson=f'Lesson {i}')
            self.orchestrator._trade_memory.log_experience(exp)
        current = TradeExperience(decision_id='test-decision', timestamp=datetime.now(), symbol='AAPL', regime='R01', regime_probabilities={'R01': 0.8}, agent_signals={'agent1': 0.5}, ensemble_signal=0.5, disagreement=0.2, effective_confidence=0.8, kalman_gain=0.5, kalman_price=100.0, kalman_trend=0.01, capital_gate_verdict='ALLOW', effective_cap=0.5, state_charges={'economic': 1.0}, position_action='BUY', quantity=1.0, confidence=0.8, expected_outcome='Price up', realized_outcome='PENDING', pnl=0.0, lesson='')
        context = self.orchestrator.get_historical_context(current, top_k=10)
        self.assertIn('similar_trades', context)
        self.assertIn('historical_win_rate', context)
        self.assertIn('avg_pnl', context)
        self.assertIn('lessons', context)
        self.assertIn('confidence_adjustment', context)
        self.assertEqual(context['wins'], 5)
        self.assertEqual(context['losses'], 5)

class TestLifecycleAndMemoryFirst(unittest.TestCase):
    """P0-1, P0-2, P1-3, P1-4, P1-5."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.memory_file = os.path.join(self.tmp, "trade_memory.json")
        self.audit_file = os.path.join(self.tmp, "audit_log.jsonl")
        self.orchestrator = XQuantXOrchestrator(
            agent_ids=AGENT_IDS,
            symbol="AAPL",
            use_hmm=False,
            enable_trading=False,
            memory_file=self.memory_file,
        )
        # Override audit log file
        self.orchestrator._audit_log = AuditLog(log_file=self.audit_file)

    def tearDown(self):
        for f in [self.memory_file, self.audit_file]:
            try:
                os.remove(f)
            except OSError:
                pass
        try:
            os.rmdir(self.tmp)
        except OSError:
            pass

    def test_lifecycle_pending_fill_then_closed(self):
        """P0-1: pending experience becomes CLOSED only on close_trade()."""
        prices = [100.0 + i * 0.1 for i in range(45)]
        volumes = [1000.0] * 45
        agents = make_agent_outputs(signals=[0.5] * 7, confidences=[0.9] * 7)
        from investment_agent.capital.capital_gate import SevenStateVector
        states = SevenStateVector(
            economic=1.0, financial=1.0, fiscal=1.0,
            portfolio=1.0, fundamental=1.0, market=1.0, sector=1.0,
        )
        result = self.orchestrator.run_cycle(
            prices=prices, volumes=volumes, agent_outputs=agents, states=states,
            portfolio_context={
                "position_pct": 0.05, "gross_leverage": 0.5, "entropy": 0.1,
                "drawdown_pct": 0.01, "execution_timeout_seconds": 5.0,
                "sector_exposure_pct": 0.1, "is_new_long": False, "regime": "R01",
                "available_liquidity": 100000.0,
            },
        )
        self.assertEqual(result.experience.lifecycle_status, TradeLifecycle.PENDING_FILL.value)
        # Now close
        closed = self.orchestrator.close_trade(
            decision_id=result.decision.decision_id,
            realized_outcome="Hit target",
            pnl=150.0,
            lesson="Pattern held",
        )
        self.assertEqual(closed.lifecycle_status, TradeLifecycle.CLOSED.value)
        self.assertEqual(closed.pnl, 150.0)
        self.assertEqual(closed.closed_at is not None, True)

    def test_pending_fill_does_not_update_reputation(self):
        """P0-1: reputation tracker must not be mutated by a pending trade."""
        prices = [100.0 + i * 0.1 for i in range(45)]
        volumes = [1000.0] * 45
        agents = make_agent_outputs(signals=[0.5] * 7, confidences=[0.9] * 7)
        from investment_agent.capital.capital_gate import SevenStateVector
        states = SevenStateVector(
            economic=1.0, financial=1.0, fiscal=1.0,
            portfolio=1.0, fundamental=1.0, market=1.0, sector=1.0,
        )
        before = {aid: self.orchestrator._reputation_tracker.get_reputation_weight(aid, "R01")
                  for aid in AGENT_IDS}
        self.orchestrator.run_cycle(
            prices=prices, volumes=volumes, agent_outputs=agents, states=states,
            portfolio_context={
                "position_pct": 0.05, "gross_leverage": 0.5, "entropy": 0.1,
                "drawdown_pct": 0.01, "execution_timeout_seconds": 5.0,
                "sector_exposure_pct": 0.1, "is_new_long": False, "regime": "R01",
                "available_liquidity": 100000.0,
            },
        )
        after = {aid: self.orchestrator._reputation_tracker.get_reputation_weight(aid, "R01")
                 for aid in AGENT_IDS}
        self.assertEqual(before, after, "Pending trade must not affect reputation")

    def test_memory_retrieved_before_capital_gate(self):
        """P0-2: provenance must include memory fields when similar trades exist."""
        # Seed 3 closed wins so retrieval triggers a non-empty context.
        for i in range(3):
            exp = TradeExperience(
                decision_id=f"seed-{i}", timestamp=datetime.now(), symbol="AAPL",
                regime="R01", regime_probabilities={"R01": 0.8},
                agent_signals={a: 0.5 for a in AGENT_IDS}, ensemble_signal=0.5,
                disagreement=0.2, effective_confidence=0.8, kalman_gain=0.5,
                kalman_price=100.0, kalman_trend=0.01, capital_gate_verdict="ALLOW",
                effective_cap=0.5, state_charges={"economic": 1.0},
                position_action="BUY", quantity=1.0, confidence=0.8,
                expected_outcome="Price up", realized_outcome="Hit target",
                pnl=100.0, lesson="Pattern held", lifecycle_status="CLOSED",
            )
            self.orchestrator._trade_memory.log_experience(exp)

        prices = [100.0 + i * 0.1 for i in range(45)]
        volumes = [1000.0] * 45
        agents = make_agent_outputs(signals=[0.5] * 7, confidences=[0.9] * 7)
        from investment_agent.capital.capital_gate import SevenStateVector
        states = SevenStateVector(
            economic=1.0, financial=1.0, fiscal=1.0,
            portfolio=1.0, fundamental=1.0, market=1.0, sector=1.0,
        )
        result = self.orchestrator.run_cycle(
            prices=prices, volumes=volumes, agent_outputs=agents, states=states,
            portfolio_context={
                "position_pct": 0.05, "gross_leverage": 0.5, "entropy": 0.1,
                "drawdown_pct": 0.01, "execution_timeout_seconds": 5.0,
                "sector_exposure_pct": 0.1, "is_new_long": False, "regime": "R01",
                "available_liquidity": 100000.0,
            },
        )
        self.assertTrue(result.decision.provenance["memory_used"])
        self.assertGreaterEqual(result.decision.provenance["similar_trade_count"], 1)
        self.assertGreaterEqual(result.decision.provenance["historical_win_rate"], 0.0)

    def test_audit_log_persists_across_restart(self):
        """P1-3: events written before restart are still queryable after restart."""
        from investment_agent.orchestrator import AuditEvent
        log1 = AuditLog(log_file=self.audit_file)
        ev = log1.record_decision("d-1", "AAPL", {"x": 1})
        # Force a fresh AuditLog instance pointing at the same file
        log2 = AuditLog(log_file=self.audit_file)
        queried = log2.query_by_decision_id("d-1")
        self.assertGreaterEqual(len(queried), 1)
        self.assertEqual(queried[0].symbol, "AAPL")

    def test_similarity_market_vs_portfolio_split(self):
        """P1-4: SimilarExperience must report market and portfolio scores."""
        memory = TradeMemory(memory_file=self.memory_file)
        hist = TradeExperience(
            decision_id="h-1", timestamp=datetime.now(), symbol="AAPL",
            regime="R01", regime_probabilities={"R01": 0.9, "R02": 0.1},
            agent_signals={a: 0.5 for a in AGENT_IDS}, ensemble_signal=0.5,
            disagreement=0.2, effective_confidence=0.8, kalman_gain=0.5,
            kalman_price=100.0, kalman_trend=0.01, capital_gate_verdict="ALLOW",
            effective_cap=0.5, state_charges={"economic": 1.0},
            position_action="BUY", quantity=1.0, confidence=0.8,
            expected_outcome="x", realized_outcome="x", pnl=0.0, lesson="",
            lifecycle_status="CLOSED",
        )
        memory.log_experience(hist)
        current = hist  # identical -> max similarity
        # Override capital_gate_verdict / effective_cap to break portfolio sim
        current_diff_cap = TradeExperience(
            **{
                **{f: getattr(hist, f) for f in hist.__dataclass_fields__},
                "decision_id": "cur-1",
                "effective_cap": 0.05,  # very different from hist
            }
        )
        results = memory.find_similar(current_diff_cap)
        self.assertEqual(len(results), 1)
        s = results[0]
        self.assertGreater(s.market_similarity, 0.8)
        self.assertLess(s.portfolio_similarity, s.market_similarity + 0.5)

    def test_find_similar_excludes_self(self):
        """P1-5: passing exclude_decision_id hides the just-recorded experience."""
        memory = TradeMemory(memory_file=self.memory_file)
        exp = TradeExperience(
            decision_id="self", timestamp=datetime.now(), symbol="AAPL",
            regime="R01", regime_probabilities={"R01": 0.8},
            agent_signals={a: 0.5 for a in AGENT_IDS}, ensemble_signal=0.5,
            disagreement=0.2, effective_confidence=0.8, kalman_gain=0.5,
            kalman_price=100.0, kalman_trend=0.01, capital_gate_verdict="ALLOW",
            effective_cap=0.5, state_charges={"economic": 1.0},
            position_action="BUY", quantity=1.0, confidence=0.8,
            expected_outcome="x", realized_outcome="x", pnl=0.0, lesson="",
            lifecycle_status="CLOSED",
        )
        memory.log_experience(exp)
        results = memory.find_similar(exp, exclude_decision_id="self")
        self.assertEqual(len(results), 0)

    def test_liquidity_from_provider(self):
        """P1-1: liquidity provider overrides portfolio_context default."""
        provider_calls = []
        def provider():
            provider_calls.append(1)
            return 12345.0
        self.orchestrator.set_liquidity_provider(provider)
        prices = [100.0 + i * 0.1 for i in range(45)]
        volumes = [1000.0] * 45
        agents = make_agent_outputs(signals=[0.5] * 7, confidences=[0.9] * 7)
        from investment_agent.capital.capital_gate import SevenStateVector
        states = SevenStateVector(
            economic=1.0, financial=1.0, fiscal=1.0,
            portfolio=1.0, fundamental=1.0, market=1.0, sector=1.0,
        )
        self.orchestrator.run_cycle(
            prices=prices, volumes=volumes, agent_outputs=agents, states=states,
            portfolio_context={
                "position_pct": 0.05, "gross_leverage": 0.5, "entropy": 0.1,
                "drawdown_pct": 0.01, "execution_timeout_seconds": 5.0,
                "sector_exposure_pct": 0.1, "is_new_long": False, "regime": "R01",
                # available_liquidity NOT provided -> provider should be used
            },
        )
        self.assertGreater(len(provider_calls), 0)


if __name__ == '__main__':
    unittest.main()