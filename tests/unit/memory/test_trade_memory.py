"""Unit tests for trade_memory.py -- TradeExperience lifecycle, memory persistence, and retrieval."""

import os
import unittest
from datetime import datetime, timezone
from investment_agent.memory.trade_memory import (
    TradeExperience,
    TradeLifecycle,
    TradeMemory,
    SimilarExperience,
)

TEST_MEMORY_FILE = "test_trade_memory.json"


class TestTradeLifecycle(unittest.TestCase):
    def test_terminal_states(self):
        self.assertTrue(TradeLifecycle.CLOSED.is_terminal)
        self.assertTrue(TradeLifecycle.REJECTED.is_terminal)
        self.assertTrue(TradeLifecycle.CANCELLED.is_terminal)
        self.assertFalse(TradeLifecycle.PENDING_FILL.is_terminal)
        self.assertFalse(TradeLifecycle.OPEN.is_terminal)

    def test_has_realized_pnl(self):
        self.assertTrue(TradeLifecycle.CLOSED.has_realized_pnl)
        self.assertFalse(TradeLifecycle.OPEN.has_realized_pnl)
        self.assertFalse(TradeLifecycle.REJECTED.has_realized_pnl)


class TestTradeMemoryOperations(unittest.TestCase):
    def setUp(self):
        if os.path.exists(TEST_MEMORY_FILE):
            os.remove(TEST_MEMORY_FILE)
        self.memory = TradeMemory(memory_file=TEST_MEMORY_FILE)

    def tearDown(self):
        if os.path.exists(TEST_MEMORY_FILE):
            os.remove(TEST_MEMORY_FILE)

    def _make_experience(self, decision_id: str = "dec-1", symbol: str = "AAPL", regime: str = "R01"):
        return TradeExperience(
            decision_id=decision_id,
            timestamp=datetime.now(timezone.utc),
            symbol=symbol,
            regime=regime,
            regime_probabilities={"R01": 0.8, "R02": 0.2},
            agent_signals={"agent1": 0.5, "agent2": -0.2},
            ensemble_signal=0.4,
            disagreement=0.1,
            effective_confidence=0.85,
            kalman_gain=0.75,
            kalman_price=150.0,
            kalman_trend=0.01,
            capital_gate_verdict="ALLOW",
            effective_cap=1.0,
            state_charges={"market": 1.0},
            position_action="BUY",
            quantity=10.0,
            confidence=0.8,
            expected_outcome="Positive return",
            realized_outcome="",
            pnl=0.0,
            lesson="",
            lifecycle_status=TradeLifecycle.PENDING_FILL.value,
        )

    def test_log_and_retrieve_experience(self):
        exp = self._make_experience(decision_id="dec-100")
        self.memory.log_experience(exp)

        retrieved = self.memory.get_by_decision_id("dec-100")
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.decision_id, "dec-100")
        self.assertEqual(retrieved.lifecycle_status, TradeLifecycle.PENDING_FILL.value)

    def test_update_lifecycle_status(self):
        exp = self._make_experience(decision_id="dec-101")
        self.memory.log_experience(exp)

        # Transition PENDING_FILL -> OPEN
        updated = self.memory.update_experience(
            "dec-101",
            lifecycle_status=TradeLifecycle.OPEN.value,
            order_id="ord-999",
            fill_price=150.25,
        )
        self.assertEqual(updated.lifecycle_status, TradeLifecycle.OPEN.value)
        self.assertEqual(updated.order_id, "ord-999")
        self.assertEqual(updated.fill_price, 150.25)

    def test_close_trade_realized_pnl(self):
        exp = self._make_experience(decision_id="dec-102")
        self.memory.log_experience(exp)

        closed = self.memory.close_trade(
            decision_id="dec-102",
            realized_outcome="Profit taking",
            pnl=250.0,
            lesson="Good trend follow",
        )
        self.assertEqual(closed.lifecycle_status, TradeLifecycle.CLOSED.value)
        self.assertEqual(closed.pnl, 250.0)
        self.assertEqual(closed.lesson, "Good trend follow")
        self.assertTrue(closed.closed_at is not None)

    def test_find_similar_excludes_self(self):
        exp1 = self._make_experience(decision_id="dec-200")
        exp2 = self._make_experience(decision_id="dec-201")
        self.memory.log_experience(exp1)
        self.memory.log_experience(exp2)

        similar = self.memory.find_similar(exp1, top_k=5, exclude_decision_id="dec-200")
        decision_ids = [s.experience.decision_id for s in similar]
        self.assertNotIn("dec-200", decision_ids)
        self.assertIn("dec-201", decision_ids)


if __name__ == "__main__":
    unittest.main()
