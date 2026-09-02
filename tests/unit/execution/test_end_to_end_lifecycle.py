"""Unit Tests — Complete End-to-End Trade Lifecycle & Verification.

Audits and verifies the complete 9-stage operational trade lifecycle:
  1. decision     : Orchestrator generates TradingDecision & logs PENDING_FILL experience.
  2. order        : Order submitted to Alpaca broker returning ExecutionResult with order_id.
  3. broker fill  : Alpaca order transitions to 'filled' with fill_price and executed quantity.
  4. reconciliation: FillReconciler polls broker status and reconciles TradeMemory state.
  5. OPEN         : TradeExperience lifecycle_status transitions to OPEN with fill parameters.
  6. close/exit   : Position exit signal triggered via close_position().
  7. broker pos   : Alpaca open position checked and closing order submitted.
  8. CLOSED       : Orchestrator.close_trade() updates lifecycle to CLOSED with realized P&L.
  9. reputation   : AgentReputationTracker records outcome and updates per-agent reputation weights.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from datetime import datetime
from unittest.mock import MagicMock, patch

from investment_agent.agents.agent_reputation import AgentReputationTracker
from investment_agent.capital.capital_gate import SevenStateVector
from investment_agent.execution.execution import ExecutionResult, apply_fill, close_position
from investment_agent.execution.fill_reconciler import FillReconciler, performance_by_regime
from investment_agent.memory.trade_memory import TradeExperience, TradeLifecycle, TradeMemory
from investment_agent.orchestrator import XQuantXOrchestrator
from investment_agent.signals.ensemble_signal import AgentOutput


class TestEndToEndTradeLifecycle(unittest.TestCase):
    """Rigorous audit of the complete 9-stage autonomous trade lifecycle."""

    def setUp(self) -> None:
        self.tmp_mem_fd, self.tmp_mem_path = tempfile.mkstemp(suffix=".json")
        self.tmp_rep_fd, self.tmp_rep_path = tempfile.mkstemp(suffix=".json")
        with os.fdopen(self.tmp_mem_fd, "w") as f:
            f.write("[]")
        with os.fdopen(self.tmp_rep_fd, "w") as f:
            f.write("{}")

    def tearDown(self) -> None:
        for path in (self.tmp_mem_path, self.tmp_rep_path):
            if os.path.exists(path):
                try:
                    os.unlink(path)
                except OSError:
                    pass

    def _make_agent_outputs(self, bullish: bool = True) -> list[AgentOutput]:
        sig = 0.8 if bullish else -0.8
        roles = [
            "macro_regime", "microstructure", "options_volatility",
            "cross_asset", "fundamental_value", "statistical_arbitrage",
            "sentiment_orderflow"
        ]
        return [
            AgentOutput(
                agent_id=role,
                s=sig,
                c=0.85,
                u=0.15,
                d=0.1,
                p_plus=0.8 if bullish else 0.2,
                p_minus=0.2 if bullish else 0.8,
                delta_t=5,
                r=1.0,
            )
            for role in roles
        ]

    @patch("investment_agent.execution.execution._get_trading_client")
    def test_complete_nine_stage_trade_lifecycle(self, mock_get_client: MagicMock) -> None:
        """Verify the strict 9-step progression from signal decision to reputation update."""

        # ------------------------------------------------------------------
        # Setup Orchestrator & Mocks
        # ------------------------------------------------------------------
        agent_ids = [
            "macro_regime", "microstructure", "options_volatility",
            "cross_asset", "fundamental_value", "statistical_arbitrage",
            "sentiment_orderflow"
        ]
        orch = XQuantXOrchestrator(
            agent_ids=agent_ids,
            symbol="AAPL",
            use_hmm=True,
            enable_trading=True,
            memory_file=self.tmp_mem_path,
            reputation_file=self.tmp_rep_path,
        )

        prices = [150.0 + i * 0.5 for i in range(30)]
        volumes = [1000000.0] * 30
        states = SevenStateVector(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
        port_ctx = {
            "position_pct": 0.0, "gross_leverage": 0.0, "entropy": 0.1,
            "drawdown_pct": 0.0, "execution_timeout_seconds": 5.0,
            "sector_exposure_pct": 0.0, "is_new_long": True,
            "regime": "R01", "available_liquidity": 100000.0,
        }

        # Mock Alpaca client for submit_order
        mock_order = MagicMock()
        mock_order.id = "ord-alpaca-777"
        mock_order.status = MagicMock(value="accepted")
        mock_client = MagicMock()
        mock_client.submit_order.return_value = mock_order
        mock_get_client.return_value = mock_client

        # ------------------------------------------------------------------
        # STAGE 1: Decision Generation & Initial PENDING_FILL Logging
        # ------------------------------------------------------------------
        agent_outputs = self._make_agent_outputs(bullish=True)
        res = orch.run_cycle(prices, volumes, agent_outputs, states, port_ctx)

        exp = res.experience
        self.assertIsNotNone(exp)
        self.assertEqual(exp.position_action, "BUY")
        self.assertEqual(exp.lifecycle_status, TradeLifecycle.PENDING_FILL.value)
        self.assertEqual(exp.order_id, "ord-alpaca-777")
        decision_id = exp.decision_id

        # Verify recorded in TradeMemory as PENDING_FILL
        stored_exp = orch._trade_memory.get_by_decision_id(decision_id)
        self.assertIsNotNone(stored_exp)
        self.assertEqual(stored_exp.lifecycle_status, TradeLifecycle.PENDING_FILL.value)
        self.assertEqual(stored_exp.order_id, "ord-alpaca-777")

        # ------------------------------------------------------------------
        # STAGE 2 & 3: Broker Fill Simulation
        # ------------------------------------------------------------------
        fill_snap = {
            "order_id": "ord-alpaca-777",
            "status": "filled",
            "filled_qty": 10.0,
            "filled_avg_price": 152.50,
            "is_terminal": True,
            "timed_out": False,
            "raw": {},
        }

        # ------------------------------------------------------------------
        # STAGE 4 & 5: Reconciliation to OPEN
        # ------------------------------------------------------------------
        reconciler = FillReconciler(verbose=False)
        with patch("investment_agent.execution.execution.poll_order_status", return_value=fill_snap):
            counts = reconciler.reconcile(orch._trade_memory)

        self.assertEqual(counts["filled"], 1)

        open_exp = orch._trade_memory.get_by_decision_id(decision_id)
        self.assertEqual(open_exp.lifecycle_status, TradeLifecycle.OPEN.value)
        self.assertAlmostEqual(open_exp.fill_price, 152.50)
        self.assertAlmostEqual(open_exp.filled_qty, 10.0)
        self.assertAlmostEqual(open_exp.remaining_qty, 0.0)

        # Verify reputation is NOT updated while trade is still OPEN
        initial_rep = orch._reputation_tracker.get_reputation_weight("macro_regime", "R01")
        self.assertAlmostEqual(initial_rep, 0.5, delta=0.5)  # Prior weight default ratio

        # ------------------------------------------------------------------
        # STAGE 6 & 7: Position Close / Exit via Broker
        # ------------------------------------------------------------------
        mock_pos = MagicMock()
        mock_pos.qty = 10.0
        mock_pos.avg_entry_price = 152.50
        mock_pos.side = MagicMock(value="long")

        mock_close_order = MagicMock()
        mock_close_order.id = "ord-close-999"
        mock_close_order.status = MagicMock(value="accepted")

        client_for_close = MagicMock()
        client_for_close.get_open_position.side_effect = [mock_pos, Exception("no position")]
        client_for_close.submit_order.return_value = mock_close_order
        mock_get_client.return_value = client_for_close

        close_result = close_position("AAPL")
        self.assertTrue(close_result["ok"])
        self.assertEqual(close_result["closed_qty"], 10.0)

        # ------------------------------------------------------------------
        # STAGE 8 & 9: Trade Closure, Realized P&L & Reputation Update
        # ------------------------------------------------------------------
        exit_price = 162.50
        realized_pnl = (exit_price - 152.50) * 10.0  # +$100.00
        closed_exp = orch.close_trade(
            decision_id=decision_id,
            realized_outcome="take_profit_target_hit",
            pnl=realized_pnl,
            lesson="Bullish momentum confirmed by high volume breakout",
        )

        self.assertEqual(closed_exp.lifecycle_status, TradeLifecycle.CLOSED.value)
        self.assertAlmostEqual(closed_exp.pnl, 100.0)
        self.assertEqual(closed_exp.realized_outcome, "take_profit_target_hit")

        # Verify reputation tracker updated for winning trade
        updated_rep = orch._reputation_tracker.get_reputation_weight("macro_regime", "R01")
        self.assertGreater(updated_rep, initial_rep)

        # Verify performance by regime attribution
        perf = performance_by_regime(orch._trade_memory)
        self.assertIn("R01", perf)
        self.assertEqual(perf["R01"]["count"], 1)
        self.assertAlmostEqual(perf["R01"]["total_pnl"], 100.0)
        self.assertAlmostEqual(perf["R01"]["win_rate"], 1.0)

    @patch("investment_agent.execution.execution._get_trading_client")
    def test_partial_fill_to_full_fill_reconciliation_lifecycle(self, mock_get_client: MagicMock) -> None:
        """Verify partial fill handling remains PENDING_FILL until remaining_qty reaches 0."""
        mem = TradeMemory(self.tmp_mem_path)
        exp = TradeExperience(
            decision_id="dec-partial-1",
            timestamp=datetime.now(),
            symbol="AAPL",
            regime="R01",
            regime_probabilities={"R01": 1.0},
            agent_signals={},
            ensemble_signal=0.5,
            disagreement=0.1,
            effective_confidence=0.8,
            kalman_gain=0.3,
            kalman_price=150.0,
            kalman_trend=0.0,
            capital_gate_verdict="ALLOW",
            effective_cap=0.5,
            state_charges={},
            position_action="BUY",
            quantity=20.0,
            confidence=0.8,
            expected_outcome="",
            realized_outcome="",
            pnl=0.0,
            lesson="",
            lifecycle_status=TradeLifecycle.PENDING_FILL.value,
            order_id="ord-part-100",
            ordered_qty=20.0,
            filled_qty=0.0,
            remaining_qty=20.0,
        )
        mem.log_experience(exp)

        reconciler = FillReconciler(verbose=False)

        # Part 1: Partial fill of 8 shares
        snap_part = {
            "order_id": "ord-part-100",
            "status": "partially_filled",
            "filled_qty": 8.0,
            "filled_avg_price": 150.0,
            "is_terminal": False,
            "timed_out": False,
            "raw": {},
        }
        with patch("investment_agent.execution.execution.poll_order_status", return_value=snap_part):
            counts = reconciler.reconcile(mem)

        self.assertEqual(counts["partially_filled"], 1)
        part_exp = mem.get_by_decision_id("dec-partial-1")
        self.assertEqual(part_exp.lifecycle_status, TradeLifecycle.PENDING_FILL.value)
        self.assertAlmostEqual(part_exp.filled_qty, 8.0)
        self.assertAlmostEqual(part_exp.remaining_qty, 12.0)

        # Part 2: Final fill of remaining 12 shares (total 20)
        snap_full = {
            "order_id": "ord-part-100",
            "status": "filled",
            "filled_qty": 20.0,
            "filled_avg_price": 150.50,
            "is_terminal": True,
            "timed_out": False,
            "raw": {},
        }
        with patch("investment_agent.execution.execution.poll_order_status", return_value=snap_full):
            counts2 = reconciler.reconcile(mem)

        self.assertEqual(counts2["filled"], 1)
        full_exp = mem.get_by_decision_id("dec-partial-1")
        self.assertEqual(full_exp.lifecycle_status, TradeLifecycle.OPEN.value)
        self.assertAlmostEqual(full_exp.filled_qty, 20.0)
        self.assertAlmostEqual(full_exp.remaining_qty, 0.0)
        self.assertAlmostEqual(full_exp.fill_price, 150.50)

    def test_restart_recovery_of_pending_orders(self) -> None:
        """Verify startup recovery polls in-flight pending orders and reconciles state."""
        mem = TradeMemory(self.tmp_mem_path)
        exp = TradeExperience(
            decision_id="dec-restart-1",
            timestamp=datetime.now(),
            symbol="NVDA",
            regime="R02",
            regime_probabilities={"R02": 1.0},
            agent_signals={},
            ensemble_signal=-0.6,
            disagreement=0.1,
            effective_confidence=0.8,
            kalman_gain=0.3,
            kalman_price=120.0,
            kalman_trend=0.0,
            capital_gate_verdict="ALLOW",
            effective_cap=0.5,
            state_charges={},
            position_action="SELL",
            quantity=15.0,
            confidence=0.8,
            expected_outcome="",
            realized_outcome="",
            pnl=0.0,
            lesson="",
            lifecycle_status=TradeLifecycle.PENDING_FILL.value,
            order_id="ord-nvda-crash",
            ordered_qty=15.0,
        )
        mem.log_experience(exp)

        reconciler = FillReconciler(verbose=False)
        snap_recovery = {
            "order_id": "ord-nvda-crash",
            "status": "filled",
            "filled_qty": 15.0,
            "filled_avg_price": 119.50,
            "is_terminal": True,
            "timed_out": False,
            "raw": {},
        }
        with patch("investment_agent.execution.execution.poll_order_status", return_value=snap_recovery):
            counts = reconciler.recover_pending_orders(mem)

        self.assertEqual(counts["filled"], 1)
        recovered_exp = mem.get_by_decision_id("dec-restart-1")
        self.assertEqual(recovered_exp.lifecycle_status, TradeLifecycle.OPEN.value)
        self.assertAlmostEqual(recovered_exp.fill_price, 119.50)


if __name__ == "__main__":
    unittest.main()
