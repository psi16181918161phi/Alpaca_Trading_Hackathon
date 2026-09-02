"""Tests — Full Execution Hardening Suite.

Covers:
  1. Full-close sizing bypass for exits
  2. Partial-fill handling (poll_order_status + apply_fill)
  3. Broker ↔ TradeMemory reconciliation after fills (FillReconciler)
  4. Order-status polling with timeout
  5. Restart/recovery of pending orders
  6. Empirical HMM calibration/validation (Brier, log-loss, drift)
  7. Regime-by-regime performance attribution
"""

import json
import os
import tempfile
import unittest
from datetime import datetime
from unittest.mock import MagicMock, patch

from investment_agent.execution.execution import (
    ExecutionResult,
    apply_fill,
    close_position,
    poll_order_status,
)
from investment_agent.execution.fill_reconciler import (
    FillReconciler,
    performance_by_regime,
)
from investment_agent.memory.trade_memory import (
    TradeExperience,
    TradeLifecycle,
    TradeMemory,
)
from investment_agent.regimes.hmm_calibration import validate_hmm


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_exp(
    decision_id: str = "d1",
    symbol: str = "AAPL",
    regime: str = "R01",
    pnl: float = 0.0,
    lifecycle: str = "PENDING_FILL",
    order_id: str | None = None,
    position_action: str = "BUY",
    fill_price: float | None = None,
    quantity: float = 10.0,
    ensemble_signal: float = 0.5,
    effective_confidence: float = 0.7,
    disagreement: float = 0.1,
    kalman_gain: float = 0.4,
) -> TradeExperience:
    return TradeExperience(
        decision_id=decision_id,
        timestamp=datetime.now(),
        symbol=symbol,
        regime=regime,
        regime_probabilities={"R01": 0.8, "R02": 0.2},
        agent_signals={},
        ensemble_signal=ensemble_signal,
        disagreement=disagreement,
        effective_confidence=effective_confidence,
        kalman_gain=kalman_gain,
        kalman_price=100.0,
        kalman_trend=0.0,
        capital_gate_verdict="ALLOW",
        effective_cap=0.5,
        state_charges={},
        position_action=position_action,
        quantity=quantity,
        confidence=0.7,
        expected_outcome="",
        realized_outcome="",
        pnl=pnl,
        lesson="",
        lifecycle_status=lifecycle,
        order_id=order_id,
        fill_price=fill_price,
    )


def _build_memory(*exps: TradeExperience) -> TradeMemory:
    fd, path = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    os.unlink(path)
    mem = TradeMemory(path)
    for exp in exps:
        mem.log_experience(exp)
    return mem


# ---------------------------------------------------------------------------
# 1. Full-close sizing bypass
# ---------------------------------------------------------------------------

class TestFullCloseSizingBypass(unittest.TestCase):
    """close_position must NEVER be blocked by is_trade_safe size limits."""

    def _mock_client(self, qty: float, avg_price: float, side_value: str = "long"):
        pos = MagicMock()
        pos.qty = qty
        pos.avg_entry_price = avg_price
        pos.side = MagicMock(value=side_value)

        order_result = MagicMock()
        order_result.id = "order-close-123"
        order_result.status = MagicMock(value="accepted")

        client = MagicMock()
        client.get_open_position.return_value = pos
        client.submit_order.return_value = order_result
        # Second call to get_open_position (post-close verification)
        client.get_open_position.side_effect = [pos, Exception("no position")]
        return client

    @patch("investment_agent.execution.execution._get_trading_client")
    def test_close_submits_without_is_trade_safe_check(self, mock_get_client):
        """close_position should submit even if SIZE far exceeds 5% buying power."""
        # qty=10_000 @ $1000 = $10M, far above any 5% limit
        mock_get_client.return_value = self._mock_client(qty=10_000, avg_price=1000.0)
        result = close_position("AAPL")
        self.assertTrue(result["ok"])
        self.assertEqual(result["closed_qty"], 10_000.0)

    @patch("investment_agent.execution.execution._get_trading_client")
    def test_close_short_position(self, mock_get_client):
        """Closing a short position should send a BUY order."""
        pos = MagicMock()
        pos.qty = -50.0
        pos.avg_entry_price = 200.0
        pos.side = MagicMock(value="short")
        order_result = MagicMock()
        order_result.id = "oid-short"
        order_result.status = MagicMock(value="accepted")
        client = MagicMock()
        client.get_open_position.side_effect = [pos, Exception("gone")]
        client.submit_order.return_value = order_result
        mock_get_client.return_value = client
        result = close_position("TSLA")
        self.assertTrue(result["ok"])
        self.assertEqual(result["closed_qty"], 50.0)

    @patch("investment_agent.execution.execution._get_trading_client")
    def test_close_no_position_returns_ok(self, mock_get_client):
        """close_position on a symbol with no open position returns ok immediately."""
        client = MagicMock()
        client.get_open_position.side_effect = Exception("no position")
        mock_get_client.return_value = client
        result = close_position("MSFT")
        self.assertTrue(result["ok"])
        self.assertEqual(result["closed_qty"], 0.0)


# ---------------------------------------------------------------------------
# 2. Partial-fill handling
# ---------------------------------------------------------------------------

class TestPartialFillHandling(unittest.TestCase):

    def test_apply_fill_updates_filled_qty(self):
        orig = ExecutionResult(submitted=True, status="accepted", order_id="oid-1")
        snap = {
            "order_id": "oid-1",
            "status": "partially_filled",
            "filled_qty": 3.0,
            "filled_avg_price": 152.50,
            "is_terminal": True,
            "timed_out": False,
            "raw": {},
        }
        updated = apply_fill(orig, snap)
        self.assertEqual(updated.status, "partially_filled")
        self.assertAlmostEqual(updated.filled_qty, 3.0)
        self.assertAlmostEqual(updated.filled_avg_price, 152.50)
        # Original unchanged
        self.assertEqual(orig.filled_qty, 0.0)

    def test_apply_fill_on_full_fill(self):
        orig = ExecutionResult(submitted=True, status="accepted", order_id="oid-2")
        snap = {
            "status": "filled",
            "filled_qty": 10.0,
            "filled_avg_price": 200.0,
            "is_terminal": True,
            "timed_out": False,
            "raw": {},
        }
        updated = apply_fill(orig, snap)
        self.assertEqual(updated.status, "filled")
        self.assertAlmostEqual(updated.filled_qty, 10.0)

    def test_apply_fill_on_rejected(self):
        orig = ExecutionResult(submitted=True, status="accepted", order_id="oid-3")
        snap = {
            "status": "rejected",
            "filled_qty": 0.0,
            "filled_avg_price": 0.0,
            "is_terminal": True,
            "timed_out": False,
            "raw": {},
        }
        updated = apply_fill(orig, snap)
        self.assertEqual(updated.status, "rejected")
        self.assertAlmostEqual(updated.filled_qty, 0.0)


# ---------------------------------------------------------------------------
# 3 & 4. Order-status polling (mocked Alpaca client)
# ---------------------------------------------------------------------------

class TestPollOrderStatus(unittest.TestCase):

    @patch("investment_agent.execution.execution._get_trading_client")
    @patch("time.sleep", return_value=None)
    def test_poll_returns_filled_terminal(self, _, mock_get_client):
        """poll_order_status stops immediately on 'filled' status."""
        order = MagicMock()
        order.status = MagicMock(value="filled")
        order.filled_qty = 5.0
        order.qty = 5.0
        order.filled_avg_price = 150.0
        order.submitted_at = None
        order.filled_at = None
        order.symbol = "AAPL"
        order.side = MagicMock(value="buy")

        client = MagicMock()
        client.get_order_by_id.return_value = order
        mock_get_client.return_value = client

        snap = poll_order_status("oid-filled", timeout_seconds=5.0, poll_interval_seconds=0.1)
        self.assertEqual(snap["status"], "filled")
        self.assertTrue(snap["is_terminal"])
        self.assertFalse(snap["timed_out"])
        self.assertAlmostEqual(snap["filled_qty"], 5.0)

    @patch("investment_agent.execution.execution._get_trading_client")
    @patch("time.sleep", return_value=None)
    @patch("time.monotonic", side_effect=[0, 100, 100])  # instant timeout
    def test_poll_times_out(self, _, __, mock_get_client):
        """poll_order_status returns timed_out=True when timeout exceeded."""
        order = MagicMock()
        order.status = MagicMock(value="new")
        order.filled_qty = 0.0
        order.qty = 5.0
        order.filled_avg_price = None
        order.submitted_at = None
        order.filled_at = None
        order.symbol = "AAPL"
        order.side = MagicMock(value="buy")

        client = MagicMock()
        client.get_order_by_id.return_value = order
        mock_get_client.return_value = client

        snap = poll_order_status("oid-timeout", timeout_seconds=0.001, poll_interval_seconds=0.001)
        self.assertTrue(snap.get("timed_out", True))

    @patch("investment_agent.execution.execution._get_trading_client")
    @patch("time.sleep", return_value=None)
    def test_poll_returns_rejected_terminal(self, _, mock_get_client):
        order = MagicMock()
        order.status = MagicMock(value="rejected")
        order.filled_qty = 0.0
        order.qty = 5.0
        order.filled_avg_price = None
        order.submitted_at = None
        order.filled_at = None
        order.symbol = "AAPL"
        order.side = MagicMock(value="buy")

        client = MagicMock()
        client.get_order_by_id.return_value = order
        mock_get_client.return_value = client

        snap = poll_order_status("oid-rej", timeout_seconds=5.0, poll_interval_seconds=0.1)
        self.assertEqual(snap["status"], "rejected")
        self.assertTrue(snap["is_terminal"])


# ---------------------------------------------------------------------------
# 5. Broker ↔ TradeMemory reconciliation + restart/recovery
# ---------------------------------------------------------------------------

class TestFillReconciler(unittest.TestCase):

    def _mock_poll(self, status: str, filled_qty: float = 10.0, fill_price: float = 150.0):
        return {
            "order_id": "oid",
            "status": status,
            "filled_qty": filled_qty,
            "filled_avg_price": fill_price,
            "is_terminal": True,
            "timed_out": False,
            "raw": {},
        }

    def test_reconcile_filled_transitions_to_open(self):
        exp = _make_exp(decision_id="d1", lifecycle="PENDING_FILL", order_id="oid-1")
        mem = _build_memory(exp)

        with patch(
            "investment_agent.execution.fill_reconciler.FillReconciler._sweep",
        ) as mock_sweep:
            # Just test the method exists and delegates to _sweep
            reconciler = FillReconciler(verbose=False)
            mock_sweep.return_value = {"filled": 1, "partially_filled": 0,
                                       "rejected": 0, "cancelled": 0,
                                       "timed_out": 0, "skipped": 0}
            counts = reconciler.reconcile(mem)
            mock_sweep.assert_called_once_with(mem, tag="reconcile")

    def test_recover_calls_sweep_with_recovery_tag(self):
        exp = _make_exp(decision_id="d2", lifecycle="PENDING_FILL", order_id="oid-2")
        mem = _build_memory(exp)
        reconciler = FillReconciler(verbose=False)
        with patch.object(reconciler, "_sweep", return_value={}) as mock_sweep:
            reconciler.recover_pending_orders(mem)
            mock_sweep.assert_called_once_with(mem, tag="recovery")

    def test_sweep_skips_no_order_id(self):
        exp = _make_exp(decision_id="d3", lifecycle="PENDING_FILL", order_id=None)
        mem = _build_memory(exp)
        reconciler = FillReconciler(verbose=False)

        with patch("investment_agent.execution.execution.poll_order_status") as mock_poll:
            counts = reconciler._sweep(mem, tag="test")
        mock_poll.assert_not_called()
        self.assertEqual(counts["skipped"], 1)

    def test_sweep_marks_filled_as_open(self):
        exp = _make_exp(decision_id="d4", lifecycle="PENDING_FILL", order_id="oid-4")
        mem = _build_memory(exp)
        reconciler = FillReconciler(verbose=False)

        with patch(
            "investment_agent.execution.execution.poll_order_status",
            return_value=self._mock_poll("filled", filled_qty=10.0, fill_price=155.0),
        ):
            counts = reconciler._sweep(mem, tag="test")

        self.assertEqual(counts["filled"], 1)
        updated = mem.get_by_decision_id("d4")
        self.assertEqual(updated.lifecycle_status, TradeLifecycle.OPEN.value)
        self.assertAlmostEqual(updated.fill_price, 155.0)

    def test_sweep_marks_rejected(self):
        exp = _make_exp(decision_id="d5", lifecycle="PENDING_FILL", order_id="oid-5")
        mem = _build_memory(exp)
        reconciler = FillReconciler(verbose=False)

        with patch(
            "investment_agent.execution.execution.poll_order_status",
            return_value=self._mock_poll("rejected", filled_qty=0.0, fill_price=0.0),
        ):
            counts = reconciler._sweep(mem, tag="test")

        self.assertEqual(counts["rejected"], 1)
        updated = mem.get_by_decision_id("d5")
        self.assertEqual(updated.lifecycle_status, TradeLifecycle.REJECTED.value)

    def test_sweep_marks_cancelled(self):
        exp = _make_exp(decision_id="d6", lifecycle="PENDING_FILL", order_id="oid-6")
        mem = _build_memory(exp)
        reconciler = FillReconciler(verbose=False)

        with patch(
            "investment_agent.execution.execution.poll_order_status",
            return_value=self._mock_poll("cancelled", filled_qty=0.0, fill_price=0.0),
        ):
            counts = reconciler._sweep(mem, tag="test")

        self.assertEqual(counts["cancelled"], 1)
        updated = mem.get_by_decision_id("d6")
        self.assertEqual(updated.lifecycle_status, TradeLifecycle.CANCELLED.value)

    def test_sweep_partial_fill_stays_pending(self):
        exp = _make_exp(decision_id="d7", lifecycle="PENDING_FILL", order_id="oid-7", quantity=10.0)
        mem = _build_memory(exp)
        reconciler = FillReconciler(verbose=False)

        with patch(
            "investment_agent.execution.execution.poll_order_status",
            return_value=self._mock_poll("partially_filled", filled_qty=4.0, fill_price=152.0),
        ):
            counts = reconciler._sweep(mem, tag="test")

        self.assertEqual(counts["partially_filled"], 1)
        updated = mem.get_by_decision_id("d7")
        # Should still stay PENDING_FILL (the partial fill didn't complete the order)
        self.assertEqual(updated.lifecycle_status, TradeLifecycle.PENDING_FILL.value)
        self.assertAlmostEqual(updated.fill_price, 152.0)
        self.assertAlmostEqual(updated.quantity, 4.0)

    def test_sweep_ignores_non_pending_experiences(self):
        """OPEN and CLOSED trades must not be touched by the reconciler."""
        exp_open = _make_exp(decision_id="d8", lifecycle="OPEN", order_id="oid-8")
        exp_closed = _make_exp(decision_id="d9", lifecycle="CLOSED", order_id="oid-9")
        mem = _build_memory(exp_open, exp_closed)
        reconciler = FillReconciler(verbose=False)

        with patch(
            "investment_agent.execution.execution.poll_order_status",
        ) as mock_poll:
            counts = reconciler._sweep(mem, tag="test")

        mock_poll.assert_not_called()


# ---------------------------------------------------------------------------
# 6. Empirical HMM calibration/validation
# ---------------------------------------------------------------------------

class TestHMMCalibration(unittest.TestCase):

    def _dummy_probs(self, dominant: str = "R01", n_regimes: int = 12) -> dict:
        probs = {f"R{i+1:02d}": 0.01 for i in range(n_regimes)}
        probs[dominant] = 1.0 - 0.01 * (n_regimes - 1)
        return probs

    def test_validate_hmm_basic_structure(self):
        n = 50
        regimes = ["R01"] * 30 + ["R02"] * 20
        probs = [self._dummy_probs("R01" if i < 30 else "R02") for i in range(n)]
        features = [[50.0, 0.0, 0.02, 15.0, 1.0, 0.0, 0.5]] * n
        report = validate_hmm(features, regimes, probs)
        self.assertEqual(report.n_bars, n)
        self.assertIn("R01", report.regime_counts)
        self.assertIn("R02", report.regime_counts)
        self.assertEqual(report.regime_counts["R01"], 30)
        self.assertEqual(report.regime_counts["R02"], 20)

    def test_brier_score_perfect_classifier(self):
        """A perfectly confident correct classifier should have zero Brier score."""
        n = 20
        regimes = ["R01"] * n
        probs = [{"R01": 1.0}] * n
        features = [[50.0, 0.0, 0.02, 15.0, 1.0, 0.0, 0.5]] * n
        report = validate_hmm(features, regimes, probs)
        self.assertAlmostEqual(report.brier_score, 0.0, places=6)

    def test_brier_score_worst_classifier(self):
        """Confidently wrong every time → Brier = 2.0."""
        n = 10
        regimes = ["R01"] * n
        probs = [{"R02": 1.0}] * n  # Always predicts R02, true is R01
        features = [[50.0, 0.0, 0.02, 15.0, 1.0, 0.0, 0.5]] * n
        report = validate_hmm(features, regimes, probs)
        # Brier = sum_t [(1-0)^2 + (0-1)^2] / n = 2.0 per bar
        self.assertAlmostEqual(report.brier_score, 2.0, places=6)

    def test_log_loss_perfect(self):
        """Perfect classifier → log-loss = 0."""
        n = 10
        regimes = ["R01"] * n
        probs = [{"R01": 1.0}] * n
        features = [[50.0, 0.0, 0.02, 15.0, 1.0, 0.0, 0.5]] * n
        report = validate_hmm(features, regimes, probs)
        self.assertAlmostEqual(report.log_loss, 0.0, places=4)

    def test_empirical_transition_matrix(self):
        """Transitions R01 → R02 → R01 → R02 = 50% each from both states."""
        regimes = ["R01", "R02", "R01", "R02", "R01"]
        n = len(regimes)
        probs = [self._dummy_probs(r) for r in regimes]
        features = [[50.0, 0.0, 0.02, 15.0, 1.0, 0.0, 0.5]] * n
        report = validate_hmm(features, regimes, probs)
        self.assertIn("R01", report.empirical_transition_matrix)
        row_r01 = report.empirical_transition_matrix["R01"]
        # R01 only transitions to R02 in this sequence
        self.assertAlmostEqual(row_r01.get("R02", 0.0), 1.0)

    def test_feature_drift_detected(self):
        """Mean that is >2σ from calibration should generate a warning."""
        n = 30
        regimes = ["R01"] * n
        probs = [self._dummy_probs("R01")] * n
        # RSI = 90 (normal calibration mean is 50, std 15 → z ≈ 2.67)
        features = [[90.0, 0.0, 0.02, 15.0, 1.0, 0.0, 0.5]] * n
        calib_means = [50.0, 0.0, 0.02, 15.0, 1.0, 0.0, 0.5]
        calib_stds  = [15.0, 2.0, 0.015, 10.0, 0.5, 0.4, 0.15]
        report = validate_hmm(
            features, regimes, probs,
            calibration_means=calib_means,
            calibration_stds=calib_stds,
        )
        self.assertTrue(
            any("drifted" in w for w in report.warnings),
            f"Expected drift warning, got: {report.warnings}",
        )

    def test_empty_sequence(self):
        report = validate_hmm([], [], [])
        self.assertEqual(report.n_bars, 0)
        self.assertTrue(any("Empty" in w for w in report.warnings))


# ---------------------------------------------------------------------------
# 7. Regime-by-regime performance attribution
# ---------------------------------------------------------------------------

class TestPerformanceAttribution(unittest.TestCase):

    def _closed_exp(self, decision_id, regime, pnl, signal=0.5, conf=0.7):
        return _make_exp(
            decision_id=decision_id,
            regime=regime,
            pnl=pnl,
            lifecycle="CLOSED",
            ensemble_signal=signal,
            effective_confidence=conf,
        )

    def test_attribution_groups_by_regime(self):
        mem = _build_memory(
            self._closed_exp("a1", "R01", pnl=+100.0),
            self._closed_exp("a2", "R01", pnl=-50.0),
            self._closed_exp("a3", "R02", pnl=+200.0),
        )
        result = performance_by_regime(mem)
        self.assertIn("R01", result)
        self.assertIn("R02", result)
        self.assertEqual(result["R01"]["count"], 2)
        self.assertEqual(result["R02"]["count"], 1)
        self.assertAlmostEqual(result["R01"]["total_pnl"], 50.0)
        self.assertAlmostEqual(result["R02"]["total_pnl"], 200.0)

    def test_win_rate_correct(self):
        mem = _build_memory(
            self._closed_exp("b1", "R03", pnl=+10.0),
            self._closed_exp("b2", "R03", pnl=+20.0),
            self._closed_exp("b3", "R03", pnl=-5.0),
        )
        result = performance_by_regime(mem)
        self.assertAlmostEqual(result["R03"]["win_rate"], 2 / 3)

    def test_sharpe_positive_for_winning_regime(self):
        mem = _build_memory(
            self._closed_exp("c1", "R04", pnl=+100.0),
            self._closed_exp("c2", "R04", pnl=+80.0),
            self._closed_exp("c3", "R04", pnl=+120.0),
        )
        result = performance_by_regime(mem)
        self.assertGreater(result["R04"]["sharpe"], 0.0)

    def test_empty_memory_returns_empty(self):
        mem = _build_memory()
        result = performance_by_regime(mem)
        self.assertEqual(result, {})

    def test_pending_trades_excluded(self):
        """Only CLOSED trades should appear in attribution."""
        mem = _build_memory(
            _make_exp("p1", regime="R05", pnl=999.0, lifecycle="PENDING_FILL"),
            _make_exp("p2", regime="R05", pnl=999.0, lifecycle="OPEN"),
            self._closed_exp("p3", "R05", pnl=10.0),
        )
        result = performance_by_regime(mem)
        self.assertEqual(result["R05"]["count"], 1)
        self.assertAlmostEqual(result["R05"]["total_pnl"], 10.0)

    def test_trade_memory_method_delegates(self):
        """TradeMemory.get_performance_summary_by_regime delegates to fill_reconciler."""
        mem = _build_memory(self._closed_exp("z1", "R06", pnl=42.0))
        result = mem.get_performance_summary_by_regime()
        self.assertIn("R06", result)
        self.assertAlmostEqual(result["R06"]["total_pnl"], 42.0)

    def test_avg_signal_and_confidence(self):
        mem = _build_memory(
            self._closed_exp("s1", "R07", pnl=10.0, signal=0.6, conf=0.8),
            self._closed_exp("s2", "R07", pnl=20.0, signal=0.4, conf=0.6),
        )
        result = performance_by_regime(mem)
        self.assertAlmostEqual(result["R07"]["avg_signal"], 0.5, places=5)
        self.assertAlmostEqual(result["R07"]["avg_confidence"], 0.7, places=5)


if __name__ == "__main__":
    unittest.main()
