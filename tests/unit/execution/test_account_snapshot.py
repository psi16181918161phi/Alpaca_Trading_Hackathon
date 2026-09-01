"""Unit tests for the extended Alpaca account snapshot helpers.

Covers ``get_account_summary()`` (full broker snapshot),
``load_account_baseline`` / ``save_account_baseline`` (atomic baseline
file), and ``get_account_snapshot()`` (Total P&L against the baseline).
"""
import json
import os
import tempfile
import unittest
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from investment_agent.execution import execution


def _fake_account(**kwargs):
    """Build a MagicMock that mimics an alpaca Account object."""
    defaults = dict(
        status="ACTIVE",
        buying_power=351619.88,
        equity=100145.03,
        last_equity=100000.00,
        cash=87904.97,
        portfolio_value=100145.03,
        account_blocked=False,
        pattern_day_trader=False,
        trading_blocked=False,
        transfers_blocked=False,
    )
    defaults.update(kwargs)
    a = MagicMock(spec=list(defaults.keys()))
    for k, v in defaults.items():
        setattr(a, k, v)
    return a


class TestGetAccountSummary(unittest.TestCase):
    def setUp(self):
        self._patcher = patch.object(execution, "_get_trading_client")
        self.mock_get_client = self._patcher.start()

    def tearDown(self):
        self._patcher.stop()

    def test_returns_full_broker_snapshot(self):
        self.mock_get_client.return_value = MagicMock(
            get_account=lambda: _fake_account()
        )
        snap = execution.get_account_summary()
        self.assertEqual(snap["status"], "ACTIVE")
        self.assertAlmostEqual(snap["buying_power"], 351619.88)
        self.assertAlmostEqual(snap["equity"], 100145.03)
        self.assertAlmostEqual(snap["cash"], 87904.97)
        self.assertAlmostEqual(snap["last_equity"], 100000.00)
        self.assertAlmostEqual(snap["portfolio_value"], 100145.03)
        # Daily P&L uses Alpaca's own equity - last_equity semantics.
        self.assertAlmostEqual(snap["daily_pnl"], 145.03, places=2)
        self.assertAlmostEqual(snap["daily_pnl_pct"], 0.0014503, places=6)
        self.assertIn("snapshot_at", snap)
        # The string ISO timestamp parses cleanly.
        datetime.fromisoformat(snap["snapshot_at"])

    def test_handles_missing_equity_fields(self):
        self.mock_get_client.return_value = MagicMock(
            get_account=lambda: _fake_account(equity=None, last_equity=None, cash=None)
        )
        snap = execution.get_account_summary()
        self.assertIsNone(snap["equity"])
        self.assertIsNone(snap["cash"])
        self.assertIsNone(snap["last_equity"])
        self.assertIsNone(snap["daily_pnl"])
        self.assertIsNone(snap["daily_pnl_pct"])

    def test_handles_string_decimal_values(self):
        # Alpaca sometimes returns strings (e.g. from older SDK paths).
        a = _fake_account()
        a.equity = "100145.03"
        a.last_equity = "100000.00"
        a.cash = "87904.97"
        a.buying_power = "351619.88"
        self.mock_get_client.return_value = MagicMock(get_account=lambda: a)
        snap = execution.get_account_summary()
        self.assertAlmostEqual(snap["equity"], 100145.03)
        self.assertAlmostEqual(snap["daily_pnl"], 145.03, places=2)

    def test_flags_blocked_account(self):
        a = _fake_account(account_blocked=True, trading_blocked=True)
        self.mock_get_client.return_value = MagicMock(get_account=lambda: a)
        snap = execution.get_account_summary()
        self.assertTrue(snap["account_blocked"])
        self.assertTrue(snap["trading_blocked"])


class TestAccountBaseline(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="xqx_alpaca_baseline_")
        self.path = os.path.join(self.tmpdir, "baseline.json")

    def tearDown(self):
        if os.path.exists(self.path):
            os.unlink(self.path)
        os.rmdir(self.tmpdir)

    def test_save_then_load(self):
        saved = execution.save_account_baseline(100000.0, custom_path=self.path)
        self.assertAlmostEqual(saved["baseline_equity"], 100000.0)
        loaded = execution.load_account_baseline(self.path)
        self.assertAlmostEqual(loaded["baseline_equity"], 100000.0)
        self.assertIn("saved_at", loaded)

    def test_second_save_is_noop(self):
        execution.save_account_baseline(100000.0, custom_path=self.path)
        # A second save should NOT overwrite the baseline (it would
        # otherwise reset the "since we started" window on every poll).
        execution.save_account_baseline(200000.0, custom_path=self.path)
        loaded = execution.load_account_baseline(self.path)
        self.assertAlmostEqual(loaded["baseline_equity"], 100000.0)

    def test_load_returns_empty_when_missing(self):
        self.assertEqual(execution.load_account_baseline(self.path), {})

    def test_load_handles_corrupt_file(self):
        with open(self.path, "w") as f:
            f.write("not json")
        self.assertEqual(execution.load_account_baseline(self.path), {})

    def test_atomic_write_uses_tempfile(self):
        # The implementation uses os.replace on a temp file; verify the
        # final file is a valid JSON document with the baseline value.
        execution.save_account_baseline(12345.67, custom_path=self.path)
        with open(self.path, "r") as f:
            data = json.load(f)
        self.assertAlmostEqual(data["baseline_equity"], 12345.67)


class TestGetAccountSnapshot(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="xqx_alpaca_snapshot_")
        self.baseline_path = os.path.join(self.tmpdir, "baseline.json")
        self._patcher = patch.object(execution, "_get_trading_client")
        self.mock_get_client = self._patcher.start()

    def tearDown(self):
        self._patcher.stop()
        if os.path.exists(self.baseline_path):
            os.unlink(self.baseline_path)
        os.rmdir(self.tmpdir)

    def test_first_call_seeds_baseline(self):
        self.mock_get_client.return_value = MagicMock(
            get_account=lambda: _fake_account(equity=100000.0, last_equity=99500.0)
        )
        snap = execution.get_account_snapshot(custom_baseline_path=self.baseline_path)
        self.assertTrue(snap["ok"])
        # First poll: Total P&L is 0 (we just seeded the baseline).
        self.assertAlmostEqual(snap["total_pnl"], 0.0)
        self.assertAlmostEqual(snap["baseline_equity"], 100000.0)
        # Daily P&L is still derived from last_equity.
        self.assertAlmostEqual(snap["daily_pnl"], 500.0)

    def test_subsequent_call_measures_total_pnl(self):
        self.mock_get_client.return_value = MagicMock(
            get_account=lambda: _fake_account(equity=100000.0, last_equity=99500.0)
        )
        # First call seeds baseline at 100000.
        execution.get_account_snapshot(custom_baseline_path=self.baseline_path)
        # Simulate market moving: now equity is 100500.
        self.mock_get_client.return_value = MagicMock(
            get_account=lambda: _fake_account(equity=100500.0, last_equity=99500.0)
        )
        snap = execution.get_account_snapshot(custom_baseline_path=self.baseline_path)
        self.assertTrue(snap["ok"])
        self.assertAlmostEqual(snap["total_pnl"], 500.0)
        self.assertAlmostEqual(snap["baseline_equity"], 100000.0)
        # Daily P&L still uses last_equity.
        self.assertAlmostEqual(snap["daily_pnl"], 1000.0)

    def test_broker_failure_returns_ok_false(self):
        # The mock client itself raises when get_account() is invoked.
        self.mock_get_client.return_value.get_account.side_effect = \
            Exception("network down")
        snap = execution.get_account_snapshot(custom_baseline_path=self.baseline_path)
        self.assertFalse(snap["ok"])
        self.assertIn("network down", snap["error"])
        self.assertIn("snapshot_at", snap)


if __name__ == "__main__":
    unittest.main()
