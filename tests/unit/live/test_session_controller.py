"""Unit tests for the SessionController.

The controller owns the LiveOrchestrator thread, persists status to
``session_status.json`` on every transition, and reacts to commands
written by the dashboard to ``session_command.json``. The tests use
a fake orchestrator so they don't need a real broker connection.
"""
from __future__ import annotations

import json
import os
import tempfile
import threading
import time
import unittest
from pathlib import Path

from investment_agent.live import (
    DEFAULT_COMMAND_FILE,
    DEFAULT_STATUS_FILE,
    SessionController,
    SessionState,
    read_session_status,
)


class _FakeReport:
    def __init__(self, decisions=1, orders=0, exits=0, summary="BUY AAPL (equity)"):
        self.decisions = [{"symbol": "AAPL", "action": "BUY",
                           "product": "equity", "reason": ""}]
        self.orders = []
        self.exits = []
        self.summary = summary


class _FakeOrchestrator:
    """A drop-in for LiveOrchestrator with a run_once that just sleeps."""
    def __init__(self, sleep_s: float = 0.05):
        self._sleep = sleep_s
        self._count = 0

    def run_once(self):
        time.sleep(self._sleep)
        self._count += 1
        return _FakeReport(decisions=self._count)


def _wait_for(predicate, timeout_s: float = 5.0, poll_s: float = 0.05) -> bool:
    """Block up to ``timeout_s`` until ``predicate()`` is truthy."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(poll_s)
    return False


class TestSessionController(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="xqx_session_test_")
        self.status_file = os.path.join(self.tmpdir, "session_status.json")
        self.command_file = os.path.join(self.tmpdir, "session_command.json")
        self.completed = []
        self.controller = SessionController(
            build_orchestrator=lambda: _FakeOrchestrator(sleep_s=0.02),
            status_file=self.status_file,
            command_file=self.command_file,
            on_session_complete=self.completed.append,
            preflight=lambda params: None,  # tests skip broker check
        )

    def tearDown(self):
        # Make sure no threads leak.
        if self.controller._thread is not None and self.controller._thread.is_alive():
            self.controller.stop()
            self.controller._thread.join(timeout=3)
        for f in os.listdir(self.tmpdir):
            try:
                os.unlink(os.path.join(self.tmpdir, f))
            except OSError:
                pass
        os.rmdir(self.tmpdir)

    def _wait_for_state(self, state_value: str, timeout_s: float = 5.0) -> bool:
        return _wait_for(
            lambda: self.controller.status.state == state_value,
            timeout_s=timeout_s,
        )

    def test_initial_state_is_stopped(self):
        self.assertEqual(self.controller.status.state, SessionState.STOPPED.value)
        # The controller eagerly writes the initial STOPPED state so
        # the dashboard can render without a race.
        self.assertTrue(os.path.exists(self.status_file))
        self.assertEqual(read_session_status(self.status_file)["state"],
                         SessionState.STOPPED.value)

    def test_start_then_stop_lifecycle(self):
        self.assertTrue(self.controller.start(params={
            "stage": "dry_run",
            "decision_interval_seconds": 60,
            "symbol_universe": ["AAPL", "SPY"],
            "max_lookups_per_interval": 2,
        }))
        # Idempotent: a second start() while running is a no-op.
        self.assertFalse(self.controller.start(params={}))

        # Lifecycle should progress: STARTING -> RUNNING.
        self.assertTrue(self._wait_for_state(SessionState.RUNNING.value))
        # Status file reflects the running state.
        on_disk = read_session_status(self.status_file)
        self.assertEqual(on_disk["state"], SessionState.RUNNING.value)
        self.assertEqual(on_disk["symbol_universe"], ["AAPL", "SPY"])
        self.assertEqual(on_disk["decision_interval_seconds"], 60)
        self.assertEqual(on_disk["stage"], "dry_run")
        # Cycle counter is incrementing.
        self.assertTrue(_wait_for(
            lambda: self.controller.status.cycle_index >= 1,
            timeout_s=4.0,
        ))

        # Stop.
        self.assertTrue(self.controller.stop())
        self.assertTrue(self._wait_for_state(SessionState.STOPPED.value, timeout_s=10))
        # The session thread must fully exit before on_session_complete
        # fires. Join with a short grace so the assertion is deterministic.
        if self.controller._thread is not None:
            self.controller._thread.join(timeout=5)
        # on_session_complete was called with a STOPPED status.
        self.assertEqual(len(self.completed), 1)
        self.assertEqual(self.completed[0].state, SessionState.STOPPED.value)
        # Cycle + decision counters persisted.
        on_disk = read_session_status(self.status_file)
        self.assertGreaterEqual(on_disk["cycle_index"], 1)
        self.assertGreaterEqual(on_disk["total_decisions"], 1)
        self.assertIn("stopped_at", on_disk)

    def test_request_via_command_file(self):
        self.controller.request_start({
            "stage": "paper",
            "decision_interval_seconds": 30,
            "symbol_universe": ["MSFT"],
        })
        # Controller consumes the command and starts the loop.
        self.assertTrue(self.controller.poll_and_apply_commands())
        self.assertTrue(self._wait_for_state(SessionState.RUNNING.value, timeout_s=4))
        # The command file is cleared after consumption.
        self.assertFalse(os.path.exists(self.command_file))

    def test_emergency_stop_persists_emergency_state(self):
        self.controller.start(params={"decision_interval_seconds": 60,
                                        "symbol_universe": ["AAPL"]})
        self.assertTrue(self._wait_for_state(SessionState.RUNNING.value))
        self.assertTrue(self.controller.stop(emergency=True))
        self.assertTrue(self._wait_for_state(SessionState.STOPPED.value, timeout_s=10))
        on_disk = read_session_status(self.status_file)
        # The transient state must show EMERGENCY_HALT at some point
        # during the shutdown, even if the controller has already
        # reconciled it back to STOPPED.
        self.assertEqual(on_disk["state"], SessionState.STOPPED.value)
        self.assertIn("stopped_at", on_disk)

    def test_preflight_failure_marks_error(self):
        """A pre-flight that fails (e.g. unreachable broker) marks ERROR."""
        def bad_orch():
            raise RuntimeError("Alpaca unreachable")

        bad = SessionController(
            build_orchestrator=bad_orch,
            status_file=self.status_file,
            command_file=self.command_file,
        )
        # Patch preflight to fail without contacting the broker.
        original = bad._validate_preflight
        bad._validate_preflight = lambda params: "credentials missing"
        try:
            bad.start(params={"decision_interval_seconds": 60,
                                "symbol_universe": ["AAPL"]})
            self.assertTrue(bad._wait_for_state_unsafe(SessionState.ERROR.value))
        finally:
            bad._validate_preflight = original

    def test_thread_does_not_leak_after_stop(self):
        self.controller.start(params={"decision_interval_seconds": 60,
                                        "symbol_universe": ["AAPL"]})
        self.assertTrue(self._wait_for_state(SessionState.RUNNING.value))
        thread_ref = self.controller._thread
        self.assertIsNotNone(thread_ref)
        self.controller.stop()
        thread_ref.join(timeout=5)
        self.assertFalse(thread_ref.is_alive())


# Helper added to the controller only in test scope.
def _wait_for_state_unsafe(self, value, timeout_s=5.0):
    return _wait_for(lambda: self.status.state == value, timeout_s=timeout_s)


SessionController._wait_for_state_unsafe = _wait_for_state_unsafe


if __name__ == "__main__":
    unittest.main()
