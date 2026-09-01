"""Smoke tests for ``scripts/run_session.py``.

The daemon is a thin process wrapper around SessionController; the
real behavior we want to verify is:

* a ``start`` command in the command file actually launches the
  controller's session thread,
* a ``stop`` command asks the controller to stop and the daemon
  shuts down cleanly,
* a crash inside the orchestrator surfaces in the session status
  (state==ERROR, last_error populated).

We bypass the real Alpaca + LLM stack with a custom
``build_orchestrator`` factory and a fake preflight.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPTS_DIR = REPO_ROOT / "scripts"
SRC_DIR = REPO_ROOT / "src"


def _python_path() -> str:
    return os.pathsep.join([str(SRC_DIR), str(REPO_ROOT), os.environ.get("PATH", "")])


def _build_fake_orchestrator_factory(state_path: Path):
    """Return a factory that builds an orchestrator-shaped fake.

    The fake's ``run_once`` sleeps a little then returns a report
    with the attributes SessionController._on_cycle_complete reads
    (decisions, orders, exits). For the smoke test we just want
    the session thread to be alive while the daemon polls.
    """
    from dataclasses import dataclass, field
    from typing import List
    @dataclass
    class FakeReport:
        decisions: List[dict] = field(default_factory=list)
        orders: List[dict] = field(default_factory=list)
        exits: List[dict] = field(default_factory=list)
        def __post_init__(self):
            self.decisions = [{"symbol": "AAPL", "action": "HOLD", "product": "none"}]
            self.orders = []
            self.exits = []
    class FakeOrchestrator:
        def __init__(self):
            self.cycles = 0
        def run_once(self):
            self.cycles += 1
            return FakeReport()
    def factory():
        return FakeOrchestrator()
    return factory


def _make_controller_for_daemon(tmp_path, monkeypatch):
    """Build a SessionController in-process with a fake preflight."""
    monkeypatch.syspath_prepend(str(SRC_DIR))
    from investment_agent.live import SessionController
    status_file = str(tmp_path / "session_status.json")
    command_file = str(tmp_path / "session_command.json")
    state_path = tmp_path / "live_state.json"
    return SessionController(
        build_orchestrator=_build_fake_orchestrator_factory(state_path),
        status_file=status_file,
        command_file=command_file,
        preflight=lambda params: None,  # always pass
    )


def test_start_command_launches_session(tmp_path, monkeypatch):
    """Daemon polls command file, applies start, session reaches RUNNING."""
    controller = _make_controller_for_daemon(tmp_path, monkeypatch)
    from scripts.run_session import SessionDaemon
    daemon = SessionDaemon(
        controller=controller,
        status_file=controller.status_file,
        command_file=controller.command_file,
        poll_interval_s=0.05,
    )
    # Drop a start command and run the daemon briefly in a thread.
    Path(controller.command_file).write_text(json.dumps({
        "action": "start",
        "params": {
            "stage": "dry_run",
            "decision_interval_seconds": 1,
            "symbol_universe": ["AAPL"],
            "max_lookups_per_interval": 1,
        },
    }))
    t = __import__("threading").Thread(target=daemon.run, daemon=True)
    t.start()
    try:
        # Wait up to 2s for the session to reach RUNNING.
        deadline = time.time() + 2
        while time.time() < deadline:
            if controller.status.state == "RUNNING":
                break
            time.sleep(0.05)
        assert controller.status.state == "RUNNING", (
            f"expected RUNNING, got {controller.status.state}; "
            f"last_error={controller.status.last_error!r}")
        # Status file should reflect the running state.
        persisted = json.loads(Path(controller.status_file).read_text())
        assert persisted["state"] == "RUNNING"
        assert persisted["symbol_universe"] == ["AAPL"]
        assert persisted["decision_interval_seconds"] == 1
    finally:
        # Ask the daemon to stop and clean up the session.
        controller.request_emergency_stop()
        # Drop a stop command in case the thread is still alive.
        Path(controller.command_file).write_text(json.dumps({
            "action": "emergency_stop",
        }))
        daemon.request_stop()
        t.join(timeout=5)
    # After the daemon has stopped, the controller should be STOPPED
    # or EMERGENCY_HALT -> STOPPED.
    final = json.loads(Path(controller.status_file).read_text())
    assert final["state"] in {"STOPPED", "EMERGENCY_HALT"}, (
        f"expected STOPPED/EMERGENCY_HALT, got {final['state']}")


def test_unknown_command_is_skipped(tmp_path, monkeypatch):
    """Garbage command file should not crash the daemon."""
    controller = _make_controller_for_daemon(tmp_path, monkeypatch)
    from scripts.run_session import SessionDaemon
    daemon = SessionDaemon(
        controller=controller,
        status_file=controller.status_file,
        command_file=controller.command_file,
        poll_interval_s=0.05,
    )
    Path(controller.command_file).write_text(json.dumps({"action": "wat"}))
    t = __import__("threading").Thread(target=daemon.run, daemon=True)
    t.start()
    try:
        time.sleep(0.3)
        # The command file should have been cleared without a state change.
        assert not Path(controller.command_file).exists()
        assert controller.status.state == "STOPPED"
    finally:
        daemon.request_stop()
        t.join(timeout=3)


def test_preflight_failure_surfaces_in_status(tmp_path, monkeypatch):
    """If preflight fails, the controller's status should be ERROR."""
    monkeypatch.syspath_prepend(str(SRC_DIR))
    from investment_agent.live import SessionController
    from scripts.run_session import SessionDaemon

    def boom(params):
        return "Alpaca is on fire"

    controller = SessionController(
        build_orchestrator=_build_fake_orchestrator_factory(tmp_path / "live.json"),
        status_file=str(tmp_path / "session_status.json"),
        command_file=str(tmp_path / "session_command.json"),
        preflight=boom,
    )
    daemon = SessionDaemon(
        controller=controller,
        status_file=controller.status_file,
        command_file=controller.command_file,
        poll_interval_s=0.05,
    )
    Path(controller.command_file).write_text(json.dumps({
        "action": "start",
        "params": {"stage": "paper"},
    }))
    t = __import__("threading").Thread(target=daemon.run, daemon=True)
    t.start()
    try:
        deadline = time.time() + 2
        while time.time() < deadline:
            if controller.status.state == "ERROR":
                break
            time.sleep(0.05)
        persisted = json.loads(Path(controller.status_file).read_text())
        assert persisted["state"] == "ERROR"
        assert "Alpaca is on fire" in persisted["last_error"]
    finally:
        daemon.request_stop()
        t.join(timeout=3)


def test_daemon_help_cli_runs():
    """The CLI's --help should work without a real broker."""
    result = subprocess.run(
        [sys.executable, str(SCRIPTS_DIR / "run_session.py"), "--help"],
        capture_output=True, text=True, timeout=10,
        env={**os.environ, "PYTHONPATH": _python_path()},
        cwd=str(REPO_ROOT),
    )
    assert result.returncode == 0, (result.stdout, result.stderr)
    assert "--stage" in result.stdout
    assert "--status-file" in result.stdout
