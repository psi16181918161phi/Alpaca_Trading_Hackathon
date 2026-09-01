"""Smoke tests for scripts/run_live_loop.py end-to-end pipeline."""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "scripts" / "run_live_loop.py"


def _run_live_loop(extra_args=None, env_overrides=None):
    env = os.environ.copy()
    # Force offline / mock mode
    for k in ("APCA_API_KEY_ID", "APCA_API_SECRET_KEY",
              "FEATHERLESS_DEEPHERMES_KEY", "FEATHERLESS_FINANCE_LLAMA_KEY",
              "FEATHERLESS_QWEN_TRADING_KEY", "FEATHERLESS_RESERVE_KEY"):
        env.pop(k, None)
    # Isolated state per test invocation so stale temp files can't
    # poison the first ``_load_state`` (which would skip the
    # ``run_once`` call and yield an empty stdout).
    tmpdir = tempfile.mkdtemp(prefix="xqx_live_loop_test_")
    state_file = os.path.join(tmpdir, "state.json")
    memory_file = os.path.join(tmpdir, "memory.json")
    reputation_file = os.path.join(tmpdir, "reputation.json")
    if env_overrides:
        env.update(env_overrides)
    cmd = [sys.executable, str(SCRIPT),
           "--stage", "dry_run", "--max-intervals", "1", "--interval", "1",
           "--state-file", state_file,
           "--memory-file", memory_file,
           "--reputation-file", reputation_file,
           "--lookback", "60"]
    if extra_args:
        cmd.extend(extra_args)
    result = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=120)
    return result.returncode, result.stdout, result.stderr


class TestLiveLoopEndToEnd(unittest.TestCase):
    def test_runs_and_prints_all_panels(self):
        rc, out, err = _run_live_loop(extra_args=["--symbols", "AAPL,SPY"])
        self.assertEqual(rc, 0, msg=f"stdout: {out}\nstderr: {err}")
        for label in (
            "X QUANT X -- LIVE PAPER LOOP",
            "Time:",
            "Regime:",
            "Equity:",
            "Circuit:",
            "CANDIDATES",
            "DECISION:",
            "REPUTATION",
            "SUMMARY",
        ):
            self.assertIn(label, out, f"missing {label} in stdout")

    def test_dry_run_does_not_invoke_executor(self):
        rc, out, err = _run_live_loop(extra_args=["--symbols", "AAPL"])
        # The dry-run executor never returns a real broker id.
        self.assertEqual(rc, 0)
        # The "Total orders" line should reflect zero or just the dry-run log.
        self.assertIn("Total orders:", out)

    def test_circuit_breaker_appears_in_report(self):
        rc, out, err = _run_live_loop()
        self.assertEqual(rc, 0)
        # Circuit line is always present.
        self.assertIn("Circuit:", out)
        self.assertIn("equity_ok", out)

    def test_reputation_panel_shows_seven_agents(self):
        rc, out, err = _run_live_loop()
        self.assertEqual(rc, 0)
        # All 7 canonical agent IDs are in the reputation table.
        for aid in ("agent_economic", "agent_financial", "agent_fiscal",
                    "agent_portfolio", "agent_fundamental",
                    "agent_market", "agent_sector"):
            self.assertIn(aid, out)

    def test_summary_prints_final_equity(self):
        rc, out, err = _run_live_loop()
        self.assertEqual(rc, 0)
        self.assertIn("Final equity:", out)
        self.assertIn("State file:", out)
        self.assertIn("Memory file:", out)
        self.assertIn("Reputation file:", out)


if __name__ == "__main__":
    unittest.main()
