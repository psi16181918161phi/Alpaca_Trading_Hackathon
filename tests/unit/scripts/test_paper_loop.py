"""Smoke tests for scripts/run_paper_loop.py end-to-end pipeline."""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "scripts" / "run_paper_loop.py"


def _run_paper_loop(extra_args=None, env_overrides=None):
    """Run the paper-loop script in a subprocess and return (returncode, stdout, stderr)."""
    env = os.environ.copy()
    # Force offline mode (no Alpaca keys)
    env.pop("APCA_API_KEY_ID", None)
    env.pop("APCA_API_SECRET_KEY", None)
    env.pop("FEATHERLESS_DEEPHERMES_KEY", None)
    env.pop("FEATHERLESS_FINANCE_LLAMA_KEY", None)
    env.pop("FEATHERLESS_QWEN_TRADING_KEY", None)
    env.pop("FEATHERLESS_RESERVE_KEY", None)
    if env_overrides:
        env.update(env_overrides)
    cmd = [sys.executable, str(SCRIPT), "--symbol", "AAPL", "--no-execute"]
    if extra_args:
        cmd.extend(extra_args)
    result = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=60)
    return result.returncode, result.stdout, result.stderr


class TestPaperLoopEndToEnd(unittest.TestCase):
    def test_runs_and_prints_all_stages(self):
        with tempfile.TemporaryDirectory() as d:
            mem = os.path.join(d, "trade_memory.json")
            rep = os.path.join(d, "reputation_state.json")
            rc, out, err = _run_paper_loop(extra_args=[
                "--memory", mem, "--reputation", rep,
            ])
        self.assertEqual(rc, 0, msg=f"stdout: {out}\nstderr: {err}")
        # Each stage label must be present in stdout.
        for label in ("[1] Market data", "[2] LLM", "[3] Regime",
                      "[4] Agent signals", "[5] Decision",
                      "[6] Product gate", "[7] Reputation persisted",
                      "[8]"):
            self.assertIn(label, out, f"missing {label} in stdout:\n{out}")

    def test_persists_reputation_and_memory(self):
        with tempfile.TemporaryDirectory() as d:
            mem = os.path.join(d, "trade_memory.json")
            rep = os.path.join(d, "reputation_state.json")
            rc, out, err = _run_paper_loop(extra_args=[
                "--memory", mem, "--reputation", rep,
            ])
            self.assertEqual(rc, 0)
            self.assertTrue(os.path.exists(rep), "reputation file not written")
            self.assertTrue(os.path.exists(mem), "trade memory file not written")

    def test_seven_agents_emit_distinct_signals(self):
        rc, out, err = _run_paper_loop()
        self.assertEqual(rc, 0)
        # Parse the agent signals line
        line = next((l for l in out.splitlines() if l.startswith("[4] Agent signals")), "")
        self.assertTrue(line)
        # Each of the seven canonical agent_ids should appear
        for aid in ("agent_economic", "agent_financial", "agent_fiscal",
                    "agent_portfolio", "agent_fundamental",
                    "agent_market", "agent_sector"):
            self.assertIn(aid, line, f"{aid} missing from agent signals line")


if __name__ == "__main__":
    unittest.main()
