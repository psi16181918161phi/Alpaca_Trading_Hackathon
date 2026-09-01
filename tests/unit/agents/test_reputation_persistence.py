"""Tests for ``reputation_persistence``."""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from investment_agent.agents.agent_reputation import AgentReputationTracker
from investment_agent.agents.reputation_persistence import (
    load_reputation,
    save_reputation,
)


class TestReputationPersistence(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.path = os.path.join(self.tmpdir, "reputation.json")
        self.t = AgentReputationTracker(
            agent_ids=["a1", "a2", "a3"],
            regimes=["R01", "R02", "R03"],
        )
        for _ in range(2):
            self.t.record_outcome("a1", "R01", True)
        self.t.record_outcome("a1", "R01", False)
        for _ in range(3):
            self.t.record_outcome("a2", "R02", True)
        self.t.record_outcome("a3", "R03", False)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_round_trip_preserves_state(self):
        save_reputation(self.t, self.path)
        loaded = load_reputation(self.path)
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.to_dict(), self.t.to_dict())

    def test_weights_preserved(self):
        save_reputation(self.t, self.path)
        loaded = load_reputation(self.path)
        self.assertEqual(
            loaded.get_normalized_weights("R01"),
            self.t.get_normalized_weights("R01"),
        )

    def test_missing_file_returns_none(self):
        self.assertIsNone(load_reputation(os.path.join(self.tmpdir, "nope.json")))

    def test_corrupt_file_returns_none(self):
        Path(self.path).write_text("not json{{{")
        self.assertIsNone(load_reputation(self.path))

    def test_schema_violation_returns_none(self):
        # Valid JSON but wrong shape
        Path(self.path).write_text(json.dumps({"foo": "bar"}))
        self.assertIsNone(load_reputation(self.path))

    def test_atomic_write_no_leftover_tmp(self):
        save_reputation(self.t, self.path)
        # Only the canonical file should remain (no .tmp leftovers)
        leftover = [
            p for p in os.listdir(self.tmpdir)
            if p.endswith(".tmp")
        ]
        self.assertEqual(leftover, [])

    def test_overwrite_existing_file(self):
        save_reputation(self.t, self.path)
        # New tracker with a different state
        t2 = AgentReputationTracker(agent_ids=["a1"], regimes=["R01"])
        for _ in range(5):
            t2.record_outcome("a1", "R01", True)
        save_reputation(t2, self.path)
        loaded = load_reputation(self.path)
        self.assertEqual(
            loaded.get_posterior_parameters("a1", "R01"),
            {"alpha": 6.0, "beta": 1.0},
        )


if __name__ == "__main__":
    unittest.main()
