"""Tests for the historical replay / backtest engine."""
from __future__ import annotations

import tempfile
import unittest
from datetime import datetime
from pathlib import Path

import pandas as pd

from investment_agent.capital.capital_gate import SevenStateVector
from investment_agent.data.market_data import FakeMarketDataClient
from investment_agent.orchestrator import XQuantXOrchestrator
from investment_agent.replay import ReplayConfig, ReplayEngine
from investment_agent.signals.ensemble_signal import AgentOutput


def _trending_series(n: int = 60, start: float = 100.0, step: float = 0.4) -> pd.DataFrame:
    """Monotonically rising series so the agents should BUY and win."""
    idx = pd.date_range("2024-01-02", periods=n, freq="D")
    closes = [start + i * step for i in range(n)]
    return pd.DataFrame({
        "open": [c - 0.1 for c in closes],
        "high": [c + 0.2 for c in closes],
        "low": [c - 0.2 for c in closes],
        "close": closes,
        "volume": [1_000_000.0] * n,
    }, index=idx)


def _choppy_series(n: int = 60) -> pd.DataFrame:
    """Oscillating series that should produce some HOLDs and small P&L."""
    idx = pd.date_range("2024-01-02", periods=n, freq="D")
    closes = [100.0 + ((i % 6) - 3) * 0.4 for i in range(n)]
    return pd.DataFrame({
        "open": [c - 0.1 for c in closes],
        "high": [c + 0.2 for c in closes],
        "low": [c - 0.2 for c in closes],
        "close": closes,
        "volume": [1_000_000.0] * n,
    }, index=idx)


def _bullish_agent_factory(bar_ctx):
    """All seven agents agree on BUY with strong confidence."""
    return [
        AgentOutput(
            s=0.6, c=0.9, u=0.1, d=0.05, p_plus=0.7, p_minus=0.2,
            delta_t=1.0, r=0.3, agent_id=f"agent{i + 1}",
        )
        for i in range(7)
    ]


def _hold_agent_factory(bar_ctx):
    """All agents say HOLD (signal near zero)."""
    return [
        AgentOutput(
            s=0.0, c=0.5, u=0.5, d=0.5, p_plus=0.5, p_minus=0.5,
            delta_t=1.0, r=0.0, agent_id=f"agent{i + 1}",
        )
        for i in range(7)
    ]


def _build_orchestrator(tmpdir: str, reputation_file: Path) -> XQuantXOrchestrator:
    return XQuantXOrchestrator(
        agent_ids=[f"agent{i + 1}" for i in range(7)],
        symbol="AAPL",
        use_hmm=False,
        enable_trading=False,
        memory_file=str(Path(tmpdir) / "mem.json"),
        reputation_file=str(reputation_file),
    )


class TestReplayEngine(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.rep_file = Path(self.tmpdir) / "reputation.json"

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_replay_bullish_market_accumulates_pnl(self):
        fake = FakeMarketDataClient()
        fake.set_series("AAPL", _trending_series(60, start=100.0, step=0.5))
        orch = _build_orchestrator(self.tmpdir, self.rep_file)
        engine = ReplayEngine(orch, market_data=fake, reputation_file=str(self.rep_file))
        result = engine.run(
            ReplayConfig(symbol="AAPL", lookback=30),
            agent_factory=_bullish_agent_factory,
        )
        self.assertGreater(result.bars_processed, 30)
        self.assertGreater(result.decisions, 0)
        self.assertGreater(result.buys, 0)
        # Bull market + bullish agents -> positive P&L on a closed trade
        self.assertGreater(result.realized_pnl, 0.0)
        self.assertGreater(result.final_equity, 100_000.0)
        self.assertGreater(result.win_rate, 0.0)
        # Reputation got persisted
        self.assertTrue(self.rep_file.exists())

    def test_replay_hold_only_has_no_realized_pnl(self):
        fake = FakeMarketDataClient()
        fake.set_series("AAPL", _choppy_series(60))
        orch = _build_orchestrator(self.tmpdir, self.rep_file)
        engine = ReplayEngine(orch, market_data=fake, reputation_file=str(self.rep_file))
        result = engine.run(
            ReplayConfig(symbol="AAPL", lookback=30),
            agent_factory=_hold_agent_factory,
        )
        self.assertEqual(result.closed_trades, 0)
        self.assertEqual(result.realized_pnl, 0.0)

    def test_replay_too_short_returns_empty(self):
        fake = FakeMarketDataClient()
        fake.set_series("AAPL", _trending_series(20))  # less than lookback
        orch = _build_orchestrator(self.tmpdir, self.rep_file)
        engine = ReplayEngine(orch, market_data=fake, reputation_file=str(self.rep_file))
        result = engine.run(
            ReplayConfig(symbol="AAPL", lookback=30),
            agent_factory=_bullish_agent_factory,
        )
        self.assertEqual(result.bars_processed, 0)

    def test_replay_records_decisions_in_log(self):
        fake = FakeMarketDataClient()
        fake.set_series("AAPL", _trending_series(60))
        orch = _build_orchestrator(self.tmpdir, self.rep_file)
        engine = ReplayEngine(orch, market_data=fake, reputation_file=str(self.rep_file))
        result = engine.run(
            ReplayConfig(symbol="AAPL", lookback=30),
            agent_factory=_bullish_agent_factory,
        )
        self.assertGreater(len(result.decisions_log), 0)
        for d in result.decisions_log:
            self.assertIn("action", d)
            self.assertIn("regime", d)
            self.assertIn("kalman_posterior", d)

    def test_replay_updates_reputation(self):
        fake = FakeMarketDataClient()
        fake.set_series("AAPL", _trending_series(60))
        orch = _build_orchestrator(self.tmpdir, self.rep_file)
        engine = ReplayEngine(orch, market_data=fake, reputation_file=str(self.rep_file))
        result = engine.run(
            ReplayConfig(symbol="AAPL", lookback=30),
            agent_factory=_bullish_agent_factory,
        )
        # Reputation file was saved
        self.assertTrue(self.rep_file.exists())
        # And the in-memory tracker was updated (close_trade -> record_outcome)
        weights = orch._reputation_tracker.get_normalized_weights("R01")
        total = sum(weights.values())
        self.assertAlmostEqual(total, 1.0, places=4)

    def test_replay_factory_count_mismatch_raises(self):
        def _bad_factory(bar_ctx):
            return [AgentOutput(s=0.0, c=0.0, u=0.0, d=0.0, p_plus=0.0,
                                p_minus=0.0, delta_t=1.0, r=0.0, agent_id="x")]
        fake = FakeMarketDataClient()
        fake.set_series("AAPL", _trending_series(60))
        orch = _build_orchestrator(self.tmpdir, self.rep_file)
        engine = ReplayEngine(orch, market_data=fake, reputation_file=str(self.rep_file))
        with self.assertRaises(ValueError):
            engine.run(
                ReplayConfig(symbol="AAPL", lookback=30),
                agent_factory=_bad_factory,
            )

    def test_replay_result_to_dict_serializable(self):
        fake = FakeMarketDataClient()
        fake.set_series("AAPL", _trending_series(60))
        orch = _build_orchestrator(self.tmpdir, self.rep_file)
        engine = ReplayEngine(orch, market_data=fake, reputation_file=str(self.rep_file))
        result = engine.run(
            ReplayConfig(symbol="AAPL", lookback=30),
            agent_factory=_bullish_agent_factory,
        )
        d = result.to_dict()
        import json
        # Should be JSON-serializable
        json.dumps(d)
        self.assertEqual(d["symbol"], "AAPL")
        self.assertGreater(d["bars_processed"], 0)


if __name__ == "__main__":
    unittest.main()
