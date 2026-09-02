"""Scalability test: the replay engine must process a modest backtest
window within a bounded wall-clock budget.

Investigation note (see CHATS/2026-09-02_archive-consolidation-regression-suite
follow-up): ``ReplayEngine`` logs one ``TradeMemory`` experience per bar, and
``TradeMemory._save()`` performs a full JSON re-serialize + ``fsync`` of the
*entire* experience list on every write (an intentional durability choice,
not a bug -- it guarantees crash-safe persistence). That makes per-bar cost
proportional to how much history has accumulated so far, so total replay
cost is super-linear in bar count. This is an acceptable characteristic for
live trading (one write per ~60-300s decision interval) but a genuine
throughput ceiling for large in-process backtests. Rather than assert a
misleading "near-linear" ratio, this test asserts a concrete, generous wall-
clock budget for a modest, CI-safe window size -- it still catches an
accidental further regression without being flaky about disk I/O variance.
"""
from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

import pandas as pd

from investment_agent.data.market_data import FakeMarketDataClient
from investment_agent.orchestrator import XQuantXOrchestrator
from investment_agent.replay import ReplayConfig, ReplayEngine
from investment_agent.signals.ensemble_signal import AgentOutput

N_BARS = 100
WALL_CLOCK_BUDGET_S = 30.0


def _series(n: int) -> pd.DataFrame:
    idx = pd.date_range("2020-01-02", periods=n, freq="D")
    closes = [100.0 + ((i % 20) - 10) * 0.3 for i in range(n)]
    return pd.DataFrame({
        "open": [c - 0.1 for c in closes],
        "high": [c + 0.2 for c in closes],
        "low": [c - 0.2 for c in closes],
        "close": closes,
        "volume": [1_000_000.0] * n,
    }, index=idx)


def _hold_agent_factory(bar_ctx):
    return [
        AgentOutput(s=0.0, c=0.5, u=0.5, d=0.5, p_plus=0.5, p_minus=0.5,
                    delta_t=1.0, r=0.0, agent_id=f"agent{i + 1}")
        for i in range(7)
    ]


def test_replay_engine_completes_within_wall_clock_budget():
    with tempfile.TemporaryDirectory() as tmpdir:
        rep_file = Path(tmpdir) / "reputation.json"
        fake = FakeMarketDataClient()
        fake.set_series("AAPL", _series(N_BARS))
        orch = XQuantXOrchestrator(
            agent_ids=[f"agent{i + 1}" for i in range(7)],
            symbol="AAPL", use_hmm=False, enable_trading=False,
            memory_file=str(Path(tmpdir) / "mem.json"),
            reputation_file=str(rep_file),
        )
        engine = ReplayEngine(orch, market_data=fake, reputation_file=str(rep_file))
        start = time.perf_counter()
        result = engine.run(ReplayConfig(symbol="AAPL", lookback=30), agent_factory=_hold_agent_factory)
        elapsed = time.perf_counter() - start

    assert result.bars_processed == N_BARS
    assert elapsed < WALL_CLOCK_BUDGET_S, (
        f"replay of {N_BARS} bars took {elapsed:.1f}s, budget is {WALL_CLOCK_BUDGET_S}s"
    )
