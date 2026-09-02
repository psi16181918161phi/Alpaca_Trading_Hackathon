"""Memory-budget performance test for a sustained replay session.

Asserts the historical replay engine does not leak unbounded memory
across a multi-hundred-bar run, per
``alpaca_paper_trading_specifications_x_quant_x/012_xquantx_performance_tests.txt``
(RSS budget). Uses ``tracemalloc`` (stdlib) rather than an external
profiler so this test has no new runtime dependency.
"""
from __future__ import annotations

import sys
import tempfile
import tracemalloc
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

import pandas as pd

from investment_agent.data.market_data import FakeMarketDataClient
from investment_agent.orchestrator import XQuantXOrchestrator
from investment_agent.replay import ReplayConfig, ReplayEngine
from investment_agent.signals.ensemble_signal import AgentOutput

MEMORY_GROWTH_BUDGET_MB = 50.0


def _long_series(n: int) -> pd.DataFrame:
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


def test_replay_engine_long_window_memory_stays_bounded():
    # NOTE: kept modest (not 1000+ bars) because TradeMemory._save() does a
    # full fsync'd rewrite per bar (see tests/scalability/test_replay_engine_scaling.py
    # for the throughput characterization); this test targets peak memory,
    # not wall-clock, so a smaller window is sufficient and keeps CI fast.
    n_bars = 150
    with tempfile.TemporaryDirectory() as tmpdir:
        rep_file = Path(tmpdir) / "reputation.json"
        fake = FakeMarketDataClient()
        fake.set_series("AAPL", _long_series(n_bars))
        orch = XQuantXOrchestrator(
            agent_ids=[f"agent{i + 1}" for i in range(7)],
            symbol="AAPL", use_hmm=False, enable_trading=False,
            memory_file=str(Path(tmpdir) / "mem.json"),
            reputation_file=str(rep_file),
        )
        engine = ReplayEngine(orch, market_data=fake, reputation_file=str(rep_file))

        tracemalloc.start()
        engine.run(ReplayConfig(symbol="AAPL", lookback=30), agent_factory=_hold_agent_factory)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

    peak_mb = peak / (1024 * 1024)
    assert peak_mb < MEMORY_GROWTH_BUDGET_MB, (
        f"replay of {n_bars} bars peaked at {peak_mb:.1f}MB, "
        f"budget is {MEMORY_GROWTH_BUDGET_MB}MB"
    )
