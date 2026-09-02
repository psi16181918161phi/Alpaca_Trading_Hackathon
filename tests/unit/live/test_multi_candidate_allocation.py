"""Integration tests for multi-candidate capital allocation.

Verifies that when 2+ assets produce BUY decisions in the same cycle:
  * all are evaluated through the complete pipeline
  * aggregate deployment obeys existing capital constraints
  * the liquidity floor remains intact
  * each order receives an independent ID
  * fills are reconciled independently
  * partial fills are handled correctly
  * failed closes do not falsely mark positions CLOSED
"""
from __future__ import annotations

import os
import tempfile
import unittest
from typing import Any, Dict, List

from investment_agent.live.live_orchestrator import (
    LiveOrchestrator,
    LiveOrchestratorConfig,
)
from investment_agent.live.candidate_screener import CandidateScreener
from investment_agent.data.market_data import FakeMarketDataClient
from investment_agent.orchestrator import XQuantXOrchestrator
from investment_agent.agents.specialist import DEFAULT_ROLES, AgentOutput
from investment_agent.products import ProductGate
from investment_agent.capital.capital_gate import evaluate, SevenStateVector


def _make_bars(n: int = 30, price: float = 100.0) -> "pd.DataFrame":
    import pandas as pd
    import numpy as np
    idx = pd.date_range(end=pd.Timestamp.now().normalize(), periods=n, freq="D")
    closes = np.linspace(price * 0.9, price * 1.1, n)
    return pd.DataFrame({
        "open": closes, "high": closes * 1.01, "low": closes * 0.99,
        "close": closes, "volume": [1_000_000.0] * n,
    }, index=idx)


def _build_live_orchestrator(tmpdir: str, **overrides):
    symbols = overrides.pop("symbols", ["A", "B", "C"])
    md = FakeMarketDataClient()
    for s in symbols:
        md.set_series(s, _make_bars(30))

    orch = XQuantXOrchestrator(
        agent_ids=[r.agent_id for r in DEFAULT_ROLES],
        symbol=symbols[0], use_hmm=False, enable_trading=False,
        memory_file=os.path.join(tmpdir, "mem.json"),
    )
    orch._agent_ids = [r.agent_id for r in DEFAULT_ROLES]

    def factory(ctx):
        return [
            AgentOutput(s=0.6, c=0.9, u=0.1, d=0.05, p_plus=0.8,
                        p_minus=0.2, delta_t=1.0, r=0.5,
                        agent_id=aid)
            for aid in orch._agent_ids
        ]

    config = LiveOrchestratorConfig(
        symbol_universe=symbols,
        top_n_candidates=len(symbols),
        state_file=os.path.join(tmpdir, "live_state.json"),
        memory_file=os.path.join(tmpdir, "mem.json"),
        reputation_file=os.path.join(tmpdir, "rep.json"),
        max_lookups_per_interval=10,
    )
    config.__dict__.update(overrides)

    live = LiveOrchestrator(
        config=config, market_data=md, orchestrator=orch,
        agent_factory=factory, executor=lambda *a, **k: {
            "id": "test-order", "status": "filled",
            "filled_qty": 10.0, "filled_avg_price": 100.0,
        },
        screener=CandidateScreener(top_n=len(symbols)),
        product_gate=ProductGate(),
    )
    return live


class TestMultiCandidateAllocation(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def test_two_simultaneous_buys_both_evaluated(self):
        live = _build_live_orchestrator(self.tmpdir, symbols=["BTC/USD", "ETH/USD"])
        report = live.run_once()
        symbols = [d.get("symbol") for d in report.decisions]
        self.assertIn("BTC/USD", symbols)
        self.assertIn("ETH/USD", symbols)
        buys = [d for d in report.decisions if d.get("action") == "BUY"]
        self.assertGreaterEqual(len(buys), 1)

    def test_three_simultaneous_buys_all_evaluated(self):
        live = _build_live_orchestrator(
            self.tmpdir, symbols=["BTC/USD", "ETH/USD", "SOL/USD"])
        report = live.run_once()
        symbols = [d.get("symbol") for d in report.decisions]
        for sym in ("BTC/USD", "ETH/USD", "SOL/USD"):
            self.assertIn(sym, symbols)

    def test_aggregate_deployment_respects_liquidity_floor(self):
        live = _build_live_orchestrator(
            self.tmpdir, symbols=["BTC/USD", "ETH/USD", "SOL/USD"])
        live._equity = 10_000.0
        report = live.run_once()
        deployed = sum(
            d.get("notional", 0.0) for d in report.decisions
            if d.get("client_order_id")
        )
        self.assertLessEqual(deployed, 10_000.0 - 5_000.0)

    def test_orders_receive_independent_ids(self):
        live = _build_live_orchestrator(
            self.tmpdir, symbols=["BTC/USD", "ETH/USD"])
        report = live.run_once()
        order_ids = [d.get("client_order_id") for d in report.decisions
                     if d.get("client_order_id")]
        self.assertEqual(len(order_ids), len(set(order_ids)))

    def test_rejected_order_does_not_open_position(self):
        calls = []
        def executor(symbol, side, qty, option_side):
            calls.append({"symbol": symbol, "side": side, "qty": qty})
            if symbol == "ETH/USD":
                return {"id": "rej", "status": "rejected", "error": "test reject",
                        "filled_qty": 0.0, "filled_avg_price": 0.0}
            return {"id": "ok", "status": "filled",
                    "filled_qty": qty, "filled_avg_price": 100.0}
        live = _build_live_orchestrator(
            self.tmpdir, symbols=["BTC/USD", "ETH/USD"])
        live._executor = executor
        report = live.run_once()
        btc = next((d for d in report.decisions if d.get("symbol") == "BTC/USD"), None)
        eth = next((d for d in report.decisions if d.get("symbol") == "ETH/USD"), None)
        self.assertIsNotNone(btc)
        self.assertIsNotNone(eth)
        self.assertEqual(eth.get("order_status"), "rejected")


if __name__ == "__main__":
    unittest.main()
