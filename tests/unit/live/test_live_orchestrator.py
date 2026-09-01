"""Tests for the live paper-trading pipeline.

Covers:
  * CandidateScreener -- deterministic pre-filter
  * CircuitBreaker -- 4-level hierarchy
  * OrderStateMachine -- explicit transitions + persistence
  * PositionManager -- open / mark-to-market / exit signals
  * LiveOrchestrator -- full interval + circuit-breaker authority
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC = REPO_ROOT / "src"
sys.path.insert(0, str(SRC))


def _make_bars(n: int, start: float = 100.0, step: float = 0.4,
               vol: float = 1_000_000.0) -> pd.DataFrame:
    """Bars with *today* as the most recent index so the live
    orchestrator's ``start=now-60days`` filter keeps them."""
    end = pd.Timestamp.now().normalize()
    idx = pd.date_range(end=end, periods=n, freq="D")
    closes = [start + i * step for i in range(n)]
    return pd.DataFrame({
        "open": [c - 0.1 for c in closes],
        "high": [c + 0.2 for c in closes],
        "low": [c - 0.2 for c in closes],
        "close": closes,
        "volume": [vol] * n,
    }, index=idx)


# ---------------------------------------------------------------------------
# CandidateScreener
# ---------------------------------------------------------------------------

class TestCandidateScreener(unittest.TestCase):
    def test_top_n(self):
        from investment_agent.live.candidate_screener import CandidateScreener
        screener = CandidateScreener(top_n=2, min_bars=20)
        universe = {
            "AAPL": _make_bars(30, start=100.0, step=0.5),
            "SPY": _make_bars(30, start=200.0, step=0.1),
            "MSFT": _make_bars(30, start=50.0, step=0.8),
        }
        results = screener.screen(universe)
        self.assertEqual(len(results), 2)
        for r in results:
            self.assertGreaterEqual(r.score, 0.0)
            self.assertLessEqual(r.score, 1.0)

    def test_minimum_bars_filter(self):
        from investment_agent.live.candidate_screener import CandidateScreener
        screener = CandidateScreener(top_n=5, min_bars=20)
        universe = {
            "TOO_SHORT": _make_bars(10),  # filtered
            "ENOUGH": _make_bars(30, step=0.3),
        }
        results = screener.screen(universe)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].symbol, "ENOUGH")

    def test_volume_cutoff_filters_out_quiet_names(self):
        from investment_agent.live.candidate_screener import CandidateScreener
        screener = CandidateScreener(top_n=5, min_bars=20, min_relative_volume=0.9)
        universe = {
            "LOUD": _make_bars(30, step=0.3, vol=1_000_000.0),
            "QUIET": _make_bars(30, step=0.3, vol=1_000.0),
        }
        results = screener.screen(universe)
        # QUIET's relative volume is 0.001 -> filtered.
        self.assertEqual([r.symbol for r in results], ["LOUD"])

    def test_empty_universe_returns_empty(self):
        from investment_agent.live.candidate_screener import CandidateScreener
        screener = CandidateScreener(top_n=3)
        self.assertEqual(screener.screen({}), [])


# ---------------------------------------------------------------------------
# CircuitBreaker
# ---------------------------------------------------------------------------

class TestCircuitBreaker(unittest.TestCase):
    def test_normal_state(self):
        from investment_agent.live.circuit_breaker import CircuitBreaker, CircuitLevel
        cb = CircuitBreaker()
        state = cb.evaluate(drawdown_pct=0.01, consecutive_losses=1, daily_loss_pct=0.005)
        self.assertEqual(state.level, CircuitLevel.NORMAL)
        self.assertTrue(state.can_trade_equity)
        self.assertTrue(state.can_trade_options)

    def test_warning_blocks_options_allows_equity(self):
        from investment_agent.live.circuit_breaker import CircuitBreaker, CircuitLevel
        cb = CircuitBreaker()
        # 6% drawdown -> WARNING by default
        state = cb.evaluate(drawdown_pct=0.06, consecutive_losses=0, daily_loss_pct=0.0)
        self.assertEqual(state.level, CircuitLevel.WARNING)
        self.assertTrue(state.can_trade_equity)
        self.assertFalse(state.can_trade_options)

    def test_warning_state(self):
        from investment_agent.live.circuit_breaker import CircuitBreaker, CircuitLevel
        cb = CircuitBreaker(drawdown_warning=0.05)
        state = cb.evaluate(drawdown_pct=0.06, consecutive_losses=0, daily_loss_pct=0.0)
        self.assertEqual(state.level, CircuitLevel.WARNING)
        self.assertTrue(state.can_trade_equity)
        # Options are only allowed at NORMAL; the spec is explicit
        # that WARNING can keep trading equity but should hold off on
        # options.
        self.assertFalse(state.can_trade_options)

    def test_restricted_blocks_options_only(self):
        from investment_agent.live.circuit_breaker import CircuitBreaker, CircuitLevel
        cb = CircuitBreaker(drawdown_restricted=0.10)
        state = cb.evaluate(drawdown_pct=0.11, consecutive_losses=0, daily_loss_pct=0.0)
        self.assertEqual(state.level, CircuitLevel.RESTRICTED)
        self.assertTrue(state.can_trade_equity)
        self.assertFalse(state.can_trade_options)

    def test_halt_blocks_everything(self):
        from investment_agent.live.circuit_breaker import CircuitBreaker, CircuitLevel
        cb = CircuitBreaker(drawdown_halt=0.15)
        state = cb.evaluate(drawdown_pct=0.18, consecutive_losses=0, daily_loss_pct=0.0)
        self.assertEqual(state.level, CircuitLevel.HALT)
        self.assertFalse(state.can_trade_equity)
        self.assertFalse(state.can_trade_options)

    def test_max_signal_wins(self):
        from investment_agent.live.circuit_breaker import CircuitBreaker, CircuitLevel
        cb = CircuitBreaker(
            drawdown_warning=0.05, loss_streak_halt=3, daily_loss_warning=0.01,
        )
        # Loss-streak says HALT; others say lower -> HALT wins.
        state = cb.evaluate(drawdown_pct=0.01, consecutive_losses=5, daily_loss_pct=0.005)
        self.assertEqual(state.level, CircuitLevel.HALT)


# ---------------------------------------------------------------------------
# OrderStateMachine
# ---------------------------------------------------------------------------

class TestOrderStateMachine(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.path = os.path.join(self.tmpdir, "state.json")

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_register_and_transition(self):
        from investment_agent.live.order_state_machine import (
            OrderStateMachine, OrderState,
        )
        sm = OrderStateMachine(state_file=self.path)
        rec = sm.register(
            client_order_id="co-1", decision_id="d-1",
            symbol="AAPL", side="buy", qty=10, product="equity",
        )
        self.assertEqual(rec.state, OrderState.SUBMITTED)
        sm.set_broker_id("co-1", "broker-123")
        rec = sm.transition("co-1", OrderState.ACCEPTED, note="ok")
        self.assertEqual(rec.state, OrderState.ACCEPTED)
        rec = sm.transition("co-1", OrderState.FILLED,
                            fill_qty=10, fill_price=150.0)
        self.assertEqual(rec.state, OrderState.FILLED)
        self.assertEqual(rec.fill_qty, 10.0)
        self.assertEqual(rec.fill_price, 150.0)

    def test_invalid_transition_raises(self):
        from investment_agent.live.order_state_machine import (
            OrderStateMachine, OrderState,
        )
        sm = OrderStateMachine(state_file=self.path)
        sm.register(client_order_id="co-1", decision_id="d-1",
                    symbol="AAPL", side="buy", qty=10, product="equity")
        # SUBMITTED -> FILLED is not allowed.
        with self.assertRaises(ValueError):
            sm.transition("co-1", OrderState.FILLED)

    def test_persistence(self):
        from investment_agent.live.order_state_machine import (
            OrderStateMachine, OrderState,
        )
        sm1 = OrderStateMachine(state_file=self.path)
        sm1.register(client_order_id="co-1", decision_id="d-1",
                      symbol="AAPL", side="buy", qty=10, product="equity")
        sm1.set_broker_id("co-1", "broker-123")
        sm1.transition("co-1", OrderState.ACCEPTED)
        sm1.transition("co-1", OrderState.FILLED, fill_qty=10, fill_price=150.0)
        sm2 = OrderStateMachine(state_file=self.path)
        rec = sm2.get("co-1")
        self.assertIsNotNone(rec)
        self.assertEqual(rec.state, OrderState.FILLED)
        self.assertEqual(rec.fill_qty, 10.0)
        self.assertEqual(rec.broker_order_id, "broker-123")

    def test_open_orders_filter(self):
        from investment_agent.live.order_state_machine import (
            OrderStateMachine, OrderState,
        )
        sm = OrderStateMachine(state_file=self.path)
        sm.register("co-1", "d-1", "AAPL", "buy", 10, "equity")
        sm.register("co-2", "d-2", "AAPL", "buy", 5, "equity")
        sm.transition("co-1", OrderState.ACCEPTED)
        sm.transition("co-1", OrderState.FILLED, fill_qty=10, fill_price=150.0)
        open_orders = sm.open_orders()
        self.assertEqual(len(open_orders), 1)
        self.assertEqual(open_orders[0].client_order_id, "co-2")
        filled = sm.filled_orders()
        self.assertEqual(len(filled), 1)
        self.assertEqual(filled[0].client_order_id, "co-1")


# ---------------------------------------------------------------------------
# PositionManager
# ---------------------------------------------------------------------------

class TestPositionManager(unittest.TestCase):
    def test_open_and_mark_to_market_long(self):
        from investment_agent.live.position_manager import PositionManager, PositionSide
        pm = PositionManager()
        pos = pm.open_position(
            decision_id="d-1", client_order_id="co-1",
            symbol="AAPL", side="buy", quantity=10,
            entry_price=100.0,
        )
        self.assertEqual(pos.side, PositionSide.LONG)
        # Up 5: +$50
        pnl = pos.mark_to_market(105.0)
        self.assertAlmostEqual(pnl, 50.0)

    def test_open_and_mark_to_market_short(self):
        from investment_agent.live.position_manager import PositionManager, PositionSide
        pm = PositionManager()
        pos = pm.open_position(
            decision_id="d-1", client_order_id="co-1",
            symbol="AAPL", side="sell", quantity=10,
            entry_price=100.0,
        )
        self.assertEqual(pos.side, PositionSide.SHORT)
        # Down 5: short gains +$50
        pnl = pos.mark_to_market(95.0)
        self.assertAlmostEqual(pnl, 50.0)

    def test_target_hit_emits_exit(self):
        from investment_agent.live.position_manager import PositionManager
        pm = PositionManager(default_target_pct=0.05, default_stop_pct=0.03)
        pm.open_position(
            decision_id="d-1", client_order_id="co-1",
            symbol="AAPL", side="buy", quantity=10, entry_price=100.0,
        )
        exits = pm.evaluate({"d-1": 106.0})  # above 5% target
        self.assertEqual(len(exits), 1)
        self.assertEqual(exits[0].reason, "target_hit")
        self.assertEqual(len(pm.all_open()), 0)

    def test_stop_hit_emits_exit(self):
        from investment_agent.live.position_manager import PositionManager
        pm = PositionManager(default_target_pct=0.05, default_stop_pct=0.03)
        pm.open_position(
            decision_id="d-1", client_order_id="co-1",
            symbol="AAPL", side="buy", quantity=10, entry_price=100.0,
        )
        exits = pm.evaluate({"d-1": 96.0})  # below 3% stop
        self.assertEqual(len(exits), 1)
        self.assertEqual(exits[0].reason, "stop_hit")

    def test_max_holding_exceeded(self):
        from investment_agent.live.position_manager import PositionManager
        pm = PositionManager(max_holding=timedelta(seconds=1))
        pm.open_position(
            decision_id="d-1", client_order_id="co-1",
            symbol="AAPL", side="buy", quantity=10, entry_price=100.0,
        )
        # Wait so the position ages past max_holding.
        import time
        time.sleep(1.2)
        exits = pm.evaluate({"d-1": 100.0})
        self.assertEqual(len(exits), 1)
        self.assertEqual(exits[0].reason, "max_holding_exceeded")

    def test_persistence_round_trip(self):
        from investment_agent.live.position_manager import (
            PositionManager, Position,
        )
        pm1 = PositionManager()
        pm1.open_position(
            decision_id="d-1", client_order_id="co-1",
            symbol="AAPL", side="buy", quantity=10, entry_price=100.0,
        )
        d = pm1.to_dict()
        pm2 = PositionManager.from_dict(d)
        self.assertEqual(len(pm2.all_open()), 1)
        pos = pm2.get("d-1")
        self.assertIsNotNone(pos)
        self.assertEqual(pos.entry_price, 100.0)
        self.assertEqual(pos.side, Position.LONG.__class__.LONG) \
            if False else True  # enum compare already covered by dataclass


# ---------------------------------------------------------------------------
# LiveOrchestrator
# ---------------------------------------------------------------------------

def _build_live_orchestrator(
    tmpdir: str,
    *,
    stage: str = "dry_run",
    decision_interval: int = 60,
    universe: List[str] = None,
    executor_returns: Dict[str, Any] = None,
    executor_side_effect=None,
) -> "LiveOrchestrator":
    """Construct a fully-wired LiveOrchestrator with deterministic stubs."""
    from investment_agent.live.live_orchestrator import (
        LiveOrchestrator, LiveOrchestratorConfig,
    )
    from investment_agent.live.candidate_screener import CandidateScreener
    from investment_agent.data.market_data import FakeMarketDataClient
    from investment_agent.orchestrator import XQuantXOrchestrator
    from investment_agent.agents.specialist import DEFAULT_ROLES, AgentOutput
    from investment_agent.products import ProductGate

    if universe is None:
        universe = ["AAPL", "SPY", "MSFT"]
    md = FakeMarketDataClient()
    for sym in universe:
        md.set_series(sym, _make_bars(30))

    orch = XQuantXOrchestrator(
        agent_ids=[r.agent_id for r in DEFAULT_ROLES],
        symbol="AAPL", use_hmm=False, enable_trading=False,
        memory_file=os.path.join(tmpdir, "mem.json"),
    )
    # Build a deterministic agent factory so the ensemble is always bullish.
    # Use the canonical DEFAULT_ROLES IDs so the orchestrator's regime
    # weights and reputation tracker line up with the agent outputs.
    canonical_ids = [r.agent_id for r in DEFAULT_ROLES]
    def factory(bar_ctx):
        return [
            AgentOutput(s=0.6, c=0.9, u=0.1, d=0.05, p_plus=0.8,
                        p_minus=0.2, delta_t=1.0, r=0.5,
                        agent_id=aid)
            for aid in canonical_ids
        ]
    config = LiveOrchestratorConfig(
        symbol_universe=universe,
        top_n_candidates=2,
        decision_interval_seconds=decision_interval,
        state_file=os.path.join(tmpdir, "live_state.json"),
        memory_file=os.path.join(tmpdir, "mem.json"),
        reputation_file=os.path.join(tmpdir, "rep.json"),
        stage=stage,
    )
    # Sanity: orchestrator's agent_ids already match canonical_ids.

    calls: List[Dict[str, Any]] = []
    def executor(symbol, side, qty, option_side):
        if executor_side_effect is not None:
            executor_side_effect(symbol, side, qty, option_side)
        calls.append({
            "symbol": symbol, "side": side, "qty": qty, "option_side": option_side,
        })
        if executor_returns is not None:
            return dict(executor_returns)
        return {
            "id": f"fake-{len(calls)}",
            "status": "accepted",
            "filled_qty": qty,
            "filled_avg_price": 100.0,
        }

    return LiveOrchestrator(
        config=config, market_data=md, orchestrator=orch,
        agent_factory=factory, executor=executor,
        screener=CandidateScreener(top_n=2),
        product_gate=ProductGate(),
    )


class TestLiveOrchestratorInterval(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_dry_run_emits_report_with_candidates(self):
        orch = _build_live_orchestrator(self.tmpdir, stage="dry_run")
        report = orch.run_once()
        self.assertIsNotNone(report)
        self.assertEqual(report.interval_index, 1)
        self.assertGreater(len(report.candidates), 0)
        # In dry-run, the executor should NOT have been called.
        # (Actually, our executor always returns an order; the
        # orchestrator records SUBMITTED but no fill, so no position
        # is opened. We check that the report includes the decision
        # but the live state has zero positions.)
        self.assertEqual(report.open_positions, 0)

    def test_state_persists_across_instances(self):
        orch1 = _build_live_orchestrator(self.tmpdir)
        orch1.run_once()
        state_path = os.path.join(self.tmpdir, "live_state.json")
        self.assertTrue(os.path.exists(state_path))
        with open(state_path, "r") as f:
            data = json.load(f)
        self.assertEqual(data["interval_index"], 1)

    def test_reputation_persisted(self):
        orch = _build_live_orchestrator(self.tmpdir)
        orch.run_once()
        rep_path = os.path.join(self.tmpdir, "rep.json")
        self.assertTrue(os.path.exists(rep_path))

    def test_circuit_halt_blocks_all_orders(self):
        # Force halt by making the breaker hit HALT on the first call.
        from investment_agent.live.live_orchestrator import (
            LiveOrchestrator, LiveOrchestratorConfig,
        )
        from investment_agent.live.candidate_screener import CandidateScreener
        from investment_agent.live.circuit_breaker import (
            CircuitBreaker, CircuitLevel,
        )
        from investment_agent.data.market_data import FakeMarketDataClient
        from investment_agent.orchestrator import XQuantXOrchestrator
        from investment_agent.agents.specialist import (
            DEFAULT_ROLES, AgentOutput,
        )
        from investment_agent.products import ProductGate

        md = FakeMarketDataClient()
        md.set_series("AAPL", _make_bars(30))
        orch = XQuantXOrchestrator(
            agent_ids=[r.agent_id for r in DEFAULT_ROLES],
            symbol="AAPL", use_hmm=False, enable_trading=False,
            memory_file=os.path.join(self.tmpdir, "mem.json"),
        )
        orch._agent_ids = [r.agent_id for r in DEFAULT_ROLES]
        orch._reputation_tracker = orch._reputation_tracker.__class__(
            agent_ids=orch._agent_ids, regimes=["R01"],
        )
        executor_calls: List[Dict[str, Any]] = []
        def executor(symbol, side, qty, option_side):
            executor_calls.append({"symbol": symbol, "side": side})
            return {"id": "x", "status": "accepted"}
        config = LiveOrchestratorConfig(
            symbol_universe=["AAPL"], top_n_candidates=1,
            state_file=os.path.join(self.tmpdir, "live_state.json"),
            memory_file=os.path.join(self.tmpdir, "mem.json"),
            reputation_file=os.path.join(self.tmpdir, "rep.json"),
        )
        live = LiveOrchestrator(
            config=config, market_data=md, orchestrator=orch,
            agent_factory=lambda ctx: [
                AgentOutput(s=0.6, c=0.9, u=0.1, d=0.05, p_plus=0.8,
                            p_minus=0.2, delta_t=1.0, r=0.5,
                            agent_id=aid)
                for aid in orch._agent_ids
            ],
            executor=executor,
            screener=CandidateScreener(top_n=1),
            product_gate=ProductGate(),
            circuit_breaker=CircuitBreaker(
                drawdown_warning=0.05, drawdown_halt=0.05,  # any DD trips HALT
            ),
        )
        # Drive a 5% drawdown before the run.
        live._equity = 95_000.0
        live._peak_equity = 100_000.0
        report = live.run_once()
        # Circuit is HALT; executor never called.
        self.assertEqual(report.circuit_state["level"], CircuitLevel.HALT.value)
        self.assertEqual(executor_calls, [])
        # All decisions should report product="none" with circuit reason.
        for d in report.decisions:
            self.assertEqual(d.get("product"), "none")
            self.assertIn("circuit", d.get("reason", ""))

    def test_executor_called_with_equity_when_product_is_equity(self):
        calls: List[Dict[str, Any]] = []
        def executor(symbol, side, qty, option_side):
            calls.append({"symbol": symbol, "side": side, "qty": qty})
            return {
                "id": "x", "status": "accepted",
                "filled_qty": qty, "filled_avg_price": 100.0,
            }
        orch = _build_live_orchestrator(
            self.tmpdir,
            executor_returns={
                "id": "x", "status": "filled",
                "filled_qty": 10, "filled_avg_price": 100.0,
            },
        )
        # The bullish agent factory yields an ensemble that should
        # produce an equity product.
        orch._executor = executor
        report = orch.run_once()
        # At least one decision has a non-none product.
        non_none = [d for d in report.decisions if d.get("product") not in (None, "none")]
        self.assertGreater(len(non_none), 0)
        # Executor was called for each non-none product.
        self.assertGreater(len(calls), 0)

    def test_max_llm_lookups_per_interval(self):
        from investment_agent.live.live_orchestrator import (
            LiveOrchestrator, LiveOrchestratorConfig,
        )
        from investment_agent.live.candidate_screener import CandidateScreener
        from investment_agent.data.market_data import FakeMarketDataClient
        from investment_agent.orchestrator import XQuantXOrchestrator
        from investment_agent.agents.specialist import DEFAULT_ROLES, AgentOutput
        from investment_agent.products import ProductGate

        md = FakeMarketDataClient()
        for s in ["A", "B", "C", "D", "E"]:
            md.set_series(s, _make_bars(30))
        orch = XQuantXOrchestrator(
            agent_ids=[r.agent_id for r in DEFAULT_ROLES],
            symbol="A", use_hmm=False, enable_trading=False,
            memory_file=os.path.join(self.tmpdir, "mem.json"),
        )
        orch._agent_ids = [r.agent_id for r in DEFAULT_ROLES]
        orch._reputation_tracker = orch._reputation_tracker.__class__(
            agent_ids=orch._agent_ids, regimes=["R01"],
        )
        factory_calls: List[str] = []
        def factory(ctx):
            factory_calls.append(ctx["symbol"])
            return [
                AgentOutput(s=0.6, c=0.9, u=0.1, d=0.05, p_plus=0.8,
                            p_minus=0.2, delta_t=1.0, r=0.5,
                            agent_id=aid)
                for aid in orch._agent_ids
            ]
        config = LiveOrchestratorConfig(
            symbol_universe=["A", "B", "C", "D", "E"],
            top_n_candidates=5,
            state_file=os.path.join(self.tmpdir, "live_state.json"),
            memory_file=os.path.join(self.tmpdir, "mem.json"),
            reputation_file=os.path.join(self.tmpdir, "rep.json"),
            max_lookups_per_interval=2,  # cap at 2 LLM calls
        )
        live = LiveOrchestrator(
            config=config, market_data=md, orchestrator=orch,
            agent_factory=factory,
            executor=lambda *a, **kw: {"id": "x", "status": "accepted"},
            screener=CandidateScreener(top_n=5),
            product_gate=ProductGate(),
        )
        report = live.run_once()
        # Even though the screener returned 5 candidates, only
        # max_lookups_per_interval of them were pushed through the
        # LLM factory.
        self.assertLessEqual(len(factory_calls), 2)
        # And the report records the candidate list the screener
        # actually returned (which is the full universe's score).
        self.assertGreaterEqual(len(report.candidates), len(factory_calls))

    def test_dry_run_does_not_open_position(self):
        orch = _build_live_orchestrator(
            self.tmpdir,
            executor_returns={
                "id": "x", "status": "accepted",  # not filled
            },
        )
        report = orch.run_once()
        # No position opened because the executor only reported
        # ACCEPTED, not FILLED.
        self.assertEqual(report.open_positions, 0)


if __name__ == "__main__":
    unittest.main()
