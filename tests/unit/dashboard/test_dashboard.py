"""Tests for the new 'control-room' dashboard data_loader + chart builders."""

import json
import os
import sys
import tempfile
import unittest
from unittest.mock import patch, MagicMock
from typing import List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))

from investment_agent.dashboard import charts, data_loader, layout


def _write_trade_memory(tmpdir, rows: List[dict]) -> str:
    path = os.path.join(tmpdir, "trade_memory.json")
    with open(path, "w") as f:
        json.dump(rows, f)
    return path


def _write_audit_log(tmpdir, events: List[dict]) -> str:
    path = os.path.join(tmpdir, "audit_log.jsonl")
    with open(path, "w") as f:
        for ev in events:
            f.write(json.dumps(ev) + "\n")
    return path


def _write_llm_usage(tmpdir, records: List[dict]) -> str:
    path = os.path.join(tmpdir, "llm_usage.jsonl")
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    return path


def _sample_trade_row() -> dict:
    return {
        "decision_id": "d-1",
        "timestamp": "2026-09-01T08:00:00",
        "symbol": "AAPL",
        "regime": "R03",
        "regime_probabilities": {"R01": 0.1, "R02": 0.15, "R03": 0.6, "R04": 0.15},
        "agent_signals": {
            "agent_economic": 0.5,
            "agent_financial": 0.3,
            "agent_fundamental": 0.7,
            "agent_market": 0.2,
            "agent_sector": 0.4,
            "agent_portfolio": -0.1,
            "agent_fiscal": 0.0,
        },
        "ensemble_signal": 0.45,
        "disagreement": 0.15,
        "effective_confidence": 0.81,
        "kalman_gain": 0.73,
        "kalman_price": 100.5,
        "kalman_trend": 0.01,
        "capital_gate_verdict": "ALLOW",
        "effective_cap": 0.085,
        "state_charges": {"S1": 0.91, "S2": 0.74, "S3": 0.82, "S4": 0.55, "S5": 0.88, "S6": 0.67, "S7": 0.93},
        "position_action": "BUY",
        "quantity": 0.085,
        "confidence": 0.9,
        "expected_outcome": "",
        "realized_outcome": "Hit target",
        "pnl": 250.0,
        "lesson": "Trend worked",
        "lifecycle_status": "CLOSED",
    }


def _sample_audit_event(decision_id: str = "d-1") -> dict:
    return {
        "event_id": "ev-1",
        "event_type": "DECISION",
        "decision_id": decision_id,
        "timestamp": "2026-09-01T08:00:00",
        "symbol": "AAPL",
        "payload": {
            "action": "BUY",
            "quantity": 0.085,
            "confidence": 0.9,
            "verdict": "ALLOW",
            "effective_cap": 0.085,
            "kalman_gain": 0.73,
            "ensemble_signal": 0.45,
            "disagreement": 0.15,
            "regime": "R03",
            "regime_probabilities": {"R01": 0.1, "R02": 0.15, "R03": 0.6, "R04": 0.15},
        },
    }


def _sample_llm_record(provider_id: str = "deephermes", success: bool = True, latency_ms: float = 4500.0) -> dict:
    return {
        "timestamp": "2026-09-01T08:00:00",
        "provider_id": provider_id,
        "model": "NousResearch/DeepHermes-3-Llama-3-8B-Preview",
        "success": success,
        "latency_ms": latency_ms,
        "prompt_tokens": 85,
        "completion_tokens": 64,
        "error": "" if success else "BACKOFF: capacity_exhausted",
    }


class TestDataLoaderAudit(unittest.TestCase):
    def test_load_audit_events_returns_newest_first(self):
        with tempfile.TemporaryDirectory() as d:
            path = _write_audit_log(d, [
                {"event_id": "1", "event_type": "DECISION", "decision_id": "a", "timestamp": "2026-09-01T07:00:00", "symbol": "AAPL", "payload": {}},
                {"event_id": "2", "event_type": "DECISION", "decision_id": "b", "timestamp": "2026-09-01T08:00:00", "symbol": "AAPL", "payload": {}},
            ])
            events = data_loader.load_audit_events(path=path)
            self.assertEqual([e["event_id"] for e in events], ["2", "1"])

    def test_latest_decision_event_returns_decision(self):
        with tempfile.TemporaryDirectory() as d:
            path = _write_audit_log(d, [
                {"event_id": "1", "event_type": "OUTCOME", "decision_id": "a", "timestamp": "2026-09-01T07:00:00", "payload": {}},
                {"event_id": "2", "event_type": "DECISION", "decision_id": "b", "timestamp": "2026-09-01T08:00:00", "symbol": "AAPL", "payload": {"action": "BUY"}},
            ])
            ev = data_loader.latest_decision_event(path=path)
            self.assertIsNotNone(ev)
            self.assertEqual(ev["decision_id"], "b")

    def test_missing_audit_log_returns_empty(self):
        self.assertEqual(data_loader.load_audit_events(path="/nonexistent.jsonl"), [])
        self.assertIsNone(data_loader.latest_decision_event(path="/nonexistent.jsonl"))


class TestDataLoaderLLM(unittest.TestCase):
    def test_summarize_llm_providers_groups_by_provider(self):
        with tempfile.TemporaryDirectory() as d:
            path = _write_llm_usage(d, [
                _sample_llm_record("deephermes", success=True, latency_ms=4500.0),
                _sample_llm_record("deephermes", success=False, latency_ms=200.0),
                _sample_llm_record("reserve", success=True, latency_ms=2500.0),
            ])
            rows = data_loader.summarize_llm_providers(path=path)
            by_pid = {r["provider_id"]: r for r in rows}
            self.assertEqual(by_pid["deephermes"]["total_calls"], 2)
            self.assertEqual(by_pid["deephermes"]["success_calls"], 1)
            self.assertEqual(by_pid["deephermes"]["failure_calls"], 1)
            self.assertEqual(by_pid["reserve"]["total_calls"], 1)
            self.assertEqual(by_pid["reserve"]["last_status"], "ok")

    def test_empty_llm_log_returns_empty(self):
        self.assertEqual(data_loader.summarize_llm_providers(path="/nonexistent.jsonl"), [])


class TestDataLoaderPanels(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.history = [_sample_trade_row()]
        self.trade_path = _write_trade_memory(self.tmpdir, self.history)
        self.cycle = data_loader.latest_cycle_snapshot(history=self.history)

    def test_get_seven_state_charges_seven_rows(self):
        charges = data_loader.get_seven_state_charges(self.cycle)
        self.assertEqual(len(charges), 7)
        self.assertEqual({c["label"].split()[0] for c in charges}, {"S1", "S2", "S3", "S4", "S5", "S6", "S7"})
        # All values are in [0, 1] (post-clip)
        for c in charges:
            self.assertGreaterEqual(c["value"], 0.0)
            self.assertLessEqual(c["value"], 1.0)

    def test_get_seven_agents_seven_rows(self):
        agents = data_loader.get_seven_agents(self.cycle, self.history)
        self.assertEqual(len(agents), 7)
        for a in agents:
            self.assertIn("agent_id", a)
            self.assertIn("signal", a)
            self.assertIn("confidence", a)

    def test_get_kalman_card_posterior_equals_blend(self):
        """When the cycle has authoritative fields, they are returned directly.

        ``posterior_authoritative=True`` proves the dashboard is reading
        the value the orchestrator wrote, not reconstructing a blend.
        """
        self.cycle["kalman_prior"] = 0.81
        self.cycle["kalman_observation"] = 0.45
        self.cycle["investment_kalman_gain"] = 0.73
        self.cycle["kalman_posterior"] = 0.5472  # authoritatively written by orchestrator
        k = data_loader.get_kalman_card(self.cycle)
        self.assertAlmostEqual(k["posterior_estimate"], 0.5472, places=4)
        self.assertTrue(k["posterior_authoritative"])
        self.assertEqual(k["kalman_gain"], 0.73)
        self.assertEqual(k["prior_confidence"], 0.81)
        self.assertEqual(k["market_observation"], 0.45)

    def test_get_kalman_card_falls_back_without_authority(self):
        """Without ``kalman_posterior``, the dashboard reports 0.0 and
        flags ``posterior_authoritative=False`` instead of fabricating
        a Bayesian blend.
        """
        self.cycle.pop("kalman_posterior", None)
        self.cycle.pop("kalman_prior", None)
        self.cycle.pop("kalman_observation", None)
        self.cycle.pop("investment_kalman_gain", None)
        k = data_loader.get_kalman_card(self.cycle)
        self.assertFalse(k["posterior_authoritative"])
        self.assertEqual(k["posterior_estimate"], 0.0)

    def test_get_regime_card_returns_top(self):
        rc = data_loader.get_regime_card(self.cycle)
        self.assertEqual(rc["regime"], "R03")
        self.assertAlmostEqual(rc["top_probability"], 0.6, places=4)

    def test_get_circuit_breaker_state_levels(self):
        for verdict, level in [("ALLOW", "NORMAL"), ("REDUCE", "WARN"),
                                ("BLOCK", "CRITICAL"), ("FLATTEN", "FLATTEN")]:
            self.cycle["capital_gate_verdict"] = verdict
            self.assertEqual(
                data_loader.get_circuit_breaker_state(self.cycle)["level"],
                level,
            )

    def test_get_risk_gates_status_liquidity_fail(self):
        self.cycle["available_liquidity"] = 1000.0
        gates = data_loader.get_risk_gates_status(self.cycle, self.history)
        liq_gate = next(g for g in gates if g["gate_id"].startswith("LIQ-001"))
        self.assertEqual(liq_gate["status"], "FAIL")

    def test_get_trade_outcome_learning_accuracy(self):
        rows = data_loader.get_trade_outcome_learning(self.history, last_n=50)
        self.assertGreater(len(rows), 0)
        for r in rows:
            self.assertIn("agent_id", r)
            self.assertGreaterEqual(r["accuracy"], 0.0)
            self.assertLessEqual(r["accuracy"], 1.0)

    def test_get_reputation_snapshot_uniform_when_no_tracker_persisted(self):
        # Force a no-tracker read so the test is hermetic (it does not
        # pick up any stray reputation_state.json in the repo root).
        import os
        non_existent = os.path.join(
            tempfile.mkdtemp(), "definitely_does_not_exist.json",
        )
        rows = data_loader.get_reputation_snapshot(
            self.history, regime="R01", reputation_path=non_existent,
        )
        self.assertEqual(len(rows), 7)
        for r in rows:
            self.assertEqual(r["alpha"], 1.0)
            self.assertEqual(r["beta"], 1.0)
            self.assertEqual(r["weight"], 0.5)
            self.assertEqual(r["source"], "uniform_prior")

    def test_get_reputation_snapshot_reads_persisted_tracker(self):
        import tempfile
        from pathlib import Path
        from investment_agent.agents.agent_reputation import AgentReputationTracker
        from investment_agent.agents.reputation_persistence import save_reputation
        # Use the agent IDs the history actually carries.
        agent_ids = ["agent_economic", "agent_financial", "agent_fundamental",
                     "agent_market", "agent_sector", "agent_portfolio", "agent_fiscal"]
        with tempfile.TemporaryDirectory() as d:
            path = str(Path(d) / "reputation.json")
            tracker = AgentReputationTracker(agent_ids=agent_ids, regimes=["R01"])
            for _ in range(3):
                tracker.record_outcome("agent_economic", "R01", True)
            tracker.record_outcome("agent_economic", "R01", False)
            save_reputation(tracker, path)
            rows = data_loader.get_reputation_snapshot(
                self.history, regime="R01", reputation_path=path,
            )
        a_econ = next(r for r in rows if r["agent_id"] == "agent_economic")
        self.assertEqual(a_econ["source"], "persisted_tracker")
        self.assertEqual(a_econ["alpha"], 4.0)
        self.assertEqual(a_econ["beta"], 2.0)
        self.assertEqual(a_econ["regime"], "R01")
        self.assertGreater(a_econ["weight"], 0.5)

    def test_get_decision_waterfall_nine_steps(self):
        ev = _sample_audit_event()
        steps = data_loader.get_decision_waterfall(self.cycle, ev)
        # 9 stages per spec
        self.assertEqual(len(steps), 9)
        self.assertEqual(steps[0]["stage"], "Market observation")
        self.assertEqual(steps[-1]["stage"], "Decision")

    def test_get_top_exposure_pct_zero_when_no_positions(self):
        pos = {"ok": True, "positions": []}
        self.assertEqual(data_loader.get_top_exposure_pct(pos, 100000.0), 0.0)

    def test_options_snapshot_returns_envelope(self):
        snap = data_loader.get_options_snapshot_safe()
        self.assertIn("ok", snap)
        self.assertIn("contracts", snap)

    def test_get_alpaca_account_snapshot_prefers_live_state_cache(self, tmp_workdir=None):
        import json
        import os
        import tempfile
        from unittest.mock import patch

        tmpdir = tempfile.mkdtemp(prefix="xqx_dash_live_state_")
        try:
            cached = {
                "ok": True,
                "equity": 100500.0,
                "cash": 87900.0,
                "buying_power": 351619.88,
                "last_equity": 100000.0,
                "portfolio_value": 100500.0,
                "daily_pnl": 500.0,
                "daily_pnl_pct": 0.005,
                "total_pnl": 500.0,
                "total_pnl_pct": 0.005,
                "baseline_equity": 100000.0,
                "snapshot_at": "2026-09-01T15:00:00+00:00",
            }
            state_path = os.path.join(tmpdir, "live_state.json")
            with open(state_path, "w") as f:
                json.dump({"alpaca_account": cached}, f)

            with patch.object(data_loader, "LIVE_STATE_PATH", state_path):
                snap = data_loader.get_alpaca_account_snapshot()
            self.assertTrue(snap["ok"])
            self.assertEqual(snap["equity"], 100500.0)
            # The cached snapshot is returned as-is; the loader does
            # NOT make a fresh broker call when a valid cache exists.
            self.assertEqual(snap["total_pnl"], 500.0)
        finally:
            os.unlink(state_path)
            os.rmdir(tmpdir)

    def test_get_alpaca_account_snapshot_falls_back_to_live_call(self):
        from unittest.mock import patch

        fake_snap = {
            "ok": True,
            "equity": 100123.45,
            "cash": 87900.0,
            "buying_power": 351619.88,
            "last_equity": 100000.0,
            "portfolio_value": 100123.45,
            "daily_pnl": 123.45,
            "daily_pnl_pct": 0.0012345,
            "total_pnl": 123.45,
            "total_pnl_pct": 0.0012345,
            "baseline_equity": 100000.0,
            "snapshot_at": "2026-09-01T15:00:00+00:00",
        }
        with patch.object(data_loader, "_load_live_state_snapshot", return_value={}), \
             patch("investment_agent.execution.execution.get_account_snapshot",
                   return_value=fake_snap):
            snap = data_loader.get_alpaca_account_snapshot()
        self.assertTrue(snap["ok"])
        self.assertEqual(snap["equity"], 100123.45)
        self.assertEqual(snap["daily_pnl"], 123.45)

    def test_get_alpaca_account_snapshot_handles_broker_error(self):
        from unittest.mock import patch

        with patch.object(data_loader, "_load_live_state_snapshot", return_value={}), \
             patch("investment_agent.execution.execution.get_account_snapshot",
                   side_effect=RuntimeError("network down")):
            snap = data_loader.get_alpaca_account_snapshot()
        self.assertFalse(snap["ok"])
        self.assertIn("network down", snap["error"])
        self.assertIn("snapshot_at", snap)

    def test_get_alpaca_account_snapshot_ignores_stale_cache_without_ok_flag(self):
        import json
        import os
        import tempfile
        from unittest.mock import patch

        tmpdir = tempfile.mkdtemp(prefix="xqx_dash_stale_live_state_")
        try:
            state_path = os.path.join(tmpdir, "live_state.json")
            # No "ok": True -- the cache should be considered invalid
            # and the loader should fall through to a live call.
            with open(state_path, "w") as f:
                json.dump({"alpaca_account": {"ok": False, "error": "old"}}, f)
            live_snap = {
                "ok": True,
                "equity": 99999.0,
                "buying_power": 300000.0,
                "last_equity": 99000.0,
                "daily_pnl": 999.0,
                "snapshot_at": "2026-09-01T15:00:00+00:00",
            }
            with patch.object(data_loader, "LIVE_STATE_PATH", state_path), \
                 patch("investment_agent.execution.execution.get_account_snapshot",
                       return_value=live_snap):
                snap = data_loader.get_alpaca_account_snapshot()
            self.assertTrue(snap["ok"])
            self.assertEqual(snap["equity"], 99999.0)
        finally:
            os.unlink(state_path)
            os.rmdir(tmpdir)


class TestChartBuilders(unittest.TestCase):
    def test_seven_state_soc_empty(self):
        fig = charts.build_seven_state_soc_chart([])
        self.assertIsNotNone(fig)

    def test_seven_state_soc_with_data(self):
        fig = charts.build_seven_state_soc_chart([
            {"label": f"S{i} X", "value": v} for i, v in enumerate([0.91, 0.74, 0.82, 0.55, 0.88, 0.67, 0.93])
        ])
        self.assertIsNotNone(fig)

    def test_seven_agents_table_empty(self):
        fig = charts.build_seven_agents_table([])
        self.assertIsNotNone(fig)

    def test_seven_agents_table_with_data(self):
        agents = [
            {"agent_id": f"a{i}", "signal": 0.5, "confidence": 0.8, "weight": 0.1, "status": "ok"}
            for i in range(7)
        ]
        fig = charts.build_seven_agents_table(agents)
        self.assertIsNotNone(fig)

    def test_kalman_chart_with_data(self):
        k = {
            "kalman_gain": 0.73,
            "prior_confidence": 0.61,
            "market_observation": 0.84,
            "posterior_estimate": 0.78,
        }
        fig = charts.build_kalman_chart(k)
        self.assertIsNotNone(fig)

    def test_kalman_chart_empty(self):
        fig = charts.build_kalman_chart({})
        self.assertIsNotNone(fig)

    def test_regime_panel_with_data(self):
        rc = {"regime": "R03", "top_probability": 0.6,
              "probabilities": {"R01": 0.1, "R02": 0.2, "R03": 0.6, "R04": 0.1}}
        fig = charts.build_regime_panel_chart(rc)
        self.assertIsNotNone(fig)

    def test_regime_panel_empty(self):
        fig = charts.build_regime_panel_chart({})
        self.assertIsNotNone(fig)

    def test_llm_providers_table_with_rows(self):
        rows = [
            {"provider_id": "deephermes", "model": "m1", "last_status": "ok",
             "last_latency_ms": 4500.0, "last_tokens": 149, "success_calls": 3, "failure_calls": 0},
            {"provider_id": "fundamentals", "model": "m2", "last_status": "fail",
             "last_latency_ms": 200.0, "last_tokens": 0, "success_calls": 0, "failure_calls": 1},
        ]
        fig = charts.build_llm_providers_table(rows)
        self.assertIsNotNone(fig)

    def test_llm_providers_table_empty(self):
        fig = charts.build_llm_providers_table([])
        self.assertIsNotNone(fig)

    def test_options_table_empty(self):
        fig = charts.build_options_table([])
        self.assertIsNotNone(fig)

    def test_options_table_with_data(self):
        rows = [{"underlying": "SPY", "symbol": "SPY260101C450", "side": "BUY", "status": "FILLED"}]
        fig = charts.build_options_table(rows)
        self.assertIsNotNone(fig)

    def test_trade_outcome_table_with_rows(self):
        rows = [
            {"agent_id": "a1", "correct": 8, "incorrect": 2, "accuracy": 0.8},
            {"agent_id": "a2", "correct": 5, "incorrect": 5, "accuracy": 0.5},
        ]
        fig = charts.build_trade_outcome_table(rows)
        self.assertIsNotNone(fig)

    def test_trade_outcome_table_empty(self):
        fig = charts.build_trade_outcome_table([])
        self.assertIsNotNone(fig)

    def test_reputation_table_with_rows(self):
        rows = [
            {"agent_id": "a1", "alpha": 9.0, "beta": 1.0, "weight": 0.9, "closed_trades": 10},
        ]
        fig = charts.build_reputation_table(rows)
        self.assertIsNotNone(fig)

    def test_reputation_table_empty(self):
        fig = charts.build_reputation_table([])
        self.assertIsNotNone(fig)

    def test_decision_waterfall_with_steps(self):
        steps = [
            {"stage": "Market observation", "value": "live", "status": "info"},
            {"stage": "Capital gate", "value": "verdict=ALLOW", "status": "pass"},
        ]
        fig = charts.build_decision_waterfall(steps)
        self.assertIsNotNone(fig)

    def test_decision_waterfall_empty(self):
        fig = charts.build_decision_waterfall([])
        self.assertIsNotNone(fig)


# ---------------------------------------------------------------------------
# P0 dashboard data-integrity additions
# ---------------------------------------------------------------------------

class TestAuthoritativeKalmanProvenance(unittest.TestCase):
    def setUp(self):
        self.cycle = {
            "regime": "R01",
            "ensemble_signal": 0.45,
            "effective_confidence": 0.81,
            "kalman_gain": 0.73,
            "kalman_price": 100.0,
            "kalman_trend": 0.01,
            "kalman_posterior": 0.5472,
            "kalman_prior": 0.81,
            "kalman_observation": 0.45,
            "investment_kalman_gain": 0.73,
        }

    def test_uses_authoritative_kalman_posterior(self):
        k = data_loader.get_kalman_card(self.cycle)
        self.assertTrue(k["posterior_authoritative"])
        self.assertEqual(k["posterior_estimate"], 0.5472)

    def test_falls_back_when_no_posterior(self):
        self.cycle.pop("kalman_posterior", None)
        k = data_loader.get_kalman_card(self.cycle)
        self.assertFalse(k["posterior_authoritative"])
        self.assertEqual(k["posterior_estimate"], 0.0)


class TestRealOptionsActivity(unittest.TestCase):
    def test_is_option_symbol_occ_format(self):
        self.assertTrue(data_loader._is_option_symbol("AAPL240119C00200000"))
        self.assertTrue(data_loader._is_option_symbol("SPY260620P00425000"))
        # 6-char underlying + 6 date + C/P + 8 strike
        self.assertFalse(data_loader._is_option_symbol("AAPL"))  # too short
        self.assertFalse(data_loader._is_option_symbol("AAPL240119X00200000"))  # X is invalid
        self.assertFalse(data_loader._is_option_symbol("BRK.B240119C00200000"))  # dot in underlying

    def test_get_recent_options_activity_filters_order_history(self):
        from unittest.mock import patch
        fake_orders = [
            {"timestamp": "2026-09-01T08:00:00", "order_id": "o1", "symbol": "AAPL240119C00200000",
             "side": "buy", "type": "market", "qty": 1.0, "filled_qty": 1.0,
             "filled_avg_price": 5.5, "status": "filled"},
            {"timestamp": "2026-09-01T08:01:00", "order_id": "o2", "symbol": "AAPL",
             "side": "buy", "type": "market", "qty": 10.0, "filled_qty": 10.0,
             "filled_avg_price": 200.0, "status": "filled"},
            {"timestamp": "2026-09-01T08:02:00", "order_id": "o3", "symbol": "SPY260620P00425000",
             "side": "sell", "type": "limit", "qty": 2.0, "filled_qty": 0.0,
             "filled_avg_price": None, "status": "new"},
        ]
        with patch.object(data_loader, "get_order_history_safe",
                          return_value={"ok": True, "orders": fake_orders}):
            payload = data_loader.get_recent_options_activity(limit=10)
        self.assertTrue(payload["ok"])
        self.assertEqual(len(payload["orders"]), 2)
        syms = {o["symbol"] for o in payload["orders"]}
        self.assertIn("AAPL240119C00200000", syms)
        self.assertIn("SPY260620P00425000", syms)
        # Underlying is the first 1-6 chars of the OCC symbol
        for o in payload["orders"]:
            self.assertEqual(o["asset_class"], "option")
            self.assertIn("underlying", o)
            self.assertIn(o["underlying"], ("AAPL", "SPY"))

    def test_get_recent_options_activity_handles_broker_error(self):
        from unittest.mock import patch
        with patch.object(data_loader, "get_order_history_safe",
                          return_value={"ok": False, "error": "no creds", "orders": []}):
            payload = data_loader.get_recent_options_activity()
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["orders"], [])


class TestAuthoritativeRiskThresholds(unittest.TestCase):
    def test_state_thresholds_loaded_from_capital_gate(self):
        rows = data_loader.get_authoritative_state_thresholds()
        # All seven canonical state names must be present
        names = {r["state"] for r in rows}
        self.assertEqual(
            names,
            {"economic", "financial", "fiscal", "portfolio", "fundamental", "market", "sector"},
        )
        for r in rows:
            self.assertIn("minimum", r)
            self.assertIn("full", r)
            self.assertGreaterEqual(r["full"], r["minimum"])
            self.assertGreater(r["full"], 0.0)
            self.assertLessEqual(r["full"], 1.0)

    def test_drawdown_thresholds_return_numbers(self):
        t = data_loader.get_drawdown_thresholds()
        self.assertIn("flatten", t)
        self.assertIn("reduce", t)
        self.assertGreater(t["flatten"], t["reduce"])
        self.assertGreater(t["flatten"], 0.0)
        self.assertLess(t["flatten"], 1.0)


class TestSevenAgentsAuthoritative(unittest.TestCase):
    def setUp(self):
        self.cycle = {
            "regime": "R01",
            "ensemble_signal": 0.45,
            "effective_confidence": 0.81,
            "agent_signals": {"agent1": 0.5, "agent2": 0.3},  # legacy
            "agent_outputs_full": {
                "agent1": {"signal": 0.5, "confidence": 0.9, "uncertainty": 0.1,
                           "doubt": 0.05, "p_plus": 0.7, "p_minus": 0.2,
                           "delta_t": 1.0, "noise": 0.3, "weight": 0.16,
                           "reputation_alpha": 9.0, "reputation_beta": 1.0},
                "agent2": {"signal": 0.3, "confidence": 0.8, "uncertainty": 0.2,
                           "doubt": 0.1, "p_plus": 0.6, "p_minus": 0.3,
                           "delta_t": 1.0, "noise": 0.4, "weight": 0.14,
                           "reputation_alpha": 5.0, "reputation_beta": 2.0},
            },
        }

    def test_full_agent_outputs_surfaced(self):
        agents = data_loader.get_seven_agents(self.cycle)
        self.assertEqual(len(agents), 2)
        for a in agents:
            for k in ("signal", "confidence", "uncertainty", "doubt",
                      "p_plus", "p_minus", "delta_t", "noise", "weight",
                      "reputation_alpha", "reputation_beta"):
                self.assertIn(k, a)
        a1 = next(a for a in agents if a["agent_id"] == "agent1")
        self.assertEqual(a1["signal"], 0.5)
        self.assertEqual(a1["reputation_alpha"], 9.0)
        self.assertEqual(a1["weight"], 0.16)

    def test_legacy_fallback_when_no_full_outputs(self):
        self.cycle.pop("agent_outputs_full", None)
        agents = data_loader.get_seven_agents(self.cycle)
        self.assertEqual(len(agents), 2)
        for a in agents:
            # Legacy fallback only carries scalar signal + coarse confidence
            self.assertIn(a["status"], ("ok", "ok (legacy)"))
            self.assertEqual(a["weight"], 0.0)


class TestEquityCurveAccounting(unittest.TestCase):
    def test_incremental_convention_default(self):
        rows = [
            {"timestamp": "t1", "pnl": 100.0},
            {"timestamp": "t2", "pnl": -50.0},
            {"timestamp": "t3", "pnl": 75.0},
        ]
        curve = data_loader.compute_equity_curve(rows, starting_equity=100000.0)
        self.assertEqual(curve[0]["equity"], 100100.0)
        self.assertEqual(curve[1]["equity"], 100050.0)
        self.assertEqual(curve[2]["equity"], 100125.0)

    def test_drawdown_thresholds_read_from_capital_gate(self):
        # When the module exposes DRAWDOWN_FLATTEN_PCT /
        # DRAWDOWN_REDUCE_PCT we should read them authoritatively
        # rather than falling back to hard-coded values.
        from investment_agent.capital import capital_gate
        if hasattr(capital_gate, "DRAWDOWN_FLATTEN_PCT"):
            result = data_loader.get_drawdown_thresholds()
            self.assertEqual(
                result["flatten"],
                float(capital_gate.DRAWDOWN_FLATTEN_PCT),
            )
        # Reduce key always present.
        result = data_loader.get_drawdown_thresholds()
        self.assertIn("reduce", result)

    def test_cumulative_convention(self):
        rows = [
            {"timestamp": "t1", "pnl": 100.0},   # running total = 100
            {"timestamp": "t2", "pnl": 100050.0},  # running total = 100050
        ]
        curve = data_loader.compute_equity_curve(rows, starting_equity=100000.0,
                                                  pnl_convention="cumulative")
        # In cumulative mode each row's pnl is the running total. We
        # use the running max because the series is monotonically
        # increasing (downstream the chart's drawdown is the peak).
        self.assertEqual(curve[0]["equity"], 100.0)
        self.assertEqual(curve[-1]["equity"], 100050.0)

    def test_unknown_convention_raises(self):
        with self.assertRaises(ValueError):
            data_loader.compute_equity_curve([], pnl_convention="bogus")

    def test_strategy_equity_summary(self):
        rows = [
            {"timestamp": "t1", "pnl": 100.0},
            {"timestamp": "t2", "pnl": -50.0},
        ]
        s = data_loader.get_strategy_equity_summary(rows, starting_equity=100000.0)
        self.assertEqual(s["current_equity"], 100050.0)
        self.assertEqual(s["realized_pnl"], 50.0)
        self.assertEqual(s["trade_count"], 2)


class TestSevenAgentsAuthoritativeBackedByOrchestrator(unittest.TestCase):
    """End-to-end: orchestrator writes the full per-agent data; the
    dashboard reads it without re-deriving anything."""

    def test_orchestrator_writes_agent_outputs_full(self):
        from investment_agent.orchestrator import XQuantXOrchestrator
        from investment_agent.capital.capital_gate import SevenStateVector
        from investment_agent.signals.ensemble_signal import AgentOutput
        from investment_agent.memory.trade_memory import TradeLifecycle
        import tempfile
        import uuid

        with tempfile.TemporaryDirectory() as d:
            mem_file = f"{d}/mem.json"
            orch = XQuantXOrchestrator(
                agent_ids=["agent1", "agent2", "agent3", "agent4",
                           "agent5", "agent6", "agent7"],
                symbol="AAPL",
                use_hmm=False,
                enable_trading=False,
                memory_file=mem_file,
            )
            agent_outputs = [
                AgentOutput(s=0.5, c=0.9, u=0.1, d=0.05, p_plus=0.7, p_minus=0.2,
                            delta_t=1.0, r=0.3, agent_id=f"agent{i + 1}")
                for i in range(7)
            ]
            states = SevenStateVector(
                economic=1.0, financial=1.0, fiscal=1.0,
                portfolio=1.0, fundamental=1.0, market=1.0, sector=1.0,
            )
            result = orch.run_cycle(
                prices=[100.0 + i * 0.1 for i in range(45)],
                volumes=[1000.0] * 45,
                agent_outputs=agent_outputs,
                states=states,
                portfolio_context={
                    "position_pct": 0.05, "gross_leverage": 0.5, "entropy": 0.1,
                    "drawdown_pct": 0.01, "execution_timeout_seconds": 5.0,
                    "sector_exposure_pct": 0.1, "is_new_long": False, "regime": "R01",
                    "available_liquidity": 100000.0,
                },
            )
            self.assertEqual(result.experience.lifecycle_status,
                             TradeLifecycle.PENDING_FILL.value)
            aof = result.experience.agent_outputs_full
            self.assertIsNotNone(aof)
            self.assertEqual(set(aof.keys()),
                             {"agent1", "agent2", "agent3", "agent4",
                              "agent5", "agent6", "agent7"})
            for aid, row in aof.items():
                self.assertIn("signal", row)
                self.assertIn("confidence", row)
                self.assertIn("uncertainty", row)
                self.assertIn("p_plus", row)
                self.assertIn("p_minus", row)
                self.assertIn("weight", row)
                self.assertIn("reputation_alpha", row)
                self.assertIn("reputation_beta", row)
            # And the kalman provenance fields are also authoritative
            self.assertIsNotNone(result.experience.investment_kalman_gain)
            self.assertIsNotNone(result.experience.kalman_posterior)
            self.assertIsNotNone(result.experience.kalman_prior)
            self.assertIsNotNone(result.experience.kalman_observation)


class TestChartBuildersAuthoritative(unittest.TestCase):
    """Chart-level tests for the P0 data-integrity changes."""

    def test_seven_agents_table_surfaces_full_channels(self):
        from investment_agent.dashboard import charts
        agents = [{
            "agent_id": "agent_economic",
            "signal": 0.42,
            "confidence": 0.81,
            "uncertainty": 0.18,
            "doubt": 0.07,
            "p_bull": 0.62,
            "p_bear": 0.38,
            "decision_time_ms": 410,
            "kalman_noise": 0.012,
            "weight": 0.14,
            "alpha": 9.0,
            "beta": 1.0,
            "status": "ok (authoritative)",
        }]
        fig = charts.build_seven_agents_table(agents)
        self.assertIsNotNone(fig)
        # The header should include the new columns.
        header_text = fig.layout.title.text
        self.assertIn("full per-agent channels", header_text)
        header_values = list(fig.data[0].header.values)
        self.assertIn("Unc", header_values)
        self.assertIn("Doubt", header_values)
        self.assertIn("p_Bull", header_values)
        self.assertIn("p_Bear", header_values)
        self.assertIn("Δt", header_values)
        self.assertIn("Reputation", header_values)

    def test_seven_agents_table_legacy_fallback_graceful(self):
        from investment_agent.dashboard import charts
        # Legacy row missing the extended fields.
        agents = [{"agent_id": "a1", "signal": 0.5, "confidence": 0.8, "weight": 0.1, "status": "ok (legacy)"}]
        fig = charts.build_seven_agents_table(agents)
        self.assertIsNotNone(fig)
        # Extended columns should render "—" rather than raise.
        cell_values = list(fig.data[0].cells.values)
        self.assertEqual(cell_values[3], ["—"])  # Unc
        self.assertEqual(cell_values[9], ["—"])  # Reputation

    def test_kalman_chart_labels_authoritative(self):
        from investment_agent.dashboard import charts
        k_auth = {
            "kalman_gain": 0.73, "prior_confidence": 0.61,
            "market_observation": 0.84, "posterior_estimate": 0.78,
            "posterior_authoritative": True,
        }
        fig = charts.build_kalman_chart(k_auth)
        self.assertIsNotNone(fig)
        self.assertIn("authoritative (state-gated)", fig.layout.title.text)

        k_legacy = dict(k_auth, posterior_authoritative=False)
        fig2 = charts.build_kalman_chart(k_legacy)
        self.assertIsNotNone(fig2)
        self.assertIn("reconstructed (legacy)", fig2.layout.title.text)

    def test_options_table_renders_error_message(self):
        from investment_agent.dashboard import charts
        fig = charts.build_options_table([], error="401 unauthorized")
        self.assertIsNotNone(fig)
        # The error message is rendered as a centred annotation.
        annotations = " ".join(a.text for a in fig.layout.annotations)
        self.assertIn("401 unauthorized", annotations)

    def test_options_table_with_broker_order(self):
        from investment_agent.dashboard import charts
        rows = [{
            "underlying": "SPY",
            "symbol": "SPY260620P00425000",
            "side": "BUY",
            "type": "limit",
            "qty": 1,
            "filled_qty": 1,
            "status": "filled",
        }]
        fig = charts.build_options_table(rows)
        self.assertIsNotNone(fig)
        cell_values = list(fig.data[0].cells.values)
        self.assertEqual(cell_values[0], ["SPY"])
        self.assertEqual(cell_values[1], ["SPY260620P00425000"])
        self.assertIn("real broker /orders filter", fig.layout.title.text)

    def test_equity_curve_chart_with_source_label(self):
        from investment_agent.dashboard import charts
        rows = [
            {"timestamp": "t1", "equity": 100100.0, "peak": 100100.0, "drawdown_pct": 0.0, "pnl": 100.0},
            {"timestamp": "t2", "equity": 100050.0, "peak": 100100.0, "drawdown_pct": -0.0005, "pnl": -50.0},
        ]
        fig = charts.build_equity_curve_chart(rows, source="strategy")
        self.assertIsNotNone(fig)
        self.assertIn("[strategy]", fig.layout.title.text)

    def test_drawdown_waterfall_uses_authoritative_thresholds(self):
        from investment_agent.dashboard import charts
        rows = [{"timestamp": "t1", "equity": 100000.0, "peak": 100000.0, "drawdown_pct": 0.0, "pnl": 0.0}]
        fig = charts.build_drawdown_waterfall_chart(rows, flatten_pct=0.20, reduce_pct=0.12)
        self.assertIsNotNone(fig)
        # Threshold annotations should reflect the override (20% / 12%).
        annotations = [a.text for a in fig.layout.annotations]
        joined = " ".join(annotations)
        self.assertIn("-20%", joined)
        self.assertIn("-12%", joined)


class TestAlpacaAccountTopPanel(unittest.TestCase):
    """Tests for the new "X QUANT X — ALPACA ACCOUNT" top panel."""

    def _flatten_text(self, component) -> str:
        """Walk a Dash component tree and concatenate every text node."""
        from dash import html
        if isinstance(component, str):
            return component
        if isinstance(component, dict):
            return " ".join(str(v) for v in component.values())
        if hasattr(component, "children"):
            kids = component.children
            if isinstance(kids, (list, tuple)):
                return " ".join(self._flatten_text(k) for k in kids)
            return self._flatten_text(kids)
        return ""

    def test_panel_renders_full_snapshot_fields(self):
        snap = {
            "ok": True,
            "equity": 100500.00,
            "cash": 87904.97,
            "buying_power": 351619.88,
            "last_equity": 100000.00,
            "portfolio_value": 100500.00,
            "daily_pnl": 500.00,
            "daily_pnl_pct": 0.005,
            "total_pnl": 500.00,
            "total_pnl_pct": 0.005,
            "baseline_equity": 100000.00,
            "snapshot_at": "2026-09-01T15:00:00+00:00",
        }
        panel = layout.alpaca_account_top_panel(snap)
        text = self._flatten_text(panel)
        # All five headline KPIs are present.
        for label in ("EQUITY", "DAILY P&L", "TOTAL P&L", "CASH", "BUYING POWER"):
            self.assertIn(label, text, f"missing label {label}")
        # And the numbers from the snapshot appear (formatted with
        # thousands separators and two decimals).
        self.assertIn("$100,500.00", text)
        self.assertIn("$87,904.97", text)
        self.assertIn("$351,619.88", text)
        # P&L is signed.
        self.assertIn("+$500.00", text)

    def test_panel_renders_unavailable_state(self):
        snap = {"ok": False, "error": "alpaca api down"}
        panel = layout.alpaca_account_top_panel(snap)
        text = self._flatten_text(panel)
        self.assertIn("X QUANT X", text)
        self.assertIn("unavailable", text)
        self.assertIn("alpaca api down", text)

    def test_panel_handles_missing_optional_fields(self):
        snap = {"ok": True, "equity": 100000.0, "buying_power": 200000.0}
        panel = layout.alpaca_account_top_panel(snap)
        text = self._flatten_text(panel)
        self.assertIn("$100,000.00", text)
        # Missing cash/PNL render as "n/a" instead of crashing.
        self.assertIn("n/a", text)

    def test_build_control_room_accepts_alpaca_snapshot(self):
        """Smoke test: build_control_room no longer crashes when the
        new ``alpaca_snapshot`` kwarg is provided."""
        from investment_agent.dashboard import layout
        snap = {
            "ok": True,
            "equity": 100000.0,
            "cash": 50000.0,
            "buying_power": 200000.0,
            "last_equity": 100000.0,
            "portfolio_value": 100000.0,
            "daily_pnl": 0.0,
            "total_pnl": 0.0,
            "snapshot_at": "2026-09-01T15:00:00+00:00",
        }
        session = {"state": "STOPPED", "stage": "paper", "cycle_index": 0}
        kwargs = dict(
            account={"ok": True, "buying_power": 200000.0, "equity": 100000.0, "cash": 50000.0},
            alpaca_snapshot=snap,
            session_status=session,
            equity_curve=[],
            cycle={},
            charges=[],
            agents=[],
            kalman={},
            regime_card={},
            circuit={"level": "NORMAL", "label": "LEVEL 0", "verdict": "ALLOW"},
            gates=[],
            llm_rows=[],
            trade_outcome_rows=[],
            reputation_rows=[],
            waterfall=[],
            options_rows=[],
            positions_payload={"ok": True, "positions": []},
            exposure_pct=0.0,
            audit_event=None,
            equity_fig=charts.build_equity_curve_chart([], source="strategy"),
            soc_fig=charts.build_seven_state_soc_chart([]),
            agents_fig=charts.build_seven_agents_table([]),
            kalman_fig=charts.build_kalman_chart({}),
            regime_fig=charts.build_regime_panel_chart({},),
            llm_fig=charts.build_llm_providers_table([]),
            options_fig=charts.build_options_table([]),
            outcome_fig=charts.build_trade_outcome_table([]),
            reputation_fig=charts.build_reputation_table([]),
            waterfall_fig=charts.build_decision_waterfall([]),
        )
        room = layout.build_control_room(**kwargs)
        text = self._flatten_text(room)
        # The new top panel's title appears in the assembled layout.
        self.assertIn("X QUANT X — ALPACA ACCOUNT", text)
        self.assertIn("X QUANT X — SESSION CONTROL", text)
        # And the old "ALPACA PAPER ACCOUNT" section is gone (the new
        # top panel is the single source of truth for the account).
        self.assertNotIn("ALPACA PAPER ACCOUNT", text)


class TestSessionControlPanel(unittest.TestCase):
    """Tests for the new 'X QUANT X — SESSION CONTROL' top panel."""

    def _flatten_text(self, component) -> str:
        from dash import html
        if isinstance(component, str):
            return component
        if isinstance(component, dict):
            return " ".join(str(v) for v in component.values())
        if hasattr(component, "children"):
            kids = component.children
            if isinstance(kids, (list, tuple)):
                return " ".join(self._flatten_text(k) for k in kids)
            return self._flatten_text(kids)
        return ""

    def test_panel_renders_idle_stopped_state(self):
        from investment_agent.dashboard import layout
        panel = layout.session_control_panel({})
        text = self._flatten_text(panel)
        self.assertIn("X QUANT X — SESSION CONTROL", text)
        self.assertIn("STOPPED", text)
        self.assertIn("START PAPER TRADING", text)
        self.assertIn("STOP", text)
        self.assertIn("EMERGENCY STOP", text)

    def test_panel_renders_running_state_with_kpis(self):
        from investment_agent.dashboard import layout
        status = {
            "state": "RUNNING",
            "stage": "paper",
            "cycle_index": 7,
            "last_decision_summary": "BUY AAPL (equity)",
            "last_cycle_at": "2026-09-01T16:34:38.207724+00:00",
            "next_cycle_at": "2026-09-01T16:39:38.207724+00:00",
            "started_at": "2026-09-01T16:00:00+00:00",
            "decision_interval_seconds": 300,
            "symbol_universe": ["AAPL", "SPY"],
            "total_decisions": 12,
            "total_orders": 3,
            "total_closed": 1,
            "pid": 12345,
        }
        panel = layout.session_control_panel(status)
        text = self._flatten_text(panel)
        self.assertIn("RUNNING", text)
        self.assertIn("PAPER", text)
        self.assertIn("#7", text)
        self.assertIn("BUY AAPL", text)
        self.assertIn("12", text)  # total_decisions
        self.assertIn("300s", text)
        self.assertIn("PID: 12345", text)

    def test_panel_renders_error_state(self):
        from investment_agent.dashboard import layout
        status = {
            "state": "ERROR",
            "last_error": "Alpaca account unreachable: 401",
        }
        panel = layout.session_control_panel(status)
        text = self._flatten_text(panel)
        self.assertIn("ERROR", text)
        self.assertIn("Alpaca account unreachable", text)

    def test_panel_renders_emergency_halt_state(self):
        from investment_agent.dashboard import layout
        status = {"state": "EMERGENCY_HALT"}
        panel = layout.session_control_panel(status)
        text = self._flatten_text(panel)
        self.assertIn("EMERGENCY_HALT", text)


class TestSessionStatusLoader(unittest.TestCase):
    """Tests for the dashboard's session-status file reader / command writer."""

    def test_get_session_status_returns_empty_when_missing(self, tmpdir=None):
        import os
        from investment_agent.dashboard import data_loader
        path = os.path.join(os.environ.get("TEMP", "/tmp"),
                            "xqx_no_such_session_status_xx.json")
        if os.path.exists(path):
            os.unlink(path)
        self.assertEqual(data_loader.get_session_status(path), {})

    def test_get_session_status_reads_existing(self):
        import json
        import os
        import tempfile
        from investment_agent.dashboard import data_loader
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "s.json")
            with open(path, "w") as f:
                json.dump({"state": "RUNNING", "cycle_index": 3}, f)
            self.assertEqual(
                data_loader.get_session_status(path),
                {"state": "RUNNING", "cycle_index": 3},
            )

    def test_get_session_status_handles_corrupt_file(self):
        import os
        import tempfile
        from investment_agent.dashboard import data_loader
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "s.json")
            with open(path, "w") as f:
                f.write("not json")
            self.assertEqual(data_loader.get_session_status(path), {})

    def test_write_session_command_writes_file(self):
        import json
        import os
        import tempfile
        from investment_agent.dashboard import data_loader
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "c.json")
            ok = data_loader.write_session_command("start", params={"a": 1},
                                                  command_path=path)
            self.assertTrue(ok)
            with open(path) as f:
                cmd = json.load(f)
            self.assertEqual(cmd["action"], "start")
            self.assertEqual(cmd["params"], {"a": 1})
            self.assertIn("requested_at", cmd)

    def test_write_session_command_rejects_invalid_action(self):
        from investment_agent.dashboard import data_loader
        self.assertFalse(data_loader.write_session_command("nuke"))


class TestOrderAndSessionCallback(unittest.TestCase):
    @patch("investment_agent.execution.execution.place_order")
    def test_handle_order_actions_buy(self, mock_place):
        from investment_agent.dashboard import app as dash_app
        mock_result = MagicMock(submitted=True, status="filled", order_id="ord-99")
        mock_place.return_value = mock_result

        with patch("investment_agent.dashboard.app.ctx") as mock_ctx:
            mock_ctx.triggered_id = "manual-buy-btn"
            msg = dash_app._handle_order_and_session_actions(
                0, 0, 0, 1, 0, 0, "AAPL", 2, 150.0
            )
            self.assertIn("BUY SUCCESS", msg)
            self.assertIn("ord-99", msg)
            mock_place.assert_called_once_with("AAPL", "buy", 2.0, 150.0)

    @patch("investment_agent.execution.execution.cancel_all_orders_and_close_positions")
    def test_handle_emergency_sell_flatten(self, mock_flatten):
        from investment_agent.dashboard import app as dash_app
        mock_flatten.return_value = {"cancelled_orders": 2, "closed_positions": 1}

        with patch("investment_agent.dashboard.app.ctx") as mock_ctx:
            mock_ctx.triggered_id = "manual-emergency-sell-btn"
            msg = dash_app._handle_order_and_session_actions(
                0, 0, 0, 0, 0, 1, "AAPL", 1, 150.0
            )
            self.assertIn("EMERGENCY SELL", msg)
            self.assertIn("closed 1 open positions", msg)


if __name__ == "__main__":
    unittest.main()
