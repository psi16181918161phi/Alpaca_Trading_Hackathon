"""Tests for the new 'control-room' dashboard data_loader + chart builders."""

import json
import os
import sys
import tempfile
import unittest
from typing import List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))

from investment_agent.dashboard import charts, data_loader


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
        k = data_loader.get_kalman_card(self.cycle)
        expected = 0.81 * (1.0 - 0.73) + 0.45 * 0.73
        self.assertAlmostEqual(k["posterior_estimate"], expected, places=6)
        self.assertEqual(k["kalman_gain"], 0.73)

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
        rows = data_loader.get_reputation_snapshot(self.history)
        self.assertEqual(len(rows), 7)
        for r in rows:
            self.assertEqual(r["alpha"], 1.0)
            self.assertEqual(r["beta"], 1.0)
            self.assertEqual(r["weight"], 0.5)

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


if __name__ == "__main__":
    unittest.main()
