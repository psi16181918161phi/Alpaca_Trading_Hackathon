"""Tests for the LLM abstraction, adapter, and specialist agents."""

import json
import os
import sys
import unittest
from typing import List
from unittest.mock import MagicMock, patch

# Make src/ importable for this test module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))

from investment_agent.llm import (
    AgentLLMAdapter,
    FeatherlessProvider,
    LLMResponse,
    MockLLMProvider,
    extract_json_object,
)
from investment_agent.agents.specialist import (
    AgentContext,
    AgentRole,
    DEFAULT_ROLES,
    SpecialistAgent,
    build_specialist_agents,
    run_agents,
)
from investment_agent.signals.ensemble_signal import AgentOutput


class TestExtractJsonObject(unittest.TestCase):
    def test_fenced_json_block(self):
        text = 'Here is the result:\n```json\n{"signal": 0.5}\n```\nDone.'
        obj = extract_json_object(text)
        self.assertEqual(obj, {"signal": 0.5})

    def test_bare_json_object(self):
        text = 'Some prose {"signal": -0.2, "confidence": 0.7} more text'
        obj = extract_json_object(text)
        self.assertEqual(obj, {"signal": -0.2, "confidence": 0.7})

    def test_nested_braces(self):
        text = '```json\n{"a": {"b": 1}, "c": [1, 2]}\n```'
        obj = extract_json_object(text)
        self.assertEqual(obj, {"a": {"b": 1}, "c": [1, 2]})

    def test_invalid_returns_none(self):
        self.assertIsNone(extract_json_object("no json here"))
        self.assertIsNone(extract_json_object(""))
        self.assertIsNone(extract_json_object("{not valid json"))

    def test_first_object_wins(self):
        text = '```json\n{"signal": 0.1}\n``` and ```json\n{"signal": 0.9}\n```'
        obj = extract_json_object(text)
        self.assertEqual(obj, {"signal": 0.1})


class TestMockLLMProvider(unittest.TestCase):
    def test_default_response_is_valid_agent_output(self):
        provider = MockLLMProvider()
        response = provider.complete("anything")
        self.assertIsInstance(response, LLMResponse)
        self.assertGreater(len(response.text), 0)
        obj = extract_json_object(response.text)
        self.assertIsNotNone(obj)
        for k in ("signal", "confidence", "uncertainty", "doubt",
                  "p_plus", "p_minus", "delta_t", "noise"):
            self.assertIn(k, obj)

    def test_custom_responder(self):
        provider = MockLLMProvider(
            responder=lambda sys, prompt: json.dumps({"signal": 0.42}),
        )
        response = provider.complete("anything")
        self.assertEqual(extract_json_object(response.text)["signal"], 0.42)

    def test_call_count(self):
        provider = MockLLMProvider()
        provider.complete("a")
        provider.complete("b")
        self.assertEqual(provider.call_count, 2)


class TestAgentLLMAdapter(unittest.TestCase):
    def setUp(self):
        self.provider = MockLLMProvider(
            responder=lambda sys, prompt: json.dumps({
                "signal": 0.3,
                "confidence": 0.8,
                "uncertainty": 0.2,
                "doubt": 0.1,
                "p_plus": 0.7,
                "p_minus": 0.2,
                "delta_t": 2.0,
                "noise": 0.5,
            })
        )
        self.adapter = AgentLLMAdapter(provider=self.provider, agent_id="agent_test")

    def test_valid_json_produces_agent_output(self):
        output, response = self.adapter.call("test prompt")
        self.assertIsInstance(output, AgentOutput)
        self.assertEqual(output.s, 0.3)
        self.assertEqual(output.c, 0.8)
        self.assertEqual(output.u, 0.2)
        self.assertEqual(output.d, 0.1)
        self.assertEqual(output.p_plus, 0.7)
        self.assertEqual(output.p_minus, 0.2)
        self.assertEqual(output.delta_t, 2.0)
        self.assertEqual(output.r, 0.5)
        self.assertEqual(output.agent_id, "agent_test")
        self.assertIsInstance(response, LLMResponse)

    def test_invalid_json_falls_back(self):
        bad_provider = MockLLMProvider(responder=lambda s, p: "not json at all")
        adapter = AgentLLMAdapter(provider=bad_provider, agent_id="fallback")
        output, _ = adapter.call("prompt")
        # Should return fallback values
        self.assertEqual(output.agent_id, "fallback")
        self.assertEqual(output.s, 0.0)
        self.assertLessEqual(output.c, 0.5)

    def test_missing_fields_use_defaults(self):
        partial = MockLLMProvider(responder=lambda s, p: json.dumps({"signal": 0.5}))
        adapter = AgentLLMAdapter(provider=partial, agent_id="partial")
        output, _ = adapter.call("prompt")
        self.assertEqual(output.s, 0.5)
        self.assertGreater(output.c, 0.0)
        self.assertGreater(output.delta_t, 0.0)

    def test_nan_signal_replaced(self):
        nan_provider = MockLLMProvider(
            responder=lambda s, p: json.dumps({"signal": float("nan"), "confidence": 0.5})
        )
        adapter = AgentLLMAdapter(provider=nan_provider, agent_id="nan_test")
        output, _ = adapter.call("prompt")
        self.assertFalse(output.s != output.s)  # not NaN

    def test_signal_clipped_to_unit_interval(self):
        huge = MockLLMProvider(responder=lambda s, p: json.dumps({"signal": 99.0}))
        adapter = AgentLLMAdapter(provider=huge, agent_id="clip")
        output, _ = adapter.call("prompt")
        self.assertEqual(output.s, 1.0)


class TestFeatherlessProvider(unittest.TestCase):
    def test_missing_api_key_raises(self):
        provider = FeatherlessProvider(api_key=None)
        provider._api_key = None
        with self.assertRaises(RuntimeError):
            provider.complete("hello")

    def test_sends_request_to_chat_completions(self):
        fake_session = MagicMock()
        fake_response = MagicMock()
        fake_response.json.return_value = {
            "choices": [{"message": {"content": '{"signal": 0.1}'}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }
        fake_response.raise_for_status.return_value = None
        fake_session.post.return_value = fake_response

        provider = FeatherlessProvider(
            api_key="test-key",
            model="Qwen/Qwen2.5-72B-Instruct",
            session_factory=lambda: fake_session,
        )
        response = provider.complete("hello", system="sys")
        self.assertEqual(response.text, '{"signal": 0.1}')
        self.assertEqual(response.model, "Qwen/Qwen2.5-72B-Instruct")
        self.assertEqual(response.prompt_tokens, 10)
        self.assertEqual(response.completion_tokens, 5)

        call = fake_session.post.call_args
        url = call[0][0]
        self.assertIn("/chat/completions", url)
        body = call[1]["json"]
        self.assertEqual(body["model"], "Qwen/Qwen2.5-72B-Instruct")
        self.assertEqual(body["messages"][0]["role"], "system")
        self.assertEqual(body["messages"][1]["content"], "hello")


class TestSpecialistAgents(unittest.TestCase):
    def setUp(self):
        self.provider = MockLLMProvider(
            responder=lambda sys, prompt: json.dumps({
                "signal": 0.5,
                "confidence": 0.7,
                "uncertainty": 0.3,
                "doubt": 0.2,
                "p_plus": 0.6,
                "p_minus": 0.3,
                "delta_t": 1.0,
                "noise": 0.5,
            })
        )
        self.agents = build_specialist_agents(self.provider)

    def test_default_seven_agents_built(self):
        self.assertEqual(len(self.agents), 7)
        for agent_id in (
            "agent_economic", "agent_financial", "agent_fiscal",
            "agent_portfolio", "agent_fundamental", "agent_market",
            "agent_sector",
        ):
            self.assertIn(agent_id, self.agents)
            self.assertIsInstance(self.agents[agent_id], SpecialistAgent)

    def test_all_roles_have_required_fields(self):
        for role in DEFAULT_ROLES:
            self.assertIsInstance(role, AgentRole)
            self.assertTrue(role.agent_id)
            self.assertTrue(role.system_prompt)
            self.assertTrue(role.user_template)
            self.assertIn("Role:", role.user_template)

    def test_run_agents_returns_seven_outputs(self):
        ctx = AgentContext(
            symbol="AAPL",
            regime="R01",
            regime_probabilities={"R01": 0.8, "R02": 0.2},
            features={"rsi": 0.5, "vix": 0.3},
            ensemble_signal=0.4,
            disagreement=0.1,
        )
        outputs = run_agents(self.agents, ctx)
        self.assertEqual(len(outputs), 7)
        for aid, out in outputs.items():
            self.assertIsInstance(out, AgentOutput)
            self.assertEqual(out.agent_id, aid)

    def test_run_agents_resilient_to_provider_failure(self):
        class FailingProvider:
            @property
            def model_id(self):
                return "failing"
            def complete(self, *args, **kwargs):
                raise RuntimeError("simulated outage")

        agents = build_specialist_agents(FailingProvider())
        ctx = AgentContext(
            symbol="AAPL", regime="R01",
            regime_probabilities={"R01": 1.0},
            features={},
        )
        outputs = run_agents(agents, ctx)
        # Should still return seven outputs, all fallbacks
        self.assertEqual(len(outputs), 7)
        for aid, out in outputs.items():
            self.assertIsInstance(out, AgentOutput)
            self.assertEqual(out.s, 0.0)

    def test_prompt_contains_memory(self):
        captured_prompts: List[str] = []
        def responder(sys, prompt):
            captured_prompts.append(prompt)
            return json.dumps({"signal": 0.1, "confidence": 0.5})

        provider = MockLLMProvider(responder=responder)
        agents = build_specialist_agents(provider)
        ctx = AgentContext(
            symbol="AAPL", regime="R01",
            regime_probabilities={"R01": 1.0},
            features={"rsi": 0.5},
        )
        run_agents(agents, ctx)
        self.assertEqual(len(captured_prompts), 7)
        for p in captured_prompts:
            self.assertIn("Symbol: AAPL", p)
            self.assertIn("Current regime: R01", p)
            self.assertIn("Memory", p)


class TestSpecialistAgentEndToEnd(unittest.TestCase):
    """Verify LLM-backed agents can drive the full pipeline."""

    def setUp(self):
        # Canned responder: positive signal
        self.responder = lambda sys, prompt: json.dumps({
            "signal": 0.6,
            "confidence": 0.9,
            "uncertainty": 0.1,
            "doubt": 0.05,
            "p_plus": 0.7,
            "p_minus": 0.2,
            "delta_t": 1.0,
            "noise": 0.3,
        })
        self.provider = MockLLMProvider(responder=self.responder)
        self.agents = build_specialist_agents(self.provider)

    def test_agents_to_ensemble_to_capital_gate(self):
        from investment_agent.signals.ensemble_signal import compute_ensemble_aggregate
        from investment_agent.capital.capital_gate import evaluate, SevenStateVector
        from investment_agent.filters.kalman_filter import KalmanState

        ctx = AgentContext(
            symbol="AAPL", regime="R01",
            regime_probabilities={"R01": 0.8, "R02": 0.2},
            features={"rsi": 0.6, "vix": 0.2},
        )
        outputs = run_agents(self.agents, ctx)
        agent_list = list(outputs.values())

        weights = {aid: 1.0 for aid in outputs.keys()}
        ensemble = compute_ensemble_aggregate(agent_list, weights)

        states = SevenStateVector(
            economic=1.0, financial=1.0, fiscal=1.0,
            portfolio=1.0, fundamental=1.0, market=1.0, sector=1.0,
        )
        result = evaluate(
            kalman_state=KalmanState(
                estimated_price=100.0, trend=0.01, uncertainty=1.0,
                trend_uncertainty=0.1, price_variance=1.0,
                trend_variance=0.01, innovation=0.0, kalman_gain_price=0.5,
            ),
            states=states,
            portfolio_context={
                "position_pct": 0.05, "gross_leverage": 0.5, "entropy": 0.1,
                "drawdown_pct": 0.01, "execution_timeout_seconds": 5.0,
                "sector_exposure_pct": 0.1, "is_new_long": False, "regime": "R01",
                "available_liquidity": 100000.0,
            },
            agents=agent_list,
            agent_weights=weights,
        )
        self.assertIsNotNone(result.verdict)
        self.assertGreater(ensemble.ensemble_signal, 0.0)


if __name__ == "__main__":
    unittest.main()
