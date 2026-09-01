"""Tests for the production LLM layer: orchestrator, snapshot, named specialists."""

import json
import os
import sys
import tempfile
import unittest
from typing import List
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))

from investment_agent.llm import (
    AgentLLMAdapter,
    DEEPHERMES_FUNDAMENTALS_ROLE,
    DEEPHERMES_REASONING_ROLE,
    FailureKind,
    FeatherlessOrchestrator,
    FINANCE_QLORA_ROLE,
    LLMResponse,
    MockLLMProvider,
    NAMED_ROLES,
    NamedSpecialist,
    PreScreenResult,
    ProviderSpec,
    SpecialistOutput,
    UsageLog,
    build_named_specialists,
    build_provider_map_from_orchestrator,
    build_snapshot,
    classify_failure,
    extract_json_object,
    load_provider_specs,
    pre_screen,
    run_named_specialists,
)
from investment_agent.signals.ensemble_signal import AgentOutput


# ---------------------------------------------------------------------------
# Snapshot
# ---------------------------------------------------------------------------

class TestBuildSnapshot(unittest.TestCase):
    def test_basic_snapshot_fields(self):
        prices = [100 + 0.1 * i for i in range(20)]
        volumes = [1000 + 5 * i for i in range(20)]
        snap = build_snapshot(
            symbol="AAPL",
            prices=prices,
            volumes=volumes,
            regime="R01",
            ensemble_signal=0.42,
            risk_flags=[],
        )
        for key in (
            "symbol", "regime", "price", "return_1d", "volatility",
            "volume_change", "ensemble_signal", "risk_flags",
        ):
            self.assertIn(key, snap)
        self.assertEqual(snap["symbol"], "AAPL")
        self.assertEqual(snap["regime"], "R01")
        self.assertAlmostEqual(snap["price"], 101.9, places=4)
        self.assertGreater(snap["return_1d"], 0.0)
        self.assertEqual(snap["risk_flags"], [])

    def test_invalid_regime_dropped(self):
        snap = build_snapshot(symbol="X", prices=[1, 2, 3], regime="R99")
        self.assertIsNone(snap["regime"])

    def test_top_regimes_keeps_three(self):
        snap = build_snapshot(
            symbol="X", prices=[1, 2, 3],
            regime_probabilities={"R01": 0.5, "R02": 0.3, "R03": 0.15, "R04": 0.05},
        )
        self.assertEqual(len(snap["top_regimes"]), 3)
        self.assertEqual(snap["top_regimes"][0]["regime"], "R01")

    def test_compact_memory(self):
        from investment_agent.memory.trade_memory import (
            TradeExperience, SimilarExperience, TradeMemory,
        )
        mem = TradeMemory(memory_file=os.path.join(tempfile.mkdtemp(), "m.json"))
        for i in range(5):
            mem.log_experience(TradeExperience(
                decision_id=f"d-{i}", timestamp=__import__("datetime").datetime.now(),
                symbol="AAPL", regime="R01", regime_probabilities={"R01": 0.8},
                agent_signals={}, ensemble_signal=0.5, disagreement=0.2,
                effective_confidence=0.8, kalman_gain=0.5, kalman_price=100.0,
                kalman_trend=0.01, capital_gate_verdict="ALLOW", effective_cap=0.5,
                state_charges={}, position_action="BUY", quantity=1.0,
                confidence=0.8, expected_outcome="", realized_outcome="x",
                pnl=10.0 * i, lesson="",
            ))
        sims = mem.find_similar(mem._experiences[0], top_k=3)
        snap = build_snapshot("AAPL", [100, 101, 102], relevant_experiences=sims)
        self.assertEqual(len(snap["relevant_experiences"]), 3)
        for entry in snap["relevant_experiences"]:
            self.assertIn("decision_id", entry)
            self.assertIn("similarity", entry)


class TestPreScreen(unittest.TestCase):
    def test_first_call_passes(self):
        result = pre_screen("AAPL", [100, 100.5, 100.6])
        self.assertTrue(result.should_call_llm)
        self.assertEqual(result.reason, "first call")

    def test_risk_flag_passes(self):
        prev = build_snapshot("AAPL", [100, 100.5])
        result = pre_screen("AAPL", [100, 100.5], risk_flags=["LIQ-001"], previous_snapshot=prev)
        self.assertTrue(result.should_call_llm)
        self.assertIn("LIQ-001", result.reason)

    def test_large_return_passes(self):
        prev = build_snapshot("AAPL", [100, 100.5])
        result = pre_screen("AAPL", [100, 100.5, 105], previous_snapshot=prev)
        self.assertTrue(result.should_call_llm)
        self.assertIn("return_1d", result.reason)

    def test_small_change_skipped(self):
        prev = build_snapshot("AAPL", [100, 100.01, 100.02])
        result = pre_screen(
            "AAPL", [100, 100.01, 100.03],
            previous_snapshot=prev,
        )
        self.assertFalse(result.should_call_llm)


# ---------------------------------------------------------------------------
# Featherless orchestrator
# ---------------------------------------------------------------------------

class _ScriptedFeatherless:
    """Fake provider that returns preset responses in order."""

    def __init__(self, responses: List, model: str = "fake") -> None:
        self._responses = list(responses)
        self._model = model
        self.calls: List[dict] = []

    @property
    def model_id(self) -> str:
        return self._model

    def complete(self, prompt, **kwargs):
        self.calls.append({"prompt": prompt, "kwargs": kwargs})
        if not self._responses:
            raise RuntimeError("no scripted responses left")
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        if isinstance(item, str):
            return LLMResponse(
                text=item, model=self._model,
                latency_ms=1.0, prompt_tokens=10, completion_tokens=5,
            )
        return item


class TestFeatherlessOrchestrator(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.usage_file = os.path.join(self.tmp, "llm_usage.jsonl")

    def test_single_provider_success(self):
        provider = _ScriptedFeatherless([json.dumps({"signal": 0.3, "confidence": 0.7})])
        orch = FeatherlessOrchestrator(
            [ProviderSpec("p1", "k1", "model", 0.15, 700, "default")],
            usage_log=UsageLog(log_file=self.usage_file),
        )
        orch._providers["p1"] = provider
        response = orch.complete("hi")
        self.assertIn('"signal": 0.3', response.text)
        self.assertEqual(len(provider.calls), 1)

    def test_failover_to_reserve(self):
        active = _ScriptedFeatherless([RuntimeError("primary down")])
        reserve = _ScriptedFeatherless([json.dumps({"signal": 0.1, "confidence": 0.5})])
        orch = FeatherlessOrchestrator(
            [
                ProviderSpec("p1", "k1", "m1", 0.15, 700, "default", is_reserve=False),
                ProviderSpec("p2", "k2", "m2", 0.15, 700, "failover", is_reserve=True),
            ],
            usage_log=UsageLog(log_file=self.usage_file),
        )
        orch._providers["p1"] = active
        orch._providers["p2"] = reserve
        response = orch.complete("hi")
        self.assertIn('"signal": 0.1', response.text)
        self.assertEqual(len(reserve.calls), 1)

    def test_all_providers_fail_raises(self):
        active = _ScriptedFeatherless([RuntimeError("nope")])
        orch = FeatherlessOrchestrator(
            [ProviderSpec("p1", "k1", "m1", 0.15, 700, "default")],
            usage_log=UsageLog(log_file=self.usage_file),
        )
        orch._providers["p1"] = active
        with self.assertRaises(RuntimeError):
            orch.complete("hi")

    def test_reserve_not_used_when_disabled(self):
        active = _ScriptedFeatherless([RuntimeError("nope")])
        reserve = _ScriptedFeatherless([json.dumps({"signal": 0.0})])
        orch = FeatherlessOrchestrator(
            [
                ProviderSpec("p1", "k1", "m1", 0.15, 700, "default", is_reserve=False),
                ProviderSpec("p2", "k2", "m2", 0.15, 700, "failover", is_reserve=True),
            ],
            use_reserve_on_failure=False,
            usage_log=UsageLog(log_file=self.usage_file),
        )
        orch._providers["p1"] = active
        orch._providers["p2"] = reserve
        with self.assertRaises(RuntimeError):
            orch.complete("hi")
        self.assertEqual(len(reserve.calls), 0)

    def test_usage_log_records_success_and_failure(self):
        active = _ScriptedFeatherless(
            [RuntimeError("boom"), json.dumps({"signal": 0.0, "confidence": 0.5})]
        )
        orch = FeatherlessOrchestrator(
            [ProviderSpec("p1", "k1", "m1", 0.15, 700, "default")],
            retries_per_provider=2,
            usage_log=UsageLog(log_file=self.usage_file),
        )
        orch._providers["p1"] = active
        response = orch.complete("hi")
        self.assertIn('"signal": 0.0', response.text)
        log = UsageLog(log_file=self.usage_file)
        self.assertGreater(log.total_tokens(), 0)


# ---------------------------------------------------------------------------
# Named specialists
# ---------------------------------------------------------------------------

class TestNamedSpecialists(unittest.TestCase):
    def setUp(self):
        # Three independent providers, one per specialist. This is the
        # production binding the audit required.
        self.reasoning = MockLLMProvider(model_id="deephermes-3-8b")
        self.fundamentals = MockLLMProvider(model_id="deephermes-fundamentals")
        self.finance_qlora = MockLLMProvider(model_id="finance-qlora")
        self.provider_map = {
            "agent_deephermes_reasoning": self.reasoning,
            "agent_deephermes_fundamentals": self.fundamentals,
            "agent_finance_qlora": self.finance_qlora,
        }

    def test_three_roles_defined(self):
        ids = {r.agent_id for r in NAMED_ROLES}
        self.assertEqual(
            ids,
            {
                "agent_deephermes_reasoning",
                "agent_deephermes_fundamentals",
                "agent_finance_qlora",
            },
        )
        self.assertNotIn("agent_deephermes_execution", ids)
        self.assertNotIn("agent_finance_llama", ids)
        self.assertNotIn("agent_qwen_trading", ids)

    def test_each_role_has_capped_signal(self):
        for role in NAMED_ROLES:
            self.assertIn("Cap absolute signal", role.system_prompt)

    def test_no_rationale_in_noise_field(self):
        """noise is a quantitative channel; the prompt must not ask the LLM
        to stuff rationale into it.
        """
        for role in NAMED_ROLES:
            self.assertNotIn("'noise' field", role.user_template)
            self.assertNotIn("noise field", role.user_template.lower())

    def test_build_named_specialists(self):
        specialists = build_named_specialists(self.provider_map)
        self.assertEqual(len(specialists), 3)
        for name, sp in specialists.items():
            self.assertIsInstance(sp, NamedSpecialist)
            self.assertGreater(sp.max_tokens, 0)
            self.assertLessEqual(sp.max_tokens, 256)  # reduced budget

    def test_specialist_bound_to_its_provider(self):
        """Each specialist must use the provider mapped to it, not a shared one."""
        specialists = build_named_specialists(self.provider_map)
        self.assertIs(specialists["agent_deephermes_reasoning"].provider, self.reasoning)
        self.assertIs(specialists["agent_deephermes_fundamentals"].provider, self.fundamentals)
        self.assertIs(specialists["agent_finance_qlora"].provider, self.finance_qlora)

    def test_missing_provider_raises(self):
        bad_map = {"agent_deephermes_reasoning": self.reasoning}
        with self.assertRaises(ValueError):
            build_named_specialists(bad_map)

    def test_run_named_specialists_routes_to_correct_providers(self):
        # Each provider records which specialist it was called for.
        calls: List[str] = []
        def make_recording(name):
            def responder(sys, prompt):
                calls.append(name)
                return json.dumps({
                    "signal": 0.4, "confidence": 0.8, "uncertainty": 0.2,
                    "doubt": 0.1, "p_plus": 0.6, "p_minus": 0.3,
                    "delta_t": 1.0, "noise": 0.3,
                })
            return responder
        self.reasoning._responder = make_recording("reasoning")
        self.fundamentals._responder = make_recording("fundamentals")
        self.finance_qlora._responder = make_recording("finance_qlora")

        specialists = build_named_specialists(self.provider_map)
        snap = build_snapshot("AAPL", [100, 100.5, 101], regime="R01")
        outputs = run_named_specialists(specialists, snap)

        self.assertEqual(set(outputs.keys()),
                         {"agent_deephermes_reasoning", "agent_deephermes_fundamentals", "agent_finance_qlora"})
        for aid, out in outputs.items():
            self.assertIsInstance(out, AgentOutput)
            self.assertEqual(out.agent_id, aid)
        # Each specialist called exactly one provider.
        self.assertEqual(calls, ["reasoning", "fundamentals", "finance_qlora"])
        # And the per-provider call counts prove no cross-talk.
        self.assertEqual(self.reasoning.call_count, 1)
        self.assertEqual(self.fundamentals.call_count, 1)
        self.assertEqual(self.finance_qlora.call_count, 1)

    def test_specialist_run_returns_specialist_output(self):
        """SpecialistOutput carries AgentOutput + rationale + raw_text."""
        specialists = build_named_specialists(self.provider_map)
        snap = build_snapshot("AAPL", [100, 100.5, 101], regime="R01")
        result = specialists["agent_deephermes_reasoning"].run(snap)
        self.assertIsInstance(result, SpecialistOutput)
        self.assertIsInstance(result.output, AgentOutput)
        self.assertEqual(result.output.agent_id, "agent_deephermes_reasoning")
        self.assertIsInstance(result.raw_text, str)

    def test_prompt_contains_compact_snapshot(self):
        captured: Dict[str, str] = {}
        def responder(sys, prompt):
            return json.dumps({
                "signal": 0.2, "confidence": 0.6, "uncertainty": 0.3,
                "doubt": 0.2, "p_plus": 0.5, "p_minus": 0.4,
                "delta_t": 1.0, "noise": 0.4,
            })
        self.reasoning._responder = lambda s, p: (captured.setdefault("r", p), responder(s, p))[1]
        self.fundamentals._responder = lambda s, p: (captured.setdefault("f", p), responder(s, p))[1]
        self.finance_qlora._responder = lambda s, p: (captured.setdefault("q", p), responder(s, p))[1]
        specialists = build_named_specialists(self.provider_map)
        snap = build_snapshot("AAPL", [100, 100.5, 101], regime="R01")
        run_named_specialists(specialists, snap)
        for prompt in captured.values():
            self.assertIn("Snapshot", prompt)
            self.assertIn("AAPL", prompt)
            self.assertIn("return_1d", prompt)

    def test_provider_failure_yields_fallback(self):
        class BoomProvider:
            @property
            def model_id(self):
                return "boom"
            def complete(self, *args, **kwargs):
                raise RuntimeError("simulated outage")
        bad_map = {
            "agent_deephermes_reasoning": BoomProvider(),
            "agent_deephermes_fundamentals": BoomProvider(),
            "agent_finance_qlora": BoomProvider(),
        }
        specialists = build_named_specialists(bad_map)
        snap = build_snapshot("AAPL", [100, 100.5, 101], regime="R01")
        outputs = run_named_specialists(specialists, snap)
        for out in outputs.values():
            self.assertEqual(out.s, 0.0)
            self.assertEqual(out.c, 0.25)


# ---------------------------------------------------------------------------
# Failure classification
# ---------------------------------------------------------------------------

class TestFailureClassification(unittest.TestCase):
    def test_http_400_is_skip(self):
        import requests as _r
        response = _r.Response()
        response.status_code = 400
        exc = _r.HTTPError("bad request", response=response)
        self.assertEqual(classify_failure(exc), FailureKind.SKIP)

    def test_capacity_exhausted_body_is_backoff(self):
        body = {"error": {"code": "capacity_exhausted", "message": "temporarily at capacity"}}
        self.assertEqual(
            classify_failure(RuntimeError("x"), response_body=body),
            FailureKind.BACKOFF,
        )

    def test_model_not_found_body_is_skip(self):
        body = {"error": {"code": "model_not_found", "message": "model not found"}}
        self.assertEqual(
            classify_failure(RuntimeError("x"), response_body=body),
            FailureKind.SKIP,
        )

    def test_timeout_is_backoff(self):
        import requests as _r
        exc = _r.exceptions.Timeout("read timeout")
        self.assertEqual(classify_failure(exc), FailureKind.BACKOFF)

    def test_connection_error_is_network(self):
        import requests as _r
        exc = _r.exceptions.ConnectionError("dns")
        self.assertEqual(classify_failure(exc), FailureKind.NETWORK)


# ---------------------------------------------------------------------------
# build_provider_map_from_orchestrator
# ---------------------------------------------------------------------------

class TestBuildProviderMapFromOrchestrator(unittest.TestCase):
    def test_default_mapping(self):
        orch = FeatherlessOrchestrator([
            ProviderSpec("deephermes", "k1", "m1", 0.15, 700, "default"),
            ProviderSpec("fundamentals", "k2", "m2", 0.15, 700, "default"),
            ProviderSpec("finance_qlora", "k3", "m3", 0.15, 700, "default"),
        ])
        provider_map = build_provider_map_from_orchestrator(orch)
        self.assertEqual(
            set(provider_map.keys()),
            {"agent_deephermes_reasoning", "agent_deephermes_fundamentals", "agent_finance_qlora"},
        )
        self.assertIs(provider_map["agent_deephermes_reasoning"], orch._providers["deephermes"])
        self.assertIs(provider_map["agent_deephermes_fundamentals"], orch._providers["fundamentals"])
        self.assertIs(provider_map["agent_finance_qlora"], orch._providers["finance_qlora"])

    def test_missing_provider_id_raises(self):
        orch = FeatherlessOrchestrator([
            ProviderSpec("deephermes", "k1", "m1", 0.15, 700, "default"),
        ])
        with self.assertRaises(ValueError):
            build_provider_map_from_orchestrator(orch)


# ---------------------------------------------------------------------------
# load_provider_specs
# ---------------------------------------------------------------------------

class TestLoadProviderSpecs(unittest.TestCase):
    def test_loads_from_file_with_env_override(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({
                "providers": {
                    "deephermes": {
                        "api_key_env": "MY_KEY",
                        "model": "m1", "temperature": 0.2, "max_tokens": 800,
                        "role": "reasoning",
                    },
                }
            }, f)
            path = f.name
        try:
            os.environ["MY_KEY"] = "env-key"
            specs = load_provider_specs(keys_file=path)
            self.assertEqual(len(specs), 1)
            self.assertEqual(specs[0].api_key, "env-key")
            self.assertEqual(specs[0].model, "m1")
            self.assertFalse(specs[0].is_reserve)
        finally:
            os.environ.pop("MY_KEY", None)
            os.remove(path)

    def test_missing_file_falls_back_to_env(self):
        with tempfile.TemporaryDirectory() as d:
            missing = os.path.join(d, "missing.json")
            os.environ["FEATHERLESS_API_KEY"] = "fallback-key"
            specs = load_provider_specs(keys_file=missing)
            self.assertEqual(len(specs), 1)
            self.assertEqual(specs[0].api_key, "fallback-key")
            self.assertEqual(specs[0].provider_id, "default")
            del os.environ["FEATHERLESS_API_KEY"]


# ---------------------------------------------------------------------------
# Orchestrator-bound NamedSpecialist
# ---------------------------------------------------------------------------
#
# These tests prove the failover chain the production wiring depends on:
#
#     SpecialistAgent.run()
#         -> orchestrator.complete(provider_id=preferred)
#         -> [preferred fails]
#         -> [other active providers tried]
#         -> [all active fail]
#         -> [reserve provider]
#         -> LLMResponse
#         -> AgentOutput
#
# The reserve is only on the hot path when the specialist is bound to
# the orchestrator with its preferred provider_id. Single-provider
# mode (the legacy wiring) must keep behaving as before.

def _valid_json_for(signal: float = 0.3, confidence: float = 0.7) -> str:
    return json.dumps({
        "signal": signal,
        "confidence": confidence,
        "uncertainty": 0.2,
        "doubt": 0.1,
        "p_plus": 0.6,
        "p_minus": 0.3,
        "delta_t": 1.0,
        "noise": 0.3,
    })


class _RecordingProvider:
    """Fake provider that records which provider_id it served."""

    def __init__(self, provider_id: str, responses, model: str = "fake") -> None:
        self._provider_id = provider_id
        self._responses = list(responses)
        self._model_id = model
        self.calls: List[Dict[str, Any]] = []

    @property
    def model_id(self) -> str:
        return self._model_id

    def complete(self, prompt, **kwargs):
        self.calls.append({"prompt": prompt, "kwargs": kwargs, "provider_id": self._provider_id})
        if not self._responses:
            raise RuntimeError(f"{self._provider_id}: no scripted responses left")
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        if isinstance(item, str):
            return LLMResponse(
                text=item, model=self._model_id,
                latency_ms=1.0, prompt_tokens=10, completion_tokens=5,
            )
        return item


def _build_orchestrator_with_providers(specs, providers):
    """Helper: build a FeatherlessOrchestrator and inject scripted providers."""
    orch = FeatherlessOrchestrator(
        specs,
        usage_log=UsageLog(log_file=os.path.join(tempfile.mkdtemp(), "u.jsonl")),
        retries_per_provider=1,
        backoff_s=0.0,
    )
    for pid, prov in providers.items():
        orch._providers[pid] = prov
    return orch


class TestNamedSpecialistOrchestratorBound(unittest.TestCase):
    """End-to-end tests: specialist -> orchestrator -> reserve."""

    def setUp(self):
        self.snapshot = build_snapshot("AAPL", [100, 100.5, 101], regime="R01")

    # ---- mode plumbing ----

    def test_requires_provider_or_orchestrator(self):
        with self.assertRaises(ValueError):
            NamedSpecialist(role=DEEPHERMES_REASONING_ROLE)

    def test_orchestrator_mode_requires_provider_id(self):
        from investment_agent.llm import FeatherlessOrchestrator, ProviderSpec
        orch = FeatherlessOrchestrator([
            ProviderSpec("deephermes", "k", "m", 0.1, 100, "default"),
        ])
        with self.assertRaises(ValueError):
            NamedSpecialist(role=DEEPHERMES_REASONING_ROLE, orchestrator=orch)

    def test_build_named_specialists_orchestrator_mode_default_mapping(self):
        deephermes = _RecordingProvider("deephermes", [_valid_json_for(0.1)])
        fundamentals = _RecordingProvider("fundamentals", [_valid_json_for(0.2)])
        finance_qlora = _RecordingProvider("finance_qlora", [_valid_json_for(0.3)])
        orch = _build_orchestrator_with_providers(
            [
                ProviderSpec("deephermes", "k1", "m1", 0.15, 700, "default"),
                ProviderSpec("fundamentals", "k2", "m2", 0.15, 700, "default"),
                ProviderSpec("finance_qlora", "k3", "m3", 0.15, 700, "default"),
            ],
            {
                "deephermes": deephermes,
                "fundamentals": fundamentals,
                "finance_qlora": finance_qlora,
            },
        )
        specialists = build_named_specialists(orch)
        self.assertEqual(
            set(specialists.keys()),
            {
                "agent_deephermes_reasoning",
                "agent_deephermes_fundamentals",
                "agent_finance_qlora",
            },
        )
        for sp in specialists.values():
            self.assertTrue(sp.is_orchestrator_bound)
            self.assertIs(sp.orchestrator, orch)

    def test_build_named_specialists_orchestrator_mode_provider_id_map_required(self):
        deephermes = _RecordingProvider("deephermes", [_valid_json_for()])
        orch = _build_orchestrator_with_providers(
            [ProviderSpec("deephermes", "k", "m", 0.15, 700, "default")],
            {"deephermes": deephermes},
        )
        with self.assertRaises(ValueError):
            # finance_qlora + fundamentals not in the map -> ValueError
            build_named_specialists(
                orch,
                provider_id_map={"agent_deephermes_reasoning": "deephermes"},
            )

    # ---- the headline behaviour ----

    def test_finance_qlora_fails_reserve_attempted_returns_valid_output(self):
        """Headline test.

        Production path:
            finance_qlora provider fails
              -> orchestrator detects failure
              -> reserve is attempted
              -> valid AgentOutput returned
        Starting from NamedSpecialist.run(), not from orchestrator.complete()
        in isolation.

        The orchestrator's existing provider-ordering policy walks all
        active providers before reaching the reserve, so to prove the
        reserve is on the actual hot path we make every active provider
        fail and only the reserve succeed. The specialist still calls
        ``orchestrator.complete(provider_id="finance_qlora", ...)`` and
        the orchestrator's own policy decides to walk through the
        remaining actives and finally the reserve.
        """
        deephermes = _RecordingProvider("deephermes", [RuntimeError("d down")])
        fundamentals = _RecordingProvider("fundamentals", [RuntimeError("f down")])
        finance_qlora = _RecordingProvider(
            "finance_qlora", [RuntimeError("finance_qlora down")],
        )
        reserve = _RecordingProvider("reserve", [_valid_json_for(0.55, 0.88)])

        orch = _build_orchestrator_with_providers(
            [
                ProviderSpec("deephermes", "k1", "m1", 0.15, 700, "default"),
                ProviderSpec("fundamentals", "k2", "m2", 0.15, 700, "default"),
                ProviderSpec("finance_qlora", "k3", "m3", 0.15, 700, "default"),
                ProviderSpec("reserve", "k4", "m4", 0.15, 700, "failover", is_reserve=True),
            ],
            {
                "deephermes": deephermes,
                "fundamentals": fundamentals,
                "finance_qlora": finance_qlora,
                "reserve": reserve,
            },
        )

        specialists = build_named_specialists(orch)
        sp_fq = specialists["agent_finance_qlora"]
        result = sp_fq.run(self.snapshot)

        # The finance_qlora provider was attempted (preferred first).
        self.assertEqual(len(finance_qlora.calls), 1)
        # The other actives were also attempted (orchestrator's existing
        # ordering walks all actives before reserve).
        self.assertEqual(len(deephermes.calls), 1)
        self.assertEqual(len(fundamentals.calls), 1)
        # The reserve provider was actually called.
        self.assertEqual(len(reserve.calls), 1)
        # The reserve returned a valid JSON, not a zero-signal fallback.
        self.assertIsInstance(result, SpecialistOutput)
        self.assertIsInstance(result.output, AgentOutput)
        self.assertEqual(result.output.agent_id, "agent_finance_qlora")
        # signal 0.55 came from the reserve's canned response.
        self.assertAlmostEqual(result.output.s, 0.55, places=4)
        self.assertAlmostEqual(result.output.c, 0.88, places=4)
        # And the raw_text reflects the reserve response, not a fallback.
        self.assertIn("0.55", result.raw_text)

    def test_preferred_provider_succeeds_reserve_not_called(self):
        deephermes = _RecordingProvider("deephermes", [_valid_json_for(0.1)])
        fundamentals = _RecordingProvider("fundamentals", [_valid_json_for(0.2)])
        finance_qlora = _RecordingProvider("finance_qlora", [_valid_json_for(0.3)])
        reserve = _RecordingProvider("reserve", [_valid_json_for(0.99)])

        orch = _build_orchestrator_with_providers(
            [
                ProviderSpec("deephermes", "k1", "m1", 0.15, 700, "default"),
                ProviderSpec("fundamentals", "k2", "m2", 0.15, 700, "default"),
                ProviderSpec("finance_qlora", "k3", "m3", 0.15, 700, "default"),
                ProviderSpec("reserve", "k4", "m4", 0.15, 700, "failover", is_reserve=True),
            ],
            {
                "deephermes": deephermes,
                "fundamentals": fundamentals,
                "finance_qlora": finance_qlora,
                "reserve": reserve,
            },
        )
        specialists = build_named_specialists(orch)
        out = run_named_specialists(specialists, self.snapshot)

        # Each preferred provider answered directly.
        self.assertAlmostEqual(out["agent_deephermes_reasoning"].s, 0.1, places=4)
        self.assertAlmostEqual(out["agent_deephermes_fundamentals"].s, 0.2, places=4)
        self.assertAlmostEqual(out["agent_finance_qlora"].s, 0.3, places=4)
        # Reserve was never touched.
        self.assertEqual(len(reserve.calls), 0)

    def test_all_active_providers_fail_reserve_attempted(self):
        deephermes = _RecordingProvider("deephermes", [RuntimeError("d down")])
        fundamentals = _RecordingProvider("fundamentals", [RuntimeError("f down")])
        finance_qlora = _RecordingProvider("finance_qlora", [RuntimeError("q down")])
        reserve = _RecordingProvider("reserve", [
            _valid_json_for(0.77), _valid_json_for(0.77), _valid_json_for(0.77),
        ])

        orch = _build_orchestrator_with_providers(
            [
                ProviderSpec("deephermes", "k1", "m1", 0.15, 700, "default"),
                ProviderSpec("fundamentals", "k2", "m2", 0.15, 700, "default"),
                ProviderSpec("finance_qlora", "k3", "m3", 0.15, 700, "default"),
                ProviderSpec("reserve", "k4", "m4", 0.15, 700, "failover", is_reserve=True),
            ],
            {
                "deephermes": deephermes,
                "fundamentals": fundamentals,
                "finance_qlora": finance_qlora,
                "reserve": reserve,
            },
        )
        specialists = build_named_specialists(orch)
        out = run_named_specialists(specialists, self.snapshot)

        # All three preferred providers were attempted by their own
        # specialists, and the orchestrator's existing ordering walked
        # through the remaining actives for each call, so each active
        # was hit 3 times in total.
        self.assertEqual(len(deephermes.calls), 3)
        self.assertEqual(len(fundamentals.calls), 3)
        self.assertEqual(len(finance_qlora.calls), 3)
        # Reserve was tried once per specialist (3 total) and answered
        # all three with a real signal.
        self.assertEqual(len(reserve.calls), 3)
        # Each specialist's output is a real reserve signal, not zero.
        for aid in ("agent_deephermes_reasoning",
                    "agent_deephermes_fundamentals",
                    "agent_finance_qlora"):
            self.assertAlmostEqual(out[aid].s, 0.77, places=4)
            self.assertEqual(out[aid].agent_id, aid)

    def test_all_providers_including_reserve_fail_returns_zero_signal(self):
        deephermes = _RecordingProvider("deephermes", [RuntimeError("d down")])
        fundamentals = _RecordingProvider("fundamentals", [RuntimeError("f down")])
        finance_qlora = _RecordingProvider("finance_qlora", [RuntimeError("q down")])
        reserve = _RecordingProvider("reserve", [RuntimeError("r down")])

        orch = _build_orchestrator_with_providers(
            [
                ProviderSpec("deephermes", "k1", "m1", 0.15, 700, "default"),
                ProviderSpec("fundamentals", "k2", "m2", 0.15, 700, "default"),
                ProviderSpec("finance_qlora", "k3", "m3", 0.15, 700, "default"),
                ProviderSpec("reserve", "k4", "m4", 0.15, 700, "failover", is_reserve=True),
            ],
            {
                "deephermes": deephermes,
                "fundamentals": fundamentals,
                "finance_qlora": finance_qlora,
                "reserve": reserve,
            },
        )
        specialists = build_named_specialists(orch)
        out = run_named_specialists(specialists, self.snapshot)

        # run_named_specialists catches the orchestrator's RuntimeError and
        # returns the deterministic zero-signal fallback per specialist.
        for aid, o in out.items():
            self.assertEqual(o.s, 0.0)
            self.assertEqual(o.c, 0.25)
            self.assertEqual(o.u, 0.75)
            self.assertEqual(o.agent_id, aid)

    def test_each_specialist_uses_its_own_provider_id(self):
        """No cross-talk: each specialist hits its own preferred provider."""
        deephermes = _RecordingProvider("deephermes", [_valid_json_for(0.11)])
        fundamentals = _RecordingProvider("fundamentals", [_valid_json_for(0.22)])
        finance_qlora = _RecordingProvider("finance_qlora", [_valid_json_for(0.33)])
        reserve = _RecordingProvider("reserve", [_valid_json_for(0.99)])

        orch = _build_orchestrator_with_providers(
            [
                ProviderSpec("deephermes", "k1", "m1", 0.15, 700, "default"),
                ProviderSpec("fundamentals", "k2", "m2", 0.15, 700, "default"),
                ProviderSpec("finance_qlora", "k3", "m3", 0.15, 700, "default"),
                ProviderSpec("reserve", "k4", "m4", 0.15, 700, "failover", is_reserve=True),
            ],
            {
                "deephermes": deephermes,
                "fundamentals": fundamentals,
                "finance_qlora": finance_qlora,
                "reserve": reserve,
            },
        )
        specialists = build_named_specialists(orch)
        run_named_specialists(specialists, self.snapshot)

        # Each preferred provider was called exactly once.
        self.assertEqual(len(deephermes.calls), 1)
        self.assertEqual(len(fundamentals.calls), 1)
        self.assertEqual(len(finance_qlora.calls), 1)
        # Reserve was not touched.
        self.assertEqual(len(reserve.calls), 0)
        # Cross-check: each call carried the correct provider_id.
        for prov in (deephermes, fundamentals, finance_qlora):
            self.assertEqual(prov.calls[0]["provider_id"], prov._provider_id)

    def test_one_specialist_failure_does_not_corrupt_others(self):
        """finance_qlora fails -> reserve answers; the other two are untouched."""
        deephermes = _RecordingProvider("deephermes", [_valid_json_for(0.1)])
        fundamentals = _RecordingProvider("fundamentals", [_valid_json_for(0.2)])
        finance_qlora = _RecordingProvider(
            "finance_qlora", [RuntimeError("q down")],
        )
        reserve = _RecordingProvider("reserve", [_valid_json_for(0.5, 0.9)])

        orch = _build_orchestrator_with_providers(
            [
                ProviderSpec("deephermes", "k1", "m1", 0.15, 700, "default"),
                ProviderSpec("fundamentals", "k2", "m2", 0.15, 700, "default"),
                ProviderSpec("finance_qlora", "k3", "m3", 0.15, 700, "default"),
                ProviderSpec("reserve", "k4", "m4", 0.15, 700, "failover", is_reserve=True),
            ],
            {
                "deephermes": deephermes,
                "fundamentals": fundamentals,
                "finance_qlora": finance_qlora,
                "reserve": reserve,
            },
        )
        specialists = build_named_specialists(orch)
        out = run_named_specialists(specialists, self.snapshot)

        # The two healthy specialists are unaffected.
        self.assertAlmostEqual(out["agent_deephermes_reasoning"].s, 0.1, places=4)
        self.assertAlmostEqual(out["agent_deephermes_fundamentals"].s, 0.2, places=4)
        # finance_qlora was answered by the reserve, not the zero-signal fallback.
        self.assertAlmostEqual(out["agent_finance_qlora"].s, 0.5, places=4)
        self.assertAlmostEqual(out["agent_finance_qlora"].c, 0.9, places=4)
        # Sanity: each output still carries the correct agent_id.
        for aid, o in out.items():
            self.assertEqual(o.agent_id, aid)

    def test_existing_eight_channel_output_contract_unchanged(self):
        """The orchestrator-bound path returns the same AgentOutput contract."""
        deephermes = _RecordingProvider("deephermes", [_valid_json_for(0.42, 0.81)])
        fundamentals = _RecordingProvider("fundamentals", [_valid_json_for(0.43, 0.82)])
        finance_qlora = _RecordingProvider("finance_qlora", [_valid_json_for(0.44, 0.83)])
        reserve = _RecordingProvider("reserve", [_valid_json_for(0.99)])

        orch = _build_orchestrator_with_providers(
            [
                ProviderSpec("deephermes", "k1", "m1", 0.15, 700, "default"),
                ProviderSpec("fundamentals", "k2", "m2", 0.15, 700, "default"),
                ProviderSpec("finance_qlora", "k3", "m3", 0.15, 700, "default"),
                ProviderSpec("reserve", "k4", "m4", 0.15, 700, "failover", is_reserve=True),
            ],
            {
                "deephermes": deephermes,
                "fundamentals": fundamentals,
                "finance_qlora": finance_qlora,
                "reserve": reserve,
            },
        )
        specialists = build_named_specialists(orch)
        out = run_named_specialists(specialists, self.snapshot)

        for aid, o in out.items():
            self.assertIsInstance(o, AgentOutput)
            self.assertEqual(o.agent_id, aid)
            for channel in ("s", "c", "u", "d", "p_plus", "p_minus", "delta_t", "r"):
                self.assertTrue(hasattr(o, channel))
            self.assertGreater(o.c, 0.0)
            self.assertLessEqual(o.c, 1.0)

    def test_specialist_routes_via_orchestrator_provider_id(self):
        """Confirm the orchestrator receives the specialist's preferred provider_id."""
        deephermes = _RecordingProvider("deephermes", [_valid_json_for(0.1)])
        fundamentals = _RecordingProvider("fundamentals", [_valid_json_for(0.2)])
        finance_qlora = _RecordingProvider("finance_qlora", [_valid_json_for(0.3)])
        orch = _build_orchestrator_with_providers(
            [
                ProviderSpec("deephermes", "k1", "m1", 0.15, 700, "default"),
                ProviderSpec("fundamentals", "k2", "m2", 0.15, 700, "default"),
                ProviderSpec("finance_qlora", "k3", "m3", 0.15, 700, "default"),
            ],
            {
                "deephermes": deephermes,
                "fundamentals": fundamentals,
                "finance_qlora": finance_qlora,
            },
        )
        specialists = build_named_specialists(orch)
        run_named_specialists(specialists, self.snapshot)

        # The orchestrator's _provider_order used the specialist's
        # preferred id. Each active provider was called exactly once.
        self.assertEqual(deephermes.calls[0]["provider_id"], "deephermes")
        self.assertEqual(fundamentals.calls[0]["provider_id"], "fundamentals")
        self.assertEqual(finance_qlora.calls[0]["provider_id"], "finance_qlora")

    def test_single_provider_mode_unchanged(self):
        """Legacy single-provider wiring must still work and still degrade to fallback."""
        class Boom:
            @property
            def model_id(self): return "boom"
            def complete(self, *a, **k): raise RuntimeError("legacy outage")

        provider_map = {
            "agent_deephermes_reasoning": Boom(),
            "agent_deephermes_fundamentals": Boom(),
            "agent_finance_qlora": Boom(),
        }
        specialists = build_named_specialists(provider_map)
        out = run_named_specialists(specialists, self.snapshot)
        for o in out.values():
            self.assertEqual(o.s, 0.0)
            self.assertEqual(o.c, 0.25)


if __name__ == "__main__":
    unittest.main()
