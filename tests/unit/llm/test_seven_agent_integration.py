"""End-to-end LLM -> 7-agent integration test.

WHAT
====
This test proves the *full* pipeline: a market snapshot, an LLM
provider, the seven specialist agents (economic, financial, fiscal,
portfolio, fundamental, market, sector), and the orchestrator's
deterministic pipeline (ensemble -> regime -> Kalman -> capital gate
-> risk -> decision). Every agent's output must satisfy the canonical
eight-channel contract:

    s, c, u, d, p+, p-, Δt, r

WHY
====
Two non-trivial bugs in the past:
  * an LLM agent returned a custom JSON shape that the deterministic
    pipeline ignored
  * one of the seven agents was silently skipped because its
    adapter raised during build

This test prevents both regressions by:
  * using a single mock LLM responder that emits the exact contract
    the adapter expects
  * asserting the orchestrator received exactly seven AgentOutput
    objects, one per default role
  * asserting each output's channels are within the documented ranges
  * asserting the orchestrator's final CycleResult populates
    agent_outputs_full with all seven agents
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

# Imports are done inside the tests to avoid the circular import
# between investment_agent.llm.named and investment_agent.agents.specialist.
# Calling them at module scope triggers the cycle.


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _strict_contract_responder(agent_id: str):
    """Mock LLM that emits a strict contract with agent-specific bias.

    The responder receives (system_prompt, user_prompt) and we
    extract the agent identity from the system prompt (each role's
    system_prompt mentions the role name).
    """
    biases = {
        "agent_economic": 0.4,
        "agent_financial": -0.2,
        "agent_fiscal": 0.1,
        "agent_portfolio": -0.3,
        "agent_fundamental": 0.5,
        "agent_market": 0.2,
        "agent_sector": 0.0,
    }

    def responder(system, prompt):
        # The system prompt contains "*Economic State Specialist*" etc.
        bias = 0.0
        if system:
            keywords = {
                "agent_economic": "Economic State",
                "agent_financial": "Financial State",
                "agent_fiscal": "Fiscal State",
                "agent_portfolio": "Portfolio State",
                "agent_fundamental": "Fundamental State",
                "agent_market": "Market Microstructure",
                "agent_sector": "Sector",
            }
            for aid, kw in keywords.items():
                if kw in system:
                    bias = biases[aid]
                    break
        s = bias
        return json.dumps({
            "signal": s,
            "confidence": 0.8,
            "uncertainty": 0.2,
            "doubt": 0.1,
            "p_plus": 0.5 + s / 2.0,
            "p_minus": 0.5 - s / 2.0,
            "delta_t": 1.0,
            "noise": 0.5,
        })

    return responder


def _noisy_responder(agent_id: str):
    """Mock LLM that emits garbage so the adapter's robustness is tested."""
    del agent_id
    def responder(system, prompt):
        return "I am the LLM. I will not give a structured answer. ✨"
    return responder


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSevenAgentLLMContract(unittest.TestCase):
    """Commit 1 of the LLM-in-the-loop roadmap: prove the contract."""

    def _imports(self):
        # Import specialist first, then llm.base -- the inverse order
        # triggers the llm.__init__ -> llm.named -> agents.specialist
        # circular import.
        from investment_agent.agents.specialist import (
            DEFAULT_ROLES,
            build_specialist_agents,
            AgentContext,
        )
        from investment_agent.llm.base import MockLLMProvider
        return DEFAULT_ROLES, build_specialist_agents, AgentContext, MockLLMProvider

    def test_default_roles_cover_seven_canonical_agents(self):
        DEFAULT_ROLES, _, _, _ = self._imports()
        ids = [r.agent_id for r in DEFAULT_ROLES]
        self.assertEqual(len(ids), 7)
        expected = {
            "agent_economic", "agent_financial", "agent_fiscal",
            "agent_portfolio", "agent_fundamental", "agent_market",
            "agent_sector",
        }
        self.assertEqual(set(ids), expected)

    def test_build_seven_agents_from_one_provider(self):
        DEFAULT_ROLES, build_specialist_agents, _, MockLLMProvider = self._imports()
        provider = MockLLMProvider(responder=_strict_contract_responder("any"))
        agents = build_specialist_agents(provider)
        self.assertEqual(set(agents.keys()),
                         {r.agent_id for r in DEFAULT_ROLES})

    def test_each_agent_returns_eight_channel_contract(self):
        DEFAULT_ROLES, build_specialist_agents, AgentContext, MockLLMProvider = self._imports()
        provider = MockLLMProvider(responder=_strict_contract_responder("any"))
        agents = build_specialist_agents(provider)
        # Run every agent against a uniform context.
        ctx = AgentContext(
            symbol="AAPL",
            regime="R01",
            regime_probabilities={"R01": 0.6, "R02": 0.4},
            features={"rsi": 0.5, "atr": 1.2},
            ensemble_signal=0.0,
            disagreement=0.0,
        )
        outputs = {aid: agent.run(ctx)[0] for aid, agent in agents.items()}
        self.assertEqual(len(outputs), 7)
        for aid, out in outputs.items():
            # All eight channels populated
            for ch in ("s", "c", "u", "d", "p_plus", "p_minus", "delta_t", "r"):
                self.assertIsNotNone(getattr(out, ch), f"{aid} missing {ch}")
            # Signal in [-1, +1]
            self.assertGreaterEqual(out.s, -1.0)
            self.assertLessEqual(out.s, 1.0)
            # Confidence in [0, 1]
            self.assertGreaterEqual(out.c, 0.0)
            self.assertLessEqual(out.c, 1.0)
            # p+ and p- sum to 1
            self.assertAlmostEqual(out.p_plus + out.p_minus, 1.0, places=4)

    def test_seven_agents_emit_distinct_signals(self):
        DEFAULT_ROLES, build_specialist_agents, AgentContext, MockLLMProvider = self._imports()
        provider = MockLLMProvider(responder=_strict_contract_responder("any"))
        agents = build_specialist_agents(provider)
        ctx = AgentContext(
            symbol="AAPL", regime="R01",
            regime_probabilities={"R01": 1.0},
            features={},
        )
        outputs = {aid: agent.run(ctx)[0] for aid, agent in agents.items()}
        signals = {aid: out.s for aid, out in outputs.items()}
        self.assertEqual(len(set(signals.values())), 7)


class TestSevenAgentLLMThroughOrchestrator(unittest.TestCase):
    """End-to-end: LLM -> 7 agents -> orchestrator -> CycleResult."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _imports(self):
        from investment_agent.agents.specialist import (
            DEFAULT_ROLES,
            build_specialist_agents,
            AgentContext,
            run_agents,
        )
        from investment_agent.capital.capital_gate import SevenStateVector
        from investment_agent.llm.base import MockLLMProvider
        from investment_agent.orchestrator import XQuantXOrchestrator
        return DEFAULT_ROLES, build_specialist_agents, AgentContext, run_agents, SevenStateVector, MockLLMProvider, XQuantXOrchestrator

    def test_llm_drives_orchestrator_with_seven_outputs(self):
        (DEFAULT_ROLES, build_specialist_agents, AgentContext, run_agents,
         SevenStateVector, MockLLMProvider, XQuantXOrchestrator) = self._imports()
        provider = MockLLMProvider(responder=_strict_contract_responder("any"))
        agents = build_specialist_agents(provider)

        orch = XQuantXOrchestrator(
            agent_ids=[r.agent_id for r in DEFAULT_ROLES],
            symbol="AAPL",
            use_hmm=False,
            enable_trading=False,
            memory_file=str(Path(self.tmpdir) / "mem.json"),
        )
        orch._specialist_agents = agents

        ctx = AgentContext(
            symbol="AAPL", regime="R01",
            regime_probabilities={"R01": 0.8, "R02": 0.2},
            features={"rsi": 0.55, "atr": 1.1, "vix": 0.18},
        )
        agent_outputs_map = run_agents(agents, ctx)
        self.assertEqual(len(agent_outputs_map), 7)
        agent_outputs = [agent_outputs_map[r.agent_id] for r in DEFAULT_ROLES]

        states = SevenStateVector(
            economic=1.0, financial=1.0, fiscal=1.0,
            portfolio=1.0, fundamental=1.0, market=1.0, sector=1.0,
        )
        result = orch.run_cycle(
            prices=[100.0 + i * 0.05 for i in range(45)],
            volumes=[1000.0] * 45,
            agent_outputs=agent_outputs,
            states=states,
            portfolio_context={
                "position_pct": 0.0, "gross_leverage": 0.0, "entropy": 0.1,
                "drawdown_pct": 0.0, "execution_timeout_seconds": 5.0,
                "sector_exposure_pct": 0.0, "is_new_long": True, "regime": "R01",
                "available_liquidity": 100000.0,
            },
        )
        aof = result.experience.agent_outputs_full
        self.assertIsNotNone(aof)
        self.assertEqual(set(aof.keys()),
                         {r.agent_id for r in DEFAULT_ROLES})
        for aid, row in aof.items():
            for ch in ("signal", "confidence", "uncertainty", "doubt",
                       "p_plus", "p_minus", "delta_t", "noise", "weight",
                       "reputation_alpha", "reputation_beta"):
                self.assertIn(ch, row, f"{aid} missing {ch}")

    def test_noisy_responder_does_not_break_pipeline(self):
        """If the LLM returns garbage, the adapter must produce a
        well-formed (zero-signal, low-confidence) fallback. The
        orchestrator must still complete its cycle."""
        (DEFAULT_ROLES, build_specialist_agents, AgentContext, run_agents,
         SevenStateVector, MockLLMProvider, XQuantXOrchestrator) = self._imports()
        provider = MockLLMProvider(responder=_noisy_responder("any"))
        agents = build_specialist_agents(provider)

        orch = XQuantXOrchestrator(
            agent_ids=[r.agent_id for r in DEFAULT_ROLES],
            symbol="AAPL",
            use_hmm=False,
            enable_trading=False,
            memory_file=str(Path(self.tmpdir) / "mem.json"),
        )
        orch._specialist_agents = agents

        ctx = AgentContext(
            symbol="AAPL", regime="R01",
            regime_probabilities={"R01": 1.0},
            features={},
        )
        agent_outputs_map = run_agents(agents, ctx)
        self.assertEqual(len(agent_outputs_map), 7)
        for out in agent_outputs_map.values():
            self.assertEqual(out.s, 0.0)
            self.assertEqual(out.p_plus + out.p_minus, 1.0)
            self.assertGreaterEqual(out.c, 0.0)
            self.assertLessEqual(out.c, 1.0)

        agent_outputs = [agent_outputs_map[r.agent_id] for r in DEFAULT_ROLES]
        states = SevenStateVector(
            economic=1.0, financial=1.0, fiscal=1.0,
            portfolio=1.0, fundamental=1.0, market=1.0, sector=1.0,
        )
        result = orch.run_cycle(
            prices=[100.0 + i * 0.05 for i in range(45)],
            volumes=[1000.0] * 45,
            agent_outputs=agent_outputs,
            states=states,
            portfolio_context={
                "position_pct": 0.0, "gross_leverage": 0.0, "entropy": 0.1,
                "drawdown_pct": 0.0, "execution_timeout_seconds": 5.0,
                "sector_exposure_pct": 0.0, "is_new_long": True, "regime": "R01",
                "available_liquidity": 100000.0,
            },
        )
        self.assertIsNotNone(result.experience)


if __name__ == "__main__":
    unittest.main()
