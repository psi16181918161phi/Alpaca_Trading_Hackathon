"""
Unit tests for Seven-State Capital Gate module (capital_gate.py).

Validates all 23 core requirements of the Seven-State Capital Gate,
plus adversarial regression tests for boolean truthiness, NaN/Inf fail-open
vulnerabilities, canonical regime validation, boundary precision, and verdict priorities.
"""

import math
import os
import tempfile
import unittest
from pathlib import Path
from typing import Dict, Any

from investment_agent.filters.kalman_filter import KalmanState
from investment_agent.signals.ensemble_signal import AgentOutput, compute_ensemble_signal, compute_disagreement
from investment_agent.capital.capital_gate import (
    RiskVerdict,
    SevenStateVector,
    CapitalGateResult,
    STATE_THRESHOLDS,
    _load_risk_thresholds_from_path,
    _resolve_risk_thresholds_path,
    compute_individual_gating,
    compute_gating_factor,
    evaluate,
    _parse_bool,
    _parse_float,
)


def full_charge_state(**overrides):
    values = {
        "economic": 1.0,
        "financial": 1.0,
        "fiscal": 1.0,
        "portfolio": 1.0,
        "fundamental": 1.0,
        "market": 1.0,
        "sector": 1.0,
    }
    values.update(overrides)
    return SevenStateVector(**values)


class TestSevenStateVectorValidation(unittest.TestCase):
    """Test validation rules for SevenStateVector instantiation."""

    def test_default_construction(self):
        v = SevenStateVector.full_charge()
        self.assertEqual(v.economic, 1.0)
        self.assertEqual(v.financial, 1.0)
        self.assertEqual(v.fiscal, 1.0)
        self.assertEqual(v.portfolio, 1.0)
        self.assertEqual(v.fundamental, 1.0)
        self.assertEqual(v.market, 1.0)
        self.assertEqual(v.sector, 1.0)

    def test_valid_custom_values(self):
        v = SevenStateVector(
            economic=0.5,
            financial=0.8,
            fiscal=0.1,
            portfolio=0.7,
            fundamental=0.3,
            market=0.9,
            sector=0.4
        )
        self.assertEqual(v.economic, 0.5)
        self.assertEqual(v.financial, 0.8)

    def test_reject_boolean_values(self):
        with self.assertRaises(TypeError):
            SevenStateVector(
                economic=True,
                financial=1.0,
                fiscal=1.0,
                portfolio=1.0,
                fundamental=1.0,
                market=1.0,
                sector=1.0,
            )
        with self.assertRaises(TypeError):
            SevenStateVector(
                economic=1.0,
                financial=1.0,
                fiscal=1.0,
                portfolio=False,
                fundamental=1.0,
                market=1.0,
                sector=1.0,
            )

    def test_reject_non_numeric_values(self):
        with self.assertRaises(TypeError):
            SevenStateVector(
                economic=1.0,
                financial="0.5",  # type: ignore
                fiscal=1.0,
                portfolio=1.0,
                fundamental=1.0,
                market=1.0,
                sector=1.0,
            )
        with self.assertRaises(TypeError):
            SevenStateVector(
                economic=1.0,
                financial=1.0,
                fiscal=1.0,
                portfolio=1.0,
                fundamental=1.0,
                market=[0.5],  # type: ignore
                sector=1.0,
            )

    def test_reject_nan(self):
        with self.assertRaises(ValueError):
            SevenStateVector(
                economic=float("nan"),
                financial=1.0,
                fiscal=1.0,
                portfolio=1.0,
                fundamental=1.0,
                market=1.0,
                sector=1.0,
            )

    def test_reject_infinity(self):
        with self.assertRaises(ValueError):
            SevenStateVector(
                economic=1.0,
                financial=1.0,
                fiscal=1.0,
                portfolio=1.0,
                fundamental=float("inf"),
                market=1.0,
                sector=1.0,
            )
        with self.assertRaises(ValueError):
            SevenStateVector(
                economic=1.0,
                financial=1.0,
                fiscal=1.0,
                portfolio=1.0,
                fundamental=1.0,
                market=1.0,
                sector=float("-inf"),
            )

    def test_reject_negative_values(self):
        with self.assertRaises(ValueError):
            SevenStateVector(
                economic=1.0,
                financial=1.0,
                fiscal=-0.01,
                portfolio=1.0,
                fundamental=1.0,
                market=1.0,
                sector=1.0,
            )

    def test_reject_values_greater_than_one(self):
        with self.assertRaises(ValueError):
            SevenStateVector(
                economic=1.0,
                financial=1.0,
                fiscal=1.0,
                portfolio=1.0001,
                fundamental=1.0,
                market=1.0,
                sector=1.0,
            )

    def test_config_threshold_loader_falls_back_to_defaults(self):
        """Without a config file present, the fallback must preserve the canonical state thresholds."""
        thresholds = _load_risk_thresholds_from_path(None)
        self.assertEqual(thresholds["economic"]["minimum"], 0.15)
        self.assertEqual(thresholds["economic"]["full"], 0.70)
        self.assertEqual(thresholds["portfolio"]["minimum"], 0.20)
        self.assertEqual(thresholds["portfolio"]["full"], 0.70)

    def test_config_threshold_loader_reads_override_values(self):
        """If a config file exists, it should override the canonical defaults without changing architecture."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "risk_rules.toml"
            path.write_text(
                """
[state_thresholds]
[economic]
minimum = 0.11
full = 0.66
[portfolio]
minimum = 0.21
full = 0.71
""".strip(),
                encoding="utf-8",
            )
            thresholds = _load_risk_thresholds_from_path(path)
            self.assertAlmostEqual(thresholds["economic"]["minimum"], 0.11)
            self.assertAlmostEqual(thresholds["economic"]["full"], 0.66)
            self.assertAlmostEqual(thresholds["portfolio"]["minimum"], 0.21)
            self.assertAlmostEqual(thresholds["portfolio"]["full"], 0.71)

    def test_state_duck_typing_rejects_invalid_values(self):
        """Regression: evaluate() must reject duck-typed state objects with invalid state-of-charge values."""
        bad_values = [
            -0.01,
            1.01,
            100,
            float("nan"),
            float("inf"),
            True,
        ]

        for bad in bad_values:
            attrs = {
                "economic": bad,
                "financial": 1.0,
                "fiscal": 1.0,
                "portfolio": 1.0,
                "fundamental": 1.0,
                "market": 1.0,
                "sector": 1.0,
            }
            FakeStates = type("FakeStates", (), attrs)
            fake = FakeStates()
            with self.assertRaises(Exception, msg=f"Allowed invalid state value: {bad}"):
                evaluate(
                    kalman_state=KalmanState(
                        estimated_price=100.0,
                        trend=0.0,
                        uncertainty=1.0,
                        trend_uncertainty=0.1,
                        price_variance=1.0,
                        trend_variance=0.01,
                        innovation=0.0,
                        kalman_gain_price=0.5,
                    ),
                    states=fake,
                    portfolio_context={
                        "position_pct": 0.05,
                        "gross_leverage": 0.5,
                        "entropy": 0.1,
                        "drawdown_pct": 0.01,
                        "execution_timeout_seconds": 5.0,
                        "is_new_long": False,
                        "sector_exposure_pct": 0.1,
                        "regime": "R01",
                        "available_liquidity": 100000.0,
                    },
                    agents=[
                        AgentOutput(s=1.0, c=0.80, u=0.0, d=0.0, p_plus=0.8, p_minus=0.1, delta_t=1.0, r=0.0, agent_id=f"agent{i}")
                        for i in range(1, 8)
                    ],
                    agent_weights={f"agent{i}": 1.0 for i in range(1, 8)},
                )


class TestIndividualStateGating(unittest.TestCase):
    """Test individual state gating piecewise-linear formula g(S)."""

    def test_at_or_above_full_threshold(self):
        self.assertEqual(compute_individual_gating("economic", 0.70), 1.0)
        self.assertEqual(compute_individual_gating("economic", 1.0), 1.0)

    def test_below_minimum_threshold(self):
        self.assertEqual(compute_individual_gating("economic", 0.14), 0.0)
        self.assertEqual(compute_individual_gating("economic", 0.0), 0.0)

    def test_exactly_at_minimum_threshold(self):
        self.assertEqual(compute_individual_gating("economic", 0.15), 0.0)

    def test_linear_interpolation_between_minimum_and_full(self):
        g_val = compute_individual_gating("financial", 0.50)
        self.assertAlmostEqual(g_val, 0.50, places=6)
        g_val_fiscal = compute_individual_gating("fiscal", 0.375)
        self.assertAlmostEqual(g_val_fiscal, 0.50, places=6)


class TestCompositeGating(unittest.TestCase):
    """Test composite product gating function G(S)."""

    def test_all_states_full_charge(self):
        states = SevenStateVector.full_charge()
        gating, gatings_map = compute_gating_factor(states)
        self.assertEqual(gating, 1.0)
        for val in gatings_map.values():
            self.assertEqual(val, 1.0)

    def test_individual_hard_stop_zero_gate(self):
        mins = {
            "economic": 0.10,
            "financial": 0.15,
            "fiscal": 0.05,
            "portfolio": 0.10,
            "fundamental": 0.10,
            "market": 0.10,
            "sector": 0.10,
        }
        for state_name, low_val in mins.items():
            kwargs = {s: 1.0 for s in mins.keys()}
            kwargs[state_name] = low_val
            v = SevenStateVector(**kwargs)
            gating, gatings_map = compute_gating_factor(v)
            self.assertEqual(gating, 0.0, f"Failed zero-gate for state {state_name}")
            self.assertEqual(gatings_map[state_name], 0.0)

    def test_composite_product_calculation(self):
        kwargs = {}
        for state_name, thresh in STATE_THRESHOLDS.items():
            kwargs[state_name] = (thresh["minimum"] + thresh["full"]) / 2.0
        v = SevenStateVector(**kwargs)
        gating, _ = compute_gating_factor(v)
        expected = 0.5 ** 7
        self.assertAlmostEqual(gating, expected, places=6)

    def test_evaluate_hard_stop_blocks(self):
        for state_name, thresh in STATE_THRESHOLDS.items():
            low_val = thresh["minimum"] - 1e-6
            kwargs = {s: 1.0 for s in STATE_THRESHOLDS.keys()}
            kwargs[state_name] = low_val
            states = SevenStateVector(**kwargs)
            base_agent = AgentOutput(s=1.0, c=0.80, u=0.0, d=0.0, p_plus=0.8, p_minus=0.1, delta_t=1.0, r=0.0, agent_id="agent1")
            dummy_agent_template = lambda i: AgentOutput(
                s=1.0, c=0.80, u=0.0, d=0.0, p_plus=0.8, p_minus=0.1, delta_t=1.0, r=0.0, agent_id=f"agent{i}"
            )
            agents = [base_agent] + [dummy_agent_template(i) for i in range(2, 8)]
            weights = {f"agent{i}": 1.0 for i in range(1, 8)}
            res = evaluate(
                kalman_state=KalmanState(
                    estimated_price=100.0,
                    trend=0.0,
                    uncertainty=1.0,
                    trend_uncertainty=0.1,
                    price_variance=1.0,
                    trend_variance=0.01,
                    innovation=0.0,
                    kalman_gain_price=0.5,
                ),
                states=states,
                portfolio_context={
                    "position_pct": 0.05,
                    "gross_leverage": 0.5,
                    "entropy": 0.1,
                    "drawdown_pct": 0.01,
                    "execution_timeout_seconds": 5.0,
                    "is_new_long": False,
                    "sector_exposure_pct": 0.1,
                    "regime": "R01",
                    "available_liquidity": 100000.0,
                },
                agents=agents,
                agent_weights=weights,
            )
            self.assertEqual(res.verdict, RiskVerdict.BLOCK, f"State {state_name} below minimum should BLOCK")
            self.assertIn(f"GATE-{state_name.upper()}-MIN", res.triggered_rules)


class TestCapitalGateEvaluation(unittest.TestCase):
    """Test evaluate() API, Kalman integration, and risk verdict hierarchy."""

    def setUp(self):
        self.kalman_state = KalmanState(
            estimated_price=100.0,
            trend=0.5,
            uncertainty=1.0,
            trend_uncertainty=0.1,
            price_variance=1.0,
            trend_variance=0.01,
            innovation=0.2,
            kalman_gain_price=0.8
        )
        self.default_context = {
            "position_pct": 0.10,
            "gross_leverage": 0.50,
            "regime": "R01",
            "entropy": 0.20,
            "drawdown_pct": 0.02,
            "execution_timeout_seconds": 5.0,
            "is_new_long": False,
            "sector_exposure_pct": 0.15,
            "available_liquidity": 100000.0,
        }
        base_agent = AgentOutput(s=1.0, c=0.80, u=0.0, d=0.0, p_plus=0.8, p_minus=0.1, delta_t=1.0, r=0.0, agent_id="agent1")
        dummy_agent_template = lambda i: AgentOutput(
            s=1.0, c=0.80, u=0.0, d=0.0, p_plus=0.8, p_minus=0.1, delta_t=1.0, r=0.0, agent_id=f"agent{i}"
        )
        self.default_agents = [base_agent] + [dummy_agent_template(i) for i in range(2, 8)]
        self.default_weights = {f"agent{i}": 1.0 for i in range(1, 8)}

    def _eval(self, states=None, ctx=None, agents=None, weights=None, k_state=None):
        return evaluate(
            kalman_state=k_state if k_state is not None else self.kalman_state,
            states=states if states is not None else SevenStateVector.full_charge(),
            portfolio_context=ctx if ctx is not None else self.default_context,
            agents=agents if agents is not None else self.default_agents,
            agent_weights=weights if weights is not None else self.default_weights,
        )

    def test_allow_verdict(self):
        res = self._eval()
        self.assertEqual(res.verdict, RiskVerdict.ALLOW)
        self.assertEqual(res.gating_factor, 1.0)
        self.assertAlmostEqual(res.effective_cap, 1.0 / 1.29, places=6)
        self.assertEqual(len(res.triggered_rules), 0)

    def test_kalman_gain_and_signal_confidence_modulation(self):
        base_agent_low = AgentOutput(s=1.0, c=0.50, u=0.0, d=0.0, p_plus=0.5, p_minus=0.1, delta_t=1.0, r=0.0, agent_id="agent1")
        dummy_agents = [
            AgentOutput(s=1.0, c=0.50, u=0.0, d=0.0, p_plus=0.5, p_minus=0.1, delta_t=1.0, r=0.0, agent_id=f"agent{i}")
            for i in range(2, 8)
        ]
        agents_low = [base_agent_low] + dummy_agents
        weights_low = {f"agent{i}": 1.0 for i in range(1, 8)}
        res = self._eval(agents=agents_low, weights=weights_low)
        self.assertAlmostEqual(res.effective_cap, 4.0 / 9.0, places=6)

    def test_concentration_cap_breach_blocks(self):
        ctx = dict(self.default_context, position_pct=0.25)
        res = self._eval(ctx=ctx)
        self.assertEqual(res.verdict, RiskVerdict.BLOCK)
        self.assertEqual(res.effective_cap, 0.0)
        self.assertIn("CONC-001", res.triggered_rules)

    def test_zero_effective_cap_on_block_and_flatten(self):
        ctx_block = dict(self.default_context, gross_leverage=1.20)
        res_block = self._eval(ctx=ctx_block)
        self.assertEqual(res_block.verdict, RiskVerdict.BLOCK)
        self.assertEqual(res_block.effective_cap, 0.0)

        ctx_flatten = dict(self.default_context, drawdown_pct=0.20)
        res_flatten = self._eval(ctx=ctx_flatten)
        self.assertEqual(res_flatten.verdict, RiskVerdict.FLATTEN)
        self.assertEqual(res_flatten.effective_cap, 0.0)

    def test_economic_threshold_spec_alignment(self):
        self.assertEqual(STATE_THRESHOLDS["economic"]["minimum"], 0.15)
        self.assertEqual(STATE_THRESHOLDS["economic"]["full"], 0.70)
        self.assertEqual(compute_individual_gating("economic", 0.70), 1.0)

    def test_gross_leverage_limit_breach_blocks(self):
        ctx = dict(self.default_context, gross_leverage=1.10)
        res = self._eval(ctx=ctx)
        self.assertEqual(res.verdict, RiskVerdict.BLOCK)
        self.assertIn("LEV-001", res.triggered_rules)

    def test_regime_r04_r07_new_long_blocks(self):
        ctx_r04 = dict(self.default_context, regime="R04", is_new_long=True)
        res_r04 = self._eval(ctx=ctx_r04)
        self.assertEqual(res_r04.verdict, RiskVerdict.BLOCK)
        self.assertIn("REGM-001", res_r04.triggered_rules)

        base_agent_high = AgentOutput(s=1.0, c=0.90, u=0.0, d=0.0, p_plus=0.9, p_minus=0.0, delta_t=1.0, r=0.1, agent_id="agent1")
        dummy_agents = [
            AgentOutput(s=1.0, c=0.90, u=0.0, d=0.0, p_plus=0.9, p_minus=0.0, delta_t=1.0, r=0.1, agent_id=f"agent{i}")
            for i in range(2, 8)
        ]
        agents_high = [base_agent_high] + dummy_agents
        weights_high = {f"agent{i}": 1.0 for i in range(1, 8)}
        res_r04_high = self._eval(ctx=ctx_r04, agents=agents_high, weights=weights_high)
        self.assertEqual(res_r04_high.verdict, RiskVerdict.ALLOW)

    def test_entropy_triggers(self):
        ctx_high_ent = dict(self.default_context, entropy=0.92)
        res_high_ent = self._eval(ctx=ctx_high_ent)
        self.assertEqual(res_high_ent.verdict, RiskVerdict.BLOCK)
        self.assertIn("ENT-002", res_high_ent.triggered_rules)

        ctx_med_ent = dict(self.default_context, entropy=0.80)
        res_med_ent = self._eval(ctx=ctx_med_ent)
        self.assertEqual(res_med_ent.verdict, RiskVerdict.REDUCE)
        self.assertIn("ENT-001", res_med_ent.triggered_rules)

    def test_drawdown_flattens(self):
        ctx_dd = dict(self.default_context, drawdown_pct=0.16)
        res_dd = self._eval(ctx=ctx_dd)
        self.assertEqual(res_dd.verdict, RiskVerdict.FLATTEN)
        self.assertIn("DD-001", res_dd.triggered_rules)

        ctx_peak = dict(self.default_context, session_peak_equity=100000.0, current_equity=84000.0)
        res_peak = self._eval(ctx=ctx_peak)
        self.assertEqual(res_peak.verdict, RiskVerdict.FLATTEN)
        self.assertIn("DD-001", res_peak.triggered_rules)

    def test_execution_timeout_flattens(self):
        ctx_timeout = dict(self.default_context, execution_timeout_seconds=31.0)
        res = self._eval(ctx=ctx_timeout)
        self.assertEqual(res.verdict, RiskVerdict.FLATTEN)
        self.assertIn("EXEC-001", res.triggered_rules)

    def test_sector_exposure_reduces(self):
        ctx_sector = dict(self.default_context, sector_exposure_pct=0.40)
        res = self._eval(ctx=ctx_sector)
        self.assertEqual(res.verdict, RiskVerdict.REDUCE)
        self.assertIn("SECT-001", res.triggered_rules)

    def test_caution_range_state_reduces(self):
        v = SevenStateVector(economic=0.50, financial=1.0, fiscal=1.0, portfolio=1.0, fundamental=1.0, market=1.0, sector=1.0)
        res = self._eval(states=v)
        self.assertEqual(res.verdict, RiskVerdict.REDUCE)
        self.assertIn("GATE-ECONOMIC-CAUTION", res.triggered_rules)

    def test_disagreement_and_effective_confidence_rules(self):
        base_agent_low = AgentOutput(s=1.0, c=0.30, u=0.0, d=0.0, p_plus=0.3, p_minus=0.1, delta_t=1.0, r=0.0, agent_id="a1")
        dummy_low = [
            AgentOutput(s=1.0, c=0.30, u=0.0, d=0.0, p_plus=0.3, p_minus=0.1, delta_t=1.0, r=0.0, agent_id=f"a{i}")
            for i in range(2, 8)
        ]
        agents_low_conf = [base_agent_low] + dummy_low
        weights_low = {f"a{i}": 1.0 for i in range(1, 8)}
        res_econf = self._eval(agents=agents_low_conf, weights=weights_low)
        self.assertEqual(res_econf.verdict, RiskVerdict.REDUCE)
        self.assertIn("ECONF-001", res_econf.triggered_rules)

        a1 = AgentOutput(s=1.0, c=0.8, u=0.0, d=0.0, p_plus=0.8, p_minus=0.0, delta_t=1.0, r=0.1, agent_id="a1")
        a2 = AgentOutput(s=-1.0, c=0.8, u=0.0, d=0.0, p_plus=0.0, p_minus=0.8, delta_t=1.0, r=0.1, agent_id="a2")
        a3 = AgentOutput(s=-1.0, c=0.8, u=0.0, d=0.0, p_plus=0.0, p_minus=0.8, delta_t=1.0, r=0.1, agent_id="a3")
        a4 = AgentOutput(s=-1.0, c=0.8, u=0.0, d=0.0, p_plus=0.0, p_minus=0.8, delta_t=1.0, r=0.1, agent_id="a4")
        a5 = AgentOutput(s=-1.0, c=0.8, u=0.0, d=0.0, p_plus=0.0, p_minus=0.8, delta_t=1.0, r=0.1, agent_id="a5")
        a6 = AgentOutput(s=-1.0, c=0.8, u=0.0, d=0.0, p_plus=0.0, p_minus=0.8, delta_t=1.0, r=0.1, agent_id="a6")
        a7 = AgentOutput(s=-1.0, c=0.8, u=0.0, d=0.0, p_plus=0.0, p_minus=0.8, delta_t=1.0, r=0.1, agent_id="a7")
        disag_agents = [a1, a2, a3, a4, a5, a6, a7]
        disag_weights = {"a1": 1.0, "a2": 2.0, "a3": 2.0, "a4": 1.0, "a5": 1.0, "a6": 1.0, "a7": 1.0}
        ensemble_signal = compute_ensemble_signal(disag_agents, disag_weights)
        disagreement = compute_disagreement(disag_agents, disag_weights, ensemble_signal)
        self.assertGreater(disagreement, 0.50)
        res_disag = self._eval(agents=disag_agents, weights=disag_weights)
        self.assertEqual(res_disag.verdict, RiskVerdict.REDUCE)
        self.assertIn("DISAG-001", res_disag.triggered_rules)

    def test_verdict_priority_flatten_overrides_block_and_reduce(self):
        ctx_combo = dict(self.default_context, drawdown_pct=0.20, position_pct=0.30, sector_exposure_pct=0.50)
        res = self._eval(ctx=ctx_combo)
        self.assertEqual(res.verdict, RiskVerdict.FLATTEN)
        self.assertIn("DD-001", res.triggered_rules)

    def test_verdict_priority_block_overrides_reduce(self):
        ctx_combo = dict(self.default_context, position_pct=0.30, sector_exposure_pct=0.50)
        res = self._eval(ctx=ctx_combo)
        self.assertEqual(res.verdict, RiskVerdict.BLOCK)
        self.assertIn("CONC-001", res.triggered_rules)

    def test_immutability_and_purity(self):
        states = SevenStateVector(
            economic=0.9,
            financial=0.9,
            fiscal=1.0,
            portfolio=1.0,
            fundamental=1.0,
            market=1.0,
            sector=1.0,
        )
        ctx = dict(self.default_context)
        ctx_copy = dict(ctx)
        res = self._eval(states=states, ctx=ctx)
        self.assertEqual(states.economic, 0.9)
        self.assertEqual(states.financial, 0.9)
        self.assertEqual(self.kalman_state.kalman_gain_price, 0.8)
        self.assertEqual(ctx, ctx_copy)

    def test_output_bounds(self):
        states = SevenStateVector(
            economic=0.3,
            financial=1.0,
            fiscal=1.0,
            portfolio=1.0,
            fundamental=1.0,
            market=0.4,
            sector=1.0,
        )
        res = self._eval(states=states)
        self.assertTrue(0.0 <= res.gating_factor <= 1.0)
        self.assertTrue(0.0 <= res.effective_cap <= 1.0)


class TestAdversarialAuditingAndEdgeCases(unittest.TestCase):
    """Adversarial unit tests covering boolean safety, NaN/Inf fail-open prevention, and regime validation."""

    def setUp(self):
        self.kalman_state = KalmanState(
            estimated_price=100.0,
            trend=0.5,
            uncertainty=1.0,
            trend_uncertainty=0.1,
            price_variance=1.0,
            trend_variance=0.01,
            innovation=0.2,
            kalman_gain_price=0.8
        )
        self.default_context = {
            "position_pct": 0.10,
            "gross_leverage": 0.50,
            "regime": "R01",
            "entropy": 0.20,
            "drawdown_pct": 0.02,
            "execution_timeout_seconds": 5.0,
            "is_new_long": False,
            "sector_exposure_pct": 0.15,
            "available_liquidity": 100000.0,
        }
        base_agent = AgentOutput(s=1.0, c=0.80, u=0.0, d=0.0, p_plus=0.8, p_minus=0.1, delta_t=1.0, r=0.1, agent_id="agent1")
        dummy_agent_template = lambda i: AgentOutput(
            s=1.0, c=0.80, u=0.0, d=0.0, p_plus=0.8, p_minus=0.1, delta_t=1.0, r=0.1, agent_id=f"agent{i}"
        )
        self.default_agents = [base_agent] + [dummy_agent_template(i) for i in range(2, 8)]
        self.default_weights = {f"agent{i}": 1.0 for i in range(1, 8)}

    def _eval(self, states=None, ctx=None, agents=None, weights=None, k_state=None):
        return evaluate(
            kalman_state=k_state if k_state is not None else self.kalman_state,
            states=states if states is not None else SevenStateVector.full_charge(),
            portfolio_context=ctx if ctx is not None else self.default_context,
            agents=agents if agents is not None else self.default_agents,
            agent_weights=weights if weights is not None else self.default_weights,
        )

    def test_boolean_string_parsing_safety(self):
        self.assertFalse(_parse_bool("False", "is_new_long"))
        self.assertFalse(_parse_bool("false", "is_new_long"))
        self.assertFalse(_parse_bool("0", "is_new_long"))
        self.assertTrue(_parse_bool("True", "is_new_long"))
        self.assertTrue(_parse_bool("true", "is_new_long"))
        self.assertTrue(_parse_bool("1", "is_new_long"))
        self.assertFalse(_parse_bool(0, "is_new_long"))
        self.assertTrue(_parse_bool(1, "is_new_long"))
        with self.assertRaises(ValueError):
            _parse_bool("foobar", "is_new_long")
        with self.assertRaises(ValueError):
            _parse_bool(2, "is_new_long")

    def test_reject_nan_and_inf_in_portfolio_context(self):
        bad_values = [float("nan"), float("inf"), float("-inf")]
        numeric_fields = [
            "position_pct",
            "gross_leverage",
            "entropy",
            "drawdown_pct",
            "execution_timeout_seconds",
            "sector_exposure_pct",
            "session_peak_equity",
            "current_equity",
            "available_liquidity",
        ]
        for field in numeric_fields:
            for bad_val in bad_values:
                ctx = dict(self.default_context, **{field: bad_val})
                with self.assertRaises(ValueError, msg=f"Field {field} allowed {bad_val}"):
                    self._eval(ctx=ctx)

    def test_canonical_regime_validation(self):
        valid_regimes = [f"R{i:02d}" for i in range(1, 13)]
        for r in valid_regimes:
            ctx = dict(self.default_context, regime=r)
            res = self._eval(ctx=ctx)
            self.assertIn(res.verdict, [RiskVerdict.ALLOW, RiskVerdict.REDUCE, RiskVerdict.BLOCK])

        invalid_regimes = ["R00", "R13", "BULL_QUIET", "INVALID", "R01_EXTRA"]
        for bad_r in invalid_regimes:
            ctx = dict(self.default_context, regime=bad_r)
            with self.assertRaises(ValueError, msg=f"Allowed invalid regime: {bad_r}"):
                self._eval(ctx=ctx)

    def test_out_of_bounds_percentages(self):
        with self.assertRaises(ValueError):
            self._eval(ctx=dict(self.default_context, position_pct=-0.1))
        with self.assertRaises(ValueError):
            self._eval(ctx=dict(self.default_context, position_pct=1.5))
        with self.assertRaises(ValueError):
            self._eval(ctx=dict(self.default_context, entropy=1.05))
        with self.assertRaises(ValueError):
            self._eval(ctx=dict(self.default_context, sector_exposure_pct=-0.01))

    def test_reject_unknown_portfolio_context_keys(self):
        bad_keys = ["postion_pct", "drawdow_pct", "leverag", "unknown_field"]
        for bad_k in bad_keys:
            ctx = dict(self.default_context, **{bad_k: 0.5})
            with self.assertRaises(KeyError, msg=f"Allowed unknown typo key: {bad_k}"):
                self._eval(ctx=ctx)

    def test_exact_threshold_boundary_conditions(self):
        ctx_exact_dd = dict(self.default_context, drawdown_pct=0.15)
        self.assertEqual(self._eval(ctx=ctx_exact_dd).verdict, RiskVerdict.ALLOW)

        ctx_above_dd = dict(self.default_context, drawdown_pct=0.150001)
        self.assertEqual(self._eval(ctx=ctx_above_dd).verdict, RiskVerdict.FLATTEN)

        ctx_exact_conc = dict(self.default_context, position_pct=0.20)
        self.assertEqual(self._eval(ctx=ctx_exact_conc).verdict, RiskVerdict.ALLOW)

        ctx_above_conc = dict(self.default_context, position_pct=0.200001)
        self.assertEqual(self._eval(ctx=ctx_above_conc).verdict, RiskVerdict.BLOCK)

        ctx_exact_lev = dict(self.default_context, gross_leverage=1.0)
        self.assertEqual(self._eval(ctx=ctx_exact_lev).verdict, RiskVerdict.ALLOW)

        ctx_above_lev = dict(self.default_context, gross_leverage=1.00001)
        self.assertEqual(self._eval(ctx=ctx_above_lev).verdict, RiskVerdict.BLOCK)

        ctx_ent_75 = dict(self.default_context, entropy=0.75)
        self.assertEqual(self._eval(ctx=ctx_ent_75).verdict, RiskVerdict.ALLOW)

        ctx_ent_76 = dict(self.default_context, entropy=0.75001)
        self.assertEqual(self._eval(ctx=ctx_ent_76).verdict, RiskVerdict.REDUCE)

        ctx_ent_90 = dict(self.default_context, entropy=0.90)
        self.assertEqual(self._eval(ctx=ctx_ent_90).verdict, RiskVerdict.REDUCE)

        ctx_ent_91 = dict(self.default_context, entropy=0.90001)
        self.assertEqual(self._eval(ctx=ctx_ent_91).verdict, RiskVerdict.BLOCK)

    def test_empty_agents_list_raises_via_evaluate(self):
        with self.assertRaises(ValueError):
            self._eval(agents=[])

    def test_agent_cardinality_enforcement(self):
        single_agent = [AgentOutput(s=1.0, c=0.8, u=0.0, d=0.0, p_plus=0.8, p_minus=0.0, delta_t=1.0, r=0.0, agent_id="a1")]
        with self.assertRaises(ValueError):
            self._eval(agents=single_agent, weights={"a1": 1.0})

        six_agents = [
            AgentOutput(s=1.0, c=0.8, u=0.0, d=0.0, p_plus=0.8, p_minus=0.0, delta_t=1.0, r=0.0, agent_id=f"a{i}")
            for i in range(1, 7)
        ]
        weights_6 = {f"a{i}": 1.0 for i in range(1, 7)}
        with self.assertRaises(ValueError):
            self._eval(agents=six_agents, weights=weights_6)

        eight_agents = [
            AgentOutput(s=1.0, c=0.8, u=0.0, d=0.0, p_plus=0.8, p_minus=0.0, delta_t=1.0, r=0.0, agent_id=f"a{i}")
            for i in range(1, 9)
        ]
        weights_8 = {f"a{i}": 1.0 for i in range(1, 9)}
        with self.assertRaises(ValueError):
            self._eval(agents=eight_agents, weights=weights_8)

    def test_missing_agent_weight_raises_via_evaluate(self):
        seven_agents = [
            AgentOutput(s=1.0, c=0.8, u=0.0, d=0.0, p_plus=0.8, p_minus=0.0, delta_t=1.0, r=0.0, agent_id=f"agent{i}")
            for i in range(1, 8)
        ]
        incomplete_weights = {f"agent{i}": 1.0 for i in range(1, 7)}
        with self.assertRaises(ValueError):
            self._eval(agents=seven_agents, weights=incomplete_weights)


class TestCapitalGateBoundaryConditions(unittest.TestCase):
    """Test boundary conditions for capital gate evaluation."""

    def test_state_at_minimum_returns_zero_gate(self):
        """Verify state at exactly minimum returns 0.0 gating via linear branch."""
        from investment_agent.capital.capital_gate import compute_individual_gating
        g_val = compute_individual_gating("economic", 0.15)
        self.assertEqual(g_val, 0.0)

    def test_state_below_minimum_returns_zero_gate(self):
        """Verify state below minimum returns 0.0 gating."""
        from investment_agent.capital.capital_gate import compute_individual_gating
        g_val = compute_individual_gating("economic", 0.149999)
        self.assertEqual(g_val, 0.0)

    def test_state_at_full_returns_one_gate(self):
        """Verify state at exactly full returns 1.0 gating."""
        from investment_agent.capital.capital_gate import compute_individual_gating
        g_val = compute_individual_gating("economic", 0.70)
        self.assertEqual(g_val, 1.0)

    def test_state_above_full_returns_one_gate(self):
        """Verify state above full returns 1.0 gating."""
        from investment_agent.capital.capital_gate import compute_individual_gating
        g_val = compute_individual_gating("economic", 0.700001)
        self.assertEqual(g_val, 1.0)

    def test_liquidity_floor_blocks_below_5000(self):
        """Verify available_liquidity < $5,000 triggers LIQ-001 BLOCK."""
        from investment_agent.capital.capital_gate import evaluate
        from investment_agent.filters.kalman_filter import KalmanState
        from investment_agent.signals.ensemble_signal import AgentOutput
        from investment_agent.capital.capital_gate import SevenStateVector
        ctx = {
            "position_pct": 0.10,
            "gross_leverage": 0.50,
            "regime": "R01",
            "entropy": 0.20,
            "drawdown_pct": 0.02,
            "execution_timeout_seconds": 5.0,
            "is_new_long": False,
            "sector_exposure_pct": 0.15,
            "available_liquidity": 4999.99,
        }
        res = evaluate(
            kalman_state=KalmanState(
                estimated_price=100.0,
                trend=0.5,
                uncertainty=1.0,
                trend_uncertainty=0.1,
                price_variance=1.0,
                trend_variance=0.01,
                innovation=0.2,
                kalman_gain_price=0.8,
            ),
            states=SevenStateVector(
                economic=1.0, financial=1.0, fiscal=1.0,
                portfolio=1.0, fundamental=1.0, market=1.0, sector=1.0
            ),
            portfolio_context=ctx,
            agents=[
                AgentOutput(s=1.0, c=0.80, u=0.0, d=0.0, p_plus=0.8, p_minus=0.1, delta_t=1.0, r=0.0, agent_id=f"agent{i}")
                for i in range(1, 8)
            ],
            agent_weights={f"agent{i}": 1.0 for i in range(1, 8)},
        )
        self.assertEqual(res.verdict, RiskVerdict.BLOCK)
        self.assertIn("LIQ-001", res.triggered_rules)
        self.assertEqual(res.effective_cap, 0.0)

    def test_liquidity_floor_allows_at_5000(self):
        """Verify available_liquidity >= $5,000 does not trigger LIQ-001."""
        from investment_agent.capital.capital_gate import evaluate
        from investment_agent.filters.kalman_filter import KalmanState
        from investment_agent.signals.ensemble_signal import AgentOutput
        from investment_agent.capital.capital_gate import SevenStateVector
        ctx = {
            "position_pct": 0.10,
            "gross_leverage": 0.50,
            "regime": "R01",
            "entropy": 0.20,
            "drawdown_pct": 0.02,
            "execution_timeout_seconds": 5.0,
            "is_new_long": False,
            "sector_exposure_pct": 0.15,
            "available_liquidity": 5000.00,
        }
        res = evaluate(
            kalman_state=KalmanState(
                estimated_price=100.0,
                trend=0.5,
                uncertainty=1.0,
                trend_uncertainty=0.1,
                price_variance=1.0,
                trend_variance=0.01,
                innovation=0.2,
                kalman_gain_price=0.8,
            ),
            states=SevenStateVector(
                economic=1.0, financial=1.0, fiscal=1.0,
                portfolio=1.0, fundamental=1.0, market=1.0, sector=1.0
            ),
            portfolio_context=ctx,
            agents=[
                AgentOutput(s=1.0, c=0.80, u=0.0, d=0.0, p_plus=0.8, p_minus=0.1, delta_t=1.0, r=0.0, agent_id=f"agent{i}")
                for i in range(1, 8)
            ],
            agent_weights={f"agent{i}": 1.0 for i in range(1, 8)},
        )
        self.assertNotIn("LIQ-001", res.triggered_rules)

    def test_liquidity_floor_blocks_at_zero(self):
        """Verify zero liquidity triggers LIQ-001."""
        from investment_agent.capital.capital_gate import evaluate
        from investment_agent.filters.kalman_filter import KalmanState
        from investment_agent.signals.ensemble_signal import AgentOutput
        from investment_agent.capital.capital_gate import SevenStateVector
        ctx = {
            "position_pct": 0.10,
            "gross_leverage": 0.50,
            "regime": "R01",
            "entropy": 0.20,
            "drawdown_pct": 0.02,
            "execution_timeout_seconds": 5.0,
            "is_new_long": False,
            "sector_exposure_pct": 0.15,
            "available_liquidity": 0.0,
        }
        res = evaluate(
            kalman_state=KalmanState(
                estimated_price=100.0,
                trend=0.5,
                uncertainty=1.0,
                trend_uncertainty=0.1,
                price_variance=1.0,
                trend_variance=0.01,
                innovation=0.2,
                kalman_gain_price=0.8,
            ),
            states=SevenStateVector(
                economic=1.0, financial=1.0, fiscal=1.0,
                portfolio=1.0, fundamental=1.0, market=1.0, sector=1.0
            ),
            portfolio_context=ctx,
            agents=[
                AgentOutput(s=1.0, c=0.80, u=0.0, d=0.0, p_plus=0.8, p_minus=0.1, delta_t=1.0, r=0.0, agent_id=f"agent{i}")
                for i in range(1, 8)
            ],
            agent_weights={f"agent{i}": 1.0 for i in range(1, 8)},
        )
        self.assertEqual(res.verdict, RiskVerdict.BLOCK)
        self.assertIn("LIQ-001", res.triggered_rules)


class TestCapitalGateResultFields(unittest.TestCase):
    """Test CapitalGateResult includes kalman_gain and ensemble_agg."""

    def test_result_includes_kalman_gain(self):
        """Verify CapitalGateResult includes kalman_gain field."""
        from investment_agent.capital.capital_gate import evaluate
        from investment_agent.filters.kalman_filter import KalmanState
        from investment_agent.signals.ensemble_signal import AgentOutput
        agents = [
            AgentOutput(s=1.0, c=0.8, u=0.0, d=0.0, p_plus=0.8, p_minus=0.0, delta_t=1.0, r=0.0, agent_id=f"agent{i}")
            for i in range(1, 8)
        ]
        weights = {f"agent{i}": 1.0 for i in range(1, 8)}
        result = evaluate(
            kalman_state=KalmanState(
                estimated_price=100.0,
                trend=0.01,
                uncertainty=1.0,
                trend_uncertainty=0.1,
                price_variance=1.0,
                trend_variance=0.01,
                innovation=0.0,
                kalman_gain_price=0.5,
            ),
            states=SevenStateVector(
                economic=1.0, financial=1.0, fiscal=1.0,
                portfolio=1.0, fundamental=1.0, market=1.0, sector=1.0
            ),
            portfolio_context={
                "position_pct": 0.05,
                "gross_leverage": 0.5,
                "entropy": 0.1,
                "drawdown_pct": 0.01,
                "execution_timeout_seconds": 5.0,
                "sector_exposure_pct": 0.1,
                "is_new_long": False,
                "regime": "R01",
                "available_liquidity": 100000.0,
            },
            agents=agents,
            agent_weights=weights,
        )
        self.assertTrue(hasattr(result, "kalman_gain"))
        self.assertIsInstance(result.kalman_gain, float)
        self.assertGreaterEqual(result.kalman_gain, 0.0)
        self.assertLessEqual(result.kalman_gain, 1.0)

    def test_result_includes_ensemble_agg(self):
        """Verify CapitalGateResult includes ensemble_agg field."""
        from investment_agent.capital.capital_gate import evaluate
        from investment_agent.filters.kalman_filter import KalmanState
        from investment_agent.signals.ensemble_signal import AgentOutput, EnsembleAggregate
        agents = [
            AgentOutput(s=1.0, c=0.8, u=0.0, d=0.0, p_plus=0.8, p_minus=0.0, delta_t=1.0, r=0.0, agent_id=f"agent{i}")
            for i in range(1, 8)
        ]
        weights = {f"agent{i}": 1.0 for i in range(1, 8)}
        result = evaluate(
            kalman_state=KalmanState(
                estimated_price=100.0,
                trend=0.01,
                uncertainty=1.0,
                trend_uncertainty=0.1,
                price_variance=1.0,
                trend_variance=0.01,
                innovation=0.0,
                kalman_gain_price=0.5,
            ),
            states=SevenStateVector(
                economic=1.0, financial=1.0, fiscal=1.0,
                portfolio=1.0, fundamental=1.0, market=1.0, sector=1.0
            ),
            portfolio_context={
                "position_pct": 0.05,
                "gross_leverage": 0.5,
                "entropy": 0.1,
                "drawdown_pct": 0.01,
                "execution_timeout_seconds": 5.0,
                "sector_exposure_pct": 0.1,
                "is_new_long": False,
                "regime": "R01",
                "available_liquidity": 100000.0,
            },
            agents=agents,
            agent_weights=weights,
        )
        self.assertTrue(hasattr(result, "ensemble_agg"))
        self.assertIsInstance(result.ensemble_agg, EnsembleAggregate)


class TestPropertyInvariantTests(unittest.TestCase):
    """Property-based invariant tests for capital gate evaluation."""

    def setUp(self):
        self.kalman_state = KalmanState(
            estimated_price=100.0,
            trend=0.01,
            uncertainty=1.0,
            trend_uncertainty=0.1,
            price_variance=1.0,
            trend_variance=0.01,
            innovation=0.0,
            kalman_gain_price=0.5,
        )
        self.agents = [
            AgentOutput(s=1.0, c=0.8, u=0.0, d=0.0, p_plus=0.8, p_minus=0.0, delta_t=1.0, r=0.0, agent_id=f"agent{i}")
            for i in range(1, 8)
        ]
        self.weights = {f"agent{i}": 1.0 for i in range(1, 8)}
        self.default_context = {
            "position_pct": 0.05,
            "gross_leverage": 0.5,
            "entropy": 0.1,
            "drawdown_pct": 0.01,
            "execution_timeout_seconds": 5.0,
            "sector_exposure_pct": 0.1,
            "is_new_long": False,
            "regime": "R01",
            "available_liquidity": 100000.0,
        }

    def _eval(self, **overrides):
        ctx = dict(self.default_context)
        ctx.update(overrides)
        return evaluate(
            kalman_state=self.kalman_state,
            states=SevenStateVector(
                economic=1.0, financial=1.0, fiscal=1.0,
                portfolio=1.0, fundamental=1.0, market=1.0, sector=1.0
            ),
            portfolio_context=ctx,
            agents=self.agents,
            agent_weights=self.weights,
        )

    def test_gating_factor_in_zero_one(self):
        """Gating factor must be in [0.0, 1.0] for any valid input."""
        for entropy in [0.0, 0.5, 0.75, 0.90]:
            for drawdown in [0.0, 0.10, 0.15]:
                res = self._eval(entropy=entropy, drawdown_pct=drawdown)
                self.assertGreaterEqual(res.gating_factor, 0.0)
                self.assertLessEqual(res.gating_factor, 1.0)

    def test_effective_cap_in_zero_one(self):
        """Effective cap must be in [0.0, 1.0] for any valid input."""
        for verdict in [RiskVerdict.ALLOW, RiskVerdict.REDUCE, RiskVerdict.BLOCK, RiskVerdict.FLATTEN]:
            if verdict == RiskVerdict.ALLOW:
                res = self._eval()
            elif verdict == RiskVerdict.REDUCE:
                res = self._eval(entropy=0.80)
            elif verdict == RiskVerdict.BLOCK:
                res = self._eval(entropy=0.95)
            else:
                res = self._eval(drawdown_pct=0.20)
            self.assertGreaterEqual(res.effective_cap, 0.0)
            self.assertLessEqual(res.effective_cap, 1.0)

    def test_reduce_factor_in_zero_one(self):
        """Reduce factor must be in [0.0, 1.0]."""
        res = self._eval(entropy=0.80, sector_exposure_pct=0.40)
        self.assertGreaterEqual(res.reduce_factor, 0.0)
        self.assertLessEqual(res.reduce_factor, 1.0)

    def test_kalman_gain_non_negative(self):
        """Kalman gain must be >= 0."""
        res = self._eval()
        self.assertGreaterEqual(res.kalman_gain, 0.0)

    def test_ensemble_signal_in_range(self):
        """Ensemble signal must be in [-1.0, 1.0]."""
        res = self._eval()
        self.assertGreaterEqual(res.ensemble_agg.ensemble_signal, -1.0)
        self.assertLessEqual(res.ensemble_agg.ensemble_signal, 1.0)

    def test_effective_confidence_in_range(self):
        """Effective confidence must be in [0.0, 1.0]."""
        res = self._eval()
        self.assertGreaterEqual(res.ensemble_agg.effective_confidence, 0.0)
        self.assertLessEqual(res.ensemble_agg.effective_confidence, 1.0)

    def test_disagreement_in_range(self):
        """Disagreement must be in [0.0, 2.0]."""
        res = self._eval()
        self.assertGreaterEqual(res.ensemble_agg.disagreement, 0.0)
        self.assertLessEqual(res.ensemble_agg.disagreement, 2.0)

    def test_state_charges_in_zero_one(self):
        """All state charges must be in [0.0, 1.0]."""
        res = self._eval()
        for name, val in res.state_charges.items():
            self.assertGreaterEqual(val, 0.0, msg=f"State {name} below 0")
            self.assertLessEqual(val, 1.0, msg=f"State {name} above 1")

    def test_agent_weights_non_negative(self):
        """Agent weights used in evaluation must be non-negative."""
        for wid, wval in self.weights.items():
            self.assertGreaterEqual(wval, 0.0, msg=f"Weight {wid} negative")

    def test_flatten_sets_effective_cap_zero(self):
        """FLATTEN verdict must force effective_cap to 0.0."""
        res = self._eval(drawdown_pct=0.20)
        self.assertEqual(res.verdict, RiskVerdict.FLATTEN)
        self.assertEqual(res.effective_cap, 0.0)

    def test_block_sets_effective_cap_zero(self):
        """BLOCK verdict must force effective_cap to 0.0."""
        res = self._eval(entropy=0.95)
        self.assertEqual(res.verdict, RiskVerdict.BLOCK)
        self.assertEqual(res.effective_cap, 0.0)

    def test_reduce_preserves_raw_cap_relationship(self):
        """REDUCE effective_cap should be <= raw_cap (kalman_gain * gating_factor)."""
        res = self._eval(entropy=0.80)
        raw_cap = res.kalman_gain * res.gating_factor
        self.assertLessEqual(res.effective_cap, raw_cap + 1e-9)

    def test_all_state_gatings_in_zero_one(self):
        """Individual state gatings must be in [0.0, 1.0]."""
        res = self._eval()
        for name, val in res.state_gatings.items():
            self.assertGreaterEqual(val, 0.0)
            self.assertLessEqual(val, 1.0)


if __name__ == "__main__":
    unittest.main()
