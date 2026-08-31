"""Adversarial unit test suite for agent_reputation.py.

Verifies Bayesian Beta-prior reputation tracking (Definition 8.1), parameter & variance inspection,
state serialization, agent/regime isolation, and end-to-end integration into investment Kalman gain.
"""

import random
import unittest
from investment_agent.agents.agent_reputation import AgentReputationTracker
from investment_agent.signals.ensemble_signal import AgentOutput, compute_ensemble_signal, compute_disagreement, compute_effective_confidence
from investment_agent.filters.investment_kalman_gain import compute_investment_kalman_gain
from investment_agent.capital.capital_gate import evaluate, SevenStateVector, KalmanState, RiskVerdict


class TestAgentReputationTracker(unittest.TestCase):
    """Test AgentReputationTracker initialization, outcome updates, and key validations."""

    def test_prior_initialization_and_exact_arithmetic(self):
        """Verify default prior Beta(1,1) -> weight 0.5, and single update arithmetic."""
        tracker = AgentReputationTracker(agent_ids=["agent1"], regimes=["R01"], prior_alpha=1.0, prior_beta=1.0)
        self.assertEqual(tracker.get_reputation_weight("agent1", "R01"), 0.5)

        # 1 correct outcome -> alpha=2, beta=1 -> weight = 2 / 3 = 0.666667
        tracker.record_outcome("agent1", "R01", was_correct=True)
        self.assertAlmostEqual(tracker.get_reputation_weight("agent1", "R01"), 2.0 / 3.0, places=6)

        # 1 incorrect outcome -> alpha=2, beta=2 -> weight = 2 / 4 = 0.50
        tracker.record_outcome("agent1", "R01", was_correct=False)
        self.assertEqual(tracker.get_reputation_weight("agent1", "R01"), 0.50)

    def test_duplicate_agent_ids_rejected(self):
        """Verify duplicate agent IDs raise ValueError at initialization."""
        with self.assertRaises(ValueError):
            AgentReputationTracker(agent_ids=["a1", "a1", "a2"], regimes=["R01"])

    def test_duplicate_regimes_rejected(self):
        """Verify duplicate regime identifiers raise ValueError at initialization."""
        with self.assertRaises(ValueError):
            AgentReputationTracker(agent_ids=["a1"], regimes=["R01", "R01"])

    def test_empty_agent_ids_or_regimes_rejected(self):
        """Verify empty lists for agent_ids or regimes raise ValueError."""
        with self.assertRaises(ValueError):
            AgentReputationTracker(agent_ids=[], regimes=["R01"])
        with self.assertRaises(ValueError):
            AgentReputationTracker(agent_ids=["a1"], regimes=[])

    def test_invalid_regimes_rejected(self):
        """Verify invalid regime strings (e.g. R00 or INVALID) raise ValueError."""
        with self.assertRaises(ValueError):
            AgentReputationTracker(agent_ids=["a1"], regimes=["INVALID_REGIME"])
        with self.assertRaises(ValueError):
            AgentReputationTracker(agent_ids=["a1"], regimes=["R00"])

    def test_prior_boundary_and_invalid_type_validations(self):
        """Verify prior_alpha and prior_beta <= 0, NaN, Inf, bool, non-numeric raise appropriate errors."""
        invalid_vals = [0.0, -1.0, float("nan"), float("inf"), float("-inf")]
        for val in invalid_vals:
            with self.assertRaises(ValueError):
                AgentReputationTracker(agent_ids=["a1"], regimes=["R01"], prior_alpha=val)
            with self.assertRaises(ValueError):
                AgentReputationTracker(agent_ids=["a1"], regimes=["R01"], prior_beta=val)

        # Bool or non-numeric types raise TypeError
        with self.assertRaises(TypeError):
            AgentReputationTracker(agent_ids=["a1"], regimes=["R01"], prior_alpha=True)
        with self.assertRaises(TypeError):
            AgentReputationTracker(agent_ids=["a1"], regimes=["R01"], prior_beta="1.0")

    def test_was_correct_non_boolean_rejected(self):
        """Verify record_outcome rejects non-boolean was_correct values (e.g., 1 or 0 or 'True')."""
        tracker = AgentReputationTracker(agent_ids=["a1"], regimes=["R01"])
        with self.assertRaises(TypeError):
            tracker.record_outcome("a1", "R01", was_correct=1)
        with self.assertRaises(TypeError):
            tracker.record_outcome("a1", "R01", was_correct="True")

    def test_whitespace_normalization(self):
        """Verify leading/trailing whitespace in agent IDs is normalized cleanly."""
        tracker = AgentReputationTracker(agent_ids=[" agent1 "], regimes=["R01"])
        self.assertAlmostEqual(tracker.get_reputation_weight("agent1", "R01"), 0.5)
        tracker.record_outcome(" agent1 ", "R01", was_correct=True)
        self.assertAlmostEqual(tracker.get_reputation_weight("agent1", "R01"), 2.0 / 3.0)

    def test_custom_prior_and_multiple_outcomes(self):
        """Verify Beta(2,3) prior with 4 correct and 1 incorrect outcome produces alpha=6, beta=4, weight=0.6."""
        tracker = AgentReputationTracker(
            agent_ids=["a1"],
            regimes=["R01"],
            prior_alpha=2.0,
            prior_beta=3.0,
        )

        for _ in range(4):
            tracker.record_outcome("a1", "R01", was_correct=True)
        tracker.record_outcome("a1", "R01", was_correct=False)

        params = tracker.get_posterior_parameters("a1", "R01")
        self.assertEqual(params["alpha"], 6.0)
        self.assertEqual(params["beta"], 4.0)
        self.assertAlmostEqual(tracker.get_reputation_weight("a1", "R01"), 0.6)

    def test_all_reputation_weights_are_strictly_positive(self):
        """Verify all reputation weights remain strictly bounded in (0.0, 1.0)."""
        tracker = AgentReputationTracker(agent_ids=["a1", "a2"], regimes=["R01", "R02"])
        for r in ["R01", "R02"]:
            weights = tracker.get_all_weights(r)
            for w in weights.values():
                self.assertGreater(w, 0.0)
                self.assertLess(w, 1.0)

    def test_monotonicity_invariants(self):
        """Verify that a correct observation strictly increases the posterior mean, while an incorrect observation strictly decreases the posterior mean relative to the posterior immediately before that observation."""
        tracker = AgentReputationTracker(agent_ids=["a1"], regimes=["R01"])
        w0 = tracker.get_reputation_weight("a1", "R01")

        tracker.record_outcome("a1", "R01", was_correct=True)
        w1 = tracker.get_reputation_weight("a1", "R01")
        self.assertGreater(w1, w0)

        tracker.record_outcome("a1", "R01", was_correct=False)
        w2 = tracker.get_reputation_weight("a1", "R01")
        self.assertLess(w2, w1)

    def test_long_run_extreme_outcomes(self):
        """Verify 1000 correct outcomes approach 1.0 and 1000 incorrect outcomes approach 0.0."""
        tracker_good = AgentReputationTracker(agent_ids=["good"], regimes=["R01"])
        for _ in range(1000):
            tracker_good.record_outcome("good", "R01", was_correct=True)
        w_good = tracker_good.get_reputation_weight("good", "R01")
        self.assertGreater(w_good, 0.99)
        self.assertLess(w_good, 1.0)

        tracker_bad = AgentReputationTracker(agent_ids=["bad"], regimes=["R01"])
        for _ in range(1000):
            tracker_bad.record_outcome("bad", "R01", was_correct=False)
        w_bad = tracker_bad.get_reputation_weight("bad", "R01")
        self.assertLess(w_bad, 0.01)
        self.assertGreater(w_bad, 0.0)

    def test_regime_specific_updates_are_isolated(self):
        """Verify outcome recorded for (agent1, R01) does not mutate (agent1, R02)."""
        tracker = AgentReputationTracker(agent_ids=["a1"], regimes=["R01", "R02"])
        tracker.record_outcome("a1", "R01", was_correct=True)

        self.assertAlmostEqual(tracker.get_reputation_weight("a1", "R01"), 2.0 / 3.0)
        self.assertAlmostEqual(tracker.get_reputation_weight("a1", "R02"), 0.5)

    def test_agent_isolation(self):
        """Verify outcome recorded for (a1, R01) does not mutate (a2, R01)."""
        tracker = AgentReputationTracker(agent_ids=["a1", "a2"], regimes=["R01"])
        tracker.record_outcome("a1", "R01", was_correct=True)

        self.assertAlmostEqual(tracker.get_reputation_weight("a1", "R01"), 2.0 / 3.0)
        self.assertAlmostEqual(tracker.get_reputation_weight("a2", "R01"), 0.5)

    def test_inspection_apis(self):
        """Verify get_posterior_parameters, get_posterior_variance, and get_observation_count."""
        tracker = AgentReputationTracker(agent_ids=["a1"], regimes=["R01"], prior_alpha=1.0, prior_beta=1.0)
        self.assertEqual(tracker.get_observation_count("a1", "R01"), 0)

        # Initial Beta(1,1) variance = (1*1) / (2^2 * 3) = 1/12 = 0.083333...
        self.assertAlmostEqual(tracker.get_posterior_variance("a1", "R01"), 1.0 / 12.0)

        tracker.record_outcome("a1", "R01", was_correct=True)
        self.assertEqual(tracker.get_observation_count("a1", "R01"), 1)
        params = tracker.get_posterior_parameters("a1", "R01")
        self.assertEqual(params, {"alpha": 2.0, "beta": 1.0})

        # Beta(2,1) variance = (2*1) / (3^2 * 4) = 2/36 = 1/18 = 0.05555...
        self.assertAlmostEqual(tracker.get_posterior_variance("a1", "R01"), 1.0 / 18.0)

    def test_serialization_to_and_from_dict(self):
        """Verify to_dict and from_dict serialize and restore complete tracker state faithfully."""
        tracker = AgentReputationTracker(agent_ids=["a1", "a2"], regimes=["R01", "R02"], prior_alpha=2.0, prior_beta=2.0)
        tracker.record_outcome("a1", "R01", was_correct=True)
        tracker.record_outcome("a2", "R02", was_correct=False)

        state_dict = tracker.to_dict()
        restored_tracker = AgentReputationTracker.from_dict(state_dict)

        self.assertAlmostEqual(restored_tracker.get_reputation_weight("a1", "R01"), 3.0 / 5.0)
        self.assertAlmostEqual(restored_tracker.get_reputation_weight("a2", "R02"), 2.0 / 5.0)
        self.assertEqual(restored_tracker.get_observation_count("a1", "R01"), 1)
        self.assertEqual(restored_tracker.get_observation_count("a2", "R02"), 1)

    def test_simulated_posterior_tracks_true_reliability(self):
        """Verify posterior mean consistency and variance reduction under simulated Bernoulli outcomes across seeds."""
        true_reliability = 0.80
        episodes = 500

        for seed in [42, 123, 999]:
            rng = random.Random(seed)
            tracker = AgentReputationTracker(agent_ids=["agent_80"], regimes=["R01"])

            for step in range(episodes):
                outcome = rng.random() < true_reliability
                tracker.record_outcome("agent_80", "R01", was_correct=outcome)

            final_weight = tracker.get_reputation_weight("agent_80", "R01")
            self.assertAlmostEqual(final_weight, true_reliability, delta=0.05)

            # Check variance reduction
            var_final = tracker.get_posterior_variance("agent_80", "R01")
            self.assertLess(var_final, 0.001)
            self.assertEqual(tracker.get_observation_count("agent_80", "R01"), episodes)

    def test_end_to_end_reputation_to_kalman_gain_pipeline(self):
        """Verify end-to-end integration: reputation updates -> agent weights -> ensemble signal -> K_t -> CapitalGate."""
        agent_ids = [f"a{i}" for i in range(1, 8)]
        tracker = AgentReputationTracker(agent_ids=agent_ids, regimes=["R01"])
        
        # Initial weights are equal (0.5 each)
        w_init = tracker.get_all_weights("R01")

        a1_output = AgentOutput(s=1.0, c=0.9, u=0.0, d=0.0, p_plus=0.9, p_minus=0.1, delta_t=1.0, r=0.1, agent_id="a1")
        a2_output = AgentOutput(s=-1.0, c=0.5, u=0.0, d=0.0, p_plus=0.1, p_minus=0.9, delta_t=1.0, r=0.1, agent_id="a2")
        dummy_agents = [
            AgentOutput(s=0.0, c=0.8, u=0.0, d=0.0, p_plus=0.5, p_minus=0.5, delta_t=1.0, r=0.1, agent_id=f"a{i}")
            for i in range(3, 8)
        ]
        agents = [a1_output, a2_output] + dummy_agents

        # Equal weights -> compute initial metrics
        s_init = compute_ensemble_signal(agents, w_init)
        d_init = compute_disagreement(agents, w_init, s_init)
        c_init = compute_effective_confidence(agents, w_init)
        k_init = compute_investment_kalman_gain(prediction_covariance=1.0, effective_confidence=c_init, disagreement=d_init)

        # Now agent a1 gets 10 correct outcomes in R01 -> a1 reputation weight increases
        for _ in range(10):
            tracker.record_outcome("a1", "R01", was_correct=True)

        w_updated = tracker.get_all_weights("R01")
        self.assertGreater(w_updated["a1"], w_updated["a2"])

        # Re-evaluate pipeline with updated weights
        s_upd = compute_ensemble_signal(agents, w_updated)
        d_upd = compute_disagreement(agents, w_updated, s_upd)
        c_upd = compute_effective_confidence(agents, w_updated)
        k_upd = compute_investment_kalman_gain(prediction_covariance=1.0, effective_confidence=c_upd, disagreement=d_upd)

        # Verify end-to-end propagation: each stage must change when reputation changes
        self.assertGreater(w_updated["a1"], w_init["a1"], "Reputation weight should increase after correct outcomes")
        self.assertNotAlmostEqual(s_upd, s_init, msg="Ensemble signal should change when weights change")
        self.assertNotAlmostEqual(d_upd, d_init, msg="Disagreement should change when weights change")
        self.assertNotAlmostEqual(c_upd, c_init, msg="Effective confidence should change when weights change")
        self.assertNotAlmostEqual(k_upd, k_init, msg="Kalman gain should change when confidence/disagreement change")

        # Because a1 (bullish) dominates, aggregate signal becomes bullish (s_upd > s_init)
        self.assertGreater(s_upd, s_init)

        k_state = KalmanState(
            estimated_price=100.0,
            trend=0.0,
            uncertainty=1.0,
            trend_uncertainty=0.1,
            price_variance=1.0,
            trend_variance=0.01,
            innovation=0.2,
            kalman_gain_price=0.8,
        )
        res = evaluate(
            kalman_state=k_state,
            states=SevenStateVector(
                economic=1.0,
                financial=1.0,
                fiscal=1.0,
                portfolio=1.0,
                fundamental=1.0,
                market=1.0,
                sector=1.0,
            ),
            portfolio_context={"position_pct": 0.05, "gross_leverage": 0.5, "entropy": 0.5, "drawdown_pct": 0.02, "execution_timeout_seconds": 5.0, "is_new_long": False, "sector_exposure_pct": 0.15, "regime": "R01", "available_liquidity": 100000.0},
            agents=agents,
            agent_weights=w_updated,
        )
        self.assertIsInstance(res.verdict, RiskVerdict)

    def test_corrupted_persistence_from_dict(self):
        """Adversarial suite testing rejection of corrupted, malformed, or invariant-violating state dictionaries in from_dict()."""
        base_tracker = AgentReputationTracker(agent_ids=["a1"], regimes=["R01"], prior_alpha=1.0, prior_beta=1.0)
        base_tracker.record_outcome("a1", "R01", was_correct=True)
        valid_dict = base_tracker.to_dict()

        # Non-dict data
        with self.assertRaises(TypeError):
            AgentReputationTracker.from_dict("not_a_dict")

        # Missing top-level keys
        incomplete_dict = dict(valid_dict)
        del incomplete_dict["state"]
        with self.assertRaises(ValueError):
            AgentReputationTracker.from_dict(incomplete_dict)

        # Non-dict state map
        bad_state_type = dict(valid_dict, state="not_a_dict")
        with self.assertRaises(TypeError):
            AgentReputationTracker.from_dict(bad_state_type)

        # Malformed key format (missing |)
        bad_key_fmt = dict(valid_dict, state={"a1_R01": {"alpha": 2.0, "beta": 1.0, "observations": 1}})
        with self.assertRaises(ValueError):
            AgentReputationTracker.from_dict(bad_key_fmt)

        # Key containing unregistered agent or regime
        unreg_agent = dict(valid_dict, state={"unregistered|R01": {"alpha": 2.0, "beta": 1.0, "observations": 1}})
        with self.assertRaises(ValueError):
            AgentReputationTracker.from_dict(unreg_agent)

        unreg_regime = dict(valid_dict, state={"a1|R99": {"alpha": 2.0, "beta": 1.0, "observations": 1}})
        with self.assertRaises(ValueError):
            AgentReputationTracker.from_dict(unreg_regime)

        # Missing required field in entry
        missing_alpha = dict(valid_dict, state={"a1|R01": {"beta": 1.0, "observations": 1}})
        with self.assertRaises(ValueError):
            AgentReputationTracker.from_dict(missing_alpha)

        # Negative alpha / beta
        neg_alpha = dict(valid_dict, state={"a1|R01": {"alpha": -1.0, "beta": 1.0, "observations": 1}})
        with self.assertRaises(ValueError):
            AgentReputationTracker.from_dict(neg_alpha)

        neg_beta = dict(valid_dict, state={"a1|R01": {"alpha": 2.0, "beta": -1.0, "observations": 1}})
        with self.assertRaises(ValueError):
            AgentReputationTracker.from_dict(neg_beta)

        # Zero alpha / beta
        zero_alpha = dict(valid_dict, state={"a1|R01": {"alpha": 0.0, "beta": 1.0, "observations": 1}})
        with self.assertRaises(ValueError):
            AgentReputationTracker.from_dict(zero_alpha)

        # NaN / Inf in alpha / beta
        nan_alpha = dict(valid_dict, state={"a1|R01": {"alpha": float("nan"), "beta": 1.0, "observations": 1}})
        with self.assertRaises(ValueError):
            AgentReputationTracker.from_dict(nan_alpha)

        inf_beta = dict(valid_dict, state={"a1|R01": {"alpha": 2.0, "beta": float("inf"), "observations": 1}})
        with self.assertRaises(ValueError):
            AgentReputationTracker.from_dict(inf_beta)

        # Alpha < prior_alpha
        alpha_below_prior = dict(valid_dict, state={"a1|R01": {"alpha": 0.5, "beta": 1.0, "observations": 0}})
        with self.assertRaises(ValueError):
            AgentReputationTracker.from_dict(alpha_below_prior)

        # Negative / non-integer / bool observations
        neg_obs = dict(valid_dict, state={"a1|R01": {"alpha": 2.0, "beta": 1.0, "observations": -1}})
        with self.assertRaises(ValueError):
            AgentReputationTracker.from_dict(neg_obs)

        float_obs = dict(valid_dict, state={"a1|R01": {"alpha": 2.0, "beta": 1.0, "observations": 1.5}})
        with self.assertRaises(TypeError):
            AgentReputationTracker.from_dict(float_obs)

        bool_obs = dict(valid_dict, state={"a1|R01": {"alpha": 2.0, "beta": 1.0, "observations": True}})
        with self.assertRaises(TypeError):
            AgentReputationTracker.from_dict(bool_obs)

        # Invariant breach: (alpha - prior_alpha) + (beta - prior_beta) != observations
        invariant_breach = dict(valid_dict, state={"a1|R01": {"alpha": 2.0, "beta": 1.0, "observations": 500}})
        with self.assertRaises(ValueError):
            AgentReputationTracker.from_dict(invariant_breach)

        # Missing state entry for registered pair
        two_agent_dict = AgentReputationTracker(agent_ids=["a1", "a2"], regimes=["R01"]).to_dict()
        del two_agent_dict["state"]["a2|R01"]
        with self.assertRaises(ValueError):
            AgentReputationTracker.from_dict(two_agent_dict)

    def test_fractional_posterior_deltas_rejected(self):
        """Verify that fractional posterior parameter deltas (e.g. alpha=1.5, beta=1.5, obs=1) are rejected."""
        tracker = AgentReputationTracker(agent_ids=["a1"], regimes=["R01"], prior_alpha=1.0, prior_beta=1.0)
        state = tracker.to_dict()
        state["state"]["a1|R01"] = {
            "alpha": 1.5,
            "beta": 1.5,
            "observations": 1,
        }
        with self.assertRaises(ValueError):
            AgentReputationTracker.from_dict(state)

    def test_agent_id_delimiter_round_trip_rejected(self):
        """Verify that agent IDs containing the state-key delimiter '|' are rejected at initialization."""
        with self.assertRaises(ValueError):
            AgentReputationTracker(agent_ids=["agent|1"], regimes=["R01"])

    def test_unexpected_top_level_and_entry_fields_rejected(self):
        """Verify closed-schema validation rejects unknown top-level or per-state entry keys."""
        tracker = AgentReputationTracker(agent_ids=["a1"], regimes=["R01"])
        valid_dict = tracker.to_dict()

        # Extra top-level field
        extra_top = dict(valid_dict, unknown_field="malicious")
        with self.assertRaises(ValueError):
            AgentReputationTracker.from_dict(extra_top)

        # Extra entry field in state map
        extra_entry = dict(valid_dict)
        extra_entry["state"] = {
            "a1|R01": {"alpha": 1.0, "beta": 1.0, "observations": 0, "extra_key": 123}
        }
        with self.assertRaises(ValueError):
            AgentReputationTracker.from_dict(extra_entry)

    def test_parameter_upper_bound_and_overflow_protection(self):
        """Verify parameters exceeding MAX_PARAM_VALUE (1e12) are rejected at init and update."""
        with self.assertRaises(ValueError):
            AgentReputationTracker(agent_ids=["a1"], regimes=["R01"], prior_alpha=1e13)

        tracker = AgentReputationTracker(agent_ids=["a1"], regimes=["R01"], prior_alpha=1e12, prior_beta=1.0)
        with self.assertRaises(OverflowError):
            tracker.record_outcome("a1", "R01", was_correct=True)

    def test_max_param_value_is_implementation_boundary_not_mathematical_requirement(self):
        """Verify MAX_PARAM_VALUE = 1e12 is an explicit implementation constraint.

        The authoritative Beta-Bernoulli reputation model defines:
            α = α_0 + k
            β = β_0 + (n - k)
        with no upper bound. The 1e12 cap is an engineering safeguard for float64 stability,
        not a mathematical requirement of the model. This test proves that mathematically
        valid states beyond 1e12 are rejected solely due to implementation limits.
        """
        # 1e12 is accepted (at implementation capacity boundary)
        tracker_at_cap = AgentReputationTracker(
            agent_ids=["a1"], regimes=["R01"],
            prior_alpha=1e12, prior_beta=1.0
        )
        self.assertAlmostEqual(tracker_at_cap.get_reputation_weight("a1", "R01"), 1.0, places=6)

        # 1e12 + 1 is rejected by the implementation cap, not model invalidity
        with self.assertRaises(ValueError):
            AgentReputationTracker(
                agent_ids=["a1"], regimes=["R01"],
                prior_alpha=1e12 + 1, prior_beta=1.0
            )

        # Similarly, a state with alpha > 1e12 is rejected during deserialization
        valid_dict = tracker_at_cap.to_dict()
        over_capacity_dict = dict(valid_dict)
        over_capacity_dict["state"]["a1|R01"] = {
            "alpha": 1e12 + 1.0,
            "beta": 1.0,
            "observations": 1e12 + 1.0,
        }
        with self.assertRaises(ValueError):
            AgentReputationTracker.from_dict(over_capacity_dict)

    def test_pre_mutation_failure_preserves_state(self):
        """Verify that a failed update (OverflowError) leaves internal state untouched."""
        tracker = AgentReputationTracker(agent_ids=["a1"], regimes=["R01"], prior_alpha=1.0, prior_beta=1.0)
        key = ("a1", "R01")
        tracker._alpha[key] = 1e12
        tracker._beta[key] = 5.0
        tracker._observations[key] = 10

        with self.assertRaises(OverflowError):
            tracker.record_outcome("a1", "R01", was_correct=True)

        # Assert no state variables were modified
        self.assertEqual(tracker._alpha[key], 1e12)
        self.assertEqual(tracker._beta[key], 5.0)
        self.assertEqual(tracker._observations[key], 10)

    def test_exact_capacity_boundary(self):
        """Verify MAX - 1.0 -> MAX succeeds, and the next update at MAX fails."""
        tracker = AgentReputationTracker(agent_ids=["a1"], regimes=["R01"], prior_alpha=1.0, prior_beta=1.0)
        key = ("a1", "R01")
        tracker._alpha[key] = 1e12 - 1.0

        # MAX - 1 -> MAX transition succeeds
        tracker.record_outcome("a1", "R01", was_correct=True)
        self.assertEqual(tracker._alpha[key], 1e12)

        # Update at MAX fails
        with self.assertRaises(OverflowError):
            tracker.record_outcome("a1", "R01", was_correct=True)

    def test_defensive_getters_against_corrupted_internal_state(self):
        """Verify defensive getters raise ValueError when internal state contains corrupted values."""
        tracker = AgentReputationTracker(agent_ids=["a1"], regimes=["R01"])
        key = ("a1", "R01")

        for corrupted_val in [float("nan"), float("inf"), 0.0, -1.0]:
            tracker._alpha[key] = corrupted_val
            tracker._beta[key] = 1.0
            with self.assertRaises(ValueError):
                tracker.get_reputation_weight("a1", "R01")
            with self.assertRaises(ValueError):
                tracker.get_posterior_parameters("a1", "R01")
            with self.assertRaises(ValueError):
                tracker.get_posterior_variance("a1", "R01")

        tracker._alpha[key] = 1.0
        for corrupted_obs in [True, False, -1, -10]:
            tracker._observations[key] = corrupted_obs
            with self.assertRaises(ValueError):
                tracker.get_observation_count("a1", "R01")

    def test_huge_integer_overflow_conversion(self):
        """Verify huge integer inputs causing OverflowError during float conversion raise ValueError."""
        huge_int = 10**1000
        with self.assertRaises(ValueError):
            AgentReputationTracker(agent_ids=["a1"], regimes=["R01"], prior_alpha=huge_int)

    def test_non_canonical_state_keys_rejected(self):
        """Verify from_dict rejects state keys with non-canonical whitespace around delimiter or regime."""
        tracker = AgentReputationTracker(agent_ids=["a1"], regimes=["R01"])
        valid_dict = tracker.to_dict()

        for bad_key in ["a1 | R01 ", "a1| R01", "a1 |R01", " a1 | R01 "]:
            bad_dict = dict(valid_dict)
            bad_dict["state"] = {
                bad_key: {"alpha": 1.0, "beta": 1.0, "observations": 0}
            }
            with self.assertRaises(ValueError):
                AgentReputationTracker.from_dict(bad_dict)

    def test_adversarial_extreme_valid_parameters(self):
        """Verify that mathematically valid extreme parameters do not trigger spurious boundary rejection."""
        # alpha at capacity, beta at minimum
        tracker_max_alpha = AgentReputationTracker(
            agent_ids=["a1"], regimes=["R01"],
            prior_alpha=1e12, prior_beta=1.0
        )
        # Weight should be extremely close to 1.0 but strictly less
        w = tracker_max_alpha.get_reputation_weight("a1", "R01")
        self.assertGreaterEqual(w, 0.999999999998)
        self.assertLess(w, 1.0)

        # beta at capacity, alpha at minimum
        tracker_max_beta = AgentReputationTracker(
            agent_ids=["a1"], regimes=["R01"],
            prior_alpha=1.0, prior_beta=1e12
        )
        w2 = tracker_max_beta.get_reputation_weight("a1", "R01")
        self.assertLessEqual(w2, 0.000000000002)
        self.assertGreater(w2, 0.0)

        # Both at capacity -> weight = 0.5 exactly
        tracker_both_max = AgentReputationTracker(
            agent_ids=["a1"], regimes=["R01"],
            prior_alpha=1e12, prior_beta=1e12
        )
        w3 = tracker_both_max.get_reputation_weight("a1", "R01")
        self.assertAlmostEqual(w3, 0.5, places=6)

    def test_all_12_regimes_tracked(self):
        """Verify tracker correctly handles all 12 canonical regimes."""
        regimes = [f"R{i:02d}" for i in range(1, 13)]
        tracker = AgentReputationTracker(agent_ids=["a1"], regimes=regimes)

        # Record one correct outcome per regime
        for r in regimes:
            tracker.record_outcome("a1", r, was_correct=True)

        # Each regime should have weight 2/3
        for r in regimes:
            self.assertAlmostEqual(tracker.get_reputation_weight("a1", r), 2.0 / 3.0)
            self.assertEqual(tracker.get_observation_count("a1", r), 1)

    def test_maximum_valid_state(self):
        """Verify tracker operates correctly at MAX_PARAM_VALUE capacity boundary."""
        tracker = AgentReputationTracker(
            agent_ids=["a1"], regimes=["R01"],
            prior_alpha=1e12, prior_beta=1e12
        )
        params = tracker.get_posterior_parameters("a1", "R01")
        self.assertEqual(params["alpha"], 1e12)
        self.assertEqual(params["beta"], 1e12)
        self.assertAlmostEqual(tracker.get_reputation_weight("a1", "R01"), 0.5)

        # Serialize and deserialize at max state
        state_dict = tracker.to_dict()
        restored = AgentReputationTracker.from_dict(state_dict)
        self.assertAlmostEqual(restored.get_reputation_weight("a1", "R01"), 0.5)

    def test_asymmetric_extreme_parameters(self):
        """Verify asymmetric extreme alpha/beta values produce correct weights and variances."""
        tracker = AgentReputationTracker(
            agent_ids=["a1"], regimes=["R01"],
            prior_alpha=1e12, prior_beta=1.0
        )
        # Weight should be near 1.0
        w = tracker.get_reputation_weight("a1", "R01")
        self.assertGreaterEqual(w, 0.999999999998)
        self.assertLess(w, 1.0)

        # Variance for Beta(1e12, 1) should be extremely small
        var = tracker.get_posterior_variance("a1", "R01")
        self.assertGreater(var, 0.0)
        self.assertLess(var, 1e-12)

    def test_long_run_persistence(self):
        """Verify weights remain stable and bounded after many sequential updates."""
        tracker = AgentReputationTracker(agent_ids=["a1", "a2"], regimes=["R01"])
        for _ in range(500):
            tracker.record_outcome("a1", "R01", was_correct=True)
            tracker.record_outcome("a2", "R01", was_correct=False)

        w1 = tracker.get_reputation_weight("a1", "R01")
        w2 = tracker.get_reputation_weight("a2", "R01")
        self.assertGreater(w1, 0.99)
        self.assertLess(w2, 0.01)
        self.assertAlmostEqual(w1 + w2, 1.0, places=6)

    def test_sequence_order_equivalence(self):
        """Verify that recording outcomes in different orders produces identical final state."""
        regimes = ["R01", "R02", "R03"]
        agents = ["a1", "a2"]

        # Order 1: regime-major
        tracker1 = AgentReputationTracker(agent_ids=agents, regimes=regimes)
        for r in regimes:
            for a in agents:
                tracker1.record_outcome(a, r, was_correct=(a == "a1"))

        # Order 2: agent-major
        tracker2 = AgentReputationTracker(agent_ids=agents, regimes=regimes)
        for a in agents:
            for r in regimes:
                tracker2.record_outcome(a, r, was_correct=(a == "a1"))

        # Final states must be identical
        for a in agents:
            for r in regimes:
                self.assertAlmostEqual(
                    tracker1.get_reputation_weight(a, r),
                    tracker2.get_reputation_weight(a, r)
                )
                self.assertEqual(
                    tracker1.get_observation_count(a, r),
                    tracker2.get_observation_count(a, r)
                )

    def test_kalman_gain_propagates_from_reputation_change(self):
        """Verify that reputation weight changes propagate to investment Kalman gain."""
        agent_ids = ["a1", "a2"]
        tracker = AgentReputationTracker(agent_ids=agent_ids, regimes=["R01"])

        def compute_k(weights):
            a1 = AgentOutput(s=1.0, c=0.8, u=0.0, d=0.0, p_plus=0.8, p_minus=0.0, delta_t=1.0, r=0.1, agent_id="a1")
            a2 = AgentOutput(s=-1.0, c=0.8, u=0.0, d=0.0, p_plus=0.0, p_minus=0.8, delta_t=1.0, r=0.1, agent_id="a2")
            s = compute_ensemble_signal([a1, a2], weights)
            d = compute_disagreement([a1, a2], weights, s)
            c = compute_effective_confidence([a1, a2], weights)
            return compute_investment_kalman_gain(prediction_covariance=1.0, effective_confidence=c, disagreement=d)

        k_before = compute_k(tracker.get_all_weights("R01"))

        for _ in range(20):
            tracker.record_outcome("a1", "R01", was_correct=True)

        k_after = compute_k(tracker.get_all_weights("R01"))

        # Kalman gain must change when reputation weights change
        self.assertNotAlmostEqual(k_after, k_before)
        # More confidence should increase Kalman gain
        self.assertGreater(k_after, k_before)

    def test_capital_gate_effect_from_reputation_change(self):
        """Verify that reputation changes can alter the capital gate verdict."""
        agent_ids = ["a1", "a2", "a3", "a4", "a5", "a6", "a7"]
        tracker = AgentReputationTracker(agent_ids=agent_ids, regimes=["R01"])

        def get_verdict(weights):
            agents = [
                AgentOutput(s=1.0, c=0.1, u=0.0, d=0.0, p_plus=0.9, p_minus=0.1, delta_t=1.0, r=0.1, agent_id="a1"),
                AgentOutput(s=-1.0, c=0.1, u=0.0, d=0.0, p_plus=0.1, p_minus=0.9, delta_t=1.0, r=0.1, agent_id="a2"),
                AgentOutput(s=0.0, c=0.9, u=0.0, d=0.0, p_plus=0.5, p_minus=0.5, delta_t=1.0, r=0.1, agent_id="a3"),
                AgentOutput(s=0.0, c=0.9, u=0.0, d=0.0, p_plus=0.5, p_minus=0.5, delta_t=1.0, r=0.1, agent_id="a4"),
                AgentOutput(s=0.0, c=0.9, u=0.0, d=0.0, p_plus=0.5, p_minus=0.5, delta_t=1.0, r=0.1, agent_id="a5"),
                AgentOutput(s=0.0, c=0.9, u=0.0, d=0.0, p_plus=0.5, p_minus=0.5, delta_t=1.0, r=0.1, agent_id="a6"),
                AgentOutput(s=0.0, c=0.9, u=0.0, d=0.0, p_plus=0.5, p_minus=0.5, delta_t=1.0, r=0.1, agent_id="a7"),
            ]
            k_state = KalmanState(
                estimated_price=100.0, trend=0.0, uncertainty=1.0,
                trend_uncertainty=0.1, price_variance=1.0, trend_variance=0.01,
                innovation=0.2, kalman_gain_price=0.8,
            )
            return evaluate(
                kalman_state=k_state,
                states=SevenStateVector(
                    economic=0.8, financial=0.8, fiscal=0.8,
                    portfolio=0.8, fundamental=0.8, market=0.8, sector=0.8,
                ),
                portfolio_context={
                    "position_pct": 0.05, "gross_leverage": 0.5,
                    "entropy": 0.5, "drawdown_pct": 0.02,
                    "execution_timeout_seconds": 5.0, "is_new_long": True,
                    "sector_exposure_pct": 0.15, "regime": "R01",
                    "available_liquidity": 100000.0,
                },
                agents=agents,
                agent_weights=weights,
            ).verdict

        verdict_before = get_verdict(tracker.get_all_weights("R01"))

        # Drastically shift weights: a1 and a2 (low confidence) become high reputation
        for _ in range(100):
            tracker.record_outcome("a1", "R01", was_correct=True)
            tracker.record_outcome("a2", "R01", was_correct=True)
            for i in range(3, 8):
                tracker.record_outcome(f"a{i}", "R01", was_correct=False)

        verdict_after = get_verdict(tracker.get_all_weights("R01"))

        # With drastically changed weights, effective confidence drops and disagreement rises,
        # triggering ECONF-001 and/or DISAG-001, changing verdict from ALLOW to REDUCE
        self.assertNotEqual(verdict_after, verdict_before)

    def test_scalar_prior_round_trip(self):
        """Verify scalar prior serializes and deserializes faithfully."""
        tracker = AgentReputationTracker(
            agent_ids=["a1", "a2"],
            regimes=["R01", "R02"],
            prior_alpha=2.0,
            prior_beta=3.0,
        )
        tracker.record_outcome("a1", "R01", was_correct=True)
        tracker.record_outcome("a2", "R02", was_correct=False)

        state = tracker.to_dict()
        restored = AgentReputationTracker.from_dict(state)

        self.assertEqual(state["prior_alpha"], 2.0)
        self.assertEqual(state["prior_beta"], 3.0)
        # a1/R01: alpha=3, beta=3 -> weight = 3/6 = 0.5
        self.assertAlmostEqual(restored.get_reputation_weight("a1", "R01"), 0.5)
        # a2/R02: alpha=2, beta=4 -> weight = 2/6 = 1/3
        self.assertAlmostEqual(restored.get_reputation_weight("a2", "R02"), 1.0 / 3.0)

    def test_dict_prior_by_pair_round_trip(self):
        """Verify dictionary prior keyed by (agent_id, regime) serializes and round-trips."""
        tracker = AgentReputationTracker(
            agent_ids=["a1", "a2"],
            regimes=["R01", "R02"],
            prior_alpha={("a1", "R01"): 5.0, ("a1", "R02"): 5.0, ("a2", "R01"): 2.0, ("a2", "R02"): 2.0},
            prior_beta={("a1", "R01"): 2.0, ("a1", "R02"): 2.0, ("a2", "R01"): 4.0, ("a2", "R02"): 4.0},
        )

        state = tracker.to_dict()
        self.assertIsInstance(state["prior_alpha"], dict)
        self.assertIsInstance(state["prior_beta"], dict)
        self.assertEqual(state["prior_alpha"]["a1|R01"], 5.0)
        self.assertEqual(state["prior_beta"]["a1|R01"], 2.0)

        restored = AgentReputationTracker.from_dict(state)
        # a1/R01: alpha=5, beta=2 -> weight = 5/7
        self.assertAlmostEqual(restored.get_reputation_weight("a1", "R01"), 5.0 / 7.0)
        # a2/R02: alpha=2, beta=4 -> weight = 2/6 = 1/3
        self.assertAlmostEqual(restored.get_reputation_weight("a2", "R02"), 1.0 / 3.0)

    def test_dict_prior_by_agent_round_trip(self):
        """Verify dictionary prior keyed by agent_id applies to all regimes for that agent."""
        tracker = AgentReputationTracker(
            agent_ids=["a1", "a2"],
            regimes=["R01", "R02"],
            prior_alpha={"a1": 5.0, "a2": 2.0},
            prior_beta={"a1": 2.0, "a2": 4.0},
        )

        state = tracker.to_dict()
        restored = AgentReputationTracker.from_dict(state)

        # a1 gets prior 5,2 in both regimes -> weight = 5/7
        self.assertAlmostEqual(restored.get_reputation_weight("a1", "R01"), 5.0 / 7.0)
        self.assertAlmostEqual(restored.get_reputation_weight("a1", "R02"), 5.0 / 7.0)
        # a2 gets prior 2,4 in both regimes -> weight = 2/6 = 1/3
        self.assertAlmostEqual(restored.get_reputation_weight("a2", "R01"), 1.0 / 3.0)
        self.assertAlmostEqual(restored.get_reputation_weight("a2", "R02"), 1.0 / 3.0)

    def test_dict_prior_missing_key_fails_closed(self):
        """Verify dictionary prior missing a required (agent, regime) pair raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            AgentReputationTracker(
                agent_ids=["a1", "a2"],
                regimes=["R01"],
                prior_alpha={("a1", "R01"): 5.0},
                prior_beta={("a1", "R01"): 2.0},
            )
        self.assertIn("missing required entry", str(ctx.exception))

    def test_dict_prior_after_observations_round_trip(self):
        """Verify dictionary prior with observations serializes and round-trips correctly."""
        tracker = AgentReputationTracker(
            agent_ids=["a1"],
            regimes=["R01", "R02"],
            prior_alpha={("a1", "R01"): 5.0, ("a1", "R02"): 3.0},
            prior_beta={("a1", "R01"): 2.0, ("a1", "R02"): 1.0},
        )
        tracker.record_outcome("a1", "R01", was_correct=True)
        tracker.record_outcome("a1", "R01", was_correct=True)
        tracker.record_outcome("a1", "R02", was_correct=False)

        state = tracker.to_dict()
        restored = AgentReputationTracker.from_dict(state)

        # a1/R01: alpha=7, beta=2 -> weight = 7/9
        self.assertAlmostEqual(restored.get_reputation_weight("a1", "R01"), 7.0 / 9.0)
        # a1/R02: alpha=3, beta=2 -> weight = 3/5
        self.assertAlmostEqual(restored.get_reputation_weight("a1", "R02"), 3.0 / 5.0)
        self.assertEqual(restored.get_observation_count("a1", "R01"), 2)
        self.assertEqual(restored.get_observation_count("a1", "R02"), 1)

    def test_dict_prior_invariant_validation_rejects_non_integer_deltas(self):
        """Verify from_dict rejects state where deltas from dict prior are not integer steps."""
        tracker = AgentReputationTracker(
            agent_ids=["a1"],
            regimes=["R01"],
            prior_alpha={("a1", "R01"): 5.0},
            prior_beta={("a1", "R01"): 2.0},
        )
        valid_dict = tracker.to_dict()

        # Corrupt alpha to be non-integer step from prior
        bad_dict = dict(valid_dict)
        bad_dict["state"]["a1|R01"] = {
            "alpha": 5.5,
            "beta": 2.0,
            "observations": 0,
        }
        with self.assertRaises(ValueError):
            AgentReputationTracker.from_dict(bad_dict)

    def test_dict_prior_extra_keys_ignored(self):
        """Verify extra keys in dictionary prior that don't match registered pairs are ignored."""
        tracker = AgentReputationTracker(
            agent_ids=["a1"],
            regimes=["R01"],
            prior_alpha={("a1", "R01"): 5.0, ("ghost", "R99"): 10.0},
            prior_beta={("a1", "R01"): 2.0, ("ghost", "R99"): 3.0},
        )
        # Extra keys for unregistered pairs should be silently ignored
        self.assertAlmostEqual(tracker.get_reputation_weight("a1", "R01"), 5.0 / 7.0)

        # But serialization should only include the actual prior for registered pairs
        state = tracker.to_dict()
        restored = AgentReputationTracker.from_dict(state)
        self.assertAlmostEqual(restored.get_reputation_weight("a1", "R01"), 5.0 / 7.0)

    def test_dict_prior_defensive_copy_prevents_external_mutation(self):
        """Verify that mutating the caller's dictionary after construction does not affect tracker serialization."""
        prior_alpha = {("a1", "R01"): 5.0}
        prior_beta = {("a1", "R01"): 2.0}

        tracker = AgentReputationTracker(
            agent_ids=["a1"],
            regimes=["R01"],
            prior_alpha=prior_alpha,
            prior_beta=prior_beta,
        )

        # Mutate caller's dictionaries after construction
        prior_alpha[("a1", "R01")] = 999.0
        prior_beta[("a1", "R01")] = 888.0

        # Serialized state must still reflect the original values, not the mutated ones
        state = tracker.to_dict()
        self.assertEqual(state["prior_alpha"]["a1|R01"], 5.0)
        self.assertEqual(state["prior_beta"]["a1|R01"], 2.0)

        # Tracker's internal state must also remain unchanged
        self.assertAlmostEqual(tracker.get_reputation_weight("a1", "R01"), 5.0 / 7.0)


if __name__ == "__main__":
    unittest.main()

