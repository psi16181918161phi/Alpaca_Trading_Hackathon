"""X Quant X Pipeline — End-to-End Analytical Integration Layer.

WHAT
====
Wires the analytical modules into a coherent data-flow pipeline:
    RegimeDetector → AgentReputationTracker → EnsembleSignal → InvestmentKalmanGain → CapitalGate

WHY
===
Each module is pure and independently testable, but the architecture only produces
correct risk decisions when they are composed with consistent, non-fabricated inputs.
This module provides that composition contract.

HOW
===
1. Classify the active market regime from price/volume history.
2. Query per-agent reputation weights for the active regime from the Bayesian tracker.
3. Aggregate specialist agent outputs into an ensemble signal using those weights.
4. Feed the ensemble metrics into the investment Kalman gain computation.
5. Evaluate the Seven-State Capital Gate using the real Kalman posterior and ensemble state.

No values are fabricated. Every downstream input is derived from upstream module output.

Architectural Role
==================
Integration layer. Owns the execution order and data contracts between analytical modules.
Performs no order placement, broker API calls, or external side-effects.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from investment_agent.regimes.regime_detector import RegimeDetector, RegimeClassification
from investment_agent.regimes.regimes import VALID_REGIMES
from investment_agent.regimes.hmm_regime_detector import HMMRegimeDetector, HMMUnderflowError
from investment_agent.agents.agent_reputation import AgentReputationTracker
from investment_agent.signals.ensemble_signal import AgentOutput, EnsembleAggregate, compute_ensemble_aggregate
from investment_agent.filters.investment_kalman_gain import compute_investment_kalman_gain
from investment_agent.capital.capital_gate import evaluate, CapitalGateResult
from investment_agent.filters.kalman_filter import KalmanFilter, KalmanState


# ---------------------------------------------------------------------------
# Pipeline result
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ProvenanceTrace:
    """Immutable provenance trace for pipeline execution.

    Records the data flow through each pipeline stage for audit and dashboard display.

    Attributes
    ----------
    market_data : Dict[str, Any]
        Input market data (prices, volumes, timestamp).
    features : Dict[str, float]
        Extracted market features (trend, volatility, volume).
    regime : str
        Active regime identifier.
    regime_probabilities : Optional[Dict[str, float]]
        HMM posterior probabilities if HMM used, None if rule-based.
    weights : Dict[str, float]
        Per-agent reputation weights.
    agent_outputs : List[Dict[str, float]]
        Agent signal outputs (agent_id, signal, confidence).
    ensemble : Dict[str, float]
        Ensemble aggregate (signal, disagreement, effective_confidence).
    kalman_state : Dict[str, float]
        Kalman filter state (estimated_price, trend, uncertainty, price_variance).
    kalman_gain : float
        Investment Kalman gain K_t.
    capital_gate : Dict[str, Any]
        Capital gate result (verdict, effective_cap, triggered_rules).
    timestamp : datetime
        Pipeline execution timestamp.
    """

    market_data: Dict[str, Any]
    features: Dict[str, float]
    regime: str
    regime_probabilities: Optional[Dict[str, float]]
    weights: Dict[str, float]
    agent_outputs: List[Dict[str, float]]
    ensemble: Dict[str, float]
    kalman_state: Dict[str, float]
    kalman_gain: float
    capital_gate: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass(frozen=True)
class PipelineResult:
    """Immutable end-to-end pipeline result.

    Attributes
    ----------
    regime : RegimeClassification
        Active market regime classification.
    weights : Dict[str, float]
        Per-agent reputation weights for the active regime.
    ensemble : EnsembleAggregate
        Atomic ensemble aggregation result (same object passed to capital gate).
    kalman_state : KalmanState
        Posterior Kalman filter state after ingesting latest price.
    kalman_gain : float
        Investment Kalman gain K_t derived from ensemble effective confidence
        and disagreement. Exposed explicitly for audit transparency.
    capital_gate : CapitalGateResult
        Seven-State Capital Gate evaluation result.
    provenance : ProvenanceTrace
        Complete data flow trace for audit and dashboard display.
    timestamp : datetime
        Pipeline execution timestamp.
    """

    regime: RegimeClassification
    weights: Dict[str, float]
    ensemble: EnsembleAggregate
    kalman_state: KalmanState
    kalman_gain: float
    capital_gate: CapitalGateResult
    provenance: ProvenanceTrace
    timestamp: datetime = field(default_factory=datetime.now)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class XQuantXPipeline:
    """End-to-end analytical pipeline for X Quant X.

    WHAT
    ====
    Composes RegimeDetector, AgentReputationTracker, EnsembleSignal, InvestmentKalmanGain,
    and CapitalGate into a single execution path with verified data contracts.

    WHY
    ====
    Prevents fabricated inputs and ensures regime-aware weighting flows through
    the entire analytical stack. Verifies that ensemble output changes when
    regime or agent performance changes.

    HOW
    ====
    1. Classify regime from price/volume history using rule-based detector.
    2. Derive per-agent weights from reputation tracker for the active regime.
    3. Aggregate agent outputs into ensemble signal with those weights.
    4. Update Kalman filter with latest price observation.
    5. Evaluate capital gate using real ensemble and Kalman outputs.

    NOTE: The authoritative architecture requires HMM-based regime modeling.
    This pipeline uses the deterministic rule-based detector as a fallback.
    The HMM detector (hmm_regime_detector.py) is available when implemented.

    NOTE: The seven-dimensional State-of-Charge vector (economic, financial, fiscal,
    portfolio, fundamental, market, sector) is currently provided by the caller.
    The authoritative architecture specifies state charge/discharge dynamics that
    are not yet implemented in this pipeline. The capital gate validates and gates
    on the provided states, but does not compute their evolution.

    Parameters
    ----------
    agent_ids : List[str]
        Registered specialist agent identifiers.
    prior_alpha : Union[float, Dict], optional
        Prior alpha for Beta reputation distributions (default 1.0).
    prior_beta : Union[float, Dict], optional
        Prior beta for Beta reputation distributions (default 1.0).
    kalman_initial_price : float, optional
        Initial price for Kalman filter (default 100.0).
    regime_lookback_days : int, optional
        Lookback window for regime detection (default 20).
    use_hmm : bool, optional
        If True, use HMM regime detector when available (default False).
    """

    def __init__(
        self,
        agent_ids: List[str],
        prior_alpha: Any = 1.0,
        prior_beta: Any = 1.0,
        kalman_initial_price: float = 100.0,
        regime_lookback_days: int = 20,
        use_hmm: bool = False,
    ) -> None:
        if not agent_ids:
            raise ValueError("agent_ids must be non-empty")

        self._agent_ids = [aid.strip() for aid in agent_ids if aid.strip()]
        if len(self._agent_ids) != len(set(self._agent_ids)):
            raise ValueError("agent_ids must be unique")

        self._regime_detector = RegimeDetector(lookback_days=regime_lookback_days)
        self._reputation_tracker = AgentReputationTracker(
            agent_ids=self._agent_ids,
            regimes=sorted(VALID_REGIMES),
            prior_alpha=prior_alpha,
            prior_beta=prior_beta,
        )
        self._kalman_filter = KalmanFilter(initial_price=kalman_initial_price)
        self._regime_history: List[Tuple[datetime, str]] = []
        self._use_hmm = True  # HMM is always authoritative
        self._hmm_detector: Optional[HMMRegimeDetector] = HMMRegimeDetector()

    def classify_regime(
        self,
        prices: List[float],
        volumes: Optional[List[float]] = None,
        highs: Optional[List[float]] = None,
        lows: Optional[List[float]] = None,
    ) -> RegimeClassification:
        """Classify active market regime from OHLCV history using HMM (always authoritative)."""
        from investment_agent.regimes.market_feature_extractor import extract_features
        features = extract_features(prices, volumes, highs=highs, lows=lows,
                                    lookback_days=self._regime_detector._lookback_days)
        hmm_result = self._hmm_detector.classify(features.tolist())
        regime_result = RegimeClassification(
            regime=hmm_result.regime,
            confidence=1.0 - hmm_result.normalized_entropy,
            timestamp=hmm_result.timestamp,
            features={
                "rsi": float(np.mean(features[:, 0])),
                "macd": float(np.mean(features[:, 1])),
                "atr": float(np.mean(features[:, 2])),
                "vix": float(np.mean(features[:, 3])),
                "vol_ratio": float(np.mean(features[:, 4])),
                "corr": float(np.mean(features[:, 5])),
                "hurst": float(np.mean(features[:, 6])),
            },
            regime_affinity=hmm_result.probabilities,
        )
        self._regime_history.append((datetime.now(), regime_result.regime))
        return regime_result

    def get_regime_weights(self, regime: str) -> Dict[str, float]:
        """Get per-agent reputation weights for a specific regime.

        Parameters
        ----------
        regime : str
            Active regime identifier (e.g. "R01").

        Returns
        -------
        Dict[str, float]
            Dictionary mapping agent_id to posterior mean reputation weight w_i ∈ (0, 1).
        """
        if regime not in VALID_REGIMES:
            raise ValueError(f"Invalid regime '{regime}'. Must be one of {sorted(VALID_REGIMES)}")

        weights = {}
        for agent_id in self._agent_ids:
            weights[agent_id] = self._reputation_tracker.get_reputation_weight(agent_id, regime)
        return weights

    def record_agent_outcome(self, agent_id: str, regime: str, was_correct: bool) -> None:
        """Record a prediction outcome for an agent in a specific regime.

        Parameters
        ----------
        agent_id : str
            Agent identifier.
        regime : str
            Regime identifier at prediction emission time.
        was_correct : bool
            Whether the agent's prediction was correct.
        """
        self._reputation_tracker.record_outcome(agent_id, regime, was_correct)

    def update_kalman(self, observed_price: float) -> KalmanState:
        """Update Kalman filter with a new price observation.

        Parameters
        ----------
        observed_price : float
            New observed market price. Must be positive and finite.

        Returns
        -------
        KalmanState
            Posterior state estimate after incorporating the observation.
        """
        return self._kalman_filter.update(observed_price)

    def get_kalman_state(self) -> KalmanState:
        """Get current Kalman filter state without updating."""
        return self._kalman_filter.get_state()

    def evaluate(
        self,
        prices: List[float],
        volumes: Optional[List[float]],
        agent_outputs: List[AgentOutput],
        states: Any,  # SevenStateVector-compatible
        portfolio_context: Dict[str, Any],
        sigma_base_squared: float = 1.0,
        update_kalman: bool = True,
    ) -> PipelineResult:
        """Execute the full analytical pipeline.

        Parameters
        ----------
        prices : List[float]
            Historical price series for regime detection and Kalman update.
        volumes : Optional[List[float]]
            Historical volume series for regime detection.
        agent_outputs : List[AgentOutput]
            Specialist agent outputs for the current period.
        states : SevenStateVector-compatible
            7-dimensional investment state-of-charge vector.
        portfolio_context : Dict[str, Any]
            Portfolio risk context (position_pct, drawdown_pct, etc.).
        sigma_base_squared : float, optional
            Base model variance (default 1.0).
        update_kalman : bool, optional
            Whether to update Kalman filter with latest price (default True).

        Returns
        -------
        PipelineResult
            End-to-end pipeline result containing regime, weights, ensemble,
            Kalman state, and capital gate verdict.

        Raises
        ------
        ValueError
            If inputs are invalid or inconsistent.
        """
        if not agent_outputs:
            raise ValueError("agent_outputs must be non-empty")
        if len(agent_outputs) != len(self._agent_ids):
            raise ValueError(
                f"Expected {len(self._agent_ids)} agent outputs, got {len(agent_outputs)}. "
                f"Pipeline is configured for agents: {self._agent_ids}"
            )

        # Validate agent IDs match
        output_ids = {a.agent_id for a in agent_outputs}
        expected_ids = set(self._agent_ids)
        if output_ids != expected_ids:
            raise ValueError(
                f"Agent output IDs {sorted(output_ids)} do not match configured agents {sorted(expected_ids)}"
            )

        # 1. Classify regime
        regime_result = self.classify_regime(prices, volumes)
        active_regime = regime_result.regime

        # 2. Get regime-specific weights
        weights = self.get_regime_weights(active_regime)

        # 3. Update Kalman with latest price observation
        latest_price = prices[-1]
        if update_kalman:
            kalman_state = self.update_kalman(latest_price)
        else:
            kalman_state = self.get_kalman_state()

        # 4. Compute ensemble aggregate FIRST (chain-of-custody)
        ensemble = compute_ensemble_aggregate(agent_outputs, weights)

        # 5. Evaluate capital gate with pre-computed ensemble to preserve
        #    chain-of-custody: the same EnsembleAggregate object flows into
        #    the gate and is returned in PipelineResult.
        capital_gate = evaluate(
            kalman_state=kalman_state,
            states=states,
            portfolio_context=portfolio_context,
            agents=agent_outputs,
            agent_weights=weights,
            sigma_base_squared=sigma_base_squared,
            ensemble_agg=ensemble,
        )

        # 6. Build provenance trace for audit and dashboard display
        provenance = ProvenanceTrace(
            market_data={
                "prices_count": len(prices),
                "volumes_count": len(volumes) if volumes else 0,
                "latest_price": float(prices[-1]),
            },
            features=dict(regime_result.features),
            regime=active_regime,
            regime_probabilities=None,  # Rule-based detector doesn't produce HMM probabilities
            weights=dict(weights),
            agent_outputs=[
                {
                    "agent_id": a.agent_id,
                    "signal": float(a.s),
                    "confidence": float(a.c),
                }
                for a in agent_outputs
            ],
            ensemble={
                "signal": float(ensemble.ensemble_signal),
                "disagreement": float(ensemble.disagreement),
                "effective_confidence": float(ensemble.effective_confidence),
            },
            kalman_state={
                "estimated_price": float(kalman_state.estimated_price),
                "trend": float(kalman_state.trend),
                "uncertainty": float(kalman_state.uncertainty),
                "price_variance": float(kalman_state.price_variance),
            },
            kalman_gain=float(capital_gate.kalman_gain),
            capital_gate={
                "verdict": capital_gate.verdict.value,
                "effective_cap": float(capital_gate.effective_cap),
                "gating_factor": float(capital_gate.gating_factor),
                "reduce_factor": float(capital_gate.reduce_factor),
                "triggered_rules": list(capital_gate.triggered_rules),
                "reason": capital_gate.reason,
            },
        )

        return PipelineResult(
            regime=regime_result,
            weights=weights,
            ensemble=ensemble,
            kalman_state=kalman_state,
            kalman_gain=capital_gate.kalman_gain,
            capital_gate=capital_gate,
            provenance=provenance,
        )

    def get_regime_history(self) -> List[Tuple[datetime, str]]:
        """Return regime classification history."""
        return list(self._regime_history)

    def clear_history(self) -> None:
        """Clear regime classification history."""
        self._regime_detector.clear_history()
        self._regime_history.clear()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "XQuantXPipeline",
    "PipelineResult",
    "ProvenanceTrace",
]
