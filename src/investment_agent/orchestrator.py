"""X Quant X Orchestrator — Autonomous Trading Cycle.

WHAT
====
Implements the complete end-to-end trading cycle:

    Market data → Feature extraction → HMM regime → 7 specialist agents →
    Ensemble + disagreement → Kalman correction → Investment Kalman gain →
    7-state capital gate → Risk/circuit breakers → Options-aware decision →
    Alpaca order → Position/outcome tracking → Trade memory

WHY
===
Each module is independently testable, but the architecture only produces
correct trading decisions when composed as an autonomous cycle. This module
provides that composition with explicit data flow, error handling, and
provenance tracking.

HOW
===
1. Fetch market data (prices, volumes)
2. Extract features for HMM
3. Classify regime via HMM or rule-based detector
4. Run 7 specialist agents
5. Compute weighted ensemble signal with disagreement
6. Update Kalman filter with latest price
7. Compute investment Kalman gain
8. Evaluate 7-state capital gate
9. Apply risk circuit breakers
10. Make options-aware trading decision
11. Submit order via Alpaca
12. Record outcome in trade memory
13. Update agent reputations with outcome

Architectural Role
==================
Orchestration layer. Owns the execution cycle and data contracts between
all analytical and execution modules. Performs broker API calls and order
placement as its primary side-effect.
"""

from __future__ import annotations

import math
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from .regimes.regime_detector import RegimeDetector, RegimeClassification
from .regimes.regimes import VALID_REGIMES
from .regimes.hmm_regime_detector import HMMRegimeDetector, HMMUnderflowError
from .regimes.market_feature_extractor import extract_features
from .agents.agent_reputation import AgentReputationTracker
from .signals.ensemble_signal import AgentOutput, EnsembleAggregate, compute_ensemble_aggregate
from .filters.investment_kalman_gain import compute_investment_kalman_gain
from .filters.kalman_filter import KalmanFilter, KalmanState
from .capital.capital_gate import evaluate, CapitalGateResult, SevenStateVector
from .memory.trade_memory import (
    TradeMemory,
    TradeExperience,
    TradeLifecycle,
    DEFAULT_MEMORY_FILE,
)
from .execution.hedge_capital_bridge import evaluate_hedge_risk, HedgeRiskAssessment


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TradingDecision:
    """Immutable trading decision result.

    Attributes
    ----------
    decision_id : str
        Unique identifier for this decision (UUID4).
    action : str
        Trading action: BUY, SELL, HOLD, REDUCE, FLATTEN.
    symbol : str
        Trading symbol.
    quantity : float
        Position quantity (shares or contracts).
    confidence : float
        Decision confidence in [0.0, 1.0].
    reasoning : str
        Human-readable decision rationale.
    circuit_breakers : List[str]
        List of triggered circuit breaker rules.
    provenance : Dict[str, Any]
        Complete data flow trace for audit.
    """

    decision_id: str
    action: str
    symbol: str
    quantity: float
    confidence: float
    reasoning: str
    circuit_breakers: List[str]
    provenance: Dict[str, Any]


@dataclass(frozen=True)
class CycleResult:
    """Immutable result of a complete trading cycle.

    Attributes
    ----------
    decision_id : str
        Unique identifier linking this cycle to its decision and audit log.
    decision : TradingDecision
        Final trading decision.
    regime : RegimeClassification
        Market regime classification.
    weights : Dict[str, float]
        Per-agent reputation weights.
    ensemble : EnsembleAggregate
        Ensemble aggregation result.
    kalman_state : KalmanState
        Kalman filter state.
    capital_gate : CapitalGateResult
        Capital gate evaluation result.
    experience : TradeExperience
        Recorded trade experience.
    timestamp : datetime
        Cycle execution timestamp.
    """

    decision_id: str
    decision: TradingDecision
    regime: RegimeClassification
    weights: Dict[str, float]
    ensemble: EnsembleAggregate
    kalman_state: KalmanState
    capital_gate: CapitalGateResult
    experience: TradeExperience
    timestamp: datetime = field(default_factory=datetime.now)


# ---------------------------------------------------------------------------
# Audit / Event Log
# ---------------------------------------------------------------------------

AUDIT_LOG_FILE = "audit_log.jsonl"


@dataclass(frozen=True)
class AuditEvent:
    """Immutable audit event for decision/outcome tracking."""

    event_id: str
    event_type: str
    decision_id: Optional[str]
    timestamp: datetime
    symbol: str
    payload: Dict[str, Any]


class AuditLog:
    """Append-only audit log backed by a JSONL file.

    Provides:
    - best-effort atomic append writes (single ``os.write`` call)
    - event replay for deterministic testing
    - query by decision_id
    - persistent state: existing events are loaded on ``__init__`` so queries
      survive process restarts (P1-3).
    """

    def __init__(self, log_file: str = AUDIT_LOG_FILE) -> None:
        self._log_file = log_file
        self._events: List[AuditEvent] = []
        self._load()

    def _load(self) -> None:
        """Load existing events from disk on initialization (P1-3)."""
        import json

        if not os.path.exists(self._log_file):
            return
        try:
            with open(self._log_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        raw = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    self._events.append(AuditEvent(
                        event_id=raw.get("event_id", str(uuid.uuid4())),
                        event_type=raw.get("event_type", "UNKNOWN"),
                        decision_id=raw.get("decision_id"),
                        timestamp=datetime.fromisoformat(raw["timestamp"]) if raw.get("timestamp") else datetime.now(),
                        symbol=raw.get("symbol", ""),
                        payload=raw.get("payload", {}),
                    ))
        except OSError:
            return

    def record_decision(
        self,
        decision_id: str,
        symbol: str,
        payload: Dict[str, Any],
    ) -> AuditEvent:
        """Record a new decision event."""
        event = AuditEvent(
            event_id=str(uuid.uuid4()),
            event_type="DECISION",
            decision_id=decision_id,
            timestamp=datetime.now(),
            symbol=symbol,
            payload=payload,
        )
        self._events.append(event)
        self._append(event)
        return event

    def record_outcome(
        self,
        decision_id: str,
        symbol: str,
        payload: Dict[str, Any],
    ) -> AuditEvent:
        """Record an outcome event linked to a prior decision."""
        event = AuditEvent(
            event_id=str(uuid.uuid4()),
            event_type="OUTCOME",
            decision_id=decision_id,
            timestamp=datetime.now(),
            symbol=symbol,
            payload=payload,
        )
        self._events.append(event)
        self._append(event)
        return event

    def query_by_decision_id(self, decision_id: str) -> List[AuditEvent]:
        return [e for e in self._events if e.decision_id == decision_id]

    def _append(self, event: AuditEvent) -> None:
        """Best-effort atomic append: serialize JSON, then a single ``os.write``.

        Note: This is not safe for concurrent multi-process writers sharing the
        same file (P1-2). It is safe for a single writer with ``O_APPEND`` on
        POSIX, and within a single process on Windows. Concurrent writers must
        use external locking.
        """
        import json

        record = {
            "event_id": event.event_id,
            "event_type": event.event_type,
            "decision_id": event.decision_id,
            "timestamp": event.timestamp.isoformat(),
            "symbol": event.symbol,
            "payload": event.payload,
        }
        line = (json.dumps(record, default=str) + "\n").encode("utf-8")
        try:
            fd = os.open(self._log_file, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
            try:
                os.write(fd, line)
            finally:
                os.close(fd)
        except OSError:
            with open(self._log_file, "a", encoding="utf-8") as f:
                f.write(line.decode("utf-8"))


from .agents.reputation_persistence import load_reputation, save_reputation


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class XQuantXOrchestrator:
    """Autonomous trading cycle orchestrator for X Quant X.

    WHAT
    ====
    Executes the complete analytical and execution pipeline from market data
    to order placement, with explicit data flow and provenance tracking.

    WHY
    ====
    Prevents fabricated inputs and ensures regime-aware weighting flows through
    the entire stack. Provides deterministic, auditable trading decisions.

    HOW
    ====
    1. Receive market data (prices, volumes, symbol)
    2. Extract features and classify regime
    3. Generate agent outputs for the symbol
    4. Aggregate ensemble with reputation weights
    5. Update Kalman and compute investment gain
    6. Evaluate capital gate
    7. Apply risk circuit breakers
    8. Make options-aware trading decision
    9. Submit order via Alpaca (if enabled)
    10. Record experience in trade memory
    11. Update agent reputations with outcome

    The HMM is the *only* regime classifier. The rule-based detector is
    NOT used in the production path. `use_hmm` is retained as an accepted
    (but now ignored) kwarg for backward-compat; regime detection is always
    HMM-backed.

    Parameters
    ----------
    agent_ids : List[str]
        Registered specialist agent identifiers.
    symbol : str
        Trading symbol (e.g., "AAPL").
    use_hmm : bool, optional
        Deprecated — retained for backward-compat only. Ignored.
    enable_trading : bool, optional
        If True, actually submit orders via Alpaca (default False for safety).
    memory_file : str, optional
        Path to trade memory JSON file.
    reputation_file : str, optional
        Path to persistent reputation JSON file.
    """

    def __init__(
        self,
        agent_ids: List[str],
        symbol: str,
        use_hmm: bool = False,  # deprecated; ignored — HMM is always active
        enable_trading: bool = False,
        memory_file: str = DEFAULT_MEMORY_FILE,
        reputation_file: Optional[str] = "reputation_state.json",
        llm_provider: Optional[Any] = None,
    ) -> None:
        if not agent_ids:
            raise ValueError("agent_ids must be non-empty")
        if not symbol:
            raise ValueError("symbol must be non-empty")

        self._agent_ids = [aid.strip() for aid in agent_ids if aid.strip()]
        self._symbol = symbol
        self._enable_trading = enable_trading
        self._reputation_file = reputation_file

        # HMM is the single authoritative regime classifier.
        # The rule-based RegimeDetector is NOT used in the production path.
        self._hmm_detector = HMMRegimeDetector()

        # Restore persisted reputation if file exists, else initialize fresh tracker
        restored = load_reputation(reputation_file) if reputation_file else None
        if restored is not None:
            self._reputation_tracker = restored
        else:
            self._reputation_tracker = AgentReputationTracker(
                agent_ids=self._agent_ids,
                regimes=sorted(VALID_REGIMES),
                prior_alpha=1.0,
                prior_beta=1.0,
            )
        self._kalman_filter = KalmanFilter(initial_price=100.0)
        self._trade_memory = TradeMemory(memory_file)
        self._audit_log = AuditLog()

        # P1-1: liquidity must come from the broker. Caller may pass a callable
        # ``available_liquidity_provider`` that returns the current available
        # liquidity in dollars. If absent, fall back to 100,000 (canonical
        # total capital) only for offline/paper testing.
        self._liquidity_provider: Optional[Callable[[], float]] = None
        self._fallback_liquidity: float = 100000.0

        # Optional LLM-backed specialist agents. When ``llm_provider`` is
        # provided, ``run_cycle_with_llm`` can call the seven specialists
        # before falling back into the existing pipeline.
        self._llm_provider = llm_provider
        self._specialist_agents: Optional[Dict[str, Any]] = None
        if llm_provider is not None:
            try:
                from .agents.specialist import build_specialist_agents
                self._specialist_agents = build_specialist_agents(llm_provider)
            except Exception:
                # Specialist construction is best-effort: failure here must
                # not break the deterministic pipeline.
                self._specialist_agents = None

    def set_liquidity_provider(
        self,
        provider: Callable[[], float],
    ) -> None:
        """Inject a callable returning live available liquidity (P1-1)."""
        self._liquidity_provider = provider

    def _current_available_liquidity(self) -> float:
        if self._liquidity_provider is not None:
            return float(self._liquidity_provider())
        return self._fallback_liquidity

    def _preview_experience(
        self,
        regime: RegimeClassification,
        ensemble: EnsembleAggregate,
        kalman_state: KalmanState,
        capital_gate: CapitalGateResult,
        decision: TradingDecision,
        investment_kalman_gain: float,
    ) -> TradeExperience:
        """Build a PENDING_FILL TradeExperience BEFORE logging (P1-5, P0-1).

        ``investment_kalman_gain`` is the authoritative K_t computed by
        ``compute_investment_kalman_gain`` in the cycle; the legacy
        ``kalman_gain`` field is preserved (set to the same value) for
        back-compat with existing trade_memory consumers.
        """
        return TradeExperience(
            decision_id=decision.decision_id,
            timestamp=datetime.now(),
            symbol=self._symbol,
            regime=regime.regime,
            regime_probabilities=dict(regime.regime_affinity),
            agent_signals={a.agent_id: float(a.s) for a in getattr(decision, "_agent_outputs", [])},
            ensemble_signal=float(ensemble.ensemble_signal),
            disagreement=float(ensemble.disagreement),
            effective_confidence=float(ensemble.effective_confidence),
            kalman_gain=float(investment_kalman_gain),
            kalman_price=float(kalman_state.estimated_price),
            kalman_trend=float(kalman_state.trend),
            capital_gate_verdict=capital_gate.verdict.name,
            effective_cap=float(capital_gate.effective_cap),
            state_charges=dict(capital_gate.state_charges),
            position_action=decision.action,
            quantity=float(decision.quantity),
            confidence=float(decision.confidence),
            expected_outcome=decision.reasoning,
            realized_outcome="PENDING",
            pnl=0.0,
            lesson="",
            lifecycle_status=TradeLifecycle.PENDING_FILL.value,
            order_id=None,
            fill_price=None,
            closed_at=None,
            kalman_prior=float(ensemble.effective_confidence),
            kalman_observation=float(ensemble.ensemble_signal),
            investment_kalman_gain=float(investment_kalman_gain),
            kalman_posterior=float(capital_gate.effective_cap),
            state_gatings=dict(capital_gate.state_gatings),
            triggered_rules=tuple(capital_gate.triggered_rules),
        )

    def run_cycle(
        self,
        prices: List[float],
        volumes: Optional[List[float]],
        agent_outputs: List[AgentOutput],
        states: SevenStateVector,
        portfolio_context: Dict[str, Any],
        sigma_base_squared: float = 1.0,
    ) -> CycleResult:
        """Execute one complete trading cycle.

        P0-2: Historical memory retrieval happens *before* the capital gate.
        P1-1: ``available_liquidity`` is sourced from the broker via the
        liquidity provider if set, otherwise from ``portfolio_context``.
        P0-1: Experience is logged in PENDING_FILL, then optionally updated
        to OPEN / CLOSED. Reputation is updated *only* on CLOSED.
        """
        # P1-1: ensure portfolio context carries broker-sourced liquidity.
        if "available_liquidity" not in portfolio_context:
            portfolio_context = dict(portfolio_context)
            portfolio_context["available_liquidity"] = self._current_available_liquidity()

        # 1. Classify regime
        regime_result = self._classify_regime(prices, volumes)

        # 2. Get regime-specific weights
        weights = self._get_regime_weights(regime_result.regime)

        # 3. Update Kalman with latest price
        latest_price = prices[-1]
        kalman_state = self._kalman_filter.update(latest_price)

        # 4. Compute ensemble aggregate
        ensemble = compute_ensemble_aggregate(agent_outputs, weights)

        # 5. Compute investment Kalman gain
        kalman_gain = compute_investment_kalman_gain(
            prediction_covariance=kalman_state.price_variance,
            effective_confidence=ensemble.effective_confidence,
            disagreement=ensemble.disagreement,
            sigma_base_squared=sigma_base_squared,
        )

        # 6. P0-2: retrieve similar historical trades BEFORE the capital gate.
        historical_context = self._retrieve_historical_context(
            regime=regime_result,
            ensemble=ensemble,
            kalman_state=kalman_state,
            capital_gate_preview_states=states,
        )

        # 7. Evaluate capital gate (pass pre-computed ensemble to avoid recomputation)
        capital_gate = evaluate(
            kalman_state=kalman_state,
            states=states,
            portfolio_context=portfolio_context,
            agents=agent_outputs,
            agent_weights=weights,
            sigma_base_squared=sigma_base_squared,
            ensemble_agg=ensemble,
        )

        # 8. Make trading decision (uses historical_context, but capital gate is
        # the hard authority — memory may only adjust confidence/reasoning, never
        # override the gate).
        decision = self._make_decision(
            regime=regime_result,
            ensemble=ensemble,
            kalman_state=kalman_state,
            kalman_gain=kalman_gain,
            capital_gate=capital_gate,
            weights=weights,
            agent_outputs=agent_outputs,
            historical_context=historical_context,
        )

        # Build the authoritative per-agent payload ONCE; used in both
        # the audit log and the persisted TradeExperience so the
        # dashboard's 7-agent table has a single source of truth.
        agent_outputs_full: Dict[str, Dict[str, float]] = {}
        for a in agent_outputs:
            row: Dict[str, float] = {
                "signal": float(a.s),
                "confidence": float(a.c),
                "uncertainty": float(a.u),
                "doubt": float(a.d),
                "p_plus": float(a.p_plus),
                "p_minus": float(a.p_minus),
                "delta_t": float(a.delta_t),
                "noise": float(a.r),
                "weight": float(weights.get(a.agent_id, 0.0)),
            }
            try:
                params = self._reputation_tracker.get_posterior_parameters(
                    a.agent_id, regime_result.regime
                )
                row["reputation_alpha"] = float(params["alpha"])
                row["reputation_beta"] = float(params["beta"])
            except Exception:
                row["reputation_alpha"] = 1.0
                row["reputation_beta"] = 1.0
            agent_outputs_full[a.agent_id] = row

        self._audit_log.record_decision(
            decision_id=decision.decision_id,
            symbol=self._symbol,
            payload={
                "action": decision.action,
                "quantity": decision.quantity,
                "confidence": decision.confidence,
                "verdict": capital_gate.verdict.name,
                "effective_cap": capital_gate.effective_cap,
                "kalman_gain": kalman_gain,
                "ensemble_signal": ensemble.ensemble_signal,
                "disagreement": ensemble.disagreement,
                "regime": regime_result.regime,
                "regime_probabilities": regime_result.regime_affinity,
                "state_charges": dict(capital_gate.state_charges),
                "state_gatings": dict(capital_gate.state_gatings),
                "triggered_rules": capital_gate.triggered_rules,
                "weights": weights,
                "circuit_breakers": decision.circuit_breakers,
                "reasoning": decision.reasoning,
                "provenance": decision.provenance,
                "memory_used": historical_context.get("memory_used", False),
                "similar_trade_count": historical_context.get("similar_trade_count", 0),
                "historical_win_rate": historical_context.get("historical_win_rate", 0.0),
                "historical_avg_pnl": historical_context.get("avg_pnl", 0.0),
                "memory_adjustment": historical_context.get("confidence_adjustment", 0.0),
                # Authoritative Kalman / ensemble provenance (P0 dashboard fix).
                # The dashboard reads these rather than reconstructing the posterior.
                "kalman_prior": float(ensemble.effective_confidence),
                "kalman_observation": float(ensemble.ensemble_signal),
                "investment_kalman_gain": float(kalman_gain),
                "kalman_posterior": float(capital_gate.effective_cap),
                "agent_outputs_full": agent_outputs_full,
            },
        )

        # 9. Execute order (if enabled) -> PENDING_FILL
        order_result = self._execute_order(decision)
        experience = self._record_pending_experience(
            regime=regime_result,
            ensemble=ensemble,
            kalman_state=kalman_state,
            capital_gate=capital_gate,
            decision=decision,
            agent_outputs=agent_outputs,
            order_result=order_result,
            investment_kalman_gain=kalman_gain,
            agent_weights=weights,
            agent_outputs_full=agent_outputs_full,
        )

        # 10. Reputation update ONLY on terminal lifecycle (P0-1).
        # PENDING_FILL is NOT terminal; a separate close_trade() call is
        # required to finalize reputation updates.
        if (
            experience.lifecycle_status == TradeLifecycle.CLOSED.value
            or experience.lifecycle_status == TradeLifecycle.REJECTED.value
        ):
            self._update_reputations(experience)
            self._audit_log.record_outcome(
                decision_id=decision.decision_id,
                symbol=self._symbol,
                payload={
                    "order_result": order_result,
                    "pnl": experience.pnl,
                    "realized_outcome": experience.realized_outcome,
                    "lesson": experience.lesson,
                    "lifecycle_status": experience.lifecycle_status,
                },
            )

        return CycleResult(
            decision_id=decision.decision_id,
            decision=decision,
            regime=regime_result,
            weights=weights,
            ensemble=ensemble,
            kalman_state=kalman_state,
            capital_gate=capital_gate,
            experience=experience,
        )

    def _retrieve_historical_context(
        self,
        regime: RegimeClassification,
        ensemble: EnsembleAggregate,
        kalman_state: KalmanState,
        capital_gate_preview_states: SevenStateVector,
    ) -> Dict[str, Any]:
        """Retrieve similar historical trades and summarize outcomes (P0-2)."""
        preview = TradeExperience(
            decision_id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            symbol=self._symbol,
            regime=regime.regime,
            regime_probabilities=dict(regime.regime_affinity),
            agent_signals={aid: 0.0 for aid in self._agent_ids},
            ensemble_signal=float(ensemble.ensemble_signal),
            disagreement=float(ensemble.disagreement),
            effective_confidence=float(ensemble.effective_confidence),
            kalman_gain=float(kalman_state.price_variance),
            kalman_price=float(kalman_state.estimated_price),
            kalman_trend=float(kalman_state.trend),
            capital_gate_verdict="PREVIEW",
            effective_cap=0.0,
            state_charges={},
            position_action="HOLD",
            quantity=0.0,
            confidence=float(ensemble.effective_confidence),
            expected_outcome="",
            realized_outcome="",
            pnl=0.0,
            lesson="",
        )
        similar = self._trade_memory.find_similar(preview, top_k=5, min_similarity=0.0)

        if not similar:
            return {
                "memory_used": False,
                "similar_trade_count": 0,
                "similar_trades": [],
                "historical_win_rate": 0.0,
                "avg_pnl": 0.0,
                "wins": 0,
                "losses": 0,
                "lessons": [],
                "confidence_adjustment": 0.0,
            }

        closed = [s for s in similar if s.experience.lifecycle_status == TradeLifecycle.CLOSED.value]
        if not closed:
            return {
                "memory_used": True,
                "similar_trade_count": len(similar),
                "similar_trades": [],
                "historical_win_rate": 0.0,
                "avg_pnl": 0.0,
                "wins": 0,
                "losses": 0,
                "lessons": [],
                "confidence_adjustment": 0.0,
            }

        pnls = [s.experience.pnl for s in closed]
        wins = sum(1 for p in pnls if p > 0)
        losses = sum(1 for p in pnls if p < 0)
        win_rate = wins / len(pnls)
        avg_pnl = sum(pnls) / len(pnls)
        lessons = [s.experience.lesson for s in closed if s.experience.lesson]
        unique_lessons = list(dict.fromkeys(lessons))
        adjustment = max(-0.1, min(0.1, (win_rate - 0.5) * 0.2))

        return {
            "memory_used": True,
            "similar_trade_count": len(similar),
            "similar_trades": [
                {
                    "decision_id": s.experience.decision_id,
                    "symbol": s.experience.symbol,
                    "action": s.experience.position_action,
                    "pnl": s.experience.pnl,
                    "similarity": s.similarity_score,
                    "market_similarity": s.market_similarity,
                    "portfolio_similarity": s.portfolio_similarity,
                }
                for s in similar
            ],
            "historical_win_rate": win_rate,
            "avg_pnl": avg_pnl,
            "wins": wins,
            "losses": losses,
            "lessons": unique_lessons[:5],
            "confidence_adjustment": adjustment,
        }

    def close_trade(
        self,
        decision_id: str,
        realized_outcome: str,
        pnl: float,
        lesson: str = "",
    ) -> TradeExperience:
        """Finalize a trade and trigger the reputation update (P0-1)."""
        closed = self._trade_memory.close_trade(
            decision_id=decision_id,
            realized_outcome=realized_outcome,
            pnl=pnl,
            lesson=lesson,
        )
        self._update_reputations(closed)
        self._audit_log.record_outcome(
            decision_id=decision_id,
            symbol=closed.symbol,
            payload={
                "pnl": closed.pnl,
                "realized_outcome": closed.realized_outcome,
                "lesson": closed.lesson,
                "lifecycle_status": closed.lifecycle_status,
            },
        )
        return closed

    def _classify_regime(
        self,
        prices: List[float],
        volumes: Optional[List[float]],
        highs: Optional[List[float]] = None,
        lows: Optional[List[float]] = None,
    ) -> RegimeClassification:
        """Classify market regime using HMM (always authoritative).

        The rule-based detector fallback has been removed. HMM is the
        single canonical regime pipeline. Highs and lows, when provided,
        flow into `extract_features` / ATR so the HMM sees genuine True Range.
        """
        features = extract_features(
            prices, volumes,
            highs=highs, lows=lows,
            lookback_days=20,
        )
        hmm_result = self._hmm_detector.classify(features.tolist())

        return RegimeClassification(
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

    def _get_regime_weights(self, regime: str) -> Dict[str, float]:
        """Get per-agent reputation weights for active regime."""
        weights = {}
        for agent_id in self._agent_ids:
            weights[agent_id] = self._reputation_tracker.get_reputation_weight(
                agent_id, regime
            )
        return weights

    def _make_decision(
        self,
        regime: RegimeClassification,
        ensemble: EnsembleAggregate,
        kalman_state: KalmanState,
        kalman_gain: float,
        capital_gate: CapitalGateResult,
        weights: Dict[str, float],
        agent_outputs: List[AgentOutput],
        historical_context: Optional[Dict[str, Any]] = None,
    ) -> TradingDecision:
        """Make trading decision based on analytical outputs.

        Decision logic:
        1. Check capital gate verdict
        2. Apply risk circuit breakers
        3. Determine position action and size
        4. Apply memory-derived confidence adjustment (P0-2: bounded +/-10%)
        5. Build provenance trace

        Note: The LLM/interpreter layer may provide hypothesis/lesson context,
        but the deterministic mathematical layer (ensemble → Kalman → capital gate)
        remains the ultimate risk authority. The LLM does not directly modify
        capital gate parameters or override risk controls.
        """
        historical_context = historical_context or {}
        circuit_breakers: List[str] = []
        reasoning_parts = []

        # Capital gate verdict
        verdict = capital_gate.verdict
        if verdict.name == "BLOCK":
            circuit_breakers.append("CAPITAL_GATE_BLOCK")
            reasoning_parts.append("Capital gate blocked trade")
        elif verdict.name == "FLATTEN":
            circuit_breakers.append("CAPITAL_GATE_FLATTEN")
            reasoning_parts.append("Capital gate requires flattening")
        elif verdict.name == "REDUCE":
            reasoning_parts.append(f"Capital gate reduced position to {capital_gate.effective_cap:.1%}")

        # Regime-based circuit breakers
        if regime.regime in ("R04", "R07"):
            if ensemble.effective_confidence <= 0.85:
                circuit_breakers.append(f"REGM_{regime.regime}_LOW_CONFIDENCE")
                reasoning_parts.append(f"Low confidence in {regime.regime}")

        # Entropy circuit breaker
        if capital_gate.state_gatings.get("market", 1.0) < 0.5:
            circuit_breakers.append("HIGH_MARKET_ENTROPY")
            reasoning_parts.append("High market entropy detected")

        # P0-2: memory-derived confidence adjustment is bounded ±10% and
        # explicitly NOT a risk override.
        memory_used = bool(historical_context.get("memory_used"))
        if memory_used:
            adjustment = float(historical_context.get("confidence_adjustment", 0.0))
            similar_count = int(historical_context.get("similar_trade_count", 0))
            win_rate = float(historical_context.get("historical_win_rate", 0.0))
            reasoning_parts.append(
                f"Memory: {similar_count} similar trades, historical win rate {win_rate:.1%}, "
                f"confidence adjust {adjustment:+.2%}"
            )

        # Determine action
        if circuit_breakers:
            action = "HOLD"
            quantity = 0.0
            confidence = 0.0
        else:
            # Use ensemble signal direction
            base_confidence = ensemble.effective_confidence
            if memory_used:
                base_confidence = max(0.0, min(1.0, base_confidence + float(historical_context.get("confidence_adjustment", 0.0))))
            if ensemble.ensemble_signal > 0.3:
                action = "BUY"
                quantity = self._compute_position_size(capital_gate.effective_cap)
                confidence = base_confidence
            elif ensemble.ensemble_signal < -0.3:
                action = "SELL"
                quantity = self._compute_position_size(capital_gate.effective_cap)
                confidence = base_confidence
            else:
                action = "HOLD"
                quantity = 0.0
                confidence = base_confidence

            reasoning_parts.append(
                f"Ensemble signal: {ensemble.ensemble_signal:.3f}, "
                f"disagreement: {ensemble.disagreement:.3f}"
            )

        reasoning = "; ".join(reasoning_parts) if reasoning_parts else "No specific triggers"

        return TradingDecision(
            decision_id=str(uuid.uuid4()),
            action=action,
            symbol=self._symbol,
            quantity=quantity,
            confidence=confidence,
            reasoning=reasoning,
            circuit_breakers=circuit_breakers,
            provenance={
                "regime": regime.regime,
                "ensemble_signal": ensemble.ensemble_signal,
                "kalman_gain": kalman_gain,
                "effective_cap": capital_gate.effective_cap,
                "verdict": verdict.name,
                "weights": weights,
                "memory_used": memory_used,
                "memory_adjustment": float(historical_context.get("confidence_adjustment", 0.0)),
                "similar_trade_count": int(historical_context.get("similar_trade_count", 0)),
                "historical_win_rate": float(historical_context.get("historical_win_rate", 0.0)),
            },
        )

    def _compute_position_size(self, effective_cap: float) -> float:
        """Compute position size from effective capital cap."""
        # Simplified: use effective_cap as position fraction
        # In production, this would use account buying power
        return max(0.0, min(1.0, effective_cap))

    def _execute_order(self, decision: TradingDecision) -> Optional[Any]:
        """Execute trading decision via Alpaca (if enabled)."""
        if not self._enable_trading:
            return None

        if decision.action == "HOLD":
            return None

        try:
            from investment_agent.execution.execution import place_order, get_account_summary
            price = float(getattr(decision, "price", 100.0) or 100.0)
            return place_order(
                symbol=decision.symbol,
                side=decision.action.lower(),
                qty=int(decision.quantity) if decision.quantity > 0 else 0,
                price_per_share=price,
                is_option=False,
            )
        except Exception as e:
            return {"error": str(e)}

    def _record_pending_experience(
        self,
        regime: RegimeClassification,
        ensemble: EnsembleAggregate,
        kalman_state: KalmanState,
        capital_gate: CapitalGateResult,
        decision: TradingDecision,
        agent_outputs: List[AgentOutput],
        order_result: Optional[Any],
        investment_kalman_gain: float,
        agent_weights: Optional[Dict[str, float]] = None,
        agent_outputs_full: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> TradeExperience:
        """Record a PENDING_FILL experience after order submission (P0-1).

        Authoritative Kalman/ensemble fields are written here so the
        dashboard can read them without reconstructing. The legacy
        ``kalman_gain`` field is set to the same K_t value (was previously
        set incorrectly to ``kalman_state.price_variance``).

        The full per-agent ``AgentOutput`` (all eight channels) plus
        per-agent weight and per-agent (alpha, beta) reputation are
        stored in ``agent_outputs_full`` for the dashboard's 7-agent table.
        ``agent_outputs_full`` is built once in ``run_cycle`` and passed in
        to keep the audit log and the persisted experience in lockstep.

        Returns
        -------
        TradeExperience
            Newly logged PENDING_FILL experience, or REJECTED if the order was
            rejected by the broker.
        """
        agent_signals = {a.agent_id: float(a.s) for a in agent_outputs}

        if order_result is None:
            status = "PENDING_FILL"
            order_id = None
        elif isinstance(order_result, dict) and "error" in order_result:
            status = TradeLifecycle.REJECTED.value
            order_id = None
        elif hasattr(order_result, "reason") and getattr(order_result, "reason") and not getattr(order_result, "submitted", True):
            status = TradeLifecycle.REJECTED.value
            order_id = getattr(order_result, "order_id", None)
        else:
            status = TradeLifecycle.PENDING_FILL.value
            order_id = getattr(order_result, "order_id", None) or getattr(order_result, "id", None)
            if order_id is None and isinstance(order_result, dict):
                order_id = order_result.get("order_id") or order_result.get("id")
            elif order_id is None and hasattr(order_result, "get"):
                order_id = order_result.get("order_id") or order_result.get("id")

        experience = TradeExperience(
            decision_id=decision.decision_id,
            timestamp=datetime.now(),
            symbol=self._symbol,
            regime=regime.regime,
            regime_probabilities=dict(regime.regime_affinity),
            agent_signals=agent_signals,
            ensemble_signal=float(ensemble.ensemble_signal),
            disagreement=float(ensemble.disagreement),
            effective_confidence=float(ensemble.effective_confidence),
            kalman_gain=float(investment_kalman_gain),
            kalman_price=float(kalman_state.estimated_price),
            kalman_trend=float(kalman_state.trend),
            capital_gate_verdict=capital_gate.verdict.name,
            effective_cap=float(capital_gate.effective_cap),
            state_charges=dict(capital_gate.state_charges),
            position_action=decision.action,
            quantity=float(decision.quantity),
            confidence=float(decision.confidence),
            expected_outcome=decision.reasoning,
            realized_outcome="PENDING",
            pnl=0.0,
            lesson="",
            lifecycle_status=status,
            order_id=order_id,
            fill_price=None,
            closed_at=None,
            kalman_prior=float(ensemble.effective_confidence),
            kalman_observation=float(ensemble.ensemble_signal),
            investment_kalman_gain=float(investment_kalman_gain),
            kalman_posterior=float(capital_gate.effective_cap),
            state_gatings=dict(capital_gate.state_gatings),
            triggered_rules=tuple(capital_gate.triggered_rules),
            agent_outputs_full=(
                {k: dict(v) for k, v in agent_outputs_full.items()}
                if agent_outputs_full is not None
                else None
            ),
        )
        self._trade_memory.log_experience(experience)
        return experience

    def _update_reputations(self, experience: TradeExperience) -> None:
        """Update agent reputations with trade outcome (P0-1).

        Only call this for terminal lifecycle (CLOSED or REJECTED).
        """
        if experience.lifecycle_status == TradeLifecycle.PENDING_FILL.value:
            return
        if experience.lifecycle_status == TradeLifecycle.OPEN.value:
            return
        was_correct = experience.pnl > 0
        for agent_id in self._agent_ids:
            self._reputation_tracker.record_outcome(
                agent_id, experience.regime, was_correct
            )

    def get_historical_context(
        self, current_experience: TradeExperience, top_k: int = 5
    ) -> Dict[str, Any]:
        """Get historical context for current situation.

        This is the "learning" component: find similar past trades and
        summarize what happened to inform the current decision.

        Parameters
        ----------
        current_experience : TradeExperience
            Current situation to match against.
        top_k : int
            Maximum number of similar experiences to retrieve.

        Returns
        -------
        Dict[str, Any]
            Historical context including similar trades and aggregated lessons.
        """
        similar = self._trade_memory.find_similar(current_experience, top_k=top_k)

        if not similar:
            return {
                "similar_trades": [],
                "historical_win_rate": 0.0,
                "avg_pnl": 0.0,
                "lessons": [],
                "confidence_adjustment": 0.0,
            }

        # Aggregate statistics from similar trades
        pnls = [s.experience.pnl for s in similar]
        wins = sum(1 for p in pnls if p > 0)
        losses = sum(1 for p in pnls if p < 0)
        win_rate = wins / len(pnls) if pnls else 0.0
        avg_pnl = sum(pnls) / len(pnls) if pnls else 0.0

        # Extract lessons
        lessons = [s.experience.lesson for s in similar if s.experience.lesson]
        unique_lessons = list(dict.fromkeys(lessons))  # preserve order, remove duplicates

        # Compute confidence adjustment based on historical performance
        # If similar trades had high win rate, boost confidence; otherwise reduce
        confidence_adjustment = (win_rate - 0.5) * 0.2  # ±10% max adjustment
        confidence_adjustment = max(-0.1, min(0.1, confidence_adjustment))

        return {
            "similar_trades": [
                {
                    "timestamp": s.experience.timestamp.isoformat(),
                    "symbol": s.experience.symbol,
                    "action": s.experience.position_action,
                    "pnl": s.experience.pnl,
                    "similarity": s.similarity_score,
                    "reasons": s.match_reasons,
                }
                for s in similar
            ],
            "historical_win_rate": win_rate,
            "avg_pnl": avg_pnl,
            "wins": wins,
            "losses": losses,
            "lessons": unique_lessons[:5],  # Top 5 unique lessons
            "confidence_adjustment": confidence_adjustment,
        }

    def learn_from_outcome(
        self,
        experience: TradeExperience,
        realized_pnl: float,
        realized_outcome: str,
        lesson: str = "",
    ) -> TradeExperience:
        """Backwards-compatible outcome finalization (P0-1).

        Delegates to ``close_trade`` so the lifecycle becomes CLOSED and the
        reputation update fires only after realized P&L is known.
        """
        return self.close_trade(
            decision_id=experience.decision_id,
            realized_outcome=realized_outcome,
            pnl=realized_pnl,
            lesson=lesson,
        )

    def find_similar_past_trades(
        self, current: TradeExperience, top_k: int = 5
    ) -> List[SimilarExperience]:
        """Find similar historical trades for context.

        Parameters
        ----------
        current : TradeExperience
            Current situation to match.
        top_k : int
            Maximum number of similar experiences to return.

        Returns
        -------
        List[SimilarExperience]
            Ranked similar historical experiences.
        """
        return self._trade_memory.find_similar(current, top_k=top_k)

    def get_trade_history(self, symbol: Optional[str] = None) -> List[TradeExperience]:
        """Get trade history, optionally filtered by symbol."""
        return self._trade_memory.get_history(symbol)

    def get_performance_summary(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """Get performance summary from trade history."""
        return self._trade_memory.get_performance_summary(symbol)

    def run_cycle_with_llm(
        self,
        prices: List[float],
        volumes: Optional[List[float]],
        states: SevenStateVector,
        portfolio_context: Dict[str, Any],
        *,
        regime_features: Optional[Dict[str, float]] = None,
        sigma_base_squared: float = 1.0,
    ) -> CycleResult:
        """One autonomous cycle driven by LLM-backed specialist agents.

        What
        ====
        1. Classify regime (HMM or rule-based).
        2. Retrieve similar closed trades from memory.
        3. Call every specialist LLM agent with regime + memory context.
        4. Build ``AgentOutput`` list and feed the existing pipeline
           (ensemble → Kalman → capital gate → risk → decision).

        Why
        ====
        Provides a single entry point that takes raw market data and returns
        a full ``CycleResult`` without the caller having to manually
        construct the seven ``AgentOutput`` tuples. The deterministic risk
        layer (capital gate) remains the hard authority; the LLM only
        contributes signals.

        Parameters
        ----------
        prices : List[float]
            Historical price series for regime classification.
        volumes : Optional[List[float]]
            Historical volume series.
        states : SevenStateVector
            Seven-dimensional state-of-charge vector.
        portfolio_context : Dict[str, Any]
            Portfolio risk context (must include the standard keys
            consumed by ``evaluate``).
        regime_features : Optional[Dict[str, float]]
            Optional summary of regime features forwarded to each
            specialist's user prompt.
        sigma_base_squared : float, optional
            Base model variance for the investment Kalman gain.

        Returns
        -------
        CycleResult
            Identical to ``run_cycle``. Reputation / memory updates only
            fire after ``close_trade`` is called.
        """
        if self._specialist_agents is None:
            raise RuntimeError(
                "run_cycle_with_llm requires an llm_provider at construction; "
                "pass llm_provider=... to XQuantXOrchestrator(...)"
            )

        from .agents.specialist import AgentContext, run_agents

        # 1. Classify regime (reuse existing path).
        regime_result = self._classify_regime(prices, volumes)

        # 2. Retrieve similar memory for prompt context.
        preview = TradeExperience(
            decision_id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            symbol=self._symbol,
            regime=regime_result.regime,
            regime_probabilities=dict(regime_result.regime_affinity),
            agent_signals={aid: 0.0 for aid in self._agent_ids},
            ensemble_signal=0.0,
            disagreement=0.0,
            effective_confidence=0.5,
            kalman_gain=0.0,
            kalman_price=float(prices[-1]) if prices else 0.0,
            kalman_trend=0.0,
            capital_gate_verdict="PREVIEW",
            effective_cap=0.0,
            state_charges={},
            position_action="HOLD",
            quantity=0.0,
            confidence=0.5,
            expected_outcome="",
            realized_outcome="",
            pnl=0.0,
            lesson="",
        )
        similar_memory = self._trade_memory.find_similar(preview, top_k=5, min_similarity=0.0)

        # 3. Run the seven specialists.
        ctx = AgentContext(
            symbol=self._symbol,
            regime=regime_result.regime,
            regime_probabilities=dict(regime_result.regime_affinity),
            features=regime_features or {},
            ensemble_signal=0.0,
            disagreement=0.0,
            peer_agents={},
            memory=similar_memory,
        )
        agent_output_map = run_agents(self._specialist_agents, ctx)
        agent_outputs: List[AgentOutput] = [
            agent_output_map[aid] for aid in self._agent_ids if aid in agent_output_map
        ]
        # If a caller-supplied agent_ids list doesn't align 1:1 with the
        # seven default roles, fall back to whatever the LLM produced.
        if not agent_outputs:
            agent_outputs = list(agent_output_map.values())

        # 4. Reuse the existing deterministic pipeline.
        return self.run_cycle(
            prices=prices,
            volumes=volumes,
            agent_outputs=agent_outputs,
            states=states,
            portfolio_context=portfolio_context,
            sigma_base_squared=sigma_base_squared,
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "XQuantXOrchestrator",
    "TradingDecision",
    "CycleResult",
    "TradeExperience",
    "SimilarExperience",
    "TradeMemory",
]
