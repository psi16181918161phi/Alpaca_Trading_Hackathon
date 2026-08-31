"""Trade Memory — Structured Experience Logging and Retrieval for X Quant X.

WHAT
====
Provides a JSON-backed trade memory system that records structured trading
experiences and enables retrieval of similar historical situations.

WHY
===
The architecture produces deterministic analytical outputs, but without memory
it cannot learn from past trades. This module provides:

1. Structured experience logging (TradeExperience)
2. Similarity-based retrieval of historical situations
3. Outcome tracking and P&L computation
4. Integration with agent reputation for feedback

HOW
===
- Records every completed trade as a TradeExperience dataclass
- Persists to JSON file for durability
- Computes similarity between current and historical situations
- Returns ranked historical contexts for decision support

TradeExperience Schema
======================
{
    "timestamp": ISO-8601,
    "symbol": str,
    "regime": str (R01-R12),
    "regime_probabilities": Dict[str, float],
    "agent_signals": Dict[str, float],
    "ensemble_signal": float,
    "disagreement": float,
    "effective_confidence": float,
    "kalman_gain": float,
    "kalman_price": float,
    "kalman_trend": float,
    "capital_gate_verdict": str,
    "effective_cap": float,
    "state_charges": Dict[str, float],
    "position_action": str,
    "quantity": float,
    "confidence": float,
    "expected_outcome": str,
    "realized_outcome": str,
    "pnl": float,
    "lesson": str
}

Similarity Metric
=================
Uses weighted Euclidean distance in feature space:
- Regime match (highest weight)
- Ensemble signal similarity
- Disagreement similarity
- Kalman state similarity
- Capital gate similarity

Architectural Role
==================
Experience layer. Sits between Capital Gate and Execution, recording outcomes
and providing historical context for future decisions.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Default memory file location
DEFAULT_MEMORY_FILE: str = "trade_memory.json"

# Maximum number of experiences to retain per symbol
MAX_MEMORY_PER_SYMBOL: int = 1000

# Similarity feature weights
_SIMILARITY_WEIGHTS = {
    "regime_probabilities": 3.0,
    "agent_signals": 2.0,
    "ensemble_signal": 2.0,
    "disagreement": 1.5,
    "kalman_gain": 1.5,
    "effective_cap": 1.0,
    "confidence": 1.0,
}

# Subset of weights that are *market* similarity (capital-independent).
_MARKET_SIMILARITY_FEATURES = {
    "regime_probabilities",
    "agent_signals",
    "ensemble_signal",
    "disagreement",
    "kalman_gain",
}

# Subset of weights that are *portfolio/capital* similarity.
_PORTFOLIO_SIMILARITY_FEATURES = {
    "effective_cap",
    "confidence",
}

# Normalization ranges for non-[0,1] features
_NORMALIZATION_RANGES = {
    "effective_cap": (0.0, 1.0),  # effective_cap is already in [0,1]
    "kalman_gain": (0.0, 1.0),   # Kalman gain is in [0,1]
    "ensemble_signal": (-1.0, 1.0),  # ensemble signal can be negative
    "disagreement": (0.0, 1.0),  # disagreement in [0,1]
    "confidence": (0.0, 1.0),    # confidence in [0,1]
}


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------

class TradeLifecycle(str, Enum):
    """Lifecycle state of a trade experience.

    PENDING_FILL  : order submitted, awaiting fill
    OPEN          : filled, position still held
    CLOSED        : position closed, P&L realized
    REJECTED      : order rejected by broker
    CANCELLED     : order or position cancelled
    """

    PENDING_FILL = "PENDING_FILL"
    OPEN = "OPEN"
    CLOSED = "CLOSED"
    REJECTED = "REJECTED"
    CANCELLED = "CANCELLED"

    @property
    def is_terminal(self) -> bool:
        """Whether the experience is in a final lifecycle state (no further updates)."""
        return self in (TradeLifecycle.CLOSED, TradeLifecycle.REJECTED, TradeLifecycle.CANCELLED)

    @property
    def has_realized_pnl(self) -> bool:
        """Whether this lifecycle admits a meaningful realized P&L value."""
        return self is TradeLifecycle.CLOSED


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TradeExperience:
    """Immutable structured trading experience record.

    Attributes
    ----------
    decision_id : str
        Stable identifier connecting this experience to its decision/audit/order
        record. Used to exclude self from similarity retrieval.
    timestamp : datetime
        When the trade decision was made.
    symbol : str
        Trading symbol (e.g., "AAPL", "TSLA").
    regime : str
        Active regime at time of decision (R01-R12).
    regime_probabilities : Dict[str, float]
        HMM posterior probabilities over all regimes.
    agent_signals : Dict[str, float]
        Individual agent signal outputs.
    ensemble_signal : float
        Weighted ensemble aggregate signal.
    disagreement : float
        Ensemble disagreement metric.
    effective_confidence : float
        Ensemble effective confidence.
    kalman_gain : float
        Investment Kalman gain K_t.
    kalman_price : float
        Kalman estimated price at decision time.
    kalman_trend : float
        Kalman trend estimate at decision time.
    capital_gate_verdict : str
        Capital gate verdict (ALLOW, REDUCE, BLOCK, FLATTEN).
    effective_cap : float
        Effective capital deployment cap.
    state_charges : Dict[str, float]
        Seven-state charge values at decision time.
    position_action : str
        Action taken (BUY, SELL, HOLD, etc.).
    quantity : float
        Position quantity (shares or contracts).
    confidence : float
        Decision confidence in [0.0, 1.0].
    expected_outcome : str
        Expected outcome description.
    realized_outcome : str
        Actual realized outcome (filled after trade completion).
    pnl : float
        Realized profit/loss. Meaningful only for CLOSED lifecycle.
    lesson : str
        Extracted lesson/reflection from this experience.
    lifecycle_status : str
        TradeLifecycle value: PENDING_FILL, OPEN, CLOSED, REJECTED, CANCELLED.
    order_id : Optional[str]
        Alpaca order ID, set when order submitted.
    fill_price : Optional[float]
        Average fill price when order is filled.
    closed_at : Optional[datetime]
        When the position was closed.
    """

    decision_id: str
    timestamp: datetime
    symbol: str
    regime: str
    regime_probabilities: Dict[str, float]
    agent_signals: Dict[str, float]
    ensemble_signal: float
    disagreement: float
    effective_confidence: float
    kalman_gain: float
    kalman_price: float
    kalman_trend: float
    capital_gate_verdict: str
    effective_cap: float
    state_charges: Dict[str, float]
    position_action: str
    quantity: float
    confidence: float
    expected_outcome: str
    realized_outcome: str
    pnl: float
    lesson: str
    lifecycle_status: str = "PENDING_FILL"
    order_id: Optional[str] = None
    fill_price: Optional[float] = None
    closed_at: Optional[datetime] = None


@dataclass(frozen=True)
class SimilarExperience:
    """A historically similar experience with similarity score.

    The ``similarity_score`` is the overall weighted blend. ``market_similarity``
    and ``portfolio_similarity`` decompose that score so callers can distinguish
    between market-state similarity and capital-state similarity (P1-4).

    Attributes
    ----------
    experience : TradeExperience
        The historical trade experience.
    similarity_score : float
        Overall similarity score in [0.0, 1.0], higher is more similar.
    market_similarity : float
        Similarity restricted to market-state features
        (regime, agent signals, ensemble signal, disagreement, Kalman gain).
    portfolio_similarity : float
        Similarity restricted to portfolio/capital features
        (effective_cap, confidence).
    match_reasons : List[str]
        Human-readable explanations of why this experience is similar.
    """

    experience: TradeExperience
    similarity_score: float
    market_similarity: float
    portfolio_similarity: float
    match_reasons: List[str]


# ---------------------------------------------------------------------------
# Memory Store
# ---------------------------------------------------------------------------

class TradeMemory:
    """Persistent trade memory with similarity-based retrieval.

    Stores TradeExperience records in JSON format and provides methods
    for logging new experiences and finding similar historical situations.
    """

    def __init__(self, memory_file: str = DEFAULT_MEMORY_FILE) -> None:
        """Initialize trade memory.

        Parameters
        ----------
        memory_file : str
            Path to JSON memory file.
        """
        self._memory_file = memory_file
        self._experiences: List[TradeExperience] = []
        self._load()

    def _load(self) -> None:
        """Load experiences from disk."""
        if not os.path.exists(self._memory_file):
            self._experiences = []
            return

        try:
            with open(self._memory_file, "r") as f:
                raw = json.load(f)
            if not isinstance(raw, list):
                raise ValueError(f"Memory file must contain a JSON array, got {type(raw).__name__}")
            self._experiences = [
                self._deserialize(r) for r in raw if self._is_valid_experience(r)
            ]
        except json.JSONDecodeError as e:
            raise ValueError(f"Memory file {self._memory_file} contains invalid JSON: {e}")
        except Exception as e:
            raise ValueError(f"Failed to load memory from {self._memory_file}: {e}")

    def _save(self) -> None:
        """Save experiences to disk using atomic write."""
        raw = [self._serialize(e) for e in self._experiences]
        tmp_path = f"{self._memory_file}.tmp"
        with open(tmp_path, "w") as f:
            json.dump(raw, f, indent=2, default=str)
            f.flush()
            os.fsync(f.fileno())
        try:
            os.replace(tmp_path, self._memory_file)
        except PermissionError:
            # Windows file locking fallback: write directly
            with open(self._memory_file, "w") as f:
                json.dump(raw, f, indent=2, default=str)
                f.flush()
                os.fsync(f.fileno())
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def _serialize(self, exp: TradeExperience) -> Dict[str, Any]:
        """Convert TradeExperience to JSON-serializable dict."""
        return {
            "decision_id": exp.decision_id,
            "timestamp": exp.timestamp.isoformat(),
            "symbol": exp.symbol,
            "regime": exp.regime,
            "regime_probabilities": exp.regime_probabilities,
            "agent_signals": exp.agent_signals,
            "ensemble_signal": exp.ensemble_signal,
            "disagreement": exp.disagreement,
            "effective_confidence": exp.effective_confidence,
            "kalman_gain": exp.kalman_gain,
            "kalman_price": exp.kalman_price,
            "kalman_trend": exp.kalman_trend,
            "capital_gate_verdict": exp.capital_gate_verdict,
            "effective_cap": exp.effective_cap,
            "state_charges": exp.state_charges,
            "position_action": exp.position_action,
            "quantity": exp.quantity,
            "confidence": exp.confidence,
            "expected_outcome": exp.expected_outcome,
            "realized_outcome": exp.realized_outcome,
            "pnl": exp.pnl,
            "lesson": exp.lesson,
            "lifecycle_status": exp.lifecycle_status,
            "order_id": exp.order_id,
            "fill_price": exp.fill_price,
            "closed_at": exp.closed_at.isoformat() if exp.closed_at else None,
        }

    def _deserialize(self, raw: Dict[str, Any]) -> TradeExperience:
        """Convert JSON dict to TradeExperience."""
        closed_at_raw = raw.get("closed_at")
        closed_at = datetime.fromisoformat(closed_at_raw) if closed_at_raw else None
        return TradeExperience(
            decision_id=raw.get("decision_id", ""),
            timestamp=datetime.fromisoformat(raw["timestamp"]),
            symbol=raw["symbol"],
            regime=raw["regime"],
            regime_probabilities=raw.get("regime_probabilities", {}),
            agent_signals=raw.get("agent_signals", {}),
            ensemble_signal=raw.get("ensemble_signal", 0.0),
            disagreement=raw.get("disagreement", 0.0),
            effective_confidence=raw.get("effective_confidence", 0.0),
            kalman_gain=raw.get("kalman_gain", 0.0),
            kalman_price=raw.get("kalman_price", 0.0),
            kalman_trend=raw.get("kalman_trend", 0.0),
            capital_gate_verdict=raw.get("capital_gate_verdict", "UNKNOWN"),
            effective_cap=raw.get("effective_cap", 0.0),
            state_charges=raw.get("state_charges", {}),
            position_action=raw.get("position_action", "HOLD"),
            quantity=raw.get("quantity", 0.0),
            confidence=raw.get("confidence", 0.0),
            expected_outcome=raw.get("expected_outcome", ""),
            realized_outcome=raw.get("realized_outcome", ""),
            pnl=raw.get("pnl", 0.0),
            lesson=raw.get("lesson", ""),
            lifecycle_status=raw.get("lifecycle_status", "CLOSED"),
            order_id=raw.get("order_id"),
            fill_price=raw.get("fill_price"),
            closed_at=closed_at,
        )

    def _is_valid_experience(self, raw: Dict[str, Any]) -> bool:
        """Validate raw dict has required fields."""
        required = ["timestamp", "symbol", "regime", "position_action"]
        return all(field in raw for field in required)

    def log_experience(self, experience: TradeExperience) -> None:
        """Record a new trade experience (typically in PENDING_FILL or OPEN state).

        Parameters
        ----------
        experience : TradeExperience
            The experience to record.
        """
        self._experiences.append(experience)
        self._enforce_limits()
        self._save()

    def update_experience(
        self,
        decision_id: str,
        **updates: Any,
    ) -> TradeExperience:
        """Update an existing experience in place (P0-1 lifecycle).

        Only the experience with the matching ``decision_id`` is touched.
        All other fields are preserved.

        Returns
        -------
        TradeExperience
            The updated experience.

        Raises
        ------
        KeyError
            If no experience with the given decision_id is found.
        """
        for i, exp in enumerate(self._experiences):
            if exp.decision_id == decision_id:
                data = {
                    "decision_id": exp.decision_id,
                    "timestamp": exp.timestamp,
                    "symbol": exp.symbol,
                    "regime": exp.regime,
                    "regime_probabilities": exp.regime_probabilities,
                    "agent_signals": exp.agent_signals,
                    "ensemble_signal": exp.ensemble_signal,
                    "disagreement": exp.disagreement,
                    "effective_confidence": exp.effective_confidence,
                    "kalman_gain": exp.kalman_gain,
                    "kalman_price": exp.kalman_price,
                    "kalman_trend": exp.kalman_trend,
                    "capital_gate_verdict": exp.capital_gate_verdict,
                    "effective_cap": exp.effective_cap,
                    "state_charges": exp.state_charges,
                    "position_action": exp.position_action,
                    "quantity": exp.quantity,
                    "confidence": exp.confidence,
                    "expected_outcome": exp.expected_outcome,
                    "realized_outcome": exp.realized_outcome,
                    "pnl": exp.pnl,
                    "lesson": exp.lesson,
                    "lifecycle_status": exp.lifecycle_status,
                    "order_id": exp.order_id,
                    "fill_price": exp.fill_price,
                    "closed_at": exp.closed_at,
                }
                data.update(updates)
                updated = TradeExperience(**data)
                self._experiences[i] = updated
                self._save()
                return updated
        raise KeyError(f"No experience with decision_id={decision_id!r}")

    def close_trade(
        self,
        decision_id: str,
        realized_outcome: str,
        pnl: float,
        lesson: str = "",
        closed_at: Optional[datetime] = None,
    ) -> TradeExperience:
        """Mark a trade CLOSED with realized P&L (P0-1 lifecycle close).

        Returns
        -------
        TradeExperience
            The updated experience.
        """
        return self.update_experience(
            decision_id,
            lifecycle_status=TradeLifecycle.CLOSED.value,
            realized_outcome=realized_outcome,
            pnl=pnl,
            lesson=lesson,
            closed_at=closed_at or datetime.now(),
        )

    def get_by_decision_id(self, decision_id: str) -> Optional[TradeExperience]:
        """Look up a single experience by its decision_id."""
        for exp in self._experiences:
            if exp.decision_id == decision_id:
                return exp
        return None

    def _enforce_limits(self) -> None:
        """Enforce per-symbol memory limits."""
        by_symbol: Dict[str, List[TradeExperience]] = {}
        for exp in self._experiences:
            by_symbol.setdefault(exp.symbol, []).append(exp)

        self._experiences = []
        for symbol, exps in by_symbol.items():
            exps.sort(key=lambda e: e.timestamp)
            self._experiences.extend(exps[-MAX_MEMORY_PER_SYMBOL:])

    def find_similar(
        self,
        current: TradeExperience,
        top_k: int = 5,
        min_similarity: float = 0.3,
        exclude_decision_id: Optional[str] = None,
    ) -> List[SimilarExperience]:
        """Find historically similar experiences.

        Parameters
        ----------
        current : TradeExperience
            Current situation to match against.
        top_k : int
            Maximum number of similar experiences to return.
        min_similarity : float
            Minimum similarity score threshold.
        exclude_decision_id : Optional[str]
            If provided, skip experiences with this decision_id (P1-5: avoid
            retrieving the very trade you just recorded).

        Returns
        -------
        List[SimilarExperience]
            Ranked list of similar historical experiences, each carrying both
            the overall similarity and the market-only / portfolio-only
            decomposition (P1-4).
        """
        scored: List[SimilarExperience] = []

        for hist in self._experiences:
            if exclude_decision_id and hist.decision_id == exclude_decision_id:
                continue
            score, market_sim, portfolio_sim, reasons = self._compute_similarity(current, hist)
            if score >= min_similarity:
                scored.append(SimilarExperience(
                    experience=hist,
                    similarity_score=score,
                    market_similarity=market_sim,
                    portfolio_similarity=portfolio_sim,
                    match_reasons=reasons,
                ))

        scored.sort(key=lambda s: s.similarity_score, reverse=True)
        return scored[:top_k]

    def _compute_similarity(
        self,
        current: TradeExperience,
        historical: TradeExperience,
    ) -> Tuple[float, float, float, List[str]]:
        """Compute similarity between current and historical experiences.

        Uses weighted Euclidean distance with proper normalization for each feature.
        Features are normalized to [0,1] before comparison.

        Returns
        -------
        Tuple[float, float, float, List[str]]
            (overall_similarity, market_similarity, portfolio_similarity, reasons).
        """
        reasons: List[str] = []
        total_weight = 0.0
        weighted_score = 0.0
        market_weight = 0.0
        market_score = 0.0
        portfolio_weight = 0.0
        portfolio_score = 0.0

        def _accumulate(name: str, sim: float) -> None:
            nonlocal total_weight, weighted_score
            nonlocal market_weight, market_score
            nonlocal portfolio_weight, portfolio_score
            w = _SIMILARITY_WEIGHTS[name]
            weighted_score += w * sim
            total_weight += w
            if name in _MARKET_SIMILARITY_FEATURES:
                market_score += w * sim
                market_weight += w
            elif name in _PORTFOLIO_SIMILARITY_FEATURES:
                portfolio_score += w * sim
                portfolio_weight += w

        # 1. Regime probability vector similarity (highest weight)
        prob_sim = self._compute_probability_similarity(
            current.regime_probabilities, historical.regime_probabilities
        )
        _accumulate("regime_probabilities", prob_sim)
        if prob_sim > 0.8:
            reasons.append(f"Very similar regime distribution ({prob_sim:.2f})")
        elif prob_sim > 0.5:
            reasons.append(f"Similar regime distribution ({prob_sim:.2f})")

        # 2. Agent signals similarity
        agent_sim = self._compute_agent_signal_similarity(
            current.agent_signals, historical.agent_signals
        )
        _accumulate("agent_signals", agent_sim)
        if agent_sim > 0.7:
            reasons.append(f"Similar agent signals ({agent_sim:.2f})")

        # 3. Ensemble signal similarity (normalized to [0,1])
        sig_sim = self._normalized_similarity(
            current.ensemble_signal, historical.ensemble_signal, "ensemble_signal"
        )
        _accumulate("ensemble_signal", sig_sim)
        if sig_sim > 0.7:
            reasons.append(f"Similar ensemble signal ({sig_sim:.2f})")

        # 4. Disagreement similarity
        disag_sim = self._normalized_similarity(
            current.disagreement, historical.disagreement, "disagreement"
        )
        _accumulate("disagreement", disag_sim)

        # 5. Kalman gain similarity
        gain_sim = self._normalized_similarity(
            current.kalman_gain, historical.kalman_gain, "kalman_gain"
        )
        _accumulate("kalman_gain", gain_sim)

        # 6. Effective cap similarity
        cap_sim = self._normalized_similarity(
            current.effective_cap, historical.effective_cap, "effective_cap"
        )
        _accumulate("effective_cap", cap_sim)

        # 7. Confidence similarity
        conf_sim = self._normalized_similarity(
            current.confidence, historical.confidence, "confidence"
        )
        _accumulate("confidence", conf_sim)

        score = weighted_score / total_weight if total_weight > 0 else 0.0
        mkt = market_score / market_weight if market_weight > 0 else 0.0
        ptf = portfolio_score / portfolio_weight if portfolio_weight > 0 else 0.0
        return score, mkt, ptf, reasons

    def _normalized_similarity(
        self, current_val: float, historical_val: float, feature_name: str
    ) -> float:
        """Compute normalized similarity between two feature values.

        Normalizes values to [0,1] based on known ranges before computing
        1 - |current - historical|. This prevents scale-dependent features
        from dominating similarity.
        """
        min_val, max_val = _NORMALIZATION_RANGES.get(
            feature_name, (0.0, 1.0)
        )
        range_val = max_val - min_val
        if range_val <= 0:
            return 1.0 if current_val == historical_val else 0.0

        # Normalize to [0,1]
        current_norm = (current_val - min_val) / range_val
        historical_norm = (historical_val - min_val) / range_val

        # Clamp to [0,1]
        current_norm = max(0.0, min(1.0, current_norm))
        historical_norm = max(0.0, min(1.0, historical_norm))

        diff = abs(current_norm - historical_norm)
        return max(0.0, 1.0 - diff)

    def _compute_probability_similarity(
        self, current_probs: Dict[str, float], historical_probs: Dict[str, float]
    ) -> float:
        """Compute similarity between two regime probability distributions.

        Uses cosine similarity on the full 12-dimensional probability vector.
        This captures similarity in regime confidence, not just hard assignment.
        """
        all_regimes = sorted(set(current_probs.keys()) | set(historical_probs.keys()))
        if not all_regimes:
            return 0.0

        current_vec = np.array([current_probs.get(r, 0.0) for r in all_regimes])
        historical_vec = np.array([historical_probs.get(r, 0.0) for r in all_regimes])

        # Cosine similarity
        dot = np.dot(current_vec, historical_vec)
        norm_current = np.linalg.norm(current_vec)
        norm_historical = np.linalg.norm(historical_vec)

        if norm_current < 1e-10 or norm_historical < 1e-10:
            return 0.0

        return float(dot / (norm_current * norm_historical))

    def _compute_agent_signal_similarity(
        self, current_signals: Dict[str, float], historical_signals: Dict[str, float]
    ) -> float:
        """Compute similarity between agent signal configurations.

        Compares signals for agents present in both configurations.
        Uses normalized Euclidean distance on the common agent set.
        """
        common_agents = sorted(
            set(current_signals.keys()) & set(historical_signals.keys())
        )
        if not common_agents:
            return 0.0

        current_vec = np.array([current_signals[a] for a in common_agents])
        historical_vec = np.array([historical_signals[a] for a in common_agents])

        # Normalize by signal range [-1, 1]
        current_norm = current_vec / 1.0
        historical_norm = historical_vec / 1.0

        diff = np.linalg.norm(current_norm - historical_norm)
        max_diff = math.sqrt(len(common_agents))
        if max_diff <= 0:
            return 0.0

        return max(0.0, 1.0 - diff / max_diff)

    def get_history(self, symbol: Optional[str] = None) -> List[TradeExperience]:
        """Return experience history, optionally filtered by symbol.

        Parameters
        ----------
        symbol : Optional[str]
            If provided, return only experiences for this symbol.

        Returns
        -------
        List[TradeExperience]
            Experience history.
        """
        if symbol is None:
            return list(self._experiences)
        return [e for e in self._experiences if e.symbol == symbol]

    def get_performance_summary(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """Compute performance summary from experience history.

        Parameters
        ----------
        symbol : Optional[str]
            If provided, summarize only this symbol.

        Returns
        -------
        Dict[str, Any]
            Performance summary with win rate, avg P&L, etc.
        """
        exps = self.get_history(symbol)
        if not exps:
            return {"count": 0}

        pnls = [e.pnl for e in exps]
        wins = sum(1 for p in pnls if p > 0)
        losses = sum(1 for p in pnls if p < 0)

        return {
            "count": len(exps),
            "win_rate": wins / len(exps) if exps else 0.0,
            "avg_pnl": sum(pnls) / len(pnls) if pnls else 0.0,
            "total_pnl": sum(pnls),
            "wins": wins,
            "losses": losses,
            "best_trade": max(pnls) if pnls else 0.0,
            "worst_trade": min(pnls) if pnls else 0.0,
        }

    def clear(self) -> None:
        """Clear all experiences from memory."""
        self._experiences = []
        self._save()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "TradeExperience",
    "SimilarExperience",
    "TradeMemory",
    "DEFAULT_MEMORY_FILE",
]
