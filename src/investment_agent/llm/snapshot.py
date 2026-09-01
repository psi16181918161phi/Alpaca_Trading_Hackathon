"""Compact LLM state snapshot.

WHAT
====
Builds a tiny dict-of-numbers the LLM layer can read in one glance instead
of dumping the entire price history, memory log, or portfolio state.

WHY
====
Featherless credits are scarce. Sending the model a 10 kB context every
call is wasteful. The deterministic pipeline already owns the heavy
arithmetic; the LLM only needs the *summary* numbers that drive a
specialist's directional view.

HOW
====
``build_snapshot(symbol, prices, regime, ensemble, portfolio, memory)``
returns a dict with the fields listed in the spec:

    symbol, regime, price, return_1d, volatility, volume_change,
    kalman_signal, soc_state, ensemble_signal, disagreement,
    portfolio_exposure, risk_flags, relevant_experiences

The snapshot is *read-only* — it never reaches Alpaca. The specialist
adapters pass it through their prompt templates.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from ..memory.trade_memory import SimilarExperience
from ..regimes.regimes import VALID_REGIMES


# ---------------------------------------------------------------------------
# Numerical utilities
# ---------------------------------------------------------------------------

def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(f) or math.isinf(f):
        return default
    return f


def _pct_change(current: float, prior: float) -> float:
    if prior == 0.0:
        return 0.0
    return (current - prior) / abs(prior)


def _volatility(returns: Sequence[float]) -> float:
    if len(returns) < 2:
        return 0.0
    mean = sum(returns) / len(returns)
    var = sum((r - mean) ** 2 for r in returns) / max(1, len(returns) - 1)
    return math.sqrt(var)


# ---------------------------------------------------------------------------
# Snapshot builder
# ---------------------------------------------------------------------------

def build_snapshot(
    symbol: str,
    prices: Sequence[float],
    *,
    volumes: Optional[Sequence[float]] = None,
    regime: Optional[str] = None,
    regime_probabilities: Optional[Dict[str, float]] = None,
    kalman_price: Optional[float] = None,
    kalman_trend: Optional[float] = None,
    ensemble_signal: Optional[float] = None,
    disagreement: Optional[float] = None,
    portfolio_exposure: Optional[float] = None,
    risk_flags: Optional[Sequence[str]] = None,
    relevant_experiences: Optional[Sequence[SimilarExperience]] = None,
    lookback: int = 20,
) -> Dict[str, Any]:
    """Build a compact state snapshot for an LLM prompt.

    Parameters
    ----------
    symbol : str
        Trading symbol.
    prices : Sequence[float]
        Historical close prices (most recent last). The snapshot uses only
        the last ``lookback`` values.
    volumes : Optional[Sequence[float]]
        Same alignment as ``prices``.
    regime : Optional[str]
        Active regime label (e.g. ``"R01"``).
    regime_probabilities : Optional[Dict[str, float]]
        Full regime posterior; only the top-3 entries are kept.
    kalman_price, kalman_trend : Optional[float]
        Latest Kalman filter outputs.
    ensemble_signal, disagreement : Optional[float]
        Latest ensemble metrics.
    portfolio_exposure : Optional[float]
        Current portfolio exposure as a fraction of total capital.
    risk_flags : Optional[Sequence[str]]
        Triggered risk rules (e.g. ``["LIQ-001"]``).
    relevant_experiences : Optional[Sequence[SimilarExperience]]
        Up to 3 similar past trades (top similarity, top P&L, etc).
    lookback : int
        Window size for return / volatility / volume features.

    Returns
    -------
    Dict[str, Any]
        Snapshot with primitive types only.
    """
    recent_prices = list(prices[-lookback:]) if prices else []
    if not recent_prices:
        latest_price = _safe_float(kalman_price, 0.0)
        return_1d = 0.0
        realized_vol = 0.0
    else:
        latest_price = _safe_float(recent_prices[-1], 0.0)
        return_1d = _pct_change(latest_price, recent_prices[-2]) if len(recent_prices) >= 2 else 0.0
        returns = [
            _pct_change(recent_prices[i], recent_prices[i - 1])
            for i in range(1, len(recent_prices))
        ]
        realized_vol = _volatility(returns)

    if volumes and len(volumes) == len(prices) and len(volumes) >= 2:
        last_vol = _safe_float(volumes[-1], 0.0)
        prev_vol = _safe_float(volumes[-2], 1.0) or 1.0
        volume_change = _pct_change(last_vol, prev_vol)
    else:
        volume_change = 0.0

    top_regimes = []
    if regime_probabilities:
        sorted_probs = sorted(
            regime_probabilities.items(),
            key=lambda kv: kv[1],
            reverse=True,
        )
        top_regimes = [
            {"regime": k, "p": _safe_float(v)}
            for k, v in sorted_probs[:3]
        ]

    risk_flags_list = [str(f) for f in (risk_flags or []) if f]

    compact_memory = []
    for s in (relevant_experiences or [])[:3]:
        compact_memory.append({
            "decision_id": s.experience.decision_id,
            "regime": s.experience.regime,
            "action": s.experience.position_action,
            "pnl": _safe_float(s.experience.pnl),
            "similarity": _safe_float(s.similarity_score),
        })

    if regime is not None and regime not in VALID_REGIMES:
        regime = None  # don't propagate garbage

    return {
        "symbol": str(symbol),
        "regime": regime,
        "price": latest_price,
        "return_1d": return_1d,
        "volatility": realized_vol,
        "volume_change": volume_change,
        "kalman_signal": _safe_float(kalman_trend),
        "kalman_price": _safe_float(kalman_price, latest_price),
        "soc_state": _safe_float(portfolio_exposure),
        "ensemble_signal": _safe_float(ensemble_signal),
        "disagreement": _safe_float(disagreement),
        "portfolio_exposure": _safe_float(portfolio_exposure),
        "risk_flags": risk_flags_list,
        "top_regimes": top_regimes,
        "relevant_experiences": compact_memory,
    }


# ---------------------------------------------------------------------------
# Pre-screening (skip LLM calls when nothing meaningful changed)
# ---------------------------------------------------------------------------

@dataclass
class PreScreenResult:
    """Outcome of the deterministic pre-screening step."""

    should_call_llm: bool
    reason: str
    snapshot: Dict[str, Any]


def pre_screen(
    symbol: str,
    prices: Sequence[float],
    *,
    regime: Optional[str] = None,
    ensemble_signal: Optional[float] = None,
    risk_flags: Optional[Sequence[str]] = None,
    previous_snapshot: Optional[Dict[str, Any]] = None,
    min_abs_return: float = 0.001,
    min_abs_ensemble: float = 0.10,
    lookback: int = 20,
) -> PreScreenResult:
    """Decide whether an LLM call is justified.

    Rules (any one passes → call the LLM):
        - absolute 1-day return > ``min_abs_return``
        - absolute ensemble signal > ``min_abs_ensemble``
        - any risk flag is set
        - no previous snapshot (first call)
    """
    snap = build_snapshot(
        symbol=symbol,
        prices=prices,
        regime=regime,
        ensemble_signal=ensemble_signal,
        risk_flags=risk_flags,
        lookback=lookback,
    )
    if previous_snapshot is None:
        # First call: always call the LLM.
        return PreScreenResult(True, "first call", snap)
    if risk_flags:
        return PreScreenResult(True, f"risk flag: {risk_flags[0]}", snap)
    if abs(snap["return_1d"]) >= min_abs_return:
        return PreScreenResult(True, f"|return_1d|={abs(snap['return_1d']):.4f} >= {min_abs_return}", snap)
    if abs(snap["ensemble_signal"]) >= min_abs_ensemble:
        return PreScreenResult(True, f"|ensemble|={abs(snap['ensemble_signal']):.2f} >= {min_abs_ensemble}", snap)
    return PreScreenResult(False, "no meaningful change", snap)


__all__ = [
    "build_snapshot",
    "pre_screen",
    "PreScreenResult",
]
