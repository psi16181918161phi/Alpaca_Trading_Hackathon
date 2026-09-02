"""Fill Reconciler — Broker ↔ TradeMemory reconciliation after fills.

WHAT
====
Three complementary responsibilities:

1. **FillReconciler.reconcile(memory)**
   Walk every PENDING_FILL experience in TradeMemory.  For each, poll
   the Alpaca broker for the real order status and apply the fill:
   - ``filled``          → mark OPEN, record fill_price & filled_qty
   - ``partially_filled``→ update filled_qty but stay PENDING_FILL
   - ``rejected``        → mark REJECTED, no P&L
   - ``cancelled``/``expired`` → mark CANCELLED

2. **FillReconciler.recover_pending_orders(memory)**
   Called at process startup to re-attach to any orders that were
   in-flight when the process last died.  Iterates PENDING_FILL records
   and performs the same reconciliation poll so the system can resume
   cleanly after a restart.

3. **Regime-by-regime performance attribution** (pure helper, no broker I/O)
   ``performance_by_regime(memory)`` groups CLOSED experiences by regime
   label and produces Sharpe-ratio, win-rate, avg-H/avg-L breakdowns
   comparable across the 12 HMM states.

WHY
===
Without reconciliation:
- An order that filled at $152.30 would remain PENDING_FILL forever.
- P&L would never be realized in TradeMemory.
- Reputation updates would never fire.
- After a crash, every pending order would be orphaned.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from investment_agent.memory.trade_memory import (
    TradeExperience,
    TradeLifecycle,
    TradeMemory,
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _safe_float(v: Any, default: float = 0.0) -> float:
    if v is None:
        return default
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# FillReconciler
# ---------------------------------------------------------------------------

class FillReconciler:
    """Reconcile Alpaca broker order states against TradeMemory.

    Parameters
    ----------
    poll_timeout_seconds : float
        Per-order maximum wait inside poll_order_status.
        Keep short (≤5 s) for reconciliation sweeps; longer for live waits.
    poll_interval_seconds : float
        Sleep between Alpaca GET requests.
    verbose : bool
        Print one line per reconciled order.
    """

    def __init__(
        self,
        poll_timeout_seconds: float = 5.0,
        poll_interval_seconds: float = 0.5,
        verbose: bool = True,
    ) -> None:
        self._timeout = poll_timeout_seconds
        self._interval = poll_interval_seconds
        self._verbose = verbose

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reconcile(self, memory: TradeMemory) -> Dict[str, int]:
        """Reconcile all PENDING_FILL trades against Alpaca broker state.

        Returns
        -------
        dict
            Counts of transitions: filled, partially_filled, rejected,
            cancelled, timed_out, skipped (no order_id).
        """
        return self._sweep(memory, tag="reconcile")

    def recover_pending_orders(self, memory: TradeMemory) -> Dict[str, int]:
        """Re-attach to in-flight orders after a process restart.

        Same logic as reconcile(), but logs with a 'recovery' tag so
        the distinction is visible in logs.
        """
        return self._sweep(memory, tag="recovery")

    # ------------------------------------------------------------------
    # Core sweep
    # ------------------------------------------------------------------

    def _sweep(self, memory: TradeMemory, tag: str) -> Dict[str, int]:
        from investment_agent.execution.execution import poll_order_status  # lazy import

        counts: Dict[str, int] = {
            "filled": 0,
            "partially_filled": 0,
            "rejected": 0,
            "cancelled": 0,
            "timed_out": 0,
            "skipped": 0,
        }

        pending: List[TradeExperience] = [
            e for e in memory.get_history()
            if e.lifecycle_status == TradeLifecycle.PENDING_FILL.value
        ]

        for exp in pending:
            if not exp.order_id:
                counts["skipped"] += 1
                continue

            snap = poll_order_status(
                exp.order_id,
                timeout_seconds=self._timeout,
                poll_interval_seconds=self._interval,
            )
            broker_status = snap.get("status", "unknown")
            filled_qty = _safe_float(snap.get("filled_qty"))
            fill_price = _safe_float(snap.get("filled_avg_price") or 0.0)

            if snap.get("timed_out"):
                counts["timed_out"] += 1
                if self._verbose:
                    print(f"[{tag}] {exp.order_id[:8]} TIMED_OUT (still pending)")
                continue

            if broker_status == "filled":
                memory.update_experience(
                    exp.decision_id,
                    lifecycle_status=TradeLifecycle.OPEN.value,
                    fill_price=fill_price,
                    quantity=filled_qty if filled_qty > 0 else exp.quantity,
                )
                counts["filled"] += 1
                if self._verbose:
                    print(
                        f"[{tag}] {exp.order_id[:8]} FILLED "
                        f"qty={filled_qty} @ ${fill_price:.4f}"
                    )

            elif broker_status == "partially_filled":
                memory.update_experience(
                    exp.decision_id,
                    fill_price=fill_price,
                    quantity=filled_qty if filled_qty > 0 else exp.quantity,
                )
                counts["partially_filled"] += 1
                if self._verbose:
                    print(
                        f"[{tag}] {exp.order_id[:8]} PARTIAL "
                        f"qty={filled_qty}/{exp.quantity} @ ${fill_price:.4f}"
                    )

            elif broker_status in ("rejected",):
                memory.update_experience(
                    exp.decision_id,
                    lifecycle_status=TradeLifecycle.REJECTED.value,
                    realized_outcome="rejected by broker",
                )
                counts["rejected"] += 1
                if self._verbose:
                    print(f"[{tag}] {exp.order_id[:8]} REJECTED")

            elif broker_status in ("cancelled", "expired"):
                memory.update_experience(
                    exp.decision_id,
                    lifecycle_status=TradeLifecycle.CANCELLED.value,
                    realized_outcome=f"broker status: {broker_status}",
                )
                counts["cancelled"] += 1
                if self._verbose:
                    print(f"[{tag}] {exp.order_id[:8]} CANCELLED ({broker_status})")

        return counts


# ---------------------------------------------------------------------------
# Regime performance attribution
# ---------------------------------------------------------------------------

def performance_by_regime(
    memory: TradeMemory,
    symbol: Optional[str] = None,
) -> Dict[str, Dict[str, Any]]:
    """Compute regime-by-regime performance attribution from CLOSED trades.

    Groups experiences by their HMM regime label (R01–R12) and computes:

    - ``count``         : number of closed trades
    - ``win_rate``      : fraction with pnl > 0
    - ``avg_pnl``       : mean realized P&L per trade
    - ``total_pnl``     : sum of realized P&L
    - ``best_trade``    : max single-trade P&L
    - ``worst_trade``   : min single-trade P&L
    - ``pnl_std``       : standard deviation of P&L
    - ``sharpe``        : avg_pnl / pnl_std  (0 when std ≈ 0)
    - ``avg_signal``    : mean ensemble_signal at entry
    - ``avg_confidence``: mean effective_confidence at entry
    - ``avg_disagreement``: mean disagreement at entry
    - ``avg_kalman_gain``: mean Kalman gain at entry

    Returns
    -------
    dict
        Keys are regime labels (e.g. ``"R01"``); values are the
        attribution dicts above. Only regimes with at least one
        CLOSED trade appear.
    """
    exps = [
        e for e in memory.get_history(symbol=symbol)
        if e.lifecycle_status == TradeLifecycle.CLOSED.value
    ]

    by_regime: Dict[str, List[TradeExperience]] = {}
    for exp in exps:
        by_regime.setdefault(exp.regime, []).append(exp)

    result: Dict[str, Dict[str, Any]] = {}
    for regime, group in sorted(by_regime.items()):
        pnls = [e.pnl for e in group]
        wins = sum(1 for p in pnls if p > 0)
        avg_pnl = sum(pnls) / len(pnls)
        variance = sum((p - avg_pnl) ** 2 for p in pnls) / len(pnls)
        std = math.sqrt(variance) if variance > 0 else 0.0
        sharpe = avg_pnl / std if std > 1e-9 else 0.0

        result[regime] = {
            "count": len(group),
            "win_rate": wins / len(group),
            "avg_pnl": avg_pnl,
            "total_pnl": sum(pnls),
            "best_trade": max(pnls),
            "worst_trade": min(pnls),
            "pnl_std": std,
            "sharpe": sharpe,
            "avg_signal": sum(e.ensemble_signal for e in group) / len(group),
            "avg_confidence": sum(e.effective_confidence for e in group) / len(group),
            "avg_disagreement": sum(e.disagreement for e in group) / len(group),
            "avg_kalman_gain": sum(e.kalman_gain for e in group) / len(group),
        }

    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "FillReconciler",
    "performance_by_regime",
]
