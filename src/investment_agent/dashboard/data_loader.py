"""X Quant X dashboard data loader.

WHAT
====
Read-only data access layer for the dashboard. Loads trade history from
trade_memory.json (regime, signal, and risk-gate data) and reads live account
state (equity, positions, orders) through execution.py's read-only helpers.

WHY
===
alpaca_paper_trading_specifications_x_quant_x/022 and 004 both require that
xquantx/dashboard/ never import the Alpaca TradingClient directly -- that
keeps order-submission capability out of the monitoring layer entirely. This
module only ever calls the read-only functions already exposed by
investment_agent.execution.execution (get_account_summary, get_positions,
get_order_history); it never imports alpaca.trading.client itself.

HOW
===
Every public function here is defensive: if the trade memory file is
missing/empty, or the Alpaca account call fails (no credentials, market
closed, network hiccup), the function returns an empty/placeholder result
instead of raising, so one bad data source never takes down the whole
dashboard.
"""

from __future__ import annotations

import json
import math
import os
from typing import Any, Dict, List

DEFAULT_TRADE_MEMORY_FILE = "trade_memory.json"


def load_trade_history(path: str = DEFAULT_TRADE_MEMORY_FILE) -> List[Dict[str, Any]]:
    """Load trade_memory.json as a list of experience dicts, oldest first.

    Returns an empty list if the file doesn't exist or can't be parsed --
    the dashboard should show an empty-state chart, not crash.
    """
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return []
    if not isinstance(data, list):
        return []
    return sorted(data, key=lambda row: row.get("timestamp", ""))


def compute_equity_curve(history: List[Dict[str, Any]], starting_equity: float = 100000.0) -> List[Dict[str, Any]]:
    """Derive a cumulative equity series from per-bar pnl in trade history.

    This is an approximation (starting_equity + running sum of recorded pnl),
    not the broker's own equity ledger -- it's what's actually available from
    trade_memory.json today. If Alpaca's real portfolio history endpoint gets
    wired in later, this is the function to swap out.
    """
    equity = starting_equity
    peak = starting_equity
    curve = []
    for row in history:
        equity += float(row.get("pnl", 0.0) or 0.0)
        peak = max(peak, equity)
        drawdown_pct = 0.0 if peak == 0 else (equity - peak) / peak
        curve.append({
            "timestamp": row.get("timestamp"),
            "equity": equity,
            "peak": peak,
            "drawdown_pct": drawdown_pct,
        })
    return curve


def compute_regime_entropy(regime_probabilities: Dict[str, float]) -> float:
    """Normalized Shannon entropy of a regime probability distribution, in [0, 1]."""
    probs = [p for p in regime_probabilities.values() if p and p > 0]
    if not probs:
        return 0.0
    n = len(regime_probabilities) or 1
    raw_entropy = -sum(p * math.log(p) for p in probs)
    max_entropy = math.log(n) if n > 1 else 1.0
    return raw_entropy / max_entropy if max_entropy else 0.0


def get_account_summary_safe() -> Dict[str, Any]:
    """Read-only account summary via execution.get_account_summary(), never raises."""
    try:
        from investment_agent.execution.execution import get_account_summary
        return {"ok": True, **get_account_summary()}
    except Exception as exc:  # noqa: BLE001 - dashboard must never crash on a bad API call
        return {"ok": False, "error": str(exc)}


def get_positions_safe() -> Dict[str, Any]:
    """Read-only open positions via execution.get_positions(), never raises."""
    try:
        from investment_agent.execution.execution import get_positions
        return {"ok": True, "positions": get_positions()}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc), "positions": []}


def get_order_history_safe(limit: int = 100) -> Dict[str, Any]:
    """Read-only order history via execution.get_order_history(), never raises."""
    try:
        from investment_agent.execution.execution import get_order_history
        return {"ok": True, "orders": get_order_history(limit=limit)}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc), "orders": []}


def get_risk_gate_log(history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Build the P06 risk-gate log from trade_memory.json's capital_gate_verdict per bar.

    There is no separate AUDIT_LOGS/ file in this codebase yet, so this reads
    the same verdict the capital gate actually produced per cycle (the most
    truthful source currently available) rather than fabricating a log.
    """
    rows = []
    for row in history:
        rows.append({
            "timestamp": row.get("timestamp"),
            "rule_id": "capital_gate",
            "verdict": row.get("capital_gate_verdict", "UNKNOWN"),
            "symbol": row.get("symbol", ""),
            "measured_value": row.get("effective_cap"),
            "threshold": 1.0,
        })
    return rows
