"""Order state machine: explicit lifecycle for every submitted order.

WHAT
====
Every order submitted to Alpaca goes through an explicit state
machine. The state is persisted to disk (live_state.json) so a
crash mid-order doesn't leave an unknown-position state.

States::

  SUBMITTED  -- order placed with broker, awaiting acceptance
  ACCEPTED   -- broker accepted, awaiting fill
  PARTIALLY_FILLED -- partial fill recorded
  FILLED     -- completely filled; becomes a position
  REJECTED   -- broker rejected; terminal
  CANCELLED  -- cancelled before fill; terminal
  EXPIRED    -- expired (e.g. day order); terminal
  FAILED     -- internal error; terminal

Only ``FILLED`` orders are forwarded to the position manager.
``REJECTED`` and ``FAILED`` orders trigger a reputation update
on the agents that voted for the trade (a vote for a
REJECTED trade is "wrong").
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional


logger = logging.getLogger(__name__)


class OrderState(str, Enum):
    SUBMITTED = "SUBMITTED"
    ACCEPTED = "ACCEPTED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    REJECTED = "REJECTED"
    CANCELLED = "CANCELLED"
    EXPIRED = "EXPIRED"
    FAILED = "FAILED"

    @property
    def is_terminal(self) -> bool:
        return self in {
            OrderState.FILLED,
            OrderState.REJECTED,
            OrderState.CANCELLED,
            OrderState.EXPIRED,
            OrderState.FAILED,
        }

    @property
    def is_filled(self) -> bool:
        return self == OrderState.FILLED


# Allowed transitions (from -> set of valid to-states).
ALLOWED_TRANSITIONS: Dict[OrderState, set] = {
    OrderState.SUBMITTED: {OrderState.ACCEPTED, OrderState.REJECTED, OrderState.FAILED, OrderState.EXPIRED},
    OrderState.ACCEPTED: {OrderState.FILLED, OrderState.PARTIALLY_FILLED, OrderState.CANCELLED, OrderState.EXPIRED, OrderState.FAILED},
    OrderState.PARTIALLY_FILLED: {OrderState.FILLED, OrderState.CANCELLED, OrderState.EXPIRED, OrderState.FAILED},
    OrderState.FILLED: set(),
    OrderState.REJECTED: set(),
    OrderState.CANCELLED: set(),
    OrderState.EXPIRED: set(),
    OrderState.FAILED: set(),
}


@dataclass
class OrderTransition:
    """Single transition event in the order's life."""
    from_state: OrderState
    to_state: OrderState
    timestamp: datetime
    note: str = ""
    fill_qty: Optional[float] = None
    fill_price: Optional[float] = None


@dataclass
class OrderRecord:
    """Persistent record of one order.

    Attributes
    ----------
    client_order_id : str
        Local identifier; stable across restarts.
    broker_order_id : Optional[str]
        Alpaca's order ID once SUBMITTED.
    decision_id : str
        The decision that produced this order.
    symbol : str
        Underlying or OCC option symbol.
    side : str
        "buy" / "sell".
    qty : float
        Intended quantity.
    product : str
        "equity" / "option" / "none".
    option_side : Optional[str]
        "call" / "put" when product == "option".
    created_at : datetime
    state : OrderState
    transitions : List[OrderTransition]
    fill_qty : float
        Total quantity filled so far.
    fill_price : Optional[float]
        Weighted-average fill price.
    error : Optional[str]
        Populated on REJECTED / FAILED.
    """
    client_order_id: str
    decision_id: str
    symbol: str
    side: str
    qty: float
    product: str
    created_at: datetime
    option_side: Optional[str] = None
    broker_order_id: Optional[str] = None
    state: OrderState = OrderState.SUBMITTED
    transitions: List[OrderTransition] = field(default_factory=list)
    fill_qty: float = 0.0
    fill_price: Optional[float] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "client_order_id": self.client_order_id,
            "broker_order_id": self.broker_order_id,
            "decision_id": self.decision_id,
            "symbol": self.symbol,
            "side": self.side,
            "qty": float(self.qty),
            "product": self.product,
            "option_side": self.option_side,
            "created_at": self.created_at.isoformat(),
            "state": self.state.value,
            "transitions": [
                {
                    "from": t.from_state.value,
                    "to": t.to_state.value,
                    "timestamp": t.timestamp.isoformat(),
                    "note": t.note,
                    "fill_qty": t.fill_qty,
                    "fill_price": t.fill_price,
                }
                for t in self.transitions
            ],
            "fill_qty": float(self.fill_qty),
            "fill_price": self.fill_price,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "OrderRecord":
        rec = cls(
            client_order_id=d["client_order_id"],
            decision_id=d["decision_id"],
            symbol=d["symbol"],
            side=d["side"],
            qty=float(d["qty"]),
            product=d.get("product", "equity"),
            created_at=datetime.fromisoformat(d["created_at"]),
            option_side=d.get("option_side"),
            broker_order_id=d.get("broker_order_id"),
            state=OrderState(d.get("state", "SUBMITTED")),
            fill_qty=float(d.get("fill_qty", 0.0)),
            fill_price=d.get("fill_price"),
            error=d.get("error"),
        )
        for t in d.get("transitions", []):
            rec.transitions.append(OrderTransition(
                from_state=OrderState(t["from"]),
                to_state=OrderState(t["to"]),
                timestamp=datetime.fromisoformat(t["timestamp"]),
                note=t.get("note", ""),
                fill_qty=t.get("fill_qty"),
                fill_price=t.get("fill_price"),
            ))
        return rec


class OrderStateMachine:
    """Bookkeeping for every order the orchestrator has submitted."""

    def __init__(self, state_file: str = "live_state.json") -> None:
        self._state_file = state_file
        self._orders: Dict[str, OrderRecord] = {}
        self._load()

    # ----- persistence -----

    def _load(self) -> None:
        if not os.path.exists(self._state_file):
            return
        try:
            with open(self._state_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            for d in data.get("orders", []):
                rec = OrderRecord.from_dict(d)
                self._orders[rec.client_order_id] = rec
        except (json.JSONDecodeError, OSError, KeyError, ValueError) as e:
            logger.warning("Failed to load order state machine: %s", e)

    def _save(self) -> None:
        Path(self._state_file).parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "orders": [r.to_dict() for r in self._orders.values()],
            "saved_at": datetime.now().isoformat(),
        }
        fd, tmp = tempfile.mkstemp(
            prefix=Path(self._state_file).name + ".", suffix=".tmp",
            dir=str(Path(self._state_file).parent),
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, sort_keys=True)
            os.replace(tmp, self._state_file)
        except Exception:
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise

    # ----- order management -----

    def register(
        self,
        client_order_id: str,
        decision_id: str,
        symbol: str,
        side: str,
        qty: float,
        product: str,
        option_side: Optional[str] = None,
    ) -> OrderRecord:
        """Register a new order in SUBMITTED state."""
        rec = OrderRecord(
            client_order_id=client_order_id,
            decision_id=decision_id,
            symbol=symbol,
            side=side,
            qty=float(qty),
            product=product,
            created_at=datetime.now(),
            option_side=option_side,
            state=OrderState.SUBMITTED,
        )
        rec.transitions.append(OrderTransition(
            from_state=OrderState.SUBMITTED,
            to_state=OrderState.SUBMITTED,
            timestamp=datetime.now(),
            note="registered",
        ))
        self._orders[client_order_id] = rec
        self._save()
        return rec

    def set_broker_id(self, client_order_id: str, broker_order_id: str) -> None:
        if client_order_id in self._orders:
            self._orders[client_order_id].broker_order_id = broker_order_id
            self._save()

    def transition(
        self,
        client_order_id: str,
        to_state: OrderState,
        *,
        note: str = "",
        fill_qty: Optional[float] = None,
        fill_price: Optional[float] = None,
        error: Optional[str] = None,
    ) -> OrderRecord:
        """Move an order to ``to_state``. Raises if the transition is invalid."""
        if client_order_id not in self._orders:
            raise KeyError(f"Unknown client_order_id={client_order_id!r}")
        rec = self._orders[client_order_id]
        allowed = ALLOWED_TRANSITIONS.get(rec.state, set())
        if to_state not in allowed:
            raise ValueError(
                f"Invalid transition for {client_order_id}: {rec.state.value} -> {to_state.value}"
            )
        rec.transitions.append(OrderTransition(
            from_state=rec.state,
            to_state=to_state,
            timestamp=datetime.now(),
            note=note,
            fill_qty=fill_qty,
            fill_price=fill_price,
        ))
        rec.state = to_state
        if fill_qty is not None:
            rec.fill_qty = float(fill_qty)
        if fill_price is not None:
            rec.fill_price = float(fill_price)
        if error is not None:
            rec.error = error
        self._save()
        return rec

    def get(self, client_order_id: str) -> Optional[OrderRecord]:
        return self._orders.get(client_order_id)

    def open_orders(self) -> List[OrderRecord]:
        """Return all orders that are not yet terminal."""
        return [r for r in self._orders.values() if not r.state.is_terminal]

    def filled_orders(self) -> List[OrderRecord]:
        return [r for r in self._orders.values() if r.state.is_filled]

    def all_orders(self) -> List[OrderRecord]:
        return list(self._orders.values())


__all__ = [
    "ALLOWED_TRANSITIONS",
    "OrderRecord",
    "OrderState",
    "OrderStateMachine",
    "OrderTransition",
]
