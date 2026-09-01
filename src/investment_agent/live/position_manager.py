"""Position manager: track open positions, mark to market, decide exits.

WHAT
====
For every filled order, the position manager opens a position
record. On every decision interval it re-marks to market against
the current price, and decides when to close the position:

  * target / stop hit
  * regime flipped against the original thesis
  * max-holding-time exceeded
  * explicit exit signal from the agents (SELL after a BUY)

When a position closes, the position manager emits an outcome
record the orchestrator can pass to ``close_trade`` for the
reputation / memory update.

WHY
====
Reputation must only update after a real outcome (see
"don't update reputation immediately after submission"). The
position manager is the single place that knows whether a trade
is still open or has produced a result. Everything else --
memory, reputation, the dashboard -- consumes its decisions.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional


class PositionSide(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"


@dataclass
class Position:
    """Open position record."""
    decision_id: str
    client_order_id: str
    symbol: str
    side: PositionSide
    quantity: float
    entry_price: float
    opened_at: datetime
    target_price: Optional[float] = None
    stop_price: Optional[float] = None
    product: str = "equity"
    option_side: Optional[str] = None
    last_mark_price: Optional[float] = None
    last_mark_at: Optional[datetime] = None

    def mark_to_market(self, current_price: float) -> float:
        """Update mark and return unrealized P&L in dollars."""
        self.last_mark_price = float(current_price)
        self.last_mark_at = datetime.now()
        if self.side == PositionSide.LONG:
            return (current_price - self.entry_price) * self.quantity
        return (self.entry_price - current_price) * self.quantity

    def to_dict(self) -> Dict[str, Any]:
        return {
            "decision_id": self.decision_id,
            "client_order_id": self.client_order_id,
            "symbol": self.symbol,
            "side": self.side.value,
            "quantity": float(self.quantity),
            "entry_price": float(self.entry_price),
            "opened_at": self.opened_at.isoformat(),
            "target_price": self.target_price,
            "stop_price": self.stop_price,
            "product": self.product,
            "option_side": self.option_side,
            "last_mark_price": self.last_mark_price,
            "last_mark_at": self.last_mark_at.isoformat() if self.last_mark_at else None,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Position":
        return cls(
            decision_id=d["decision_id"],
            client_order_id=d["client_order_id"],
            symbol=d["symbol"],
            side=PositionSide(d["side"]),
            quantity=float(d["quantity"]),
            entry_price=float(d["entry_price"]),
            opened_at=datetime.fromisoformat(d["opened_at"]),
            target_price=d.get("target_price"),
            stop_price=d.get("stop_price"),
            product=d.get("product", "equity"),
            option_side=d.get("option_side"),
            last_mark_price=d.get("last_mark_price"),
            last_mark_at=(
                datetime.fromisoformat(d["last_mark_at"])
                if d.get("last_mark_at") else None
            ),
        )


@dataclass
class ExitSignal:
    """Emitted by the position manager when a position should close."""
    decision_id: str
    symbol: str
    reason: str
    pnl: float
    pnl_pct: float
    holding_seconds: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "decision_id": self.decision_id,
            "symbol": self.symbol,
            "reason": self.reason,
            "pnl": float(self.pnl),
            "pnl_pct": float(self.pnl_pct),
            "holding_seconds": float(self.holding_seconds),
        }


@dataclass
class PositionManager:
    """Track open positions and emit exit signals.

    Parameters
    ----------
    default_target_pct : float
        Default take-profit as a fraction of entry price.
    default_stop_pct : float
        Default stop-loss as a fraction of entry price.
    max_holding : timedelta
        Hard cap on how long a position can stay open.
    """

    default_target_pct: float = 0.05
    default_stop_pct: float = 0.03
    max_holding: timedelta = field(default_factory=lambda: timedelta(hours=24))
    _positions: Dict[str, Position] = field(default_factory=dict)

    def open_position(
        self,
        decision_id: str,
        client_order_id: str,
        symbol: str,
        side: str,
        quantity: float,
        entry_price: float,
        product: str = "equity",
        option_side: Optional[str] = None,
        target_pct: Optional[float] = None,
        stop_pct: Optional[float] = None,
    ) -> Position:
        """Open a new position. Auto-derives target / stop from the
        default percentages when not supplied."""
        ps = PositionSide.LONG if side.lower() == "buy" else PositionSide.SHORT
        tp = target_pct if target_pct is not None else self.default_target_pct
        sp = stop_pct if stop_pct is not None else self.default_stop_pct
        if ps == PositionSide.LONG:
            target_price = entry_price * (1.0 + tp)
            stop_price = entry_price * (1.0 - sp)
        else:
            target_price = entry_price * (1.0 - tp)
            stop_price = entry_price * (1.0 + sp)
        pos = Position(
            decision_id=decision_id,
            client_order_id=client_order_id,
            symbol=symbol,
            side=ps,
            quantity=float(quantity),
            entry_price=float(entry_price),
            opened_at=datetime.now(),
            target_price=target_price,
            stop_price=stop_price,
            product=product,
            option_side=option_side,
        )
        self._positions[decision_id] = pos
        return pos

    def get(self, decision_id: str) -> Optional[Position]:
        return self._positions.get(decision_id)

    def all_open(self) -> List[Position]:
        return list(self._positions.values())

    def close(self, decision_id: str) -> Optional[Position]:
        return self._positions.pop(decision_id, None)

    def evaluate(
        self, decision_id_to_price: Dict[str, float]
    ) -> List[ExitSignal]:
        """Mark every open position to market and emit exits where
        appropriate (target / stop / max-holding)."""
        exits: List[ExitSignal] = []
        to_remove: List[str] = []
        now = datetime.now()
        for did, pos in self._positions.items():
            price = decision_id_to_price.get(did) or decision_id_to_price.get(pos.symbol)
            if price is None or price <= 0:
                continue
            pnl = pos.mark_to_market(price)
            pnl_pct = (price - pos.entry_price) / pos.entry_price
            if pos.side == PositionSide.SHORT:
                pnl_pct = -pnl_pct
            reason = ""
            if pos.target_price is not None:
                if pos.side == PositionSide.LONG and price >= pos.target_price:
                    reason = "target_hit"
                elif pos.side == PositionSide.SHORT and price <= pos.target_price:
                    reason = "target_hit"
            if not reason and pos.stop_price is not None:
                if pos.side == PositionSide.LONG and price <= pos.stop_price:
                    reason = "stop_hit"
                elif pos.side == PositionSide.SHORT and price >= pos.stop_price:
                    reason = "stop_hit"
            if not reason and (now - pos.opened_at) >= self.max_holding:
                reason = "max_holding_exceeded"
            if reason:
                exits.append(ExitSignal(
                    decision_id=did,
                    symbol=pos.symbol,
                    reason=reason,
                    pnl=float(pnl),
                    pnl_pct=float(pnl_pct),
                    holding_seconds=(now - pos.opened_at).total_seconds(),
                ))
                to_remove.append(did)
        for did in to_remove:
            self._positions.pop(did, None)
        return exits

    def to_dict(self) -> Dict[str, Any]:
        return {"positions": [p.to_dict() for p in self._positions.values()]}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PositionManager":
        pm = cls()
        for pd_ in d.get("positions", []):
            pos = Position.from_dict(pd_)
            pm._positions[pos.decision_id] = pos
        return pm


__all__ = [
    "ExitSignal",
    "Position",
    "PositionManager",
    "PositionSide",
]
