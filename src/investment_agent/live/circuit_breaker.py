"""Hierarchical circuit breaker above the execution layer.

WHAT
====
A 4-level state machine:

  NORMAL      -> unrestricted trading
  WARNING     -> reduce position sizes; tighter thresholds
  RESTRICTED  -> only high-confidence trades; no options
  HALT        -> no orders, ever; wait for human / cooldown

The breaker is evaluated against three signals:
  * current drawdown (vs peak equity)
  * consecutive losses (loss-streak)
  * daily loss cap (as fraction of starting equity)

The current level is the maximum across all signals. The breaker
is consulted *above* the execution layer -- the LLM and the
ensemble cannot override it. If HALT, no order may be submitted
regardless of the agents' opinions.

WHY
====
The user explicitly asked for the NORMAL -> WARNING -> RESTRICTED
-> HALT hierarchy. It also has to be observable: the dashboard
needs to know the current level and which signal triggered it.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional


class CircuitLevel(str, Enum):
    NORMAL = "NORMAL"
    WARNING = "WARNING"
    RESTRICTED = "RESTRICTED"
    HALT = "HALT"

    @property
    def rank(self) -> int:
        return {
            CircuitLevel.NORMAL: 0,
            CircuitLevel.WARNING: 1,
            CircuitLevel.RESTRICTED: 2,
            CircuitLevel.HALT: 3,
        }[self]


@dataclass(frozen=True)
class CircuitState:
    level: CircuitLevel
    drawdown_pct: float
    consecutive_losses: int
    daily_loss_pct: float
    triggered_signals: List[str]
    can_trade_equity: bool
    can_trade_options: bool
    can_trade_crypto: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "level": self.level.value,
            "drawdown_pct": float(self.drawdown_pct),
            "consecutive_losses": int(self.consecutive_losses),
            "daily_loss_pct": float(self.daily_loss_pct),
            "triggered_signals": list(self.triggered_signals),
            "can_trade_equity": bool(self.can_trade_equity),
            "can_trade_options": bool(self.can_trade_options),
            "can_trade_crypto": bool(self.can_trade_crypto),
        }


@dataclass
class CircuitBreaker:
    """Hierarchical circuit breaker.

    Parameters
    ----------
    drawdown_warning : float
        Drawdown at which the breaker moves to WARNING.
    drawdown_restricted : float
        Drawdown at which the breaker moves to RESTRICTED.
    drawdown_halt : float
        Drawdown at which the breaker moves to HALT.
    loss_streak_warning, loss_streak_restricted, loss_streak_halt : int
        Consecutive-loss thresholds for each level.
    daily_loss_warning, daily_loss_restricted, daily_loss_halt : float
        Daily-loss (as fraction of starting equity) thresholds.
    """

    drawdown_warning: float = 0.05
    drawdown_restricted: float = 0.10
    drawdown_halt: float = 0.15
    loss_streak_warning: int = 3
    loss_streak_restricted: int = 5
    loss_streak_halt: int = 7
    daily_loss_warning: float = 0.02
    daily_loss_restricted: float = 0.04
    daily_loss_halt: float = 0.06

    def evaluate(
        self,
        drawdown_pct: float,
        consecutive_losses: int,
        daily_loss_pct: float,
    ) -> CircuitState:
        """Evaluate the breaker and return the current state.

        Each input is mapped to a level; the overall level is the
        maximum across all three.
        """
        triggered: List[str] = []

        # Drawdown
        if drawdown_pct >= self.drawdown_halt:
            triggered.append(f"drawdown {drawdown_pct:.1%} >= {self.drawdown_halt:.0%}")
            dd_level = CircuitLevel.HALT
        elif drawdown_pct >= self.drawdown_restricted:
            triggered.append(f"drawdown {drawdown_pct:.1%} >= {self.drawdown_restricted:.0%}")
            dd_level = CircuitLevel.RESTRICTED
        elif drawdown_pct >= self.drawdown_warning:
            triggered.append(f"drawdown {drawdown_pct:.1%} >= {self.drawdown_warning:.0%}")
            dd_level = CircuitLevel.WARNING
        else:
            dd_level = CircuitLevel.NORMAL

        # Loss streak
        if consecutive_losses >= self.loss_streak_halt:
            triggered.append(f"loss_streak {consecutive_losses} >= {self.loss_streak_halt}")
            ls_level = CircuitLevel.HALT
        elif consecutive_losses >= self.loss_streak_restricted:
            triggered.append(f"loss_streak {consecutive_losses} >= {self.loss_streak_restricted}")
            ls_level = CircuitLevel.RESTRICTED
        elif consecutive_losses >= self.loss_streak_warning:
            triggered.append(f"loss_streak {consecutive_losses} >= {self.loss_streak_warning}")
            ls_level = CircuitLevel.WARNING
        else:
            ls_level = CircuitLevel.NORMAL

        # Daily loss
        if daily_loss_pct >= self.daily_loss_halt:
            triggered.append(f"daily_loss {daily_loss_pct:.1%} >= {self.daily_loss_halt:.0%}")
            dl_level = CircuitLevel.HALT
        elif daily_loss_pct >= self.daily_loss_restricted:
            triggered.append(f"daily_loss {daily_loss_pct:.1%} >= {self.daily_loss_restricted:.0%}")
            dl_level = CircuitLevel.RESTRICTED
        elif daily_loss_pct >= self.daily_loss_warning:
            triggered.append(f"daily_loss {daily_loss_pct:.1%} >= {self.daily_loss_warning:.0%}")
            dl_level = CircuitLevel.WARNING
        else:
            dl_level = CircuitLevel.NORMAL

        overall = max([dd_level, ls_level, dl_level], key=lambda lvl: lvl.rank)
        return CircuitState(
            level=overall,
            drawdown_pct=float(drawdown_pct),
            consecutive_losses=int(consecutive_losses),
            daily_loss_pct=float(daily_loss_pct),
            triggered_signals=triggered,
            can_trade_equity=overall != CircuitLevel.HALT,
            # Options are only allowed at NORMAL. WARNING can still
            # trade equity, but the spec says options need a quiet
            # regime.
            can_trade_options=overall == CircuitLevel.NORMAL,
            # Crypto is treated like equity for circuit-breaker
            # purposes: allowed at NORMAL / WARNING / RESTRICTED,
            # blocked only at HALT.
            can_trade_crypto=overall != CircuitLevel.HALT,
        )


__all__ = ["CircuitBreaker", "CircuitLevel", "CircuitState"]
