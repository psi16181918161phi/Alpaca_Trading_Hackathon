"""Live paper-trading orchestrator: continuous decision + outcome loop.

WHAT
====
A long-running driver that ties the entire pipeline into a single
process with state:

  decision interval -> candidate screener -> LLM specialists ->
  ensemble -> Kalman -> capital gate -> product gate -> risk
  gates -> order-state machine -> Alpaca paper execution
                                |
                                v
                         position manager
                                |
                                v
                on next interval: outcome evaluation
                (mark-to-market vs entry) -> close_trade
                -> reputation update -> next decision

State persisted across restarts in:
  * trade_memory.json   (TradeExperience lifecycle)
  * reputation_state.json (AgentReputationTracker)
  * live_state.json     (open positions, pending orders,
                          last decision timestamp, circuit-breaker
                          level)

WHY
====
The previous pieces (replay engine, paper-loop script) ran one
decision at a time and never reconciled what the broker actually
did. This module is the single entry point for the live paper
trading demo: it has a heartbeat, knows about open positions,
never calls the LLM more than once per decision interval, and
degrades gracefully if Alpaca is unreachable.

HOW
====
``LiveOrchestrator`` is composed of:
  * ``CandidateScreener``   - deterministic pre-filter (volume,
                              volatility, momentum) so the LLM
                              only sees the top-N candidates
  * ``OrderStateMachine``   - explicit transitions for every
                              submitted order, with terminal
                              states {FILLED, REJECTED, CANCELLED,
                              EXPIRED, FAILED}
  * ``PositionManager``     - tracks open positions, computes
                              unrealized P&L, decides when to close
  * ``CircuitBreaker``      - 4-level hierarchy
                              (NORMAL -> WARNING -> RESTRICTED
                              -> HALT) above the execution layer
  * ``LiveOrchestrator``    - ties them together with a decision
                              interval and outcome reconciliation
"""
from __future__ import annotations

from .candidate_screener import CandidateScreener, ScreenResult
from .circuit_breaker import CircuitBreaker, CircuitLevel
from .order_state_machine import OrderStateMachine, OrderTransition
from .position_manager import PositionManager
from .live_orchestrator import LiveOrchestrator, LiveOrchestratorConfig
from .session_controller import (
    DEFAULT_COMMAND_FILE,
    DEFAULT_STATUS_FILE,
    SessionController,
    SessionState,
    SessionStatus,
    read_session_status,
)

__all__ = [
    "CandidateScreener",
    "CircuitBreaker",
    "CircuitLevel",
    "DEFAULT_COMMAND_FILE",
    "DEFAULT_STATUS_FILE",
    "LiveOrchestrator",
    "LiveOrchestratorConfig",
    "OrderStateMachine",
    "OrderTransition",
    "PositionManager",
    "ScreenResult",
    "SessionController",
    "SessionState",
    "SessionStatus",
    "read_session_status",
]
