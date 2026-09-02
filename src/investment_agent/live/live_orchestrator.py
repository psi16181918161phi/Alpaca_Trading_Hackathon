"""Live paper-trading orchestrator.

WHAT
====
Single long-running process that:

  1. Pulls bars for the candidate universe via the unified
     market-data interface.
  2. Runs the deterministic screener to pick the top N symbols.
  3. For each candidate, runs the seven specialist agents (LLM
     or deterministic factory), the ensemble, the investment
     Kalman filter, the 7-state capital gate, the product gate,
     the risk gates, and the order-state machine.
  4. Reconciles open positions every interval: marks to market,
     emits exits, calls close_trade, updates reputation.
  5. Persists state across restarts in live_state.json.

WHY
====
The previous pieces ran one cycle at a time and never reconciled
broker state. The live orchestrator is the integration glue.

HOW
====
``LiveOrchestrator`` is constructed once with a market-data
client, an LLM provider (or a deterministic agent factory for
offline / CI), an executor callable, and a decision interval.
Calling ``run_once()`` runs a single decision interval; calling
``run()`` runs forever in a loop.
"""
from __future__ import annotations

import json
import logging
import math
import os
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd

from ..agents.specialist import AgentContext, run_agents
from ..capital.capital_gate import SevenStateVector
from ..data.market_data import BarRequest, MarketDataClient
from ..orchestrator import XQuantXOrchestrator
from ..products import ProductGate, ProductGateInput
from ..products.product_gate import (
    OPTION_CALL, OPTION_PUT, PRODUCT_CRYPTO, PRODUCT_EQUITY, PRODUCT_NONE, PRODUCT_OPTION,
)
from ..regimes.hmm_regime_detector import HMMRegimeDetector
from ..utils.asset_class import is_crypto_symbol
from .candidate_screener import CandidateScreener, ScreenResult
from .circuit_breaker import CircuitBreaker, CircuitLevel, CircuitState
from .order_state_machine import (
    OrderRecord, OrderState, OrderStateMachine,
)
from .position_manager import ExitSignal, PositionManager


logger = logging.getLogger(__name__)


# Callable signatures for the few dependencies that need an
# explicit injection point (LLM, executor, screener, etc.).
AgentFactory = Callable[[Dict[str, Any]], List[Any]]
"""Builds seven ``AgentOutput`` objects for a single bar context.

The dict carries: timestamp, symbol, prices_so_far, volumes_so_far,
current_price, recent_return, regime, regime_probabilities, features.
The factory must return exactly seven AgentOutput objects.
"""

Executor = Callable[[str, str, float, Optional[str]], Dict[str, Any]]
"""Submits an order. Signature:
    executor(symbol, side, qty, option_side_or_None) -> result dict
The result must contain at least ``{"id": str|None, "status": str,
"error": str|None}`` so the order state machine can record it.
"""


@dataclass
class LiveOrchestratorConfig:
    """Configuration for the live orchestrator."""
    symbol_universe: List[str] = field(default_factory=lambda: [
        "AAPL", "SPY", "MSFT", "TSLA", "NVDA",
        "BTC/USD", "ETH/USD", "SOL/USD", "AVAX/USD",
        "LINK/USD", "XRP/USD", "DOGE/USD", "RENDER/USD",
    ])
    top_n_candidates: int = 2
    decision_interval_seconds: int = 60
    state_file: str = "live_state.json"
    memory_file: str = "trade_memory.json"
    reputation_file: str = "reputation_state.json"
    lookback_days: int = 60
    starting_equity: float = 100_000.0
    max_lookups_per_interval: int = 7   # cap on total LLM calls per decision
    target_pct: float = 0.05
    stop_pct: float = 0.03
    max_holding: timedelta = field(default_factory=lambda: timedelta(hours=24))
    # Stage gating: dry_run -> log only; paper -> live Alpaca paper orders
    stage: str = "dry_run"  # "dry_run" or "paper"


@dataclass
class IntervalReport:
    """Summary of one decision interval."""
    timestamp: datetime
    interval_index: int
    candidates: List[str]
    decisions: List[Dict[str, Any]]
    exits: List[Dict[str, Any]]
    orders: List[Dict[str, Any]]
    circuit_state: Optional[Dict[str, Any]]
    regime: str
    ensemble_signal: float
    disagreement: float
    drawdown_pct: float
    consecutive_losses: int
    daily_loss_pct: float
    open_positions: int
    equity: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "interval_index": self.interval_index,
            "candidates": list(self.candidates),
            "decisions": self.decisions,
            "exits": self.exits,
            "orders": self.orders,
            "circuit_state": self.circuit_state,
            "regime": self.regime,
            "ensemble_signal": float(self.ensemble_signal),
            "disagreement": float(self.disagreement),
            "drawdown_pct": float(self.drawdown_pct),
            "consecutive_losses": int(self.consecutive_losses),
            "daily_loss_pct": float(self.daily_loss_pct),
            "open_positions": int(self.open_positions),
            "equity": float(self.equity),
        }


class LiveOrchestrator:
    """The single entry point for live paper trading."""

    def __init__(
        self,
        config: LiveOrchestratorConfig,
        market_data: MarketDataClient,
        orchestrator: XQuantXOrchestrator,
        agent_factory: AgentFactory,
        executor: Executor,
        screener: Optional[CandidateScreener] = None,
        product_gate: Optional[ProductGate] = None,
        circuit_breaker: Optional[CircuitBreaker] = None,
    ) -> None:
        self.config = config
        self._md = market_data
        self._orch = orchestrator
        self._agent_factory = agent_factory
        self._executor = executor
        self._screener = screener or CandidateScreener(
            top_n=config.top_n_candidates,
        )
        self._product_gate = product_gate or ProductGate()
        self._circuit = circuit_breaker or CircuitBreaker()
        self._positions = PositionManager(
            default_target_pct=config.target_pct,
            default_stop_pct=config.stop_pct,
            max_holding=config.max_holding,
        )
        self._orders = OrderStateMachine(state_file=config.state_file)
        self._interval_index = 0
        self._last_interval_at: Optional[datetime] = None
        self._equity = config.starting_equity
        self._peak_equity = config.starting_equity
        self._consecutive_losses = 0
        self._daily_start_equity = config.starting_equity
        self._last_reputation_save: Optional[datetime] = None
        self._last_alpaca_account: Optional[Dict[str, Any]] = None
        self._load_state()

    # ----- persistence -----

    def _load_state(self) -> None:
        path = self.config.state_file
        if not os.path.exists(path):
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._interval_index = int(data.get("interval_index", 0))
            ts = data.get("last_interval_at")
            self._last_interval_at = datetime.fromisoformat(ts) if ts else None
            self._equity = float(data.get("equity", self.config.starting_equity))
            self._peak_equity = float(data.get("peak_equity", self._equity))
            self._consecutive_losses = int(data.get("consecutive_losses", 0))
            dse = data.get("daily_start_equity")
            if dse is not None:
                self._daily_start_equity = float(dse)
            if "positions" in data:
                self._positions = PositionManager.from_dict(data["positions"])
        except (json.JSONDecodeError, OSError, KeyError, ValueError) as e:
            logger.warning("Failed to load live state: %s", e)

        # Restore reputation tracker if reputation_file exists
        if self.config.reputation_file and os.path.exists(self.config.reputation_file):
            from ..agents.reputation_persistence import load_reputation
            restored = load_reputation(self.config.reputation_file)
            if restored is not None:
                self._orch._reputation_tracker = restored
                logger.info("Restored persisted agent reputation state from %s", self.config.reputation_file)

    def _save_state(self) -> None:
        payload = {
            "interval_index": self._interval_index,
            "last_interval_at": (
                self._last_interval_at.isoformat() if self._last_interval_at else None
            ),
            "equity": float(self._equity),
            "peak_equity": float(self._peak_equity),
            "consecutive_losses": int(self._consecutive_losses),
            "daily_start_equity": float(self._daily_start_equity),
            "positions": self._positions.to_dict(),
            "alpaca_account": self._last_alpaca_account,
            "saved_at": datetime.now().isoformat(),
        }
        path = self.config.state_file
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(
            prefix=Path(path).name + ".", suffix=".tmp",
            dir=str(Path(path).parent),
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, sort_keys=True)
            os.replace(tmp, path)
        except Exception:
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise

    def _save_reputation(self) -> None:
        from ..agents.reputation_persistence import save_reputation
        try:
            save_reputation(self._orch._reputation_tracker, self.config.reputation_file)
            self._last_reputation_save = datetime.now()
        except Exception as e:
            logger.warning("Failed to persist reputation: %s", e)

    def _refresh_alpaca_account(self) -> Optional[Dict[str, Any]]:
        """Query the Alpaca account endpoint and cache the snapshot.

        Only active when the loop is in ``paper`` or ``competition``
        stage and Alpaca credentials are present. Failure is non-fatal
        -- the dashboard falls back to the previously cached snapshot,
        and the strategy side keeps running on its own accounting.
        """
        if self.config.stage not in {"paper", "competition"}:
            return None
        try:
            from ..execution.execution import get_account_snapshot
            snap = get_account_snapshot()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Alpaca account snapshot failed: %s", exc)
            return self._last_alpaca_account
        self._last_alpaca_account = snap
        # The broker's current equity becomes the strategy-side
        # ``_equity`` once we have a valid snapshot -- the live loop
        # had been bootstrapping from a synthetic starting_equity.
        if snap.get("ok") and snap.get("equity") is not None:
            try:
                eq = float(snap["equity"])
                if eq > 0:
                    self._equity = eq
                    self._peak_equity = max(self._peak_equity, eq)
            except (TypeError, ValueError):
                pass
        return snap

    # ----- main loop -----

    def _fetch_universe(self) -> Dict[str, pd.DataFrame]:
        end = datetime.now()
        start = end - timedelta(days=self.config.lookback_days)
        out: Dict[str, pd.DataFrame] = {}
        for sym in self.config.symbol_universe:
            try:
                bars = self._md.get_historical_bars(BarRequest(
                    symbol=sym, start=start, end=end, timeframe="1Day",
                ))
                if bars is not None and not bars.empty:
                    out[sym] = bars
            except Exception as e:
                logger.warning("Failed to fetch bars for %s: %s", sym, e)
        return out

    def _compute_drawdown_and_daily_loss(self) -> Tuple[float, float]:
        # Drawdown is a non-negative percentage (loss from peak). Daily
        # loss is a non-negative percentage (loss from session start).
        dd = max(0.0, (self._peak_equity - self._equity) / self._peak_equity) \
            if self._peak_equity > 0 else 0.0
        daily = max(0.0, (self._daily_start_equity - self._equity) / self._daily_start_equity) \
            if self._daily_start_equity > 0 else 0.0
        return dd, daily

    @staticmethod
    def _compute_order_notional(effective_cap: float, equity: float) -> float:
        """Return the dollar notional a candidate wants to deploy.

        ``effective_cap`` is the fraction ``[0, 1]`` produced by the
        capital gate.  The live orchestrator is the production layer,
        so we convert that fraction to an actual dollar notional here
        (the orchestrator's ``_compute_position_size`` is a test stub
        that returns the raw fraction).
        """
        return max(0.0, float(effective_cap) * float(equity))

    def _allocate_capital(
        self,
        decisions: List[Dict[str, Any]],
        liquidity_floor: float = 5_000.0,
    ) -> List[Dict[str, Any]]:
        """Apply the EXISTING capital-architecture constraints across
        multiple simultaneous BUY candidates.

        The authoritative rules preserved here are:
          * FLATTEN / BLOCK / HOLD → never submit
          * REDUCE → submit at reduced effective_cap
          * ALLOW → submit at full effective_cap
          * Aggregate notional must leave the liquidity floor intact
          * Per-position cap: ``MAX_POSITION_PCT`` of equity
          * No double-spending: each candidate's notional is measured
            against the *current* available liquidity, not the original

        Candidates are ranked by effective_cap descending so the
        capital gate's own assessment of "deserves more deployment"
        gets priority.  The ranking is NOT a new scoring formula; it
        is the capital gate's own output.
        """
        from investment_agent.execution.execution import MAX_POSITION_PCT

        equity = self._equity if self._equity > 0 else 100_000.0
        max_per_position = equity * float(MAX_POSITION_PCT)
        available = max(0.0, equity - liquidity_floor)

        # Separate by verdict tier (lexicographic priority).
        def _tier(d: Dict[str, Any]) -> int:
            v = str(d.get("verdict", "")).upper()
            if v in ("FLATTEN", "BLOCK"):
                return 3
            if v == "REDUCE":
                return 2
            if d.get("action") == "HOLD" or d.get("product") == "none":
                return 3
            return 1  # ALLOW or no verdict

        def _effective_cap(d: Dict[str, Any]) -> float:
            return float(d.get("effective_cap", 0.0) or 0.0)

        buy_decisions = [d for d in decisions if d.get("action") == "BUY"]
        buy_decisions.sort(key=lambda d: (_tier(d), -_effective_cap(d)))

        committed = 0.0
        approved: List[Dict[str, Any]] = []
        for d in buy_decisions:
            if _tier(d) == 3:
                d["_deferred_reason"] = "verdict blocks deployment"
                approved.append(d)
                continue
            cap = _effective_cap(d)
            if cap <= 0:
                d["_deferred_reason"] = "zero effective_cap"
                approved.append(d)
                continue
            notional = self._compute_order_notional(cap, equity)
            notional = min(notional, max_per_position)
            if committed + notional > available:
                d["_deferred_reason"] = (
                    f"aggregate deployment ${committed + notional:,.2f} "
                    f"exceeds available ${available:,.2f}"
                )
                approved.append(d)
                continue
            committed += notional
            d["_approved_notional"] = notional
            approved.append(d)
        return approved

    def _evaluate_candidate(
        self,
        screen: ScreenResult,
        bar_ctx: Dict[str, Any],
        circuit: CircuitState,
    ) -> Dict[str, Any]:
        """Run the full pipeline for a single screened candidate.

        Returns a dict containing the evaluated decision plus all
        intermediate results needed to submit the order later
        (``experience``, ``pg``, ``actual_qty``, etc.).
        No order is submitted and no position is opened.
        """
        # 1. Run the 7 agents via the factory.
        agent_outputs = self._agent_factory(bar_ctx)
        if len(agent_outputs) != len(self._orch._agent_ids):
            return {
                "decision": {
                    "symbol": screen.symbol,
                    "error": (
                        f"agent_factory returned {len(agent_outputs)} outputs, "
                        f"expected {len(self._orch._agent_ids)}"
                    ),
                    "agent_signals": {
                        a.agent_id: float(getattr(a, "s", 0.0))
                        for a in agent_outputs
                    },
                },
                "experience": None,
                "pg": None,
                "actual_qty": 0.0,
                "screen": screen,
                "bar_ctx": bar_ctx,
                "circuit": circuit,
                "agent_outputs": agent_outputs,
            }

        # 2. Regime + dynamic 7-State SoC + portfolio risk context.
        prices = bar_ctx["prices"]
        volumes = bar_ctx.get("volumes") or [0.0] * len(prices)
        highs = bar_ctx.get("highs")
        lows = bar_ctx.get("lows")

        regime = self._orch._classify_regime(
            prices=prices, volumes=volumes, highs=highs, lows=lows
        )

        open_pos_list = self._positions.all_open()
        equity_val = self._equity if self._equity > 0 else 100_000.0
        total_pos_val = sum(
            abs(float(p.quantity) * float(p.last_mark_price or p.entry_price or 1.0))
            for p in open_pos_list
        )
        pos_pct = min(1.0, total_pos_val / equity_val)
        gross_lev = min(2.0, total_pos_val / equity_val)
        existing_sym_pos = [p for p in open_pos_list if p.symbol == screen.symbol]
        is_new_long = len(existing_sym_pos) == 0
        avail_liq = max(0.0, equity_val - total_pos_val)
        dd, daily_loss = self._compute_drawdown_and_daily_loss()

        avg_confidence = float(
            sum(getattr(a, "confidence", 0.8) or 0.8 for a in agent_outputs)
            / max(1, len(agent_outputs))
        )
        recent_ret = float(bar_ctx.get("recent_return", 0.0) or 0.0)
        states = SevenStateVector(
            economic=max(0.1, min(1.0, 1.0 - abs(recent_ret) * 2.0)),
            financial=max(0.1, min(1.0, 1.0 - daily_loss)),
            fiscal=max(0.1, min(1.0, 1.0 - dd)),
            portfolio=max(0.1, min(1.0, 1.0 - pos_pct)),
            fundamental=max(0.1, min(1.0, avg_confidence)),
            market=max(0.1, min(1.0, 1.0 - abs(recent_ret) * 5.0)),
            sector=max(0.1, min(1.0, 1.0 - gross_lev / 2.0)),
        )

        portfolio_context = {
            "position_pct": pos_pct,
            "gross_leverage": gross_lev,
            "entropy": 0.1,
            "drawdown_pct": max(0.0, dd),
            "execution_timeout_seconds": 5.0,
            "sector_exposure_pct": pos_pct,
            "is_new_long": is_new_long,
            "regime": regime.regime,
            "available_liquidity": avail_liq,
        }

        cycle = self._orch.run_cycle(
            prices=prices, volumes=volumes,
            agent_outputs=agent_outputs, states=states,
            portfolio_context=portfolio_context,
        )
        experience = cycle.experience

        pg = self._product_gate.decide(ProductGateInput(
            action=experience.position_action,
            verdict=experience.capital_gate_verdict,
            ensemble_signal=experience.ensemble_signal,
            disagreement=experience.disagreement,
            confidence=experience.effective_confidence,
            regime=regime.regime,
            symbol=screen.symbol,
        ))

        can_equity = circuit.can_trade_equity
        can_option = circuit.can_trade_options
        can_crypto = getattr(circuit, "can_trade_crypto", True)

        current_price = float(bar_ctx.get("current_price", 0.0) or 0.0)
        raw_cap = float(getattr(experience, "quantity", 0.0) or 0.0)
        notional = self._compute_order_notional(raw_cap, self._equity)
        if pg.product == PRODUCT_OPTION:
            contract_mult = 100.0
            actual_qty = max(1.0, notional / (current_price * contract_mult)) if current_price > 0 else 1.0
        elif pg.product == PRODUCT_CRYPTO:
            actual_qty = notional / current_price if current_price > 0 else 0.0
        else:
            actual_qty = math.floor(notional / current_price) if current_price > 0 else 0.0

        # Build the decision dict (same shape as the old return value).
        if pg.product == PRODUCT_OPTION and not can_option:
            decision = {
                "symbol": screen.symbol, "regime": regime.regime,
                "action": experience.position_action,
                "verdict": experience.capital_gate_verdict,
                "ensemble_signal": experience.ensemble_signal,
                "disagreement": experience.disagreement,
                "product": "none", "reason": f"circuit={circuit.level.value} blocks options",
                "experience_id": experience.decision_id,
                "agent_signals": self._agent_signals_from(experience, agent_outputs),
            }
        elif pg.product == PRODUCT_CRYPTO and not can_crypto:
            decision = {
                "symbol": screen.symbol, "regime": regime.regime,
                "action": experience.position_action,
                "verdict": experience.capital_gate_verdict,
                "ensemble_signal": experience.ensemble_signal,
                "disagreement": experience.disagreement,
                "product": "none", "reason": f"circuit={circuit.level.value} blocks crypto",
                "experience_id": experience.decision_id,
                "agent_signals": self._agent_signals_from(experience, agent_outputs),
            }
        elif pg.product == PRODUCT_EQUITY and not can_equity:
            decision = {
                "symbol": screen.symbol, "regime": regime.regime,
                "action": experience.position_action,
                "verdict": experience.capital_gate_verdict,
                "ensemble_signal": experience.ensemble_signal,
                "disagreement": experience.disagreement,
                "product": "none", "reason": f"circuit={circuit.level.value} blocks all trades",
                "experience_id": experience.decision_id,
                "agent_signals": self._agent_signals_from(experience, agent_outputs),
            }
        elif pg.product == PRODUCT_NONE or experience.position_action == "HOLD":
            decision = {
                "symbol": screen.symbol, "regime": regime.regime,
                "action": experience.position_action,
                "verdict": experience.capital_gate_verdict,
                "ensemble_signal": experience.ensemble_signal,
                "disagreement": experience.disagreement,
                "product": "none", "reason": pg.reason,
                "experience_id": experience.decision_id,
                "agent_signals": self._agent_signals_from(experience, agent_outputs),
            }
        else:
            decision = {
                "symbol": screen.symbol, "regime": regime.regime,
                "action": experience.position_action,
                "verdict": experience.capital_gate_verdict,
                "ensemble_signal": experience.ensemble_signal,
                "disagreement": experience.disagreement,
                "product": pg.product, "option_side": pg.option_side,
                "quantity": actual_qty,
                "effective_cap": raw_cap,
                "notional": notional,
                "reason": pg.reason,
                "experience_id": experience.decision_id,
                "agent_signals": self._agent_signals_from(experience, agent_outputs),
            }

        return {
            "decision": decision,
            "experience": experience,
            "pg": pg,
            "actual_qty": actual_qty,
            "screen": screen,
            "bar_ctx": bar_ctx,
            "circuit": circuit,
            "agent_outputs": agent_outputs,
        }

    def _submit_from_evaluation(self, evaluated: Dict[str, Any]) -> Dict[str, Any]:
        """Submit the order for an already-evaluated candidate.

        Takes the dict returned by ``_evaluate_candidate``, calls the
        executor, updates the order state machine, and opens positions
        for fills.  Returns the updated decision dict with order fields.
        """
        experience = evaluated["experience"]
        pg = evaluated["pg"]
        actual_qty = evaluated["actual_qty"]
        screen = evaluated["screen"]
        bar_ctx = evaluated["bar_ctx"]
        circuit = evaluated["circuit"]
        agent_outputs = evaluated["agent_outputs"]
        decision = dict(evaluated["decision"])

        if not experience or not pg or pg.product == PRODUCT_NONE or experience.position_action == "HOLD":
            return decision

        client_order_id = f"co-{uuid.uuid4().hex[:12]}"
        option_side = pg.option_side if pg.product == PRODUCT_OPTION else None
        order_result = self._executor(
            screen.symbol, experience.position_action.lower(),
            actual_qty, option_side,
        )
        self._orders.register(
            client_order_id=client_order_id,
            decision_id=experience.decision_id,
            symbol=screen.symbol,
            side=experience.position_action.lower(),
            qty=actual_qty,
            product=pg.product,
            option_side=option_side,
        )
        if order_result.get("id"):
            self._orders.set_broker_id(client_order_id, str(order_result["id"]))
        broker_status = (order_result.get("status") or "").lower()
        if broker_status in {"rejected", "failed", "error"} or order_result.get("error"):
            self._orders.transition(
                client_order_id, OrderState.REJECTED,
                note=order_result.get("error") or broker_status,
                error=order_result.get("error"),
            )
        elif broker_status in {"accepted", "new", "pending_new"}:
            self._orders.transition(
                client_order_id, OrderState.ACCEPTED, note="broker accepted",
            )
        elif broker_status in {"filled", "partially_filled"}:
            self._orders.transition(
                client_order_id, OrderState.ACCEPTED, note="broker accepted",
            )
            self._orders.transition(
                client_order_id, OrderState.FILLED,
                note="broker reported fill",
                fill_qty=float(order_result.get("filled_qty", actual_qty)),
                fill_price=float(order_result.get("filled_avg_price", 0.0) or 0.0),
            )
            self._positions.open_position(
                decision_id=experience.decision_id,
                client_order_id=client_order_id,
                symbol=screen.symbol,
                side=experience.position_action.lower(),
                quantity=float(order_result.get("filled_qty", actual_qty)),
                entry_price=float(order_result.get("filled_avg_price", 0.0) or 0.0),
                product=pg.product,
                option_side=option_side,
            )

        decision.update({
            "client_order_id": client_order_id,
            "broker_order_id": order_result.get("id"),
            "order_status": order_result.get("status"),
            "filled_qty": float(order_result.get("filled_qty", 0.0) or 0.0),
            "filled_avg_price": float(order_result.get("filled_avg_price", 0.0) or 0.0),
            "quantity": actual_qty,
        })
        return decision

    def _run_one_candidate(
        self,
        screen: ScreenResult,
        bar_ctx: Dict[str, Any],
        circuit: CircuitState,
        submit: bool = True,
    ) -> Dict[str, Any]:
        """Run the full pipeline for a single screened candidate.

        Parameters
        ----------
        submit : bool
            When True (default) the order is submitted and positions are
            opened.  When False the method returns the evaluated decision
            without side-effects so the caller can run an aggregate
            capital-allocation pass first.
        """
        evaluated = self._evaluate_candidate(screen, bar_ctx, circuit)
        decision = evaluated["decision"]
        if submit and decision.get("product") not in (None, "none") and decision.get("action") != "HOLD":
            decision = self._submit_from_evaluation(evaluated)
        return decision

    @staticmethod
    def _agent_signals_from(experience, agent_outputs) -> Dict[str, float]:
        """Return per-agent signal map for the report printer.

        Prefers the 8-channel ``agent_outputs_full`` dict that
        ``run_cycle`` already built and persisted on the experience;
        falls back to the raw ``AgentOutput.s`` if for any reason the
        orchestrator did not populate it (e.g. factory-mismatch path).
        """
        full = getattr(experience, "agent_outputs_full", None)
        if full:
            return {aid: float(row.get("signal", 0.0))
                    for aid, row in full.items()}
        return {a.agent_id: float(getattr(a, "s", 0.0))
                for a in agent_outputs}

    def _reconcile_outcomes(self) -> List[Dict[str, Any]]:
        """Mark open positions to market, emit exits, call close_trade
        so reputation updates fire. Returns a list of exit records."""
        # Mark using last close from the most recent bar in memory
        marks: Dict[str, float] = {}
        for pos in self._positions.all_open():
            last_price = pos.last_mark_price
            if last_price is None:
                last_price = pos.entry_price
            marks[pos.decision_id] = float(last_price)
        exits = self._positions.evaluate(marks)
        out: List[Dict[str, Any]] = []
        for ex in exits:
            # Close the trade -- this fires the reputation update.
            try:
                closed = self._orch.close_trade(
                    decision_id=ex.decision_id,
                    realized_outcome=(
                        "win" if ex.pnl > 0 else "loss" if ex.pnl < 0 else "breakeven"
                    ),
                    pnl=ex.pnl,
                    lesson=f"live-loop exit ({ex.reason})",
                )
            except Exception as e:
                logger.warning("close_trade failed for %s: %s", ex.decision_id, e)
                closed = None
            # Update equity + loss streak.
            self._equity += ex.pnl
            if ex.pnl >= 0:
                self._consecutive_losses = 0
            else:
                self._consecutive_losses += 1
            self._peak_equity = max(self._peak_equity, self._equity)
            out.append({
                "decision_id": ex.decision_id,
                "symbol": ex.symbol,
                "reason": ex.reason,
                "pnl": ex.pnl,
                "pnl_pct": ex.pnl_pct,
                "holding_seconds": ex.holding_seconds,
                "realized_outcome": closed.realized_outcome if closed else None,
            })
        return out

    def run_once(self) -> IntervalReport:
        """One decision interval. Returns a structured report."""
        self._interval_index += 1
        now = datetime.now()
        self._last_interval_at = now

        # 0. Refresh the Alpaca account snapshot (paper / competition only).
        #     This populates ``_equity`` and ``_last_alpaca_account`` so the
        #     strategy-side drawdown + daily-loss math is consistent with
        #     what the broker actually reports.
        self._refresh_alpaca_account()

        # 1. Fetch universe
        universe = self._fetch_universe()
        # 2. Screen
        candidates = self._screener.screen(universe)
        # 3. Circuit breaker
        dd, daily = self._compute_drawdown_and_daily_loss()
        circuit = self._circuit.evaluate(
            drawdown_pct=dd,
            consecutive_losses=self._consecutive_losses,
            daily_loss_pct=daily,
        )

        # 4. Pass 1 — evaluate every screened candidate through the
        #    complete pipeline (7 agents → HMM → ensemble → Kalman → SoC
        #    → product gate → circuit breaker) WITHOUT submitting orders.
        #    This guarantees that capital-gate and risk-gate inputs are
        #    identical for every candidate (no earlier order skews later
        #    evaluations).
        evaluated: List[Dict[str, Any]] = []
        decisions: List[Dict[str, Any]] = []
        orders: List[Dict[str, Any]] = []
        llm_calls = 0
        ensemble_signal = 0.0
        disagreement = 0.0
        regime_label = "n/a"
        for screen in candidates:
            if llm_calls >= self.config.max_lookups_per_interval:
                break
            sym = screen.symbol
            bars = universe.get(sym)
            if bars is None or bars.empty:
                continue
            prices = bars["close"].astype(float).tolist()
            volumes = bars["volume"].astype(float).tolist() if "volume" in bars.columns else [0.0] * len(bars)
            current_price = float(prices[-1])
            recent_return = (prices[-1] - prices[-2]) / prices[-2] if len(prices) >= 2 and prices[-2] > 0 else 0.0
            from ..regimes.market_feature_extractor import compute_dict_features
            features_dict = compute_dict_features(prices, volumes)
            bar_ctx = {
                "timestamp": bars.index[-1].to_pydatetime() if hasattr(bars.index[-1], "to_pydatetime") else bars.index[-1],
                "symbol": sym,
                "prices": prices, "volumes": volumes,
                "current_price": current_price, "recent_return": recent_return,
                "regime": "R00", "regime_probabilities": {},
                "features": features_dict,
                "equity": self._equity, "drawdown_pct": -dd,
            }
            ev = self._evaluate_candidate(screen, bar_ctx, circuit)
            evaluated.append(ev)
            decisions.append(ev["decision"])
            llm_calls += 1
            ensemble_signal = ev["decision"].get("ensemble_signal", ensemble_signal)
            disagreement = ev["decision"].get("disagreement", disagreement)
            regime_label = ev["decision"].get("regime", regime_label)

        # 4b. Aggregate capital allocation — the EXISTING authoritative
        #     rules (liquidity floor, per-position cap, diversification,
        #     circuit breakers, lexicographic verdict priority) decide
        #     which of the simultaneous BUY candidates actually deploy.
        approved = self._allocate_capital(decisions)

        # 4c. Pass 2 — submit only the orders that survived allocation,
        #     reusing the pass-1 evaluation so the pipeline runs exactly
        #     once per candidate (no double LLM / HMM / Kalama calls).
        decision_to_eval = {id(ev["decision"]): ev for ev in evaluated}
        for d in approved:
            if d.get("_deferred_reason"):
                continue
            if d.get("product") in (None, "none") or d.get("action") == "HOLD":
                continue
            ev = decision_to_eval.get(id(d))
            if ev is None:
                continue
            final_decision = self._submit_from_evaluation(ev)
            orders.append({
                "client_order_id": final_decision.get("client_order_id"),
                "symbol": final_decision["symbol"],
                "product": final_decision["product"],
                "option_side": final_decision.get("option_side"),
                "order_status": final_decision.get("order_status"),
            })
            idx = next((i for i, dd in enumerate(decisions)
                        if id(dd) == id(d)), None)
            if idx is not None:
                decisions[idx] = final_decision

        # 5. Reconcile any open positions (mark to market, exits)
        exits = self._reconcile_outcomes()

        # 6. Persist state + reputation
        self._save_state()
        self._save_reputation()

        report = IntervalReport(
            timestamp=now,
            interval_index=self._interval_index,
            candidates=[c.symbol for c in candidates],
            decisions=decisions,
            exits=exits,
            orders=orders,
            circuit_state=circuit.to_dict() if circuit else None,
            regime=regime_label,
            ensemble_signal=ensemble_signal,
            disagreement=disagreement,
            drawdown_pct=dd,
            consecutive_losses=self._consecutive_losses,
            daily_loss_pct=daily,
            open_positions=len(self._positions.all_open()),
            equity=self._equity,
        )
        return report

    def run(self, max_intervals: Optional[int] = None) -> List[IntervalReport]:
        """Run forever (or up to ``max_intervals``) at the configured
        decision interval."""
        reports: List[IntervalReport] = []
        try:
            while max_intervals is None or self._interval_index < max_intervals:
                report = self.run_once()
                reports.append(report)
                logger.info(
                    "interval %d done: %d candidates, %d decisions, %d exits, equity=%.2f",
                    report.interval_index, len(report.candidates),
                    len(report.decisions), len(report.exits), report.equity,
                )
                if max_intervals is None or self._interval_index < max_intervals:
                    time.sleep(self.config.decision_interval_seconds)
        except KeyboardInterrupt:
            logger.info("live orchestrator interrupted")
        return reports


__all__ = [
    "AgentFactory",
    "Executor",
    "IntervalReport",
    "LiveOrchestrator",
    "LiveOrchestratorConfig",
]
