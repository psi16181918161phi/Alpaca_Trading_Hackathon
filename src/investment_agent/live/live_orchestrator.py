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
    OPTION_CALL, OPTION_PUT, PRODUCT_EQUITY, PRODUCT_NONE, PRODUCT_OPTION,
)
from ..regimes.hmm_regime_detector import HMMRegimeDetector
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
    symbol_universe: List[str] = field(default_factory=lambda: ["AAPL", "SPY", "MSFT"])
    top_n_candidates: int = 2
    decision_interval_seconds: int = 300
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

    def _run_one_candidate(
        self,
        screen: ScreenResult,
        bar_ctx: Dict[str, Any],
        circuit: CircuitState,
    ) -> Dict[str, Any]:
        """Run the full pipeline for a single screened candidate."""
        # 1. Run the 7 agents via the factory.
        agent_outputs = self._agent_factory(bar_ctx)
        if len(agent_outputs) != len(self._orch._agent_ids):
            return {
                "symbol": screen.symbol,
                "error": (
                    f"agent_factory returned {len(agent_outputs)} outputs, "
                    f"expected {len(self._orch._agent_ids)}"
                ),
                "agent_signals": {
                    a.agent_id: float(getattr(a, "s", 0.0))
                    for a in agent_outputs
                },
            }

        # 2. Regime + dynamic 7-State SoC + portfolio risk context.
        prices = bar_ctx["prices"]
        volumes = bar_ctx.get("volumes") or [0.0] * len(prices)
        highs = bar_ctx.get("highs")    # OHLCV: genuine True Range when available
        lows = bar_ctx.get("lows")

        # HMM is the single authoritative regime classifier — same HMM instance
        # used by the orchestrator via _classify_regime so the regime is computed
        # once and shared through the full cycle.
        regime = self._orch._classify_regime(
            prices=prices, volumes=volumes, highs=highs, lows=lows
        )

        # Dynamic portfolio metrics from actual open positions & equity
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

        # Compute 7-State SoC from evidence
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

        # 3. Product gate.
        pg = self._product_gate.decide(ProductGateInput(
            action=experience.position_action,
            verdict=experience.capital_gate_verdict,
            ensemble_signal=experience.ensemble_signal,
            disagreement=experience.disagreement,
            confidence=experience.effective_confidence,
            regime=regime.regime,
        ))

        # 4. Circuit-breaker above the execution layer.
        can_equity = circuit.can_trade_equity
        can_option = circuit.can_trade_options
        if pg.product == PRODUCT_OPTION and not can_option:
            return {
                "symbol": screen.symbol, "regime": regime.regime,
                "action": experience.position_action,
                "verdict": experience.capital_gate_verdict,
                "ensemble_signal": experience.ensemble_signal,
                "disagreement": experience.disagreement,
                "product": "none", "reason": f"circuit={circuit.level.value} blocks options",
                "experience_id": experience.decision_id,
                "agent_signals": self._agent_signals_from(experience, agent_outputs),
            }
        if pg.product == PRODUCT_EQUITY and not can_equity:
            return {
                "symbol": screen.symbol, "regime": regime.regime,
                "action": experience.position_action,
                "verdict": experience.capital_gate_verdict,
                "ensemble_signal": experience.ensemble_signal,
                "disagreement": experience.disagreement,
                "product": "none", "reason": f"circuit={circuit.level.value} blocks all trades",
                "experience_id": experience.decision_id,
                "agent_signals": self._agent_signals_from(experience, agent_outputs),
            }
        if pg.product == PRODUCT_NONE or experience.position_action == "HOLD":
            return {
                "symbol": screen.symbol, "regime": regime.regime,
                "action": experience.position_action,
                "verdict": experience.capital_gate_verdict,
                "ensemble_signal": experience.ensemble_signal,
                "disagreement": experience.disagreement,
                "product": "none", "reason": pg.reason,
                "experience_id": experience.decision_id,
                "agent_signals": self._agent_signals_from(experience, agent_outputs),
            }

        # 5. Submit the order via the executor (or simulate in dry-run).
        client_order_id = f"co-{uuid.uuid4().hex[:12]}"
        order_result = self._executor(
            screen.symbol, experience.position_action.lower(),
            experience.quantity, pg.option_side,
        )
        rec = self._orders.register(
            client_order_id=client_order_id,
            decision_id=experience.decision_id,
            symbol=screen.symbol,
            side=experience.position_action.lower(),
            qty=experience.quantity,
            product=pg.product,
            option_side=pg.option_side,
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
            fq = float(order_result.get("filled_qty", experience.quantity))
            fp = float(order_result.get("filled_avg_price", 0.0))
            self._orders.transition(
                client_order_id, OrderState.FILLED,
                note="broker reported fill", fill_qty=fq, fill_price=fp,
            )
            # Open a position
            self._positions.open_position(
                decision_id=experience.decision_id,
                client_order_id=client_order_id,
                symbol=screen.symbol,
                side=experience.position_action.lower(),
                quantity=fq,
                entry_price=fp,
                product=pg.product,
                option_side=pg.option_side,
            )

        return {
            "symbol": screen.symbol, "regime": regime.regime,
            "action": experience.position_action,
            "verdict": experience.capital_gate_verdict,
            "ensemble_signal": experience.ensemble_signal,
            "disagreement": experience.disagreement,
            "product": pg.product, "option_side": pg.option_side,
            "quantity": experience.quantity,
            "client_order_id": client_order_id,
            "broker_order_id": order_result.get("id"),
            "order_status": order_result.get("status"),
            "reason": pg.reason,
            "experience_id": experience.decision_id,
            "agent_signals": self._agent_signals_from(experience, agent_outputs),
        }

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

        # 4. Run each candidate through the pipeline (capped at max_lookups_per_interval)
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
            decision = self._run_one_candidate(screen, bar_ctx, circuit)
            decisions.append(decision)
            if decision.get("client_order_id"):
                orders.append({
                    "client_order_id": decision["client_order_id"],
                    "symbol": decision["symbol"],
                    "product": decision["product"],
                    "option_side": decision.get("option_side"),
                    "order_status": decision.get("order_status"),
                })
            llm_calls += 1
            ensemble_signal = decision.get("ensemble_signal", ensemble_signal)
            disagreement = decision.get("disagreement", disagreement)
            regime_label = decision.get("regime", regime_label)

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
