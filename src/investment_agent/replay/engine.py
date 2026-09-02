"""Historical replay / backtest engine for the trading pipeline.

WHAT
====
Drives the deterministic agent pipeline (7 specialist agents ->
ensemble -> HMM regime -> investment Kalman -> capital gate -> risk
gates -> decision) over a historical OHLCV series, one bar at a time,
and records every decision. After each bar, the previous bar's
position is closed against the realized return and the agent
reputations are updated. The replay writes to a TradeMemory JSON
file so the dashboard can render the same backtest as a live paper
session.

WHY
====
Before this module, the only way to exercise the full pipeline was
the live paper account. That made it impossible to:
  * demo the system without market hours
  * prove the reputation/memory feedback loop converges
  * test against deterministic historical prices (P&L Performance
    judging criteria)

HOW
====
``ReplayEngine`` takes a ``MarketDataClient`` (Alpaca or fake), an
``XQuantXOrchestrator``, an ``agent_factory`` callable that returns
seven ``AgentOutput`` objects per bar, and a symbol. It iterates
bars, calls ``orch.run_cycle``, and on the next bar closes the
prior experience. The agent factory is intentionally a callable
so tests can pin each agent to a deterministic policy; production
can wire it to the LLM-backed specialists.
"""
from __future__ import annotations

import json
import logging
import math
import os
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd

from ..data.market_data import BarRequest, MarketDataClient
from ..orchestrator import XQuantXOrchestrator


logger = logging.getLogger(__name__)


AgentFactory = Callable[[Dict[str, Any]], List[Any]]
"""Builds the seven ``AgentOutput`` objects for a single bar.

The dict carries the bar context: timestamp, symbol, prices_so_far,
volumes_so_far, current_price, recent_return, regime, and any other
data the caller wants to surface. The factory must return exactly
seven AgentOutput objects matching the orchestrator's agent_ids.
"""

# Trailing window (in bars) passed to the orchestrator/regime detector
# per bar. Regime feature extraction only ever consumes the trailing
# ~49 bars (_MIN_OBSERVATIONS=30 + lookback_days=20 - 1); this constant
# is set with a generous safety margin above that. Bounding the window
# (instead of slicing the full closes[:i+1]/volumes[:i+1] history every
# bar) turns the replay loop's per-bar cost from O(bars_so_far) into
# O(1), i.e. the whole replay from O(n^2) into O(n).
_HISTORY_WINDOW_BARS: int = 120


@dataclass
class ReplayConfig:
    """Replay run parameters.

    Attributes
    ----------
    symbol : str
        Symbol under simulation.
    start, end : datetime, optional
        Time bounds; default = full series.
    lookback : int
        Number of bars to feed the orchestrator on the first call.
    starting_cash : float
        Account equity for the backtest; used to translate fractional
        position size into a notional P&L.
    transaction_cost_bps : float
        Round-trip transaction cost in basis points applied to every
        closed trade.
    """
    symbol: str
    start: Optional[datetime] = None
    end: Optional[datetime] = None
    lookback: int = 30
    starting_cash: float = 100_000.0
    transaction_cost_bps: float = 5.0


@dataclass
class ReplayResult:
    """Summary of a replay run.

    All fields are populated by ``ReplayEngine.run``.
    """
    symbol: str
    bars_processed: int = 0
    decisions: int = 0
    buys: int = 0
    sells: int = 0
    holds: int = 0
    closed_trades: int = 0
    realized_pnl: float = 0.0
    transaction_costs: float = 0.0
    final_equity: float = 0.0
    max_drawdown_pct: float = 0.0
    win_rate: float = 0.0
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    audit_log: List[Dict[str, Any]] = field(default_factory=list)
    decisions_log: List[Dict[str, Any]] = field(default_factory=list)
    memory_file: str = ""
    reputation_file: str = ""
    closed_experiences: List[Any] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "bars_processed": self.bars_processed,
            "decisions": self.decisions,
            "buys": self.buys,
            "sells": self.sells,
            "holds": self.holds,
            "closed_trades": self.closed_trades,
            "realized_pnl": self.realized_pnl,
            "transaction_costs": self.transaction_costs,
            "final_equity": self.final_equity,
            "max_drawdown_pct": self.max_drawdown_pct,
            "win_rate": self.win_rate,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
            "memory_file": self.memory_file,
            "reputation_file": self.reputation_file,
            "decisions_log": self.decisions_log,
        }


class ReplayEngine:
    """Stateless runner that drives the orchestrator over historical bars.

    Example
    -------
    >>> from investment_agent.data.market_data import FakeMarketDataClient
    >>> fake = FakeMarketDataClient()
    >>> fake.set_series("AAPL", _make_series())
    >>> engine = ReplayEngine(orchestrator, market_data=fake)
    >>> result = engine.run(config, agent_factory=my_factory)
    >>> result.realized_pnl
    1234.5
    """

    DEFAULT_STATES = None  # filled in lazily

    def __init__(
        self,
        orchestrator: XQuantXOrchestrator,
        market_data: MarketDataClient,
        reputation_file: str = "reputation_state.json",
    ) -> None:
        self._orchestrator = orchestrator
        self._market_data = market_data
        self._reputation_file = reputation_file

    def _seven_state_full(self) -> Any:
        from ..capital.capital_gate import SevenStateVector
        return SevenStateVector(
            economic=1.0, financial=1.0, fiscal=1.0,
            portfolio=1.0, fundamental=1.0, market=1.0, sector=1.0,
        )

    def _portfolio_context(
        self,
        price: float,
        prices: List[float],
        drawdown_pct: float,
        equity: float,
        positions_qty: float,
    ) -> Dict[str, Any]:
        position_pct = (positions_qty * price) / max(equity, 1.0)
        return {
            "position_pct": min(max(position_pct, 0.0), 1.0),
            "gross_leverage": min(max(position_pct, 0.0), 5.0),
            "entropy": 0.0,
            "drawdown_pct": max(drawdown_pct, 0.0),
            "execution_timeout_seconds": 5.0,
            "sector_exposure_pct": min(max(position_pct, 0.0), 1.0),
            "is_new_long": positions_qty <= 0.0,
            "regime": "R01",
            "available_liquidity": equity,
        }

    def _close_previous(
        self,
        prev_experience: Any,
        prev_price: float,
        current_price: float,
        transaction_cost_bps: float,
    ) -> Tuple[float, float, str]:
        """Close the previous bar's position at the current bar's price.

        Returns (realized_pnl, transaction_cost, realized_outcome_label).
        """
        if prev_experience is None:
            return 0.0, 0.0, "no_prior_position"
        if prev_experience.position_action == "HOLD":
            return 0.0, 0.0, "hold_no_pnl"
        if prev_price <= 0:
            return 0.0, 0.0, "invalid_prev_price"

        ret = (current_price - prev_price) / prev_price
        if prev_experience.position_action == "SELL":
            ret = -ret

        notional = prev_experience.quantity * prev_experience.kalman_price
        if notional <= 0:
            notional = prev_experience.quantity * prev_price
        gross_pnl = ret * notional
        cost = abs(notional) * (transaction_cost_bps / 10_000.0) * 2  # round trip
        net_pnl = gross_pnl - cost
        if net_pnl > 0:
            outcome = "win"
        elif net_pnl < 0:
            outcome = "loss"
        else:
            outcome = "breakeven"
        return net_pnl, cost, outcome

    def run(
        self,
        config: ReplayConfig,
        agent_factory: AgentFactory,
    ) -> ReplayResult:
        """Run the replay end-to-end.

        Parameters
        ----------
        config : ReplayConfig
            Symbol, time bounds, lookback, cash.
        agent_factory : AgentFactory
            Callable returning seven ``AgentOutput`` for a given bar.
        """
        result = ReplayResult(
            symbol=config.symbol,
            started_at=datetime.now(),
            memory_file=self._orchestrator._trade_memory._memory_file,
            reputation_file=self._reputation_file,
        )

        # Pull the bar series.
        req = BarRequest(
            symbol=config.symbol,
            start=config.start or datetime(1970, 1, 1),
            end=config.end,
            timeframe="1Day",
        )
        bars: pd.DataFrame = self._market_data.get_historical_bars(req)
        if bars is None or len(bars) < config.lookback + 2:
            result.finished_at = datetime.now()
            return result

        closes = bars["close"].tolist()
        volumes = bars.get("volume", pd.Series([0.0] * len(bars))).tolist()
        timestamps = list(bars.index)

        result.bars_processed = len(bars)

        equity = float(config.starting_cash)
        peak_equity = equity
        max_drawdown_pct = 0.0
        positions_qty = 0.0
        prev_experience: Any = None
        prev_price: float = 0.0
        wins = 0
        losses = 0

        # Walk bars; the first `lookback` bars seed the regime detector
        # without producing a decision.
        states_full = self._seven_state_full()
        for i in range(config.lookback, len(bars)):
            # Bounded trailing window: still strictly historical (no
            # look-ahead) but capped so per-bar cost does not grow with
            # the size of the replay (see _HISTORY_WINDOW_BARS above).
            window_start = max(0, i + 1 - _HISTORY_WINDOW_BARS)
            recent_closes = closes[window_start : i + 1]
            recent_volumes = volumes[window_start : i + 1]
            bar_ctx = {
                "timestamp": timestamps[i].to_pydatetime() if hasattr(timestamps[i], "to_pydatetime") else timestamps[i],
                "symbol": config.symbol,
                "prices": recent_closes,
                "volumes": recent_volumes,
                "current_price": closes[i],
                "recent_return": (closes[i] - closes[i - 1]) / closes[i - 1] if closes[i - 1] > 0 else 0.0,
                "lookback_returns": [
                    (closes[j] - closes[j - 1]) / closes[j - 1] for j in range(max(1, i - 5), i + 1) if closes[j - 1] > 0
                ],
                "equity": equity,
                "drawdown_pct": max_drawdown_pct,
                "positions_qty": positions_qty,
            }
            agent_outputs = agent_factory(bar_ctx)
            if len(agent_outputs) != len(self._orchestrator._agent_ids):
                raise ValueError(
                    f"agent_factory returned {len(agent_outputs)} outputs but "
                    f"orchestrator has {len(self._orchestrator._agent_ids)} agents"
                )

            portfolio_ctx = self._portfolio_context(
                price=closes[i],
                prices=recent_closes,
                drawdown_pct=max_drawdown_pct,
                equity=equity,
                positions_qty=positions_qty,
            )

            cycle = self._orchestrator.run_cycle(
                prices=recent_closes,
                volumes=recent_volumes,
                agent_outputs=agent_outputs,
                states=states_full,
                portfolio_context=portfolio_ctx,
            )
            experience = cycle.experience
            result.decisions += 1
            decision_record = {
                "timestamp": bar_ctx["timestamp"].isoformat() if hasattr(bar_ctx["timestamp"], "isoformat") else str(bar_ctx["timestamp"]),
                "symbol": config.symbol,
                "price": closes[i],
                "action": experience.position_action,
                "quantity": experience.quantity,
                "regime": experience.regime,
                "verdict": experience.capital_gate_verdict,
                "ensemble_signal": experience.ensemble_signal,
                "disagreement": experience.disagreement,
                "kalman_gain": experience.kalman_gain,
                "kalman_posterior": experience.kalman_posterior,
            }
            result.decisions_log.append(decision_record)

            if experience.position_action == "BUY":
                result.buys += 1
                positions_qty = float(experience.quantity)
            elif experience.position_action == "SELL":
                result.sells += 1
                positions_qty = -float(experience.quantity)
            else:
                result.holds += 1

            # Close the previous bar's experience on this bar's close.
            if i > config.lookback:
                pnl, cost, outcome = self._close_previous(
                    prev_experience, prev_price, closes[i],
                    config.transaction_cost_bps,
                )
                if prev_experience is not None and prev_experience.position_action != "HOLD":
                    self._orchestrator.close_trade(
                        decision_id=prev_experience.decision_id,
                        realized_outcome=outcome,
                        pnl=pnl,
                        lesson=f"replay {config.symbol} {prev_experience.position_action} @ {prev_price:.2f} -> {closes[i]:.2f}",
                    )
                    result.closed_trades += 1
                    result.realized_pnl += pnl
                    result.transaction_costs += cost
                    equity += pnl
                    if outcome == "win":
                        wins += 1
                    elif outcome == "loss":
                        losses += 1
                    peak_equity = max(peak_equity, equity)
                    dd = (equity - peak_equity) / peak_equity if peak_equity > 0 else 0.0
                    max_drawdown_pct = min(max_drawdown_pct, dd)

            prev_experience = experience
            prev_price = closes[i]

        # Mark-to-market remaining open position.
        if prev_experience is not None and prev_experience.position_action != "HOLD" and len(bars) > config.lookback:
            pnl, cost, outcome = self._close_previous(
                prev_experience, prev_price, closes[-1],
                config.transaction_cost_bps,
            )
            if outcome != "hold_no_pnl":
                self._orchestrator.close_trade(
                    decision_id=prev_experience.decision_id,
                    realized_outcome=outcome,
                    pnl=pnl,
                    lesson=f"replay end-of-series mark @ {prev_price:.2f} -> {closes[-1]:.2f}",
                )
                result.closed_trades += 1
                result.realized_pnl += pnl
                result.transaction_costs += cost
                equity += pnl
                if outcome == "win":
                    wins += 1
                elif outcome == "loss":
                    losses += 1

        result.final_equity = equity
        result.max_drawdown_pct = max_drawdown_pct
        if (wins + losses) > 0:
            result.win_rate = wins / (wins + losses)

        # Persist reputation so the dashboard can read it back.
        try:
            from ..agents.reputation_persistence import save_reputation
            save_reputation(
                self._orchestrator._reputation_tracker,
                self._reputation_file,
            )
        except Exception as e:  # pragma: no cover
            logger.warning("Failed to persist reputation: %s", e)

        result.finished_at = datetime.now()
        return result


__all__ = [
    "AgentFactory",
    "ReplayConfig",
    "ReplayEngine",
    "ReplayResult",
]
