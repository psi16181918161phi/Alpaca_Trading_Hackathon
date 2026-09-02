"""Long-running X Quant X paper-trading session daemon.

WHAT
====
Starts a ``SessionController`` and watches ``session_command.json`` for
``start`` / ``stop`` / ``emergency_stop`` commands written by the
dashboard. When a command arrives, the daemon applies it and the
controller persists the new state to ``session_status.json`` (the same
file the dashboard reads on every refresh).

This is the only component the dashboard talks to for trading actions.
The dashboard never imports the Alpaca TradingClient and never calls
``LiveOrchestrator.run_once``; the architecture rule is
``dashboard -> session controller -> orchestrator -> execution -> Alpaca``.

WHY
====
Separates UI from trading. The dashboard stays monitoring-only; a
restart of the dashboard does not interrupt the trading session; a
restart of the session daemon does not interrupt the dashboard.

HOW
====
Run from the project root::

    python scripts/run_session.py

Optional flags::

    --stage paper           paper / dry_run / competition
    --interval 300          Decision interval in seconds
    --max-lookups 2         Max LLM calls per decision interval
    --symbols AAPL,SPY,...  Symbol universe
    --status-file PATH      Where the daemon persists status (default:
                            session_status.json in CWD)
    --command-file PATH     Where the dashboard writes commands
    --poll-interval 0.5     Seconds between command-file polls

The daemon also responds to SIGTERM / SIGINT: it asks the controller
to stop gracefully and exits after one last status persistence.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))


logger = logging.getLogger("xqx.session_daemon")


def _load_llm_provider():
    """Return a mock or real LLM provider based on env keys.

    Mirrors ``scripts/run_live_loop._load_llm_provider`` so the
    daemon's bootstrap matches the standalone CLI one-for-one.
    """
    from investment_agent.llm.base import MockLLMProvider

    keys = {
        "deephermes": os.getenv("FEATHERLESS_DEEPHERMES_KEY"),
        "fundamentals": os.getenv("FEATHERLESS_FINANCE_LLAMA_KEY"),
        "finance_qlora": os.getenv("FEATHERLESS_QWEN_TRADING_KEY"),
        "reserve": os.getenv("FEATHERLESS_RESERVE_KEY"),
    }
    have_any = any(keys.values())
    if not have_any:
        return MockLLMProvider(responder=_mock_responder), "mock"
    try:
        from investment_agent.llm.orchestrator import (
            FeatherlessOrchestrator, load_provider_specs,
        )
        for pid, key in keys.items():
            if key:
                os.environ[f"FEATHERLESS_{pid.upper()}_KEY"] = key
        specs = load_provider_specs()
        specs = [s for s in specs if s.api_key]
        if not specs:
            raise RuntimeError("no Featherless provider specs have keys set")
        return FeatherlessOrchestrator(specs=specs), "featherless"
    except Exception as e:
        print(f"WARN: failed to build Featherless orchestrator ({e}); using mock.",
              file=sys.stderr)
        return MockLLMProvider(responder=_mock_responder), "mock"


def _mock_responder(system, prompt):
    """Mock LLM with role-aware signal bias."""
    if not system:
        return json.dumps({
            "signal": 0.0, "confidence": 0.5, "uncertainty": 0.5,
            "doubt": 0.5, "p_plus": 0.5, "p_minus": 0.5,
            "delta_t": 1.0, "noise": 0.5,
        })
    bias_map = {
        "Economic State": 0.30, "Financial State": -0.10,
        "Fiscal State": 0.05, "Portfolio State": -0.20,
        "Fundamental State": 0.40, "Market Microstructure": 0.20,
        "Sector": 0.10,
    }
    s = 0.0
    for kw, b in bias_map.items():
        if kw in system:
            s = b
            break
    return json.dumps({
        "signal": s, "confidence": 0.8, "uncertainty": 0.2,
        "doubt": 0.1, "p_plus": 0.5 + s / 2, "p_minus": 0.5 - s / 2,
        "delta_t": 1.0, "noise": 0.5,
    })


def _build_agent_factory(provider, default_roles):
    from investment_agent.agents.specialist import (
        AgentContext, build_specialist_agents, run_agents,
    )
    agents = build_specialist_agents(provider)
    canonical_ids = [r.agent_id for r in default_roles]

    def factory(bar_ctx):
        ctx = AgentContext(
            symbol=bar_ctx["symbol"],
            regime=bar_ctx.get("regime", "R01"),
            regime_probabilities=bar_ctx.get("regime_probabilities", {}),
            features=bar_ctx.get("features", {}),
            ensemble_signal=0.0,
            disagreement=0.0,
        )
        out_map = run_agents(agents, ctx)
        return [out_map[aid] for aid in canonical_ids if aid in out_map]
    return factory


def _make_executor(stage: str):
    """Return a callable that places orders via Alpaca (or no-ops)."""
    if stage == "dry_run":
        def executor(symbol, side, qty, option_side):
            return {
                "id": None, "status": "dry_run",
                "filled_qty": 0.0, "filled_avg_price": 0.0,
                "error": None,
            }
        return executor

    def executor(symbol, side, qty, option_side):
        try:
            from investment_agent.execution.execution import (
                get_option_contract, is_trade_safe, place_order,
            )
            if option_side is not None:
                contract = get_option_contract(symbol, option_type=option_side)
                contract_symbol = getattr(contract, "symbol", symbol)
                close_price = float(getattr(contract, "close_price", 0.0) or 1.0)
                if not is_trade_safe(contract_symbol, max(1, int(qty)), close_price):
                    return {"id": None, "status": "rejected", "error": "is_trade_safe returned False"}
                result = place_order(
                    symbol=contract_symbol, side=side,
                    qty=max(1, int(qty)),
                    price_per_contract=close_price,
                )
            else:
                result = place_order(
                    symbol=symbol, side=side, qty=int(qty), price_per_contract=1.0)

            order_id = result.get("id") if hasattr(result, "get") else getattr(result, "order_id", None)
            status = result.get("status") if hasattr(result, "get") else getattr(result, "status", "unknown")
            error = result.get("error") if hasattr(result, "get") else getattr(result, "reason", None)
            return {
                "id": order_id,
                "status": status,
                "error": error,
                "filled_qty": float(result.get("filled_qty", 0.0) if hasattr(result, "get") else 0.0),
                "filled_avg_price": float(result.get("filled_avg_price", 0.0) if hasattr(result, "get") else 0.0),
            }
        except Exception as e:
            return {"id": None, "status": "failed", "error": str(e)}
    return executor


def _make_market_data(stage: str, lookback_days: int, symbols):
    """Real Alpaca for paper/competition (if keys present), else fake."""
    if stage in {"paper", "competition"} \
            and os.getenv("APCA_API_KEY_ID") \
            and os.getenv("APCA_API_SECRET_KEY"):
        from investment_agent.data.market_data import AlpacaMarketDataClient
        return AlpacaMarketDataClient()

    from investment_agent.data.market_data import FakeMarketDataClient
    md = FakeMarketDataClient()
    import pandas as pd
    end = pd.Timestamp.now().normalize()
    idx = pd.date_range(end=end, periods=lookback_days + 5, freq="D")
    for s in symbols:
        closes = [100.0 + 0.2 * i for i in range(len(idx))]
        df = pd.DataFrame({
            "open": closes,
            "high": [c + 0.5 for c in closes],
            "low": [c - 0.5 for c in closes],
            "close": closes,
            "volume": [1_000_000.0] * len(idx),
        }, index=idx)
        md.set_series(s, df)
    return md


def _build_orchestrator_factory(args: argparse.Namespace):
    """Return a no-arg callable that constructs a LiveOrchestrator.

    Done lazily inside the SessionController thread so the daemon's
    process startup stays light. Reads env keys / config from disk at
    the moment the user presses START, not earlier.
    """
    def factory():
        from investment_agent.agents.specialist import DEFAULT_ROLES
        from investment_agent.live import (
            CandidateScreener, LiveOrchestrator, LiveOrchestratorConfig,
        )
        from investment_agent.orchestrator import XQuantXOrchestrator
        from investment_agent.products import ProductGate

        # State files: write to the project root so the dashboard can
        # see live updates. Tests should use ``--status-file`` /
        # ``--command-file`` to point at a tempdir.
        state_file = "live_state.json"
        memory_file = "trade_memory.json"
        reputation_file = "reputation_state.json"

        md = _make_market_data(args.stage, args.lookback, args.symbols)
        provider, _ = _load_llm_provider()
        agent_factory = _build_agent_factory(provider, DEFAULT_ROLES)
        executor = _make_executor(args.stage)

        orch = XQuantXOrchestrator(
            agent_ids=[r.agent_id for r in DEFAULT_ROLES],
            symbol=args.symbols[0],
            use_hmm=False,
            enable_trading=False,
            memory_file=memory_file,
        )
        config = LiveOrchestratorConfig(
            symbol_universe=list(args.symbols),
            top_n_candidates=args.top_n,
            decision_interval_seconds=args.interval,
            state_file=state_file,
            memory_file=memory_file,
            reputation_file=reputation_file,
            lookback_days=args.lookback,
            max_lookups_per_interval=args.max_lookups,
            stage=args.stage,
        )
        return LiveOrchestrator(
            config=config,
            market_data=md,
            orchestrator=orch,
            agent_factory=agent_factory,
            executor=executor,
            screener=CandidateScreener(top_n=args.top_n),
            product_gate=ProductGate(),
        )
    return factory


class SessionDaemon:
    """Bridges the command file to a SessionController instance."""

    def __init__(
        self,
        controller,
        status_file: str = "session_status.json",
        command_file: str = "session_command.json",
        poll_interval_s: float = 0.5,
    ) -> None:
        self._controller = controller
        self.status_file = status_file
        self.command_file = command_file
        self._poll_interval_s = poll_interval_s
        self._stop_event = threading.Event()

    def request_stop(self) -> None:
        self._stop_event.set()

    def run(self) -> int:
        """Block until stopped. Returns the process exit code."""
        logger.info("daemon: watching %s -> %s",
                    self.command_file, self.status_file)
        try:
            while not self._stop_event.is_set():
                self._maybe_apply_command()
                # Wake up promptly when stop is requested.
                if self._stop_event.wait(timeout=self._poll_interval_s):
                    break
        except KeyboardInterrupt:
            logger.info("daemon: interrupted by user")
        finally:
            self._shutdown()
        return 0

    def _maybe_apply_command(self) -> None:
        if not os.path.exists(self.command_file):
            return
        try:
            with open(self.command_file, "r", encoding="utf-8") as f:
                cmd = json.load(f)
        except (OSError, json.JSONDecodeError):
            return
        if not isinstance(cmd, dict):
            return
        action = cmd.get("action")
        try:
            if action == "start":
                logger.info("daemon: applying start (params=%s)",
                            cmd.get("params"))
                self._controller.start(params=cmd.get("params") or {})
            elif action == "stop":
                logger.info("daemon: applying stop")
                self._controller.stop(emergency=False)
            elif action == "emergency_stop":
                logger.info("daemon: applying EMERGENCY STOP")
                self._controller.stop(emergency=True)
            else:
                logger.warning("daemon: unknown action %r", action)
        finally:
            self._clear_command_file()

    def _clear_command_file(self) -> None:
        try:
            os.unlink(self.command_file)
        except OSError:
            pass

    def _shutdown(self) -> None:
        st = self._controller.status.state
        if st not in {"STOPPED", "ERROR"}:
            logger.info("daemon: graceful stop on shutdown (state=%s)", st)
            self._controller.stop(emergency=False)
            # Give the thread up to 5s to wind down.
            deadline = time.time() + 5
            while time.time() < deadline:
                thread = self._controller._thread
                if thread is None or not thread.is_alive():
                    break
                time.sleep(0.1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", default="paper",
                        choices=["dry_run", "paper", "competition"])
    parser.add_argument("--interval", type=int, default=300,
                        help="Decision interval in seconds")
    parser.add_argument("--max-lookups", type=int, default=2,
                        help="Max LLM calls per decision interval")
    parser.add_argument("--symbols", default="AAPL,SPY,MSFT,TSLA,NVDA,BTC/USD,ETH/USD,SOL/USD,AVAX/USD,LINK/USD,XRP/USD,DOGE/USD,RENDER/USD")
    parser.add_argument("--top-n", type=int, default=2)
    parser.add_argument("--lookback", type=int, default=60)
    parser.add_argument("--status-file", default="session_status.json",
                        help="Where to persist session status "
                             "(read by the dashboard)")
    parser.add_argument("--command-file", default="session_command.json",
                        help="Where the dashboard writes start/stop "
                             "commands")
    parser.add_argument("--poll-interval", type=float, default=0.5)
    args = parser.parse_args()

    args.symbols = [s.strip().upper()
                    for s in args.symbols.split(",") if s.strip()]

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    from investment_agent.live import SessionController
    controller = SessionController(
        build_orchestrator=_build_orchestrator_factory(args),
        status_file=args.status_file,
        command_file=args.command_file,
    )
    daemon = SessionDaemon(
        controller=controller,
        status_file=args.status_file,
        command_file=args.command_file,
        poll_interval_s=args.poll_interval,
    )

    def _on_signal(signum, frame):  # noqa: ARG001
        logger.info("daemon: received signal %s, stopping", signum)
        daemon.request_stop()

    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGINT, _on_signal)
    return daemon.run()


if __name__ == "__main__":
    sys.exit(main())
