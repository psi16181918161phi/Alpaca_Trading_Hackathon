"""Run the live paper-trading loop in three stages.

Stage A (--stage dry_run)
    Real Alpaca market data, full architecture, NO ORDERS. Every
    decision is logged so an operator can verify the pipeline
    before touching the paper account.

Stage B (--stage paper)
    Real Alpaca market data, full architecture, Alpaca PAPER
    orders via TradingClient.submit_order. The hackathon's
    $100k PA3EGYME3NG5 paper account is the target.

Stage C (--stage competition)
    Reserved for the live competition account; not different
    from Stage B in code, but a separate --stage flag keeps
    credentials and state files isolated.

Featherless credit discipline
-----------------------------
The LLM is invoked at most ``max_lookups_per_interval`` times per
decision interval. The default is 2; raise with
``--max-lookups`` if you have a healthy budget. The screener
collapses the universe to the top N candidates so the LLM only
sees the symbols that actually cleared a deterministic filter.

Usage
-----
    # Stage A: dry run, 2 intervals, no orders
    python scripts/run_live_loop.py --stage dry_run --max-intervals 2

    # Stage B: paper trading, 30-min decision interval
    python scripts/run_live_loop.py --stage paper --interval 1800

    # With real Featherless providers
    FEATHERLESS_DEEPHERMES_KEY=... python scripts/run_live_loop.py \\
        --stage paper --interval 1800
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))


def _bar(s: str, width: int = 60, char: str = "=") -> str:
    pad = max(0, (width - len(s) - 2) // 2)
    return char * pad + f" {s} " + char * pad


def _print_report(report, last_reputations: Optional[Dict[str, Dict[str, float]]] = None) -> None:
    ts = report.timestamp.strftime("%Y-%m-%d %H:%M:%S")
    print()
    print(_bar("X QUANT X -- LIVE PAPER LOOP"))
    print(f"Time:        {ts}   Interval: #{report.interval_index}")
    print(f"Regime:      {report.regime}")
    print(f"Equity:      ${report.equity:,.2f}")
    print(f"Drawdown:    {report.drawdown_pct:.1%}    "
          f"Loss streak: {report.consecutive_losses}    "
          f"Daily loss: {report.daily_loss_pct:.1%}")
    if report.circuit_state:
        cs = report.circuit_state
        print(f"Circuit:     {cs['level']:>10}   "
              f"equity_ok={cs['can_trade_equity']}  "
              f"options_ok={cs['can_trade_options']}")
        if cs["triggered_signals"]:
            print(f"             triggers: {', '.join(cs['triggered_signals'])}")
    print(_bar("", char="-"))

    print("CANDIDATES")
    if report.candidates:
        for c in report.candidates:
            print(f"  - {c}")
    else:
        print("  (none)")
    print(_bar("", char="-"))

    for d in report.decisions:
        sym = d.get("symbol", "?")
        action = d.get("action", "HOLD")
        verdict = d.get("verdict", "n/a")
        sig = d.get("ensemble_signal", 0.0)
        disag = d.get("disagreement", 0.0)
        product = d.get("product", "none")
        side = d.get("option_side") or ""
        reason = d.get("reason", "")
        print(f"DECISION: {sym:6}  action={action:5}  verdict={verdict:7}  "
              f"ensemble={sig:+.2f}  disagreement={disag:.2f}  "
              f"product={product}{' ' + side if side else ''}")
        if reason:
            print(f"          reason: {reason}")
    print(_bar("", char="-"))

    print("7 AGENTS")
    canonical_ids = [
        "agent_economic", "agent_financial", "agent_fiscal",
        "agent_portfolio", "agent_fundamental", "agent_market",
        "agent_sector",
    ]
    last = report.decisions[-1] if report.decisions else {}
    if last:
        signals = last.get("agent_signals") or {}
        for aid in canonical_ids:
            s = signals.get(aid)
            if s is None:
                print(f"  {aid:30} (no signal)")
            else:
                print(f"  {aid:30} {float(s):+.2f}")
    print(_bar("", char="-"))

    print("KALMAN")
    last_dec = report.decisions[-1] if report.decisions else {}
    if last_dec:
        # These are written into the experience but not echoed in the
        # report; re-derive the posterior display from the run.
        pass
    if report.orders:
        for o in report.orders:
            print(f"ORDER: {o['symbol']:6}  product={o['product']:8}  "
                  f"side={o.get('option_side') or '-':4}  "
                  f"status={o.get('order_status', '?')}")
    print(_bar("", char="-"))

    if report.exits:
        print("CLOSED TRADES")
        for ex in report.exits:
            tag = "WIN " if ex["pnl"] > 0 else "LOSS" if ex["pnl"] < 0 else "FLAT"
            print(f"  {ex['symbol']:6}  {ex['reason']:20}  "
                  f"pnl=${ex['pnl']:+,.2f}  ({ex['pnl_pct']:+.2%})  "
                  f"hold={ex['holding_seconds']:.0f}s  {tag}")
    if last_reputations:
        print(_bar("", char="-"))
        print("REPUTATION (after this interval)")
        for aid, p in sorted(last_reputations.items()):
            print(f"  {aid:30}  alpha={p['alpha']:.1f}  beta={p['beta']:.1f}  "
                  f"mean={(p['alpha'] / max(1e-9, p['alpha']+p['beta'])):.2f}")


def _load_llm_provider(stage: str = "dry_run"):
    """Return a mock or real LLM provider based on env keys.

    If stage is 'paper' or 'competition', fallback to MockLLMProvider is strictly forbidden.
    """
    try:
        from investment_agent.llm.orchestrator import (
            FeatherlessOrchestrator, load_provider_specs,
        )
        specs = load_provider_specs()
        valid_specs = [s for s in specs if s.api_key]
        if valid_specs:
            return FeatherlessOrchestrator(specs=valid_specs), "featherless"
    except Exception as e:
        if stage in {"paper", "competition"}:
            raise RuntimeError(
                f"FATAL: Live paper/competition mode requires valid Featherless LLM keys. "
                f"Cannot run live trading with MockLLMProvider. Error: {e}"
            ) from e
        print(f"WARN: failed to build Featherless orchestrator ({e}); using mock.", file=sys.stderr)

    if stage in {"paper", "competition"}:
        raise RuntimeError(
            "FATAL: Live paper/competition mode requires valid Featherless LLM keys "
            "(FEATHERLESS_*_KEY). Refusing to run live trading with MockLLMProvider."
        )

    print("[llm] using MockLLMProvider for offline / dry_run testing.", file=sys.stderr)
    from investment_agent.llm.base import MockLLMProvider
    return MockLLMProvider(responder=_mock_responder), "mock"


def _mock_responder(system, prompt):
    """Mock LLM with role-aware signal bias."""
    import json
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
        "doubt": 0.1, "p_plus": 0.5 + s/2, "p_minus": 0.5 - s/2,
        "delta_t": 1.0, "noise": 0.5,
    })


def _build_agent_factory(provider, default_roles, memory_file: Optional[str] = None):
    from investment_agent.agents.specialist import (
        build_specialist_agents, AgentContext, run_agents,
    )
    from investment_agent.memory.trade_memory import TradeMemory, TradeExperience
    import uuid

    agents = build_specialist_agents(provider)
    canonical_ids = [r.agent_id for r in default_roles]
    tm = TradeMemory(memory_file) if memory_file else None

    def factory(bar_ctx):
        memories = []
        if tm and bar_ctx.get("symbol"):
            try:
                dummy_exp = TradeExperience(
                    decision_id=str(uuid.uuid4()),
                    timestamp=datetime.now(),
                    symbol=bar_ctx["symbol"],
                    regime=bar_ctx.get("regime", "R01"),
                    regime_probabilities=bar_ctx.get("regime_probabilities", {}),
                    agent_signals={}, ensemble_signal=0.0, disagreement=0.0,
                    effective_confidence=0.5, kalman_gain=0.0, kalman_price=0.0, kalman_trend=0.0,
                    capital_gate_verdict="PREVIEW", effective_cap=0.0, state_charges={},
                    position_action="HOLD", quantity=0.0, confidence=0.5, expected_outcome="",
                    realized_outcome="", pnl=0.0, lesson="",
                )
                sims = tm.find_similar(dummy_exp, top_k=3, min_similarity=0.0)
                memories = [
                    f"Past trade {s.experience.symbol} ({s.experience.regime}): "
                    f"Action={s.experience.position_action}, PnL=${s.experience.pnl:+.2f}, "
                    f"Lesson='{s.experience.lesson}'"
                    for s in sims
                ]
            except Exception:
                memories = []

        ctx = AgentContext(
            symbol=bar_ctx["symbol"],
            regime=bar_ctx.get("regime", "R01"),
            regime_probabilities=bar_ctx.get("regime_probabilities", {}),
            features=bar_ctx.get("features", {}),
            ensemble_signal=0.0,
            disagreement=0.0,
            memory=memories,
        )
        out_map = run_agents(agents, ctx)
        # Re-key to the canonical IDs the orchestrator expects.
        return [out_map[aid] for aid in canonical_ids if aid in out_map]
    return factory


def _make_executor(stage: str):
    """Return a callable that places orders via Alpaca (or no-ops)."""
    if stage == "dry_run":
        def executor(symbol, side, qty, option_side):
            print(f"[dry-run] would {side} {qty} {symbol} "
                  f"{'(option ' + option_side + ')' if option_side else ''}")
            return {
                "id": None, "status": "dry_run",
                "filled_qty": 0.0, "filled_avg_price": 0.0,
                "error": None,
            }
        return executor
    # paper / competition: real Alpaca paper
    def executor(symbol, side, qty, option_side):
        try:
            from investment_agent.execution.execution import (
                get_option_contract, place_order, is_trade_safe,
            )
            if option_side is not None:
                contract = get_option_contract(symbol, option_type=option_side)
                if not is_trade_safe(contract.symbol, max(1, int(qty)),
                                      float(contract.close_price or 0.0)):
                    return {"id": None, "status": "rejected",
                            "error": "is_trade_safe returned False"}
                result = place_order(
                    symbol=contract.symbol, side=side,
                    qty=max(1, int(qty)),
                    price_per_contract=float(contract.close_price or 0.0),
                )
                return {
                    "id": str(getattr(result, "id", "unknown")),
                    "status": str(getattr(result.status, "value", result.status)),
                    "filled_qty": float(getattr(result, "filled_qty", qty) or 0.0),
                    "filled_avg_price": float(
                        getattr(result, "filled_avg_price", 0.0) or 0.0),
                }
            # equity
            result = place_order(symbol=symbol, side=side, qty=int(qty),
                                 price_per_contract=0.0)
            return {
                "id": str(getattr(result, "id", "unknown")),
                "status": str(getattr(result.status, "value", result.status)),
                "filled_qty": float(getattr(result, "filled_qty", qty) or 0.0),
                "filled_avg_price": float(
                    getattr(result, "filled_avg_price", 0.0) or 0.0),
            }
        except Exception as e:
            return {"id": None, "status": "failed", "error": str(e)}
    return executor


def _make_market_data(stage: str, lookback_days: int):
    """Alpaca if keys present and not dry_run, else fake."""
    if stage in {"paper", "competition"} and \
            os.getenv("APCA_API_KEY_ID") and os.getenv("APCA_API_SECRET_KEY"):
        from investment_agent.data.market_data import AlpacaMarketDataClient
        return AlpacaMarketDataClient(), "alpaca"
    print("[market] using FakeMarketDataClient (offline / no Alpaca keys).",
          file=sys.stderr)
    return None, "fake"


def _bootstrap_fake_market_data(symbols, lookback_days):
    """Create a synthetic in-memory market so dry-run works offline."""
    from investment_agent.data.market_data import FakeMarketDataClient
    import pandas as pd
    md = FakeMarketDataClient()
    end = pd.Timestamp.now().normalize()
    for s in symbols:
        idx = pd.date_range(end=end, periods=lookback_days + 5, freq="D")
        closes = [100.0 + 0.2 * i for i in range(len(idx))]
        df = pd.DataFrame({
            "open": closes, "high": [c + 0.5 for c in closes],
            "low": [c - 0.5 for c in closes],
            "close": closes, "volume": [1_000_000.0] * len(idx),
        }, index=idx)
        md.set_series(s, df)
    return md


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", default="dry_run",
                        choices=["dry_run", "paper", "competition"])
    parser.add_argument("--interval", type=int, default=300,
                        help="Decision interval in seconds")
    parser.add_argument("--max-intervals", type=int, default=None,
                        help="Stop after N intervals (default: forever)")
    parser.add_argument("--max-lookups", type=int, default=2,
                        help="Max LLM calls per decision interval")
    parser.add_argument("--symbols", default="AAPL,SPY,MSFT")
    parser.add_argument("--top-n", type=int, default=2)
    parser.add_argument("--state-file", default="live_state.json")
    parser.add_argument("--memory-file", default="trade_memory.json")
    parser.add_argument("--reputation-file", default="reputation_state.json")
    parser.add_argument("--lookback", type=int, default=60)
    parser.add_argument(
        "--isolate-state", action="store_true",
        help="Write live state / memory / reputation to per-process temp files "
             "instead of the project root. Used by the smoke tests; leave off "
             "for normal runs so the dashboard can read the live updates.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    from investment_agent.agents.specialist import DEFAULT_ROLES
    from investment_agent.orchestrator import XQuantXOrchestrator
    from investment_agent.live.live_orchestrator import (
        LiveOrchestrator, LiveOrchestratorConfig,
    )
    from investment_agent.live.candidate_screener import CandidateScreener
    from investment_agent.products import ProductGate

    # By default we write the state / memory / reputation files to
    # the project root so the dashboard (which reads them on every
    # refresh) sees the live updates. ``--isolate-state`` redirects
    # them to per-process temp files so the smoke tests can run
    # without clobbering the canonical state.
    if args.isolate_state:
        if args.memory_file == "trade_memory.json":
            args.memory_file = os.path.join(tempfile.gettempdir(),
                                            "live_loop_memory.json")
        if args.reputation_file == "reputation_state.json":
            args.reputation_file = os.path.join(tempfile.gettempdir(),
                                                "live_loop_reputation.json")
        if args.state_file == "live_state.json":
            args.state_file = os.path.join(tempfile.gettempdir(), "live_loop_state.json")

    universe = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]

    # Market data
    md, md_kind = _make_market_data(args.stage, args.lookback)
    if md is None:
        md = _bootstrap_fake_market_data(universe, args.lookback)

    # Orchestrator
    canonical_ids = [r.agent_id for r in DEFAULT_ROLES]
    orch = XQuantXOrchestrator(
        agent_ids=canonical_ids, symbol=universe[0],
        use_hmm=False, enable_trading=False,
        memory_file=args.memory_file,
    )

    # LLM
    provider, llm_kind = _load_llm_provider(args.stage)
    factory = _build_agent_factory(provider, DEFAULT_ROLES, args.memory_file)
    print(f"[init] stage={args.stage}  llm={llm_kind}  market={md_kind}  "
          f"universe={','.join(universe)}  interval={args.interval}s",
          file=sys.stderr)

    # Executor
    executor = _make_executor(args.stage)

    # Build the live orchestrator
    config = LiveOrchestratorConfig(
        symbol_universe=universe,
        top_n_candidates=args.top_n,
        decision_interval_seconds=args.interval,
        state_file=args.state_file,
        memory_file=args.memory_file,
        reputation_file=args.reputation_file,
        lookback_days=args.lookback,
        max_lookups_per_interval=args.max_lookups,
        stage=args.stage,
    )
    live = LiveOrchestrator(
        config=config, market_data=md, orchestrator=orch,
        agent_factory=factory, executor=executor,
        screener=CandidateScreener(top_n=args.top_n),
        product_gate=ProductGate(),
    )

    # Run with a custom report printer
    reports = []
    import time
    try:
        while args.max_intervals is None or live._interval_index < args.max_intervals:
            report = live.run_once()
            reports.append(report)
            try:
                rep = {
                    aid: orch._reputation_tracker.get_posterior_parameters(aid, report.regime)
                    for aid in canonical_ids
                }
            except Exception:
                rep = None
            _print_report(report, rep)
            # Sleep ONLY when there's another interval still to run.
            if args.max_intervals is None or live._interval_index < args.max_intervals:
                time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\n[live-loop] interrupted by user", file=sys.stderr)

    # Final summary
    if reports:
        last = reports[-1]
        print()
        print(_bar("SUMMARY"))
        print(f"Intervals run:    {len(reports)}")
        print(f"Total decisions:  {sum(len(r.decisions) for r in reports)}")
        print(f"Total orders:     {sum(len(r.orders) for r in reports)}")
        print(f"Total closed:     {sum(len(r.exits) for r in reports)}")
        print(f"Final equity:     ${last.equity:,.2f}")
        print(f"State file:       {config.state_file}")
        print(f"Memory file:      {config.memory_file}")
        print(f"Reputation file:  {config.reputation_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
