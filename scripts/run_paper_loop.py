"""Run a full single-cycle paper-trading decision end-to-end.

This is the "one shot" demo of the entire pipeline:

  Alpaca OHLCV -> market features -> 7 LLM specialist agents ->
  signal ensemble -> HMM regime -> investment Kalman -> 7-state
  capital gate -> product gate (equity/option/none) -> Alpaca
  paper order (or skip) -> TradeMemory -> AgentReputationTracker

Each step's output is printed. If the product gate picks an option
or equity, the script will (optionally, --no-execute) submit a
paper order via Alpaca. The reputation tracker is persisted at
the end so the dashboard renders the new alpha/beta on the next
refresh.

Usage:
    # Offline mode (no Alpaca keys required; uses synthetic data):
    python scripts/run_paper_loop.py --symbol AAPL

    # Live paper mode (uses Alpaca):
    python scripts/run_paper_loop.py --symbol AAPL --live
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
from typing import Any, Dict, Optional

# Make the package importable when the script is run directly.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))


def _load_llm_provider(live: bool = False):
    """Wire the multi-provider Featherless orchestrator if keys exist.

    If live=True, mock provider fallback is strictly forbidden.
    """
    try:
        from investment_agent.llm.orchestrator import (
            FeatherlessOrchestrator, load_provider_specs,
        )
        specs = load_provider_specs()
        valid_specs = [s for s in specs if s.api_key]
        if valid_specs:
            return FeatherlessOrchestrator(specs=valid_specs)
    except Exception as e:
        if live:
            raise RuntimeError(
                f"FATAL: Live mode requires valid Featherless LLM keys. Error: {e}"
            ) from e
        print(f"WARN: failed to build Featherless orchestrator ({e}); using mock.", file=sys.stderr)

    if live:
        raise RuntimeError(
            "FATAL: Live paper mode (--live) requires valid Featherless API keys. "
            "Refusing to execute live paper trading with MockLLMProvider."
        )

    print("INFO: using MockLLMProvider for offline execution.", file=sys.stderr)
    from investment_agent.llm.base import MockLLMProvider
    return MockLLMProvider(responder=_mock_responder)


def _mock_responder(system: Optional[str], prompt: str) -> str:
    """Deterministic mock LLM response for the offline path."""
    import json
    if not system:
        return json.dumps({
            "signal": 0.0, "confidence": 0.5,
            "uncertainty": 0.5, "doubt": 0.5,
            "p_plus": 0.5, "p_minus": 0.5,
            "delta_t": 1.0, "noise": 0.5,
        })
    # Bias per role.
    if "Economic" in system:
        s = 0.3
    elif "Financial" in system:
        s = -0.1
    elif "Fiscal" in system:
        s = 0.0
    elif "Portfolio" in system:
        s = -0.2
    elif "Fundamental" in system:
        s = 0.4
    elif "Market Microstructure" in system:
        s = 0.2
    elif "Sector" in system:
        s = 0.1
    else:
        s = 0.0
    return json.dumps({
        "signal": s, "confidence": 0.8,
        "uncertainty": 0.2, "doubt": 0.1,
        "p_plus": 0.5 + s / 2.0, "p_minus": 0.5 - s / 2.0,
        "delta_t": 1.0, "noise": 0.5,
    })


def _load_market_data(symbol: str, days: int, live: bool):
    if live or (os.getenv("APCA_API_KEY_ID") and os.getenv("APCA_API_SECRET_KEY")):
        from investment_agent.data.market_data import AlpacaMarketDataClient
        return AlpacaMarketDataClient(), True
    print("INFO: using synthetic market data (set APCA_API_KEY_ID/SECRET or pass --live).",
          file=sys.stderr)
    import pandas as pd
    from investment_agent.data.market_data import FakeMarketDataClient
    idx = pd.date_range(datetime.now() - timedelta(days=days + 5),
                        periods=days + 5, freq="D")
    closes = [100.0 + 0.3 * i for i in range(len(idx))]
    df = pd.DataFrame({
        "open": closes, "high": [c + 0.5 for c in closes],
        "low": [c - 0.5 for c in closes],
        "close": closes, "volume": [1_000_000.0] * len(idx),
    }, index=idx)
    fake = FakeMarketDataClient()
    fake.set_series(symbol, df)
    return fake, False


def _maybe_execute(
    product: str,
    symbol: str,
    action: str,
    quantity: float,
    live: bool,
    option_side: Optional[str] = None,
    price: Optional[float] = None,
) -> Dict[str, Any]:
    """Submit a paper order via Alpaca. Returns the order result envelope.

    For ``product == "option"`` this looks up a real OCC-format option
    contract via ``get_option_contract`` and submits the order against
    that symbol (not the underlying equity). For ``product ==
    "equity"`` the underlying symbol is used directly.
    """
    if not live:
        return {"submitted": False, "reason": "offline mode (no --live)"}
    if product == "none":
        return {"submitted": False, "reason": "product gate returned no-trade"}
    try:
        from investment_agent.execution.execution import (
            place_order, get_option_contract,
        )
        if product == "option":
            contract = get_option_contract(
                symbol, option_type=option_side,
            )
            opt_price = float(contract.close_price or price or 0.0)
            order = place_order(
                symbol=contract.symbol,
                side=action.lower(),
                qty=max(1, int(quantity)),
                price_per_contract=opt_price,
                is_option=True,
            )
            return {
                "submitted": True,
                "order": order,
                "product": "option",
                "option_symbol": contract.symbol,
                "option_side": option_side,
            }
        # equity
        equity_price = float(price if (price is not None and price > 0) else 100.0)
        order = place_order(
            symbol=symbol,
            side=action.lower(),
            qty=int(quantity) if quantity > 0 else 0,
            price_per_share=equity_price,
            is_option=False,
        )
        return {"submitted": True, "order": order, "product": "equity"}
    except Exception as e:
        return {"submitted": False, "error": str(e), "product": product}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default="AAPL")
    parser.add_argument("--days", type=int, default=120,
                        help="Lookback window in days for market features")
    parser.add_argument("--memory", default="trade_memory.json")
    parser.add_argument("--reputation", default="reputation_state.json")
    parser.add_argument("--live", action="store_true",
                        help="Submit a real paper order via Alpaca")
    parser.add_argument("--no-execute", action="store_true",
                        help="Skip Alpaca order submission (default: skip in offline mode)")
    args = parser.parse_args()

    # Avoid clobbering any pre-existing reputation / memory files in the
    # repo root by defaulting to fresh per-process temp files. The
    # dashboard reads the canonical repo-root paths when it's running
    # for real; a script run shouldn't silently overwrite them.
    if args.memory == "trade_memory.json":
        args.memory = os.path.join(tempfile.gettempdir(), "run_paper_loop_memory.json")
    if args.reputation == "reputation_state.json":
        args.reputation = os.path.join(tempfile.gettempdir(), "run_paper_loop_reputation.json")

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    # 1. Market data
    md, is_live = _load_market_data(args.symbol, args.days, args.live)
    from investment_agent.data.market_data import BarRequest
    end = datetime.now()
    start = end - timedelta(days=args.days)
    bars = md.get_historical_bars(BarRequest(
        symbol=args.symbol, start=start, end=end, timeframe="1Day",
    ))
    if bars is None or len(bars) < 30:
        print(f"ERROR: insufficient data for {args.symbol} (got {len(bars) if bars is not None else 0} bars)")
        return 1
    prices = bars["close"].tolist()
    volumes = bars["volume"].tolist() if "volume" in bars.columns else [0.0] * len(bars)
    highs = bars["high"].tolist() if "high" in bars.columns else None
    lows = bars["low"].tolist() if "low" in bars.columns else None
    print(f"[1] Market data: {len(bars)} bars for {args.symbol} "
          f"({bars.index.min().date()} -> {bars.index.max().date()})")

    # 2. LLM provider
    provider = _load_llm_provider(live=args.live)
    from investment_agent.agents.specialist import (
        DEFAULT_ROLES, build_specialist_agents, AgentContext, run_agents,
    )
    from investment_agent.regimes.market_feature_extractor import compute_dict_features
    from investment_agent.memory.trade_memory import TradeMemory, TradeExperience

    agents = build_specialist_agents(provider)
    print(f"[2] LLM: {len(agents)} agents ready "
          f"(provider type: {type(provider).__name__})")

    # 3. Regime classification (Authoritative HMM detector).
    from investment_agent.regimes.market_feature_extractor import extract_features
    from investment_agent.regimes.hmm_regime_detector import HMMRegimeDetector
    features_matrix = extract_features(prices, volumes, highs=highs, lows=lows, lookback_days=20)
    regime = HMMRegimeDetector().classify(features_matrix.tolist())
    confidence = 1.0 - regime.normalized_entropy
    print(f"[3] Regime (HMM Authoritative): {regime.regime} (confidence={confidence:.2f})")

    # Retrieve past memories if available
    tm = TradeMemory(args.memory)
    memories = []
    try:
        dummy_exp = TradeExperience(
            decision_id="preview", timestamp=datetime.now(), symbol=args.symbol,
            regime=regime.regime, regime_probabilities=dict(regime.probabilities),
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

    features_dict = compute_dict_features(prices, volumes)

    # 4. Run the seven specialist agents.
    ctx = AgentContext(
        symbol=args.symbol,
        regime=regime.regime,
        regime_probabilities=dict(regime.probabilities),
        features=features_dict,
        ensemble_signal=0.0,
        disagreement=0.0,
        memory=memories,
    )
    agent_outputs_map = run_agents(agents, ctx)
    agent_outputs = [agent_outputs_map[r.agent_id] for r in DEFAULT_ROLES]
    signals = {r.agent_id: agent_outputs_map[r.agent_id].s for r in DEFAULT_ROLES}
    print(f"[4] Agent signals: " + ", ".join(
        f"{k}={v:+.2f}" for k, v in sorted(signals.items())))

    # 5. Run the deterministic pipeline through the orchestrator.
    from investment_agent.orchestrator import XQuantXOrchestrator
    from investment_agent.capital.capital_gate import SevenStateVector
    from investment_agent.products import (
        ProductGate, ProductGateInput, PRODUCT_OPTION, PRODUCT_EQUITY,
    )

    orch = XQuantXOrchestrator(
        agent_ids=[r.agent_id for r in DEFAULT_ROLES],
        symbol=args.symbol,
        use_hmm=True,
        enable_trading=False,
        memory_file=args.memory,
        reputation_file=args.reputation,
    )
    states = SevenStateVector(
        economic=1.0, financial=1.0, fiscal=1.0,
        portfolio=1.0, fundamental=1.0, market=1.0, sector=1.0,
    )
    result = orch.run_cycle(
        prices=prices, volumes=volumes,
        agent_outputs=agent_outputs, states=states,
        portfolio_context={
            "position_pct": 0.0, "gross_leverage": 0.0, "entropy": 0.1,
            "drawdown_pct": 0.0, "execution_timeout_seconds": 5.0,
            "sector_exposure_pct": 0.0, "is_new_long": True,
            "regime": regime.regime, "available_liquidity": 100000.0,
        },
    )
    experience = result.experience
    print(f"[5] Decision: action={experience.position_action} "
          f"qty={experience.quantity:.4f} verdict={experience.capital_gate_verdict} "
          f"signal={experience.ensemble_signal:+.3f} "
          f"disagreement={experience.disagreement:.3f}")

    # 6. Product gate: equity / option / none
    product_gate = ProductGate()
    pg_input = ProductGateInput(
        action=experience.position_action,
        verdict=experience.capital_gate_verdict,
        ensemble_signal=experience.ensemble_signal,
        disagreement=experience.disagreement,
        confidence=experience.effective_confidence,
        regime=regime.regime,
    )
    pg_result = product_gate.decide(pg_input)
    print(f"[6] Product gate: product={pg_result.product} "
          f"side={pg_result.option_side} reason={pg_result.reason}")

    # 7. Persist reputation snapshot for the dashboard.
    try:
        from investment_agent.agents.reputation_persistence import save_reputation
        save_reputation(orch._reputation_tracker, args.reputation)
        print(f"[7] Reputation persisted to {args.reputation}")
    except Exception as e:
        print(f"WARN: failed to persist reputation: {e}", file=sys.stderr)

    # 8. Optional: submit paper order.
    if args.no_execute or (not args.live and not is_live):
        print("[8] Skipping execution (offline mode or --no-execute).")
        return 0
    result = _maybe_execute(
        product=pg_result.product, symbol=args.symbol,
        action=experience.position_action, quantity=experience.quantity,
        live=args.live and is_live,
        option_side=pg_result.option_side,
    )
    print(f"[8] Execution: {json.dumps(result, default=str)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
