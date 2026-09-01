"""Run a historical replay / backtest of the trading pipeline.

Usage:
    python scripts/run_replay.py --symbol AAPL --days 180
    python scripts/run_replay.py --symbol SPY --days 90 --output replay.json

The script pulls daily bars via the unified market-data interface,
drives the orchestrator over each bar, closes the previous bar's
position at the current close, updates the reputation tracker, and
persists both ``trade_memory.json`` and ``reputation_state.json``
so the dashboard renders the same data.
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

# Ensure the package is importable when the script is run directly.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))


def _build_agent_factory(agent_ids):
    """Default factory: every agent is a small deterministic policy
    keyed off the recent return. Replace this with the LLM-backed
    factory in production.
    """
    from investment_agent.signals.ensemble_signal import AgentOutput

    def factory(bar_ctx):
        ret = bar_ctx.get("recent_return", 0.0)
        # Map recent return to a [-1, +1] signal with confidence that
        # grows as the trend persists.
        signal = max(-1.0, min(1.0, ret * 50.0))
        confidence = 0.5 + min(0.4, abs(ret) * 10.0)
        return [
            AgentOutput(
                s=signal, c=confidence, u=1.0 - confidence,
                d=0.1, p_plus=0.5 + signal / 2.0, p_minus=0.5 - signal / 2.0,
                delta_t=1.0, r=0.0, agent_id=aid,
            )
            for aid in agent_ids
        ]
    return factory


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", required=True, help="Symbol to backtest")
    parser.add_argument("--days", type=int, default=180, help="Lookback window in days")
    parser.add_argument("--lookback", type=int, default=30, help="Initial warm-up bars")
    parser.add_argument("--cash", type=float, default=100_000.0)
    parser.add_argument("--memory", default="trade_memory.json")
    parser.add_argument("--reputation", default="reputation_state.json")
    parser.add_argument("--output", help="Optional path to write the ReplayResult JSON")
    parser.add_argument("--timeframe", default="1Day")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    from investment_agent.data.market_data import (
        AlpacaMarketDataClient, BarRequest, FakeMarketDataClient,
    )
    from investment_agent.orchestrator import XQuantXOrchestrator
    from investment_agent.replay import ReplayConfig, ReplayEngine

    if os.getenv("APCA_API_KEY_ID") and os.getenv("APCA_API_SECRET_KEY"):
        md = AlpacaMarketDataClient()
    else:
        print("WARN: APCA_API_KEY_ID/SECRET not set; using a tiny synthetic series.",
              file=sys.stderr)
        import pandas as pd
        idx = pd.date_range(datetime.now() - timedelta(days=args.days + 5),
                            periods=args.days + 5, freq="D")
        closes = [100.0 + 0.2 * i for i in range(len(idx))]
        df = pd.DataFrame({"open": closes, "high": closes, "low": closes,
                           "close": closes, "volume": [1.0] * len(idx)}, index=idx)
        md = FakeMarketDataClient()
        md.set_series(args.symbol, df)

    end = datetime.now()
    start = end - timedelta(days=args.days)
    # Sanity: pull a few bars so we know the data path works.
    probe = md.get_historical_bars(BarRequest(
        symbol=args.symbol, start=start, end=end, timeframe=args.timeframe,
    ))
    print(f"Loaded {len(probe)} bars for {args.symbol} "
          f"({probe.index.min() if not probe.empty else 'n/a'} -> "
          f"{probe.index.max() if not probe.empty else 'n/a'})")

    orch = XQuantXOrchestrator(
        agent_ids=[f"agent{i + 1}" for i in range(7)],
        symbol=args.symbol,
        use_hmm=False,
        enable_trading=False,
        memory_file=args.memory,
    )

    engine = ReplayEngine(orch, market_data=md, reputation_file=args.reputation)
    result = engine.run(
        ReplayConfig(
            symbol=args.symbol,
            start=start, end=end,
            lookback=args.lookback,
            starting_cash=args.cash,
        ),
        agent_factory=_build_agent_factory(orch._agent_ids),
    )

    print()
    print("=" * 60)
    print(f"Replay complete: {result.symbol}")
    print(f"  bars processed:        {result.bars_processed}")
    print(f"  decisions:             {result.decisions}")
    print(f"  buys / sells / holds:  {result.buys} / {result.sells} / {result.holds}")
    print(f"  closed trades:         {result.closed_trades}")
    print(f"  realized P&L:          ${result.realized_pnl:,.2f}")
    print(f"  transaction costs:     ${result.transaction_costs:,.2f}")
    print(f"  final equity:          ${result.final_equity:,.2f}")
    print(f"  max drawdown:          {result.max_drawdown_pct:.1%}")
    print(f"  win rate:              {result.win_rate:.1%}")
    print(f"  memory file:           {result.memory_file}")
    print(f"  reputation file:       {result.reputation_file}")
    print("=" * 60)

    if args.output:
        Path(args.output).write_text(json.dumps(result.to_dict(), indent=2, default=str))
        print(f"Result written to {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
