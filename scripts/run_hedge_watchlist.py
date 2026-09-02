"""Run periodic hedge-signal checks across a watchlist of symbols.

Migrated from the historical ``archive/run_agent.py`` script into the
``scripts/`` convention (argparse + REPO_ROOT sys.path shim), with the
previously hardcoded watchlist/interval now overridable via flags and a
``--once`` mode so a single pass can be driven and tested without an
infinite loop.

Usage:
    python scripts/run_hedge_watchlist.py --once
    python scripts/run_hedge_watchlist.py --watchlist AAPL MSFT TSLA --interval 300
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from investment_agent.signals.hedge_signal import run_hedge_check

DEFAULT_WATCHLIST = ["AAPL", "MSFT", "TSLA"]
DEFAULT_INTERVAL_SECONDS = 300


def run_watchlist_once(watchlist: Sequence[str] = DEFAULT_WATCHLIST) -> None:
    """Run a single hedge-check pass across every symbol in the watchlist."""
    for symbol in watchlist:
        run_hedge_check(symbol)


def run_forever(watchlist: Sequence[str], interval_s: int) -> None:
    """Run hedge-check passes forever, sleeping ``interval_s`` between passes."""
    while True:
        run_watchlist_once(watchlist)
        print(f"Sleeping {interval_s}s until next check...")
        time.sleep(interval_s)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--watchlist", nargs="+", default=DEFAULT_WATCHLIST)
    parser.add_argument("--interval", type=int, default=DEFAULT_INTERVAL_SECONDS)
    parser.add_argument("--once", action="store_true", help="Run a single pass and exit.")
    args = parser.parse_args(argv)
    if args.once:
        run_watchlist_once(args.watchlist)
        return 0
    run_forever(args.watchlist, args.interval)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
