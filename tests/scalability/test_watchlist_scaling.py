"""Scalability tests: wall-clock growth should stay near-linear in input size.

Uses a mocked ``run_hedge_check`` (no network) and asserts the watchlist
loop's per-symbol overhead does not blow up as the watchlist grows from
3 to 300 symbols -- i.e. no accidental O(n^2) behavior (e.g. re-scanning
the whole list per symbol).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from scripts import run_hedge_watchlist

SMALL_N = 10
LARGE_N = 300
# Allow generous headroom: LARGE_N/SMALL_N == 30x input growth should cost
# well under 30x wall-clock (near-linear), catching an accidental
# quadratic regression without being a flaky micro-benchmark.
MAX_ACCEPTABLE_RATIO = 60.0


def _time_watchlist(n: int) -> float:
    watchlist = [f"SYM{i}" for i in range(n)]
    with patch.object(run_hedge_watchlist, "run_hedge_check", return_value=None):
        start = time.perf_counter()
        run_hedge_watchlist.run_watchlist_once(watchlist)
        return time.perf_counter() - start


def test_watchlist_scaling_is_near_linear():
    small_time = max(_time_watchlist(SMALL_N), 1e-6)
    large_time = _time_watchlist(LARGE_N)
    ratio = large_time / small_time
    assert ratio < MAX_ACCEPTABLE_RATIO, (
        f"watchlist time grew {ratio:.1f}x for a {LARGE_N / SMALL_N:.0f}x larger "
        f"input -- possible super-linear regression"
    )
