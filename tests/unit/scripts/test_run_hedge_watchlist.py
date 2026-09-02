"""Unit tests for ``scripts/run_hedge_watchlist.py``.

Migrated alongside ``scripts/run_hedge_watchlist.py`` from the historical
``archive/run_agent.py`` script. ``run_hedge_check`` is mocked so no real
Alpaca/market-data call is ever made, matching the mocking convention
already used in ``tests/unit/signals/test_hedge_signal.py``.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from scripts import run_hedge_watchlist


def test_run_watchlist_once_checks_every_symbol():
    with patch.object(run_hedge_watchlist, "run_hedge_check") as mock_check:
        run_hedge_watchlist.run_watchlist_once(["AAPL", "MSFT"])
    assert mock_check.call_count == 2
    mock_check.assert_any_call("AAPL")
    mock_check.assert_any_call("MSFT")


def test_run_watchlist_once_uses_default_watchlist():
    with patch.object(run_hedge_watchlist, "run_hedge_check") as mock_check:
        run_hedge_watchlist.run_watchlist_once()
    assert mock_check.call_count == len(run_hedge_watchlist.DEFAULT_WATCHLIST)


def test_main_once_flag_runs_single_pass_and_exits():
    with patch.object(run_hedge_watchlist, "run_hedge_check") as mock_check:
        exit_code = run_hedge_watchlist.main(["--once", "--watchlist", "TSLA"])
    assert exit_code == 0
    mock_check.assert_called_once_with("TSLA")


def test_main_forever_mode_invokes_run_forever():
    with patch.object(run_hedge_watchlist, "run_forever") as mock_forever:
        exit_code = run_hedge_watchlist.main(["--watchlist", "AAPL", "--interval", "5"])
    assert exit_code == 0
    mock_forever.assert_called_once_with(["AAPL"], 5)
