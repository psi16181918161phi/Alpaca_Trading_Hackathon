"""Unit tests for ``scripts/run_dashboard.py``.

Migrated alongside ``scripts/run_dashboard.py`` from the historical
``archive/run_dashboard.py`` entrypoint. ``app.run`` is mocked so no real
HTTP server is ever started during the test.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from scripts import run_dashboard


def test_module_exposes_dash_app():
    assert run_dashboard.app is not None


def test_main_default_args_calls_app_run():
    with patch.object(run_dashboard.app, "run") as mock_run:
        exit_code = run_dashboard.main([])
    assert exit_code == 0
    mock_run.assert_called_once_with(debug=False, host="127.0.0.1", port=8050)


def test_main_custom_args_calls_app_run():
    with patch.object(run_dashboard.app, "run") as mock_run:
        exit_code = run_dashboard.main(["--host", "0.0.0.0", "--port", "9000", "--debug"])
    assert exit_code == 0
    mock_run.assert_called_once_with(debug=True, host="0.0.0.0", port=9000)
