"""Entrypoint for the X Quant X monitoring dashboard.

Run from the repository root:  python scripts/run_dashboard.py
Then open http://127.0.0.1:8050 in a browser.

Migrated from the historical ``archive/run_dashboard.py`` scaffold script
into the ``scripts/`` convention (argparse + REPO_ROOT sys.path shim).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from investment_agent.dashboard.app import app


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args(argv)
    app.run(debug=args.debug, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
