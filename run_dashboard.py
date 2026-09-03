"""Entrypoint for the X Quant X monitoring dashboard.

Run from the repository root:  python run_dashboard.py
Then open http://127.0.0.1:8050 in a browser.
"""

import sys
from pathlib import Path

_src = str(Path(__file__).parent / "src")
if _src not in sys.path:
    sys.path.insert(0, _src)

from investment_agent.dashboard.app import app

if __name__ == "__main__":
    app.run(debug=False, host="127.0.0.1", port=8050)
