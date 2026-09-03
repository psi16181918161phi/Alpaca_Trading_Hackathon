"""Manual, one-off script that submits a REAL options order against your
Alpaca PAPER account.

WHY THIS ISN'T A PYTEST TEST
=============================
This used to live at tests/unit/test_order.py, where pytest auto-collects
and RUNS every file on `pytest` / `pytest tests/` -- with no test function,
no assertion, and no mock, it submitted a live BUY market order on every
single test run for anyone who had real APCA_API_KEY_ID / APCA_API_SECRET_KEY
credentials in their environment. Moved here so it only ever runs when
someone deliberately executes this file. The equivalent behavior already
has proper mocked unit-test coverage in tests/unit/execution/test_execution.py
(TestPlaceOrder, TestGetOptionContract) -- this script is for a human who
wants to eyeball a real paper-account fill, nothing more.

Run manually (never via pytest):
    python scripts/manual_place_test_order.py
"""

from __future__ import annotations

import os

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import GetOptionContractsRequest, MarketOrderRequest
from dotenv import load_dotenv

load_dotenv()


def main() -> int:
    key_id = os.getenv("APCA_API_KEY_ID")
    secret = os.getenv("APCA_API_SECRET_KEY")
    if not key_id or not secret:
        print("APCA_API_KEY_ID / APCA_API_SECRET_KEY missing from environment -- aborting.")
        return 1

    client = TradingClient(key_id, secret, paper=True)

    request = GetOptionContractsRequest(underlying_symbols=["AAPL"], limit=1)
    contract = client.get_option_contracts(request).option_contracts[0]

    print(f"About to submit a REAL market BUY order for 1x {contract.symbol} on your PAPER account.")
    confirm = input("Type YES to continue: ").strip()
    if confirm != "YES":
        print("Aborted -- no order submitted.")
        return 1

    order = MarketOrderRequest(
        symbol=contract.symbol,
        qty=1,
        side=OrderSide.BUY,
        time_in_force=TimeInForce.DAY,
    )
    result = client.submit_order(order)
    print("Order status:", result.status)
    print("Order id:", result.id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
