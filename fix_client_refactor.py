"""
Fixes get_positions/get_order_history (and get_account_summary) after Ameer's
refactor to a lazy _get_trading_client() instead of a module-level `client`.

Run this from the repo root:
    python fix_client_refactor.py
    python -m pytest tests/ -v
"""
import os

EXEC_PATH = os.path.join("src", "investment_agent", "execution", "execution.py")
TEST_PATH = os.path.join("tests", "unit", "execution", "test_execution.py")

EXEC_REPLACEMENTS = [
    (
        'def get_account_summary():\n'
        '    """Return current account status and buying power."""\n'
        '    account = client.get_account()',
        'def get_account_summary():\n'
        '    """Return current account status and buying power."""\n'
        '    client = _get_trading_client()\n'
        '    account = client.get_account()',
    ),
    (
        'def get_positions():\n'
        '    """Return current open positions as plain dicts (read-only, no order submission)."""\n'
        '    positions = client.get_all_positions()',
        'def get_positions():\n'
        '    """Return current open positions as plain dicts (read-only, no order submission)."""\n'
        '    client = _get_trading_client()\n'
        '    positions = client.get_all_positions()',
    ),
    (
        'def get_order_history(limit=100):\n'
        '    """Return recent orders (submitted/filled/cancelled) as plain dicts (read-only)."""\n'
        '    from alpaca.trading.requests import GetOrdersRequest\n'
        '\n'
        '    request = GetOrdersRequest(status="all", limit=limit)\n'
        '    orders = client.get_orders(request)',
        'def get_order_history(limit=100):\n'
        '    """Return recent orders (submitted/filled/cancelled) as plain dicts (read-only)."""\n'
        '    from alpaca.trading.requests import GetOrdersRequest\n'
        '\n'
        '    client = _get_trading_client()\n'
        '    request = GetOrdersRequest(status="all", limit=limit)\n'
        '    orders = client.get_orders(request)',
    ),
]

TEST_GETPOSITIONS_OLD = '''class TestGetPositions(unittest.TestCase):
    def test_returns_empty_list_when_no_positions(self):
        with patch.object(execution.client, "get_all_positions", return_value=[]):
            self.assertEqual(execution.get_positions(), [])

    def test_maps_position_fields(self):
        fake_side = MagicMock()
        fake_side.value = "long"
        fake_position = MagicMock(
            symbol="AAPL",
            side=fake_side,
            qty="10",
            avg_entry_price="150.00",
            current_price="155.00",
            market_value="1550.00",
            unrealized_pl="50.00",
            unrealized_plpc="0.0333",
        )
        with patch.object(execution.client, "get_all_positions", return_value=[fake_position]):
            result = execution.get_positions()
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0]["symbol"], "AAPL")
            self.assertEqual(result[0]["side"], "long")
            self.assertEqual(result[0]["qty"], 10.0)
            self.assertEqual(result[0]["unrealized_pl"], 50.0)

    def test_handles_none_optional_fields(self):
        fake_side = MagicMock()
        fake_side.value = "short"
        fake_position = MagicMock(
            symbol="TSLA",
            side=fake_side,
            qty="1",
            avg_entry_price="200.00",
            current_price=None,
            market_value=None,
            unrealized_pl=None,
            unrealized_plpc=None,
        )
        with patch.object(execution.client, "get_all_positions", return_value=[fake_position]):
            result = execution.get_positions()
            self.assertIsNone(result[0]["current_price"])
            self.assertIsNone(result[0]["unrealized_pl"])'''

TEST_GETPOSITIONS_NEW = '''class TestGetPositions(unittest.TestCase):
    def test_returns_empty_list_when_no_positions(self):
        fake_client = MagicMock()
        fake_client.get_all_positions.return_value = []
        with patch.object(execution, "_get_trading_client", return_value=fake_client):
            self.assertEqual(execution.get_positions(), [])

    def test_maps_position_fields(self):
        fake_side = MagicMock()
        fake_side.value = "long"
        fake_position = MagicMock(
            symbol="AAPL",
            side=fake_side,
            qty="10",
            avg_entry_price="150.00",
            current_price="155.00",
            market_value="1550.00",
            unrealized_pl="50.00",
            unrealized_plpc="0.0333",
        )
        fake_client = MagicMock()
        fake_client.get_all_positions.return_value = [fake_position]
        with patch.object(execution, "_get_trading_client", return_value=fake_client):
            result = execution.get_positions()
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0]["symbol"], "AAPL")
            self.assertEqual(result[0]["side"], "long")
            self.assertEqual(result[0]["qty"], 10.0)
            self.assertEqual(result[0]["unrealized_pl"], 50.0)

    def test_handles_none_optional_fields(self):
        fake_side = MagicMock()
        fake_side.value = "short"
        fake_position = MagicMock(
            symbol="TSLA",
            side=fake_side,
            qty="1",
            avg_entry_price="200.00",
            current_price=None,
            market_value=None,
            unrealized_pl=None,
            unrealized_plpc=None,
        )
        fake_client = MagicMock()
        fake_client.get_all_positions.return_value = [fake_position]
        with patch.object(execution, "_get_trading_client", return_value=fake_client):
            result = execution.get_positions()
            self.assertIsNone(result[0]["current_price"])
            self.assertIsNone(result[0]["unrealized_pl"])'''

TEST_GETORDERS_OLD = '''class TestGetOrderHistory(unittest.TestCase):
    def test_returns_empty_list_when_no_orders(self):
        with patch.object(execution.client, "get_orders", return_value=[]):
            self.assertEqual(execution.get_order_history(), [])

    def test_maps_order_fields(self):
        fake_side = MagicMock()
        fake_side.value = "buy"
        fake_type = MagicMock()
        fake_type.value = "market"
        fake_status = MagicMock()
        fake_status.value = "filled"
        fake_submitted_at = MagicMock()
        fake_submitted_at.isoformat.return_value = "2026-08-31T12:00:00"
        fake_order = MagicMock(
            id="order-123",
            submitted_at=fake_submitted_at,
            symbol="AAPL",
            side=fake_side,
            order_type=fake_type,
            qty="5",
            filled_qty="5",
            filled_avg_price="150.25",
            status=fake_status,
        )
        with patch.object(execution.client, "get_orders", return_value=[fake_order]):
            result = execution.get_order_history()
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0]["order_id"], "order-123")
            self.assertEqual(result[0]["symbol"], "AAPL")
            self.assertEqual(result[0]["side"], "buy")
            self.assertEqual(result[0]["status"], "filled")
            self.assertEqual(result[0]["filled_avg_price"], 150.25)

    def test_handles_missing_submitted_at(self):
        fake_side = MagicMock()
        fake_side.value = "sell"
        fake_type = MagicMock()
        fake_type.value = "market"
        fake_status = MagicMock()
        fake_status.value = "new"
        fake_order = MagicMock(
            id="order-456",
            submitted_at=None,
            symbol="MSFT",
            side=fake_side,
            order_type=fake_type,
            qty="1",
            filled_qty=None,
            filled_avg_price=None,
            status=fake_status,
        )
        with patch.object(execution.client, "get_orders", return_value=[fake_order]):
            result = execution.get_order_history()
            self.assertIsNone(result[0]["timestamp"])
            self.assertIsNone(result[0]["filled_qty"])'''

TEST_GETORDERS_NEW = '''class TestGetOrderHistory(unittest.TestCase):
    def test_returns_empty_list_when_no_orders(self):
        fake_client = MagicMock()
        fake_client.get_orders.return_value = []
        with patch.object(execution, "_get_trading_client", return_value=fake_client):
            self.assertEqual(execution.get_order_history(), [])

    def test_maps_order_fields(self):
        fake_side = MagicMock()
        fake_side.value = "buy"
        fake_type = MagicMock()
        fake_type.value = "market"
        fake_status = MagicMock()
        fake_status.value = "filled"
        fake_submitted_at = MagicMock()
        fake_submitted_at.isoformat.return_value = "2026-08-31T12:00:00"
        fake_order = MagicMock(
            id="order-123",
            submitted_at=fake_submitted_at,
            symbol="AAPL",
            side=fake_side,
            order_type=fake_type,
            qty="5",
            filled_qty="5",
            filled_avg_price="150.25",
            status=fake_status,
        )
        fake_client = MagicMock()
        fake_client.get_orders.return_value = [fake_order]
        with patch.object(execution, "_get_trading_client", return_value=fake_client):
            result = execution.get_order_history()
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0]["order_id"], "order-123")
            self.assertEqual(result[0]["symbol"], "AAPL")
            self.assertEqual(result[0]["side"], "buy")
            self.assertEqual(result[0]["status"], "filled")
            self.assertEqual(result[0]["filled_avg_price"], 150.25)

    def test_handles_missing_submitted_at(self):
        fake_side = MagicMock()
        fake_side.value = "sell"
        fake_type = MagicMock()
        fake_type.value = "market"
        fake_status = MagicMock()
        fake_status.value = "new"
        fake_order = MagicMock(
            id="order-456",
            submitted_at=None,
            symbol="MSFT",
            side=fake_side,
            order_type=fake_type,
            qty="1",
            filled_qty=None,
            filled_avg_price=None,
            status=fake_status,
        )
        fake_client = MagicMock()
        fake_client.get_orders.return_value = [fake_order]
        with patch.object(execution, "_get_trading_client", return_value=fake_client):
            result = execution.get_order_history()
            self.assertIsNone(result[0]["timestamp"])
            self.assertIsNone(result[0]["filled_qty"])'''


def main():
    if not os.path.isfile(EXEC_PATH):
        raise SystemExit("Can't find " + EXEC_PATH + " -- run this from the repo root.")

    with open(EXEC_PATH, "r", encoding="utf-8") as f:
        exec_content = f.read()
    changed = False
    for old, new in EXEC_REPLACEMENTS:
        if old in exec_content:
            exec_content = exec_content.replace(old, new)
            changed = True
    if changed:
        with open(EXEC_PATH, "w", newline="\n", encoding="utf-8") as f:
            f.write(exec_content)
        print("patched", EXEC_PATH)
    else:
        print("no change needed in", EXEC_PATH, "(already patched, or lines not found -- check manually)")

    with open(TEST_PATH, "r", encoding="utf-8") as f:
        test_content = f.read()
    test_changed = False
    if TEST_GETPOSITIONS_OLD in test_content:
        test_content = test_content.replace(TEST_GETPOSITIONS_OLD, TEST_GETPOSITIONS_NEW)
        test_changed = True
    if TEST_GETORDERS_OLD in test_content:
        test_content = test_content.replace(TEST_GETORDERS_OLD, TEST_GETORDERS_NEW)
        test_changed = True
    if test_changed:
        with open(TEST_PATH, "w", newline="\n", encoding="utf-8") as f:
            f.write(test_content)
        print("patched", TEST_PATH)
    else:
        print("no change needed in", TEST_PATH, "(already patched, or lines not found -- check manually)")

    print("done -- now run: python -m pytest tests/ -v")


if __name__ == "__main__":
    main()
