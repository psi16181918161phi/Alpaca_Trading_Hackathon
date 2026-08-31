import json
import os
from datetime import datetime, timedelta

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestTradeRequest
from dotenv import load_dotenv

load_dotenv()

_data_client = None


def _get_data_client():
    global _data_client
    if _data_client is None:
        _data_client = StockHistoricalDataClient(os.getenv("APCA_API_KEY_ID"), os.getenv("APCA_API_SECRET_KEY"))
    return _data_client


MEMORY_FILE = "memory_log.json"


def _load_memory():
    if not os.path.exists(MEMORY_FILE):
        return []
    with open(MEMORY_FILE, "r") as f:
        return json.load(f)


def _save_memory(records):
    with open(MEMORY_FILE, "w") as f:
        json.dump(records, f, indent=2, default=str)


def log_decision(symbol, drop_pct, action, order_id, price):
    """Record a hedge decision so future runs can check history."""
    records = _load_memory()
    records.append({
        "symbol": symbol,
        "drop_pct": drop_pct,
        "action": action,
        "order_id": order_id,
        "price_at_decision": price,
        "timestamp": datetime.now().isoformat(),
    })
    _save_memory(records)
    print(f"Logged decision: {action} {symbol} at drop {drop_pct:.2%}")


def already_hedged_recently(symbol, days=3):
    """Check if we already hedged this symbol within the last `days` days."""
    records = _load_memory()
    cutoff = datetime.now() - timedelta(days=days)
    for r in records:
        if r["symbol"] == symbol and r["action"] == "buy" and datetime.fromisoformat(r["timestamp"]) > cutoff:
            print(f"Already hedged {symbol} recently on {r['timestamp']}, skipping")
            return True
    return False


def _get_latest_price(symbol):
    request = StockLatestTradeRequest(symbol_or_symbols=symbol)
    client = _get_data_client()
    trade = client.get_stock_latest_trade(request)
    return float(trade[symbol].price)


def reflect(symbol):
    """Compare past decisions for this symbol against current price to see if they helped."""
    records = [r for r in _load_memory() if r["symbol"] == symbol]
    if not records:
        print(f"No past decisions logged for {symbol}")
        return []

    current_price = _get_latest_price(symbol)
    results = []
    for r in records:
        price_then = r["price_at_decision"]
        if price_then and price_then > 0:
            change_pct = (current_price - price_then) / price_then
            verdict = "helped (price recovered)" if change_pct > 0 else "didn't help (price still down)"
        else:
            verdict = "no price data to judge"
        result = {**r, "current_price": current_price, "verdict": verdict}
        results.append(result)
        print(f"{r['timestamp']}: {r['action']} {symbol} @ {price_then} -> now {current_price} -> {verdict}")

    return results


if __name__ == "__main__":
    reflect("AAPL")