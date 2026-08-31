from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from datetime import datetime, timedelta
from dotenv import load_dotenv
import os

from investment_agent.execution.execution import place_order, get_option_contract
from investment_agent.memory.memory import log_decision, already_hedged_recently
from investment_agent.execution.hedge_capital_bridge import evaluate_hedge_risk, record_hedge_placement

load_dotenv()

_data_client = None


def _get_data_client():
    global _data_client
    if _data_client is None:
        _data_client = StockHistoricalDataClient(os.getenv("APCA_API_KEY_ID"), os.getenv("APCA_API_SECRET_KEY"))
    return _data_client


DROP_THRESHOLD_PCT = 0.03  # trigger a hedge if price dropped more than 3% from its recent high


def get_recent_prices(symbol, days=10):
    request = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame.Day,
        start=datetime.now() - timedelta(days=days),
    )
    client = _get_data_client()
    bars = client.get_stock_bars(request)
    return [bar.close for bar in bars[symbol]]


def check_for_drop(symbol):
    """Return (dropped, drop_pct) -- dropped is True if price fell more than DROP_THRESHOLD_PCT from its recent high."""
    prices = get_recent_prices(symbol)
    if len(prices) < 2:
        print(f"Not enough price data for {symbol}")
        return False, 0.0

    recent_high = max(prices)
    current_price = prices[-1]
    drop_pct = (recent_high - current_price) / recent_high

    print(f"{symbol}: high={recent_high:.2f}, current={current_price:.2f}, drop={drop_pct:.2%}")
    return drop_pct >= DROP_THRESHOLD_PCT, drop_pct


def run_hedge_check(symbol):
    """If the stock has dropped enough and we haven't already hedged recently, buy a protective put
    sized according to the risk-adjusted assessment from hedge_capital_bridge."""
    assessment = evaluate_hedge_risk(symbol)

    for reason in assessment.reasons:
        print(f"[{symbol}] {reason}")

    if assessment.verdict == "BLOCK":
        print(f"Hedge blocked for {symbol}, no action taken")
        return

    if already_hedged_recently(symbol):
        return

    contract = get_option_contract(symbol, option_type="put")
    price = float(contract.close_price or 0)
    qty = assessment.adjusted_quantity
    print(f"Drop detected on {symbol} ({assessment.verdict}) -- buying {qty} protective put(s) {contract.symbol}")
    result = place_order(contract.symbol, "buy", qty=qty, price_per_contract=price)

    if result:
        log_decision(symbol, assessment.drop_pct, "buy", result.id, price)
        record_hedge_placement(symbol)
    else:
        print(f"Order for {symbol} was blocked by safety check, not logging as a hedge")


if __name__ == "__main__":
    run_hedge_check("AAPL")