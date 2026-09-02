from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, GetOptionContractsRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from dotenv import load_dotenv
import os

load_dotenv()

client = TradingClient(os.getenv("APCA_API_KEY_ID"), os.getenv("APCA_API_SECRET_KEY"), paper=True)

# grab one contract to trade
request = GetOptionContractsRequest(underlying_symbols=["AAPL"], limit=1)
contract = client.get_option_contracts(request).option_contracts[0]
print("Buying:", contract.symbol)

order = MarketOrderRequest(
    symbol=contract.symbol,
    qty=1,
    side=OrderSide.BUY,
    time_in_force=TimeInForce.DAY,
)

result = client.submit_order(order)
print("Order status:", result.status)
print("Order id:", result.id)