from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetOptionContractsRequest
from dotenv import load_dotenv
import os

load_dotenv()

client = TradingClient(os.getenv("APCA_API_KEY_ID"), os.getenv("APCA_API_SECRET_KEY"), paper=True)

request = GetOptionContractsRequest(underlying_symbols=["AAPL"], limit=5)
contracts = client.get_option_contracts(request)

for c in contracts.option_contracts:
    print(c.symbol, c.strike_price, c.expiration_date, c.type)