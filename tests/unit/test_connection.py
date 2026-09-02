from alpaca.trading.client import TradingClient
from dotenv import load_dotenv
import os

load_dotenv()

client = TradingClient(os.getenv("APCA_API_KEY_ID"), os.getenv("APCA_API_SECRET_KEY"), paper=True)
account = client.get_account()

print("Connected. Account status:", account.status)
print("Buying power:", account.buying_power)