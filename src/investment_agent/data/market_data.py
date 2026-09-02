"""Single market-data interface for the trading system.

WHAT
====
The only module that should construct an Alpaca
``StockHistoricalDataClient`` or ``CryptoHistoricalDataClient``
and fetch OHLCV bars. The orchestrator, replay engine, dashboard
loader, and feature extractor all go through this facade instead
of reaching into Alpaca independently.

WHY
====
Previously bars were pulled ad-hoc in ``signals/hedge_signal.py``
and ``memory/memory.py`` with their own client construction. That
made it impossible to:
  * swap the data source for backtest/replay (no fake-data injection)
  * reason about credentials in one place
  * unit-test the orchestrator against deterministic prices

Asset-class routing is centralised here: equity symbols go through
``StockHistoricalDataClient``, crypto symbols (e.g. ``BTC/USD``)
go through ``CryptoHistoricalDataClient``, and the fake client
keyed by raw symbol so tests can seed either asset class.

HOW
====
``MarketDataClient`` is a thin wrapper that:
  * reads ``APCA_API_KEY_ID`` / ``APCA_API_SECRET_KEY`` once on init
  * lazy-constructs the underlying ``StockHistoricalDataClient``
    or ``CryptoHistoricalDataClient`` depending on the symbol
  * exposes ``get_historical_bars`` that returns a normalized
    ``BarSeries`` (pandas DataFrame with ``open/high/low/close/volume``
    columns and a DatetimeIndex), not an opaque alpaca-py object
  * falls back to a ``FakeMarketDataClient`` for tests and replay
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Protocol

import pandas as pd
from dotenv import load_dotenv

load_dotenv()


_TIMEFRAME_MAP = {
    "1Min": "Minute",
    "5Min": "Minute",
    "15Min": "Minute",
    "1Hour": "Hour",
    "1Day": "Day",
    "1D": "Day",
    "Day": "Day",
}


def normalize_timeframe(tf: str) -> str:
    """Map a free-form timeframe string to an alpaca-py TimeFrame unit."""
    key = tf.replace(" ", "")
    if key not in _TIMEFRAME_MAP:
        raise ValueError(f"Unknown timeframe: {tf!r}. Valid: {sorted(_TIMEFRAME_MAP)}")
    return _TIMEFRAME_MAP[key]


@dataclass(frozen=True)
class BarRequest:
    """Symbol + time range + timeframe for a historical bar fetch."""
    symbol: str
    start: datetime
    end: Optional[datetime] = None
    timeframe: str = "1Day"
    limit: Optional[int] = None


class MarketDataClient(Protocol):
    """Protocol any market-data source must implement.

    Both the real Alpaca client and the in-memory fake used by tests
    satisfy this. The rest of the trading system depends only on the
    protocol, never on a concrete client.
    """

    def get_historical_bars(self, request: BarRequest) -> pd.DataFrame:
        ...

    def get_latest_price(self, symbol: str) -> Optional[float]:
        ...


@dataclass
class AlpacaMarketDataClient:
    """Production implementation backed by alpaca-py.

    Construct lazily -- the SDK only requires the env vars at first
    use, so importing this module without keys set is safe.

    Crypto symbols (e.g. ``BTC/USD``) are routed to
    ``CryptoHistoricalDataClient`` automatically; everything else
    goes to ``StockHistoricalDataClient``.
    """
    api_key: Optional[str] = field(default=None)
    api_secret: Optional[str] = field(default=None)
    _client: object = field(default=None, init=False, repr=False)
    _crypto_client: object = field(default=None, init=False, repr=False)

    def _ensure_client(self):
        if self._client is not None:
            return self._client
        from alpaca.data.historical import StockHistoricalDataClient
        self._client = StockHistoricalDataClient(
            self.api_key or os.getenv("APCA_API_KEY_ID"),
            self.api_secret or os.getenv("APCA_API_SECRET_KEY"),
        )
        return self._client

    def _ensure_crypto_client(self):
        if self._crypto_client is not None:
            return self._crypto_client
        from alpaca.data.historical import CryptoHistoricalDataClient
        self._crypto_client = CryptoHistoricalDataClient(
            self.api_key or os.getenv("APCA_API_KEY_ID"),
            self.api_secret or os.getenv("APCA_API_SECRET_KEY"),
        )
        return self._crypto_client

    @staticmethod
    def _is_crypto_symbol(symbol: str) -> bool:
        from ..utils.asset_class import is_crypto_symbol
        return is_crypto_symbol(symbol)

    def get_historical_bars(self, request: BarRequest) -> pd.DataFrame:
        from alpaca.data.requests import StockBarsRequest, CryptoBarsRequest
        from alpaca.data.timeframe import TimeFrame

        unit = normalize_timeframe(request.timeframe)
        tf = getattr(TimeFrame, unit)
        crypto = self._is_crypto_symbol(request.symbol)
        if crypto:
            client = self._ensure_crypto_client()
            kwargs = {
                "symbol_or_symbols": request.symbol,
                "timeframe": tf,
                "start": request.start,
            }
            if request.end is not None:
                kwargs["end"] = request.end
            if request.limit is not None:
                kwargs["limit"] = request.limit
            raw = client.get_crypto_bars(CryptoBarsRequest(**kwargs))
        else:
            client = self._ensure_client()
            kwargs = {
                "symbol_or_symbols": request.symbol,
                "timeframe": tf,
                "start": request.start,
                "feed": "iex",
            }
            if request.end is not None:
                kwargs["end"] = request.end
            if request.limit is not None:
                kwargs["limit"] = request.limit
            raw = client.get_stock_bars(StockBarsRequest(**kwargs))
        rows = []
        for bar in raw[request.symbol]:
            rows.append({
                "timestamp": bar.timestamp,
                "open": float(bar.open),
                "high": float(bar.high),
                "low": float(bar.low),
                "close": float(bar.close),
                "volume": float(bar.volume),
            })
        df = pd.DataFrame(rows)
        if df.empty:
            return df
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp").sort_index()
        return df

    def get_latest_price(self, symbol: str) -> Optional[float]:
        if self._is_crypto_symbol(symbol):
            from alpaca.data.requests import CryptoLatestTradeRequest
            try:
                latest = self._ensure_crypto_client().get_crypto_latest_trade(
                    CryptoLatestTradeRequest(symbol_or_symbols=symbol)
                )
            except Exception:
                return None
            trade = latest.get(symbol) if isinstance(latest, dict) else None
            if trade is None:
                return None
            return float(trade.price)
        from alpaca.data.requests import StockLatestTradeRequest
        try:
            latest = self._ensure_client().get_stock_latest_trade(
                StockLatestTradeRequest(symbol_or_symbols=symbol)
            )
        except Exception:
            return None
        trade = latest.get(symbol) if isinstance(latest, dict) else None
        if trade is None:
            return None
        return float(trade.price)


@dataclass
class FakeMarketDataClient:
    """In-memory client for tests and replay.

    Stores bars as ``{symbol: pd.DataFrame}`` so the orchestrator can
    be driven by a deterministic historical series without ever
    touching the network.
    """
    series: Dict[str, pd.DataFrame] = field(default_factory=dict)
    latest_prices: Dict[str, float] = field(default_factory=dict)

    def set_series(self, symbol: str, df: pd.DataFrame) -> None:
        if "close" not in df.columns:
            raise ValueError("FakeMarketDataClient requires a 'close' column")
        self.series[symbol] = df.sort_index()
        if not df.empty:
            self.latest_prices[symbol] = float(df["close"].iloc[-1])

    def get_historical_bars(self, request: BarRequest) -> pd.DataFrame:
        df = self.series.get(request.symbol)
        if df is None:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        mask = df.index >= pd.Timestamp(request.start)
        if request.end is not None:
            mask &= df.index <= pd.Timestamp(request.end)
        sliced = df.loc[mask]
        if request.limit is not None and len(sliced) > request.limit:
            sliced = sliced.iloc[-request.limit:]
        return sliced.copy()

    def get_latest_price(self, symbol: str) -> Optional[float]:
        return self.latest_prices.get(symbol)


def get_default_client() -> MarketDataClient:
    """Factory: returns Alpaca client if keys are present, else a fake with no data."""
    if os.getenv("APCA_API_KEY_ID") and os.getenv("APCA_API_SECRET_KEY"):
        return AlpacaMarketDataClient()
    return FakeMarketDataClient()


__all__ = [
    "AlpacaMarketDataClient",
    "BarRequest",
    "FakeMarketDataClient",
    "MarketDataClient",
    "get_default_client",
    "normalize_timeframe",
]
