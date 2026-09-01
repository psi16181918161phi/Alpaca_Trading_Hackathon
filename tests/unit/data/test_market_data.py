"""Tests for the unified market-data interface."""
from __future__ import annotations

import unittest
from datetime import datetime, timedelta

import pandas as pd

from investment_agent.data.market_data import (
    AlpacaMarketDataClient,
    BarRequest,
    FakeMarketDataClient,
    get_default_client,
    normalize_timeframe,
)


def _make_series(n: int = 30, start_price: float = 100.0) -> pd.DataFrame:
    idx = pd.date_range("2024-01-02", periods=n, freq="D")
    closes = [start_price + i * 0.5 for i in range(n)]
    return pd.DataFrame({
        "open": [c - 0.2 for c in closes],
        "high": [c + 0.4 for c in closes],
        "low": [c - 0.4 for c in closes],
        "close": closes,
        "volume": [1_000_000] * n,
    }, index=idx)


class TestNormalizeTimeframe(unittest.TestCase):
    def test_day_variants(self):
        self.assertEqual(normalize_timeframe("1Day"), "Day")
        self.assertEqual(normalize_timeframe("Day"), "Day")
        self.assertEqual(normalize_timeframe("1D"), "Day")

    def test_minute_variants(self):
        self.assertEqual(normalize_timeframe("1Min"), "Minute")
        self.assertEqual(normalize_timeframe("5Min"), "Minute")
        self.assertEqual(normalize_timeframe("15Min"), "Minute")

    def test_hour(self):
        self.assertEqual(normalize_timeframe("1Hour"), "Hour")

    def test_unknown_raises(self):
        with self.assertRaises(ValueError):
            normalize_timeframe("fortnight")


class TestFakeMarketDataClient(unittest.TestCase):
    def setUp(self):
        self.fake = FakeMarketDataClient()
        self.fake.set_series("AAPL", _make_series(30, 100.0))

    def test_get_historical_bars_returns_dataframe(self):
        req = BarRequest(
            symbol="AAPL",
            start=datetime(2024, 1, 2),
            end=datetime(2024, 1, 31),
            timeframe="1Day",
        )
        df = self.fake.get_historical_bars(req)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 30)
        self.assertIn("close", df.columns)
        self.assertIn("volume", df.columns)
        # Index is the timestamp
        self.assertIsInstance(df.index, pd.DatetimeIndex)

    def test_get_historical_bars_filters_by_start(self):
        req = BarRequest(symbol="AAPL", start=datetime(2024, 1, 20))
        df = self.fake.get_historical_bars(req)
        self.assertEqual(len(df), 12)  # Jan 20..31 inclusive

    def test_get_historical_bars_limit(self):
        req = BarRequest(symbol="AAPL", start=datetime(2024, 1, 2), limit=5)
        df = self.fake.get_historical_bars(req)
        self.assertEqual(len(df), 5)
        # Should be the last 5 rows of the series
        self.assertEqual(df["close"].iloc[-1], 100.0 + 29 * 0.5)

    def test_get_historical_bars_missing_symbol(self):
        req = BarRequest(symbol="MSFT", start=datetime(2024, 1, 2))
        df = self.fake.get_historical_bars(req)
        self.assertTrue(df.empty)
        self.assertIn("close", df.columns)

    def test_get_latest_price(self):
        self.assertEqual(self.fake.get_latest_price("AAPL"), 100.0 + 29 * 0.5)
        self.assertIsNone(self.fake.get_latest_price("MSFT"))

    def test_set_series_validates_close_column(self):
        with self.assertRaises(ValueError):
            self.fake.set_series("X", pd.DataFrame({"oops": [1, 2, 3]}))

    def test_set_series_updates_latest_price(self):
        new_df = _make_series(5, 50.0)
        new_df.index = pd.date_range("2024-06-01", periods=5, freq="D")
        self.fake.set_series("AAPL", new_df)
        self.assertEqual(self.fake.get_latest_price("AAPL"), 50.0 + 4 * 0.5)


class TestGetDefaultClient:
    def test_returns_fake_when_no_keys(self, monkeypatch):
        monkeypatch.delenv("APCA_API_KEY_ID", raising=False)
        monkeypatch.delenv("APCA_API_SECRET_KEY", raising=False)
        client = get_default_client()
        assert isinstance(client, FakeMarketDataClient)

    def test_returns_alpaca_when_keys_present(self, monkeypatch):
        monkeypatch.setenv("APCA_API_KEY_ID", "test_key")
        monkeypatch.setenv("APCA_API_SECRET_KEY", "test_secret")
        client = get_default_client()
        assert isinstance(client, AlpacaMarketDataClient)


class TestAlpacaClientLazy(unittest.TestCase):
    def test_client_construction_does_not_call_alpaca(self):
        # Constructing the client must not blow up without network
        client = AlpacaMarketDataClient(api_key="x", api_secret="y")
        # Lazy: no real client is built until the first request
        self.assertIsNone(client._client)


if __name__ == "__main__":
    unittest.main()
