"""Tests for the asset-class detection helpers in ``utils/asset_class.py``."""
from __future__ import annotations

import unittest

from investment_agent.utils.asset_class import (
    classify_symbol,
    is_crypto_symbol,
    is_equity_symbol,
    is_option_symbol,
)


class TestIsOptionSymbol(unittest.TestCase):
    def test_standard_occ_symbols(self):
        self.assertTrue(is_option_symbol("AAPL240119C00200000"))
        self.assertTrue(is_option_symbol("SPY240315P00450000"))
        self.assertTrue(is_option_symbol("TSLA260620C00150000"))

    def test_rejects_equity_tickers(self):
        self.assertFalse(is_option_symbol("AAPL"))
        self.assertFalse(is_option_symbol("SPY"))
        self.assertFalse(is_option_symbol("BRK.B"))
        self.assertFalse(is_option_symbol(""))

    def test_rejects_crypto_pairs(self):
        self.assertFalse(is_option_symbol("BTC/USD"))
        self.assertFalse(is_option_symbol("ETH/USDT"))


class TestIsCryptoSymbol(unittest.TestCase):
    def test_primary_universe_recognised(self):
        universe = [
            "BTC/USD", "ETH/USD", "SOL/USD", "AVAX/USD",
            "LINK/USD", "XRP/USD", "DOGE/USD", "RENDER/USD",
        ]
        for sym in universe:
            with self.subTest(symbol=sym):
                self.assertTrue(is_crypto_symbol(sym), f"{sym} should be crypto")

    def test_usdt_quote_accepted(self):
        self.assertTrue(is_crypto_symbol("BTC/USDT"))

    def test_rejects_equity_tickers(self):
        self.assertFalse(is_crypto_symbol("AAPL"))
        self.assertFalse(is_crypto_symbol("SPY"))
        self.assertFalse(is_crypto_symbol("NVDA"))

    def test_rejects_option_symbols(self):
        self.assertFalse(is_crypto_symbol("AAPL240119C00200000"))

    def test_rejects_invalid_formats(self):
        self.assertFalse(is_crypto_symbol("BTCUSD"))       # no slash
        self.assertFalse(is_crypto_symbol("BTC/"))         # empty quote
        self.assertFalse(is_crypto_symbol("/USD"))         # empty base
        self.assertFalse(is_crypto_symbol(""))             # empty
        self.assertFalse(is_crypto_symbol("BTC/USDX"))     # invalid quote


class TestIsEquitySymbol(unittest.TestCase):
    def test_standard_tickers(self):
        for sym in ("AAPL", "SPY", "MSFT", "TSLA", "NVDA", "BRK.B"):
            with self.subTest(symbol=sym):
                self.assertTrue(is_equity_symbol(sym))

    def test_rejects_crypto(self):
        self.assertFalse(is_equity_symbol("BTC/USD"))
        self.assertFalse(is_equity_symbol("ETH/USD"))

    def test_rejects_options(self):
        self.assertFalse(is_equity_symbol("AAPL240119C00200000"))


class TestClassifySymbol(unittest.TestCase):
    def test_equity(self):
        self.assertEqual(classify_symbol("AAPL"), "equity")
        self.assertEqual(classify_symbol("SPY"), "equity")

    def test_crypto(self):
        self.assertEqual(classify_symbol("BTC/USD"), "crypto")
        self.assertEqual(classify_symbol("ETH/USD"), "crypto")
        self.assertEqual(classify_symbol("SOL/USD"), "crypto")

    def test_option(self):
        self.assertEqual(classify_symbol("AAPL240119C00200000"), "option")


if __name__ == "__main__":
    unittest.main()
