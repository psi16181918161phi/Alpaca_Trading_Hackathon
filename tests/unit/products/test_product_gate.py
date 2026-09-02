"""Tests for the product gate (equity / option / no-trade)."""
from __future__ import annotations

import unittest

from investment_agent.products import (
    OPTION_CALL,
    OPTION_PUT,
    PRODUCT_CRYPTO,
    PRODUCT_EQUITY,
    PRODUCT_NONE,
    PRODUCT_OPTION,
    ProductGate,
    ProductGateInput,
)


class TestProductGateHoldAndBlocked(unittest.TestCase):
    def test_hold_yields_none(self):
        gate = ProductGate()
        result = gate.decide(ProductGateInput(
            action="HOLD", verdict="ALLOW", ensemble_signal=0.6,
            disagreement=0.1, confidence=0.9, regime="R01",
        ))
        self.assertEqual(result.product, PRODUCT_NONE)
        self.assertIn("HOLD", result.reason)

    def test_block_yields_none(self):
        gate = ProductGate()
        result = gate.decide(ProductGateInput(
            action="BUY", verdict="BLOCK", ensemble_signal=0.6,
            disagreement=0.1, confidence=0.9, regime="R01",
        ))
        self.assertEqual(result.product, PRODUCT_NONE)
        self.assertIn("BLOCK", result.reason)

    def test_flatten_yields_none(self):
        gate = ProductGate()
        result = gate.decide(ProductGateInput(
            action="SELL", verdict="FLATTEN", ensemble_signal=-0.6,
            disagreement=0.1, confidence=0.9, regime="R01",
        ))
        self.assertEqual(result.product, PRODUCT_NONE)
        self.assertIn("FLATTEN", result.reason)


class TestProductGateOption(unittest.TestCase):
    def test_strong_bullish_buys_call(self):
        gate = ProductGate()
        result = gate.decide(ProductGateInput(
            action="BUY", verdict="ALLOW", ensemble_signal=0.7,
            disagreement=0.1, confidence=0.85, regime="R01",
        ))
        self.assertEqual(result.product, PRODUCT_OPTION)
        self.assertEqual(result.option_side, OPTION_CALL)
        self.assertEqual(result.option_strike_offset, 0)

    def test_strong_bearish_buys_put(self):
        gate = ProductGate()
        result = gate.decide(ProductGateInput(
            action="SELL", verdict="ALLOW", ensemble_signal=-0.7,
            disagreement=0.1, confidence=0.85, regime="R01",
        ))
        self.assertEqual(result.product, PRODUCT_OPTION)
        self.assertEqual(result.option_side, OPTION_PUT)

    def test_modest_signal_falls_back_to_equity(self):
        gate = ProductGate()
        result = gate.decide(ProductGateInput(
            action="BUY", verdict="ALLOW", ensemble_signal=0.2,
            disagreement=0.1, confidence=0.9, regime="R01",
        ))
        self.assertEqual(result.product, PRODUCT_EQUITY)

    def test_low_confidence_falls_back_to_equity(self):
        gate = ProductGate()
        # Strong signal but confidence below threshold.
        result = gate.decide(ProductGateInput(
            action="BUY", verdict="ALLOW", ensemble_signal=0.6,
            disagreement=0.1, confidence=0.5, regime="R01",
        ))
        self.assertEqual(result.product, PRODUCT_EQUITY)


class TestProductGateDisagreement(unittest.TestCase):
    def test_wide_disagreement_falls_back_to_equity(self):
        gate = ProductGate()
        # Even with strong + confident signal, wide LLM spread -> equity.
        result = gate.decide(ProductGateInput(
            action="BUY", verdict="ALLOW", ensemble_signal=0.7,
            disagreement=0.5, confidence=0.9, regime="R01",
        ))
        self.assertEqual(result.product, PRODUCT_EQUITY)
        self.assertIn("disagreement", result.reason)

    def test_below_max_disagreement_still_options(self):
        gate = ProductGate()
        result = gate.decide(ProductGateInput(
            action="BUY", verdict="ALLOW", ensemble_signal=0.7,
            disagreement=0.3, confidence=0.9, regime="R01",
        ))
        self.assertEqual(result.product, PRODUCT_OPTION)


class TestProductGateResultToDict(unittest.TestCase):
    def test_to_dict_round_trip(self):
        gate = ProductGate()
        result = gate.decide(ProductGateInput(
            action="BUY", verdict="ALLOW", ensemble_signal=0.7,
            disagreement=0.1, confidence=0.85, regime="R01",
        ))
        d = result.to_dict()
        self.assertEqual(d["product"], "option")
        self.assertEqual(d["option_side"], "call")
        self.assertEqual(d["option_strike_offset"], 0)
        self.assertTrue(d["reason"])

    def test_none_result_serializes(self):
        gate = ProductGate()
        result = gate.decide(ProductGateInput(
            action="HOLD", verdict="ALLOW", ensemble_signal=0.0,
            disagreement=0.0, confidence=0.5, regime="R01",
        ))
        d = result.to_dict()
        self.assertEqual(d["product"], "none")
        self.assertIsNone(d["option_side"])


class TestProductGateConfigurableThresholds(unittest.TestCase):
    def test_custom_min_signal(self):
        gate = ProductGate(min_signal_for_option=0.8, high_confidence_threshold=0.7)
        # Signal of 0.7 < custom 0.8 -> equity, not option
        result = gate.decide(ProductGateInput(
            action="BUY", verdict="ALLOW", ensemble_signal=0.7,
            disagreement=0.1, confidence=0.9, regime="R01",
        ))
        self.assertEqual(result.product, PRODUCT_EQUITY)

    def test_custom_disagreement(self):
        gate = ProductGate(max_disagreement_for_option=0.05)
        # Default would have allowed; custom disagrees.
        result = gate.decide(ProductGateInput(
            action="BUY", verdict="ALLOW", ensemble_signal=0.7,
            disagreement=0.1, confidence=0.9, regime="R01",
        ))
        self.assertEqual(result.product, PRODUCT_EQUITY)


if __name__ == "__main__":
    unittest.main()
