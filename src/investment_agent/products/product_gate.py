"""Product gate: decide between equity, option, crypto, and no-trade.

WHAT
====
Sits between the capital gate verdict and the execution layer.
Given the post-gate decision (BUY / SELL / HOLD) and the regime /
signal strength / capital state, it picks a vehicle:

  * **equity**  -- plain stock order
  * **option**  -- call (bullish) or put (bearish)
  * **crypto**  -- spot crypto pair (e.g. BTC/USD)
  * **none**    -- do not trade

The product gate is *not* a risk override; it is a vehicle
selector. The capital gate's verdict is still the hard authority.
A BLOCK / FLATTEN verdict short-circuits to ``none`` before this
gate runs.

WHY
====
The hackathon spec requires options. The product gate turns "the
agents think we should buy" into a concrete order: a stock, a call,
or a put. Strong bullish signal + low-disagreement regime
preference for calls; bearish for puts; modest or wide
disagreement falls back to equity for cheap exposure; very high
disagreement (LLMs disagree) goes to ``none`` so the deterministic
layer can sit on its hands.

HOW
====
``ProductGate`` is a pure function over a small dataclass
``ProductGateInput``. The orchestrator's ``_make_decision`` should
call it after the capital gate and add ``product``, ``product_reason``,
and (if option) ``option_side`` / ``option_strike_offset`` to the
decision payload.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from ..utils.asset_class import is_crypto_symbol as _is_crypto_symbol


PRODUCT_EQUITY = "equity"
PRODUCT_OPTION = "option"
PRODUCT_CRYPTO = "crypto"
PRODUCT_NONE = "none"

OPTION_CALL = "call"
OPTION_PUT = "put"


@dataclass(frozen=True)
class ProductGateInput:
    """Inputs to the product gate.

    All fields come from the orchestrator's existing pipeline; this
    is a pure function over them, not a new decision authority.
    """
    action: str                          # BUY / SELL / HOLD
    verdict: str                         # ALLOW / REDUCE / BLOCK / FLATTEN
    ensemble_signal: float               # [-1, +1]
    disagreement: float                  # [0, 1]
    confidence: float                    # [0, 1]
    regime: str                          # e.g. R01..R12
    symbol: Optional[str] = None         # ticker / crypto pair / OCC symbol
    # Optional knobs. The defaults are the spec's "sensible starting
    # point" values; they can be tuned per regime later.
    min_signal_for_option: float = 0.45
    max_disagreement_for_option: float = 0.35
    high_confidence_threshold: float = 0.70


@dataclass(frozen=True)
class ProductGateResult:
    """Vehicle choice for the next order.

    ``product`` is one of ``equity`` / ``option`` / ``crypto`` / ``none``.
    For ``product == "option"``, ``option_side`` is set to ``call``
    or ``put``; ``option_strike_offset`` is a small integer
    (0 = at-the-money, 1 = slightly OTM, ...) so the execution
    layer can pick a real OCC contract.
    """
    product: str
    option_side: Optional[str] = None
    option_strike_offset: int = 0
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "product": self.product,
            "option_side": self.option_side,
            "option_strike_offset": self.option_strike_offset,
            "reason": self.reason,
        }


class ProductGate:
    """Decide equity / option / crypto / no-trade from a single decision."""

    def __init__(
        self,
        min_signal_for_option: float = 0.45,
        max_disagreement_for_option: float = 0.35,
        high_confidence_threshold: float = 0.70,
    ) -> None:
        self._min_signal = float(min_signal_for_option)
        self._max_disagreement = float(max_disagreement_for_option)
        self._high_confidence = float(high_confidence_threshold)

    def decide(self, inp: ProductGateInput) -> ProductGateResult:
        # Hard overrides first.
        if inp.action == "HOLD" or inp.verdict in {"BLOCK", "FLATTEN"}:
            return ProductGateResult(
                product=PRODUCT_NONE,
                reason=f"no-trade: action={inp.action} verdict={inp.verdict}",
            )

        signal = float(inp.ensemble_signal)
        disagreement = float(inp.disagreement)
        confidence = float(inp.confidence)
        is_crypto = bool(inp.symbol) and _is_crypto_symbol(str(inp.symbol))

        # Wide disagreement -> fall back to the cheapest liquid vehicle.
        if disagreement > self._max_disagreement:
            if is_crypto:
                return ProductGateResult(
                    product=PRODUCT_CRYPTO,
                    reason=(
                        f"crypto: disagreement {disagreement:.2f} > "
                        f"{self._max_disagreement:.2f}; liquid crypto vehicle"
                    ),
                )
            return ProductGateResult(
                product=PRODUCT_EQUITY,
                reason=(
                    f"equity: disagreement {disagreement:.2f} > "
                    f"{self._max_disagreement:.2f}; LLM spread too wide for options"
                ),
            )

        abs_sig = abs(signal)
        if abs_sig >= self._min_signal and confidence >= self._high_confidence:
            # Strong, confident, agreeing signal -> option for leverage.
            if signal > 0 and inp.action == "BUY":
                return ProductGateResult(
                    product=PRODUCT_OPTION,
                    option_side=OPTION_CALL,
                    option_strike_offset=0,
                    reason=(
                        f"call: signal {signal:+.2f} >= {self._min_signal:.2f}, "
                        f"confidence {confidence:.2f} >= {self._high_confidence:.2f}"
                    ),
                )
            if signal < 0 and inp.action == "SELL":
                return ProductGateResult(
                    product=PRODUCT_OPTION,
                    option_side=OPTION_PUT,
                    option_strike_offset=0,
                    reason=(
                        f"put: signal {signal:+.2f} <= -{self._min_signal:.2f}, "
                        f"confidence {confidence:.2f} >= {self._high_confidence:.2f}"
                    ),
                )

        # Default: crypto when the symbol is a crypto pair, else equity.
        if is_crypto:
            return ProductGateResult(
                product=PRODUCT_CRYPTO,
                reason=(
                    f"crypto: |signal|={abs_sig:.2f} < {self._min_signal:.2f} or "
                    f"confidence {confidence:.2f} < {self._high_confidence:.2f}"
                ),
            )
        return ProductGateResult(
            product=PRODUCT_EQUITY,
            reason=(
                f"equity: |signal|={abs_sig:.2f} < {self._min_signal:.2f} or "
                f"confidence {confidence:.2f} < {self._high_confidence:.2f}"
            ),
        )


__all__ = [
    "OPTION_CALL",
    "OPTION_PUT",
    "PRODUCT_CRYPTO",
    "PRODUCT_EQUITY",
    "PRODUCT_NONE",
    "PRODUCT_OPTION",
    "ProductGate",
    "ProductGateInput",
    "ProductGateResult",
]
