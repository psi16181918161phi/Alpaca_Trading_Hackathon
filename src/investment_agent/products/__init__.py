"""Product gate: pick equity vs option vs crypto vs no-trade."""
from .product_gate import (
    OPTION_CALL,
    OPTION_PUT,
    PRODUCT_CRYPTO,
    PRODUCT_EQUITY,
    PRODUCT_NONE,
    PRODUCT_OPTION,
    ProductGate,
    ProductGateInput,
    ProductGateResult,
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
