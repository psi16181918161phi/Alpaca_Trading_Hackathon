"""Asset-class detection helpers for the X Quant X trading system.

WHAT
====
Single source of truth for classifying symbols into the three
supported asset classes:

  * ``equity``  -- plain stock / ETF tickers (e.g. ``AAPL``, ``SPY``)
  * ``option``  -- OCC-formatted option contracts (e.g. ``AAPL230918C00205000``)
  * ``crypto``  -- Alpaca crypto pairs in slash form (e.g. ``BTC/USD``)

WHY
====
Previously OCC detection was duplicated in two places
(``execution.py`` and ``dashboard/data_loader.py``) and equity
was the implicit default. A single helper eliminates that
duplication and gives the rest of the system a clean way to
distinguish asset classes without scattering string checks.

HOW
====
``classify_symbol(symbol)`` returns the canonical lowercase
asset-class string. ``is_crypto_symbol`` / ``is_equity_symbol`` /
``is_option_symbol`` are convenience predicates.

Crypto detection is intentionally narrow: it requires a slash
followed by a 3-4 letter uppercase quote currency (USD, USDT).
This avoids false-positives on unrelated symbols that happen to
contain a slash.
"""
from __future__ import annotations

from typing import Literal


_ASSET_CLASS = Literal["equity", "option", "crypto"]


def is_option_symbol(symbol: str) -> bool:
    """Return True if *symbol* matches the OCC option-contract format.

    OCC symbols follow the pattern
    ``<underlying><YYMMDD><C|P><strike*1000 padded to 8 digits>``.
    The trailing 15 characters are always: 6 (date) + 1 (C/P) + 8 (strike).
    The underlying is 1-6 characters before that.
    """
    if not symbol or not isinstance(symbol, str):
        return False
    s = symbol.strip()
    if len(s) < 16:
        return False
    if "." in s:
        return False
    # The trailing 15 chars are always: 6 (date) + 1 (C/P) + 8 (strike)
    date_part = s[-15:-9]
    cp = s[-9]
    strike_part = s[-8:]
    if not date_part.isdigit():
        return False
    if cp not in ("C", "P"):
        return False
    return strike_part.isdigit()


def is_crypto_symbol(symbol: str) -> bool:
    """Return True if *symbol* is an Alpaca crypto pair.

    Alpaca crypto pairs use the form ``<BASE>/<QUOTE>`` where the
    base is 2-12 uppercase letters and the quote is a recognised
    currency suffix (currently ``USD`` / ``USDT``).

    Examples::

        BTC/USD   ETH/USD   SOL/USD   AVAX/USD
        LINK/USD  XRP/USD   DOGE/USD  RENDER/USD
    """
    if not symbol or not isinstance(symbol, str):
        return False
    s = symbol.strip()
    if "/" not in s:
        return False
    base, quote = s.rsplit("/", 1)
    if not base or not quote:
        return False
    if not (2 <= len(base) <= 12):
        return False
    if not base.isalpha() or not base.isupper():
        return False
    if quote not in {"USD", "USDT"}:
        return False
    return True


def is_equity_symbol(symbol: str) -> bool:
    """Return True if *symbol* is a plain equity / ETF ticker.

    A symbol is treated as equity when it is *not* an option and
    *not* a recognised crypto pair. Standard US equity tickers are
    1-5 uppercase letters (possibly with a dot for BRK.B-style
    tickers), but we intentionally keep the check permissive so we
    do not accidentally classify a valid non-US ticker as crypto.
    """
    if not symbol or not isinstance(symbol, str):
        return False
    if is_option_symbol(symbol):
        return False
    if is_crypto_symbol(symbol):
        return False
    return True


def classify_symbol(symbol: str) -> _ASSET_CLASS:
    """Classify *symbol* into ``"equity"``, ``"option"``, or ``"crypto"``.

    The check order matters: option symbols can look like equities
    at a glance because they embed an underlying ticker, so options
    are tested first. Crypto uses a slash separator so it cannot
    be confused with either.
    """
    if is_option_symbol(symbol):
        return "option"
    if is_crypto_symbol(symbol):
        return "crypto"
    return "equity"


__all__ = [
    "classify_symbol",
    "is_crypto_symbol",
    "is_equity_symbol",
    "is_option_symbol",
]
