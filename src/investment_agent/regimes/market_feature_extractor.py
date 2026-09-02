"""Market Feature Extractor — Converts price/volume history to HMM feature vectors.

WHAT
====
Extracts the 7 features required by the HMM regime detector from raw price and
volume history:

    [RSI, MACD, ATR(s), VIX, VolRatio, Corr, Hurst]

WHY
===
The HMM-based regime detector (hmm_regime_detector.py) requires a feature matrix
with exactly these 7 features per observation. This module provides the
translation from raw market data to that feature space.

HOW
===
- RSI: Relative Strength Index (14-period default)
- MACD: Moving Average Convergence Divergence (12, 26, 9 default)
- ATR: Average True Range (normalized by price)
- VIX: Volatility Index proxy (annualized volatility)
- VolRatio: Volume ratio (recent/long-term average)
- Corr: Correlation between price and volume changes
- Hurst: Hurst exponent approximation (rolling window)

All features are computed over a rolling window and returned as a T x 7 matrix
suitable for HMM inference.

Note: Feature scaling is critical. The HMM assumes Gaussian emissions, so
features should be standardized (zero mean, unit variance) before inference.
The configuration in config/regimes.toml provides emission means calibrated
to a specific feature scale.

Architectural Role
==================
Feature engineering layer. Consumes raw market data and produces HMM-compatible
feature vectors consumed by hmm_regime_detector.py.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Default lookback window for feature computation
_DEFAULT_LOOKBACK_DAYS: int = 20

# RSI period
_RSI_PERIOD: int = 14

# MACD parameters
_MACD_FAST: int = 12
_MACD_SLOW: int = 26
_MACD_SIGNAL: int = 9

# Minimum observations required for feature computation
_MIN_OBSERVATIONS: int = 30


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MarketFeatures:
    """Extracted market features for HMM regime classification.

    Attributes
    ----------
    rsi : float
        Relative Strength Index (0-100).
    macd : float
        MACD line value.
    atr : float
        Average True Range (normalized by price).
    vix : float
        Volatility Index proxy (annualized volatility %).
    vol_ratio : float
        Volume ratio (recent/long-term average).
    corr : float
        Correlation between price and volume changes.
    hurst : float
        Hurst exponent approximation (0.5 = random walk).
    """

    rsi: float
    macd: float
    atr: float
    vix: float
    vol_ratio: float
    corr: float
    hurst: float


# ---------------------------------------------------------------------------
# Feature extraction functions
# ---------------------------------------------------------------------------

def _compute_rsi(prices: np.ndarray, period: int = _RSI_PERIOD) -> float:
    """Compute Relative Strength Index."""
    if len(prices) < period + 1:
        return 50.0  # Neutral RSI when insufficient data
    
    deltas = np.diff(prices[-(period + 1):])
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    
    avg_gain = np.mean(gains)
    avg_loss = np.mean(losses)
    
    if avg_loss == 0:
        return 100.0
    
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def _compute_macd(prices: np.ndarray, fast: int = _MACD_FAST, 
                   slow: int = _MACD_SLOW, signal: int = _MACD_SIGNAL) -> float:
    """Compute MACD line value."""
    if len(prices) < slow + signal:
        return 0.0
    
    def ema(data: np.ndarray, period: int) -> np.ndarray:
        """Compute Exponential Moving Average."""
        result = np.zeros_like(data)
        result[0] = data[0]
        alpha = 2.0 / (period + 1)
        for i in range(1, len(data)):
            result[i] = alpha * data[i] + (1 - alpha) * result[i - 1]
        return result
    
    ema_fast = ema(prices, fast)
    ema_slow = ema(prices, slow)
    macd_line = ema_fast - ema_slow
    
    return macd_line[-1]


def _compute_atr(
    prices: np.ndarray,
    period: int = 14,
    highs: Optional[np.ndarray] = None,
    lows: Optional[np.ndarray] = None,
) -> float:
    """Compute Average True Range (normalized by price).

    If highs and lows arrays are provided, computes exact True Range using
    high, low, and previous close. Otherwise uses price-change range proxy.
    """
    if len(prices) < period + 1:
        return 0.0

    prev_closes = prices[:-1]

    if highs is not None and lows is not None and len(highs) == len(prices) and len(lows) == len(prices):
        h = highs[1:]
        l = lows[1:]
    else:
        h = np.maximum(prices[1:], prices[:-1])
        l = np.minimum(prices[1:], prices[:-1])

    tr1 = h - l
    tr2 = np.abs(h - prev_closes)
    tr3 = np.abs(l - prev_closes)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))

    atr = np.mean(tr[-period:])
    return float(atr / prices[-1]) if prices[-1] > 0 else 0.0


def _compute_vix(prices: np.ndarray, period: int = 20) -> float:
    """Compute VIX proxy (annualized volatility percentage)."""
    if len(prices) < 2:
        return 0.0
    
    returns = np.diff(np.log(prices))
    volatility = np.std(returns[-period:])
    
    # Annualize (assuming daily data)
    annualized_vol = volatility * math.sqrt(252.0)
    return annualized_vol * 100.0  # Convert to percentage


def _compute_volume_ratio(volumes: np.ndarray, short_window: int = 5, 
                          long_window: int = 20) -> float:
    """Compute volume ratio (recent average / long-term average)."""
    if len(volumes) < long_window:
        return 1.0
    
    recent_avg = np.mean(volumes[-short_window:])
    long_avg = np.mean(volumes[-long_window:])
    
    return recent_avg / long_avg if long_avg > 0 else 1.0


def _compute_correlation(prices: np.ndarray, volumes: np.ndarray, 
                         period: int = 20) -> float:
    """Compute correlation between price and volume changes."""
    if len(prices) < period + 1 or len(volumes) < period + 1:
        return 0.0
    
    price_changes = np.diff(prices[-(period + 1):])
    volume_changes = np.diff(volumes[-(period + 1):])
    
    if np.std(price_changes) == 0 or np.std(volume_changes) == 0:
        return 0.0
    
    corr = np.corrcoef(price_changes, volume_changes)[0, 1]
    return float(corr) if not math.isnan(corr) else 0.0


def _compute_hurst(prices: np.ndarray, period: int = 20) -> float:
    """Compute Hurst exponent approximation."""
    if len(prices) < period:
        return 0.5
    
    # Simplified Hurst using R/S analysis
    returns = np.diff(np.log(prices[-period:]))
    mean_return = np.mean(returns)
    deviations = returns - mean_return
    cumulative = np.cumsum(deviations)
    
    R = np.max(cumulative) - np.min(cumulative)
    S = np.std(returns)
    
    if S == 0:
        return 0.5
    
    log_RS = math.log(R / S)
    log_n = math.log(len(returns))
    
    return log_RS / log_n if log_n > 0 else 0.5


# ---------------------------------------------------------------------------
# Main extraction function
# ---------------------------------------------------------------------------

def extract_features(
    prices: List[float],
    volumes: Optional[List[float]] = None,
    lookback_days: int = _DEFAULT_LOOKBACK_DAYS,
    highs: Optional[List[float]] = None,
    lows: Optional[List[float]] = None,
) -> np.ndarray:
    """Extract HMM features from OHLCV history.

    Parameters
    ----------
    prices : List[float]
        Historical close prices (oldest first). At least _MIN_OBSERVATIONS required.
    volumes : Optional[List[float]]
        Historical volume series (oldest first).
    lookback_days : int
        Lookback window for feature computation (default 20).
    highs : Optional[List[float]]
        Historical high prices. When provided alongside `lows`, enables genuine
        True Range ATR calculation instead of the close-range proxy.
    lows : Optional[List[float]]
        Historical low prices.

    Returns
    -------
    np.ndarray
        T x 7 feature matrix: [RSI, MACD, ATR, VIX, VolRatio, Corr, Hurst]

    Raises
    ------
    ValueError
        If inputs contain invalid values or insufficient data.
    """
    if not prices:
        raise ValueError("Price series must be non-empty")
    if len(prices) < 5:
        raise ValueError("Price series must contain at least 5 observations")

    # If series is shorter than _MIN_OBSERVATIONS (30), pad by prepending earliest value
    if len(prices) < _MIN_OBSERVATIONS:
        pad_len = _MIN_OBSERVATIONS - len(prices)
        prices = [prices[0]] * pad_len + list(prices)
        if volumes is not None:
            volumes = [volumes[0]] * pad_len + list(volumes)
        if highs is not None:
            highs = [highs[0]] * pad_len + list(highs)
        if lows is not None:
            lows = [lows[0]] * pad_len + list(lows)

    prices_arr = np.array(prices, dtype=np.float64)
    if np.any(np.isnan(prices_arr)) or np.any(np.isinf(prices_arr)):
        raise ValueError("Price series contains NaN or Infinity values")
    if np.any(prices_arr <= 0):
        raise ValueError("Price series must contain positive values")

    volumes_arr = None
    if volumes is not None:
        volumes_arr = np.array(volumes, dtype=np.float64)
        if np.any(np.isnan(volumes_arr)) or np.any(np.isinf(volumes_arr)):
            raise ValueError("Volume series contains NaN or Infinity values")
        if np.any(volumes_arr < 0):
            raise ValueError("Volume series must contain non-negative values")

    # Optional OHLC arrays for genuine True Range
    highs_arr = np.array(highs, dtype=np.float64) if highs is not None else None
    lows_arr = np.array(lows, dtype=np.float64) if lows is not None else None

    T = min(lookback_days, len(prices) - _MIN_OBSERVATIONS + 1)
    if T < 1:
        raise ValueError(f"Insufficient data for feature extraction with lookback={lookback_days}")

    features = np.zeros((T, 7))

    for t in range(T):
        start_idx = len(prices) - _MIN_OBSERVATIONS - T + t + 1
        end_idx = start_idx + _MIN_OBSERVATIONS

        window_prices = prices_arr[start_idx:end_idx]

        # Genuine ATR when OHLC data is available
        window_highs = highs_arr[start_idx:end_idx] if highs_arr is not None and len(highs_arr) >= end_idx else None
        window_lows = lows_arr[start_idx:end_idx] if lows_arr is not None and len(lows_arr) >= end_idx else None

        rsi = _compute_rsi(window_prices)
        macd = _compute_macd(window_prices)
        atr = _compute_atr(window_prices, highs=window_highs, lows=window_lows)
        vix = _compute_vix(window_prices)

        if volumes_arr is not None and len(volumes_arr) >= end_idx:
            window_volumes = volumes_arr[start_idx:end_idx]
            vol_ratio = _compute_volume_ratio(window_volumes)
            corr = _compute_correlation(window_prices, window_volumes)
        else:
            vol_ratio = 1.0
            corr = 0.0

        hurst = _compute_hurst(window_prices)

        features[t] = [rsi, macd, atr, vix, vol_ratio, corr, hurst]

    return features


def extract_single_feature_vector(prices: List[float], 
                                   volumes: Optional[List[float]] = None) -> np.ndarray:
    """Extract a single feature vector from the most recent market data.

    Parameters
    ----------
    prices : List[float]
        Historical price series (oldest first).
    volumes : Optional[List[float]]
        Historical volume series (oldest first).

    Returns
    -------
    np.ndarray
        1 x 7 feature vector: [RSI, MACD, ATR, VIX, VolRatio, Corr, Hurst]

    Raises
    ------
    ValueError
        If inputs are invalid or insufficient.
    """
    features = extract_features(prices, volumes, lookback_days=1)
    return features[-1:, :]


def compute_dict_features(
    prices: List[float],
    volumes: Optional[List[float]] = None,
    highs: Optional[List[float]] = None,
    lows: Optional[List[float]] = None,
) -> Dict[str, float]:
    """Compute real market feature dictionary (RSI, ATR, VIX, etc.) from price/volume series.

    Returns a dict with 'rsi', 'atr', 'vix', 'macd', 'vol_ratio', 'corr', 'hurst'.
    Normalizes RSI to [0.0, 1.0] for model consumption.
    """
    if not prices or len(prices) < 2:
        return {"atr": 0.0, "rsi": 0.5, "vix": 0.15, "macd": 0.0, "vol_ratio": 1.0, "corr": 0.0, "hurst": 0.5}

    prices_arr = np.array(prices, dtype=np.float64)
    highs_arr = np.array(highs, dtype=np.float64) if highs is not None else None
    lows_arr = np.array(lows, dtype=np.float64) if lows is not None else None

    vix = _compute_vix(prices_arr) / 100.0  # decimal scale
    rsi = _compute_rsi(prices_arr) / 100.0  # 0.0 to 1.0
    atr = _compute_atr(prices_arr, highs=highs_arr, lows=lows_arr)

    return {
        "rsi": float(max(0.0, min(1.0, rsi))),
        "atr": float(atr),
        "vix": float(max(0.05, min(1.0, vix))),
        "macd": float(_compute_macd(prices_arr)),
        "vol_ratio": float(_compute_volume_ratio(np.array(volumes, dtype=np.float64))) if volumes and len(volumes) >= 5 else 1.0,
        "corr": float(_compute_correlation(prices_arr, np.array(volumes, dtype=np.float64))) if volumes and len(volumes) >= 5 else 0.0,
        "hurst": float(_compute_hurst(prices_arr)),
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "MarketFeatures",
    "extract_features",
    "extract_single_feature_vector",
    "compute_dict_features",
]

