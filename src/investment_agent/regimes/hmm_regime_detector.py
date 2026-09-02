"""HMM Regime Detector — Authoritative Hidden Markov Model Regime Classification for X Quant X.

WHAT
====
Operational HMM-based market regime classification implementing the authoritative
architecture specified in:
    alpaca_paper_trading_specifications_x_quant_x/027_xquantx_regime_archetypes.txt

WHY
===
The rule-based detector (regime_detector.py) provides a deterministic approximation
for auditability and testing. The HMM detector is the authoritative implementation
that matches the papers' specification:

    market observations → HMM/regime probabilities → active regime

HOW
===
- Loads regime definitions from config/regimes.toml
- Computes emission probabilities from feature vector using multivariate Gaussian
- Applies forward-backward inference for posterior regime probabilities
- Applies Viterbi decoding with dwell-time enforcement for regime classification
- Computes regime entropy H_t for uncertainty gating

The implementation provides:
1. Concrete HMMRegimeDetector with full inference
2. Configuration loading from config/regimes.toml
3. 12-state HMM with 7-feature emission model
4. Scaled forward-backward for posterior probabilities
5. Viterbi decoding with dwell-time enforcement
6. Regime entropy computation

IMPLEMENTATION STATUS
=====================
- Configuration loading: ✅ Implemented
- Interface contract: ✅ Implemented
- HMM inference (forward-backward, Viterbi): ✅ Implemented
- Dwell-time enforcement: ✅ Implemented
- Regime entropy computation: ✅ Implemented
- Baum-Welch parameter learning: ❌ Future enhancement (uses fixed priors from config)

Architectural Role
==================
Authoritative regime classification layer. Consumes market features and produces
HMM-based regime probabilities consumed by agent_reputation.py, ensemble_signal.py,
and capital_gate.py.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .regimes import VALID_REGIMES
from .hmm_inference import (
    HMMParameters,
    HMMInferenceResult,
    HMMInference,
    HMMRegimeDetectorImpl,
    load_hmm_parameters,
    HMMUnderflowError,
    MIN_DWELL_BARS,
    N_STATES,
)


# Try to import tomllib (Python 3.11+) or tomli
try:
    import tomllib
except ModuleNotFoundError:
    try:
        import tomli as tomllib  # type: ignore
    except ModuleNotFoundError:
        tomllib = None  # type: ignore


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def _load_regime_config(path: Optional[Path] = None) -> Dict[str, Any]:
    """Load regime definitions from config/regimes.toml.

    Parameters
    ----------
    path : Optional[Path]
        Custom path to regimes.toml. If None, uses default locations.

    Returns
    -------
    Dict[str, Any]
        Parsed regime configuration.
    """
    if path is None:
        candidates = [
            Path(__file__).resolve().parent.parent.parent / "config" / "regimes.toml",
            Path.cwd() / "config" / "regimes.toml",
        ]
        for candidate in candidates:
            if candidate.exists() and candidate.is_file():
                path = candidate
                break

    if path is None or not path.exists():
        return {}

    try:
        with path.open("rb") as fp:
            return tomllib.load(fp)
    except Exception:
        return {}


_REGIME_CONFIG = _load_regime_config()


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RegimeProbability:
    """Immutable HMM regime probability result.

    Attributes
    ----------
    regime : str
        Most probable regime identifier (R01-R12).
    probabilities : Dict[str, float]
        Statistically calibrated HMM posterior probabilities P(r_k|x_t) summing to 1.0.
        These ARE computed by the forward-backward algorithm, not heuristic scores.
    entropy : float
        Regime entropy H_t = -sum_k P(r_k|x_t) ln P(r_k|x_t).
    normalized_entropy : float
        Normalized entropy U_t = H_t / ln(12) in [0, 1].
    dwell_time : int
        Minimum dwell time in bars for the classified regime.
    is_confident : bool
        True if normalized entropy U_t < 0.5.
    timestamp : datetime
        Inference timestamp.
    """

    regime: str
    probabilities: Dict[str, float]
    entropy: float
    normalized_entropy: float
    dwell_time: int
    is_confident: bool
    timestamp: datetime = field(default_factory=datetime.now)


# ---------------------------------------------------------------------------
# HMM Regime Detector
# ---------------------------------------------------------------------------

class HMMRegimeDetector:
    """Concrete HMM-based regime detector.

    Uses 12-state Hidden Markov Model with:
    - Forward-backward inference for posterior probabilities
    - Viterbi decoding with dwell-time enforcement for regime classification
    - Multivariate Gaussian emission distributions
    - Regime entropy computation for uncertainty gating

    The HMM parameters are loaded from config/regimes.toml.
    """

    def __init__(self, config_path: Optional[Path] = None) -> None:
        """Initialize HMM regime detector.

        Parameters
        ----------
        config_path : Optional[Path]
            Path to regimes.toml configuration file.
        """
        config = _load_regime_config(config_path)
        if not config:
            raise ValueError(
                "Could not load regime configuration. "
                "Ensure config/regimes.toml exists."
            )

        params = load_hmm_parameters(config)
        self._impl = HMMRegimeDetectorImpl(params)
        self._config = config

        # Fixed calibration scaler for mapping raw market features to standardized HMM space:
        # [RSI, MACD, ATR, VIX, VolRatio, Corr, Hurst]
        calib_cfg = config.get("calibration", {})
        default_means = [50.0, 0.0, 0.02, 15.0, 1.0, 0.0, 0.5]
        default_stds = [15.0, 2.0, 0.015, 10.0, 0.5, 0.4, 0.15]
        means = calib_cfg.get("means", default_means)
        stds = calib_cfg.get("stds", default_stds)
        self._calibration_means = np.array(means, dtype=np.float64)
        self._calibration_stds = np.array(stds, dtype=np.float64)
        self._calibration_stds = np.where(self._calibration_stds < 1e-10, 1.0, self._calibration_stds)

    def classify(self, features: List[List[float]]) -> RegimeProbability:
        """Classify regime from feature sequence using HMM inference.

        Parameters
        ----------
        features : List[List[float]]
            Feature matrix where each row is [RSI, MACD, ATR, VIX, VolRatio, Corr, Hurst].
            Must contain at least 1 observation.

        Returns
        -------
        RegimeProbability
            HMM-based regime classification with statistically calibrated probabilities.
        
        Note: Dwell-time enforcement is applied as post-processing on the Viterbi path.
        The authoritative architecture specifies constrained Viterbi inference under dwell-time
        constraints; this implementation uses unconstrained Viterbi followed by run-length
        filtering as an approximation.

        Feature Scaling:
        The 7 features have very different natural scales (RSI: 0-100, MACD: unbounded,
        ATR: dollars, VIX: percentage, etc.). The HMM assumes Gaussian emissions, so
        features are standardized (zero mean, unit variance) before inference.
        The configuration in config/regimes.toml provides emission means in standardized space.
        """
        obs = np.array(features, dtype=np.float64)
        
        # Validate before standardization
        if obs.size == 0:
            raise ValueError("Feature sequence is empty")
        if obs.ndim != 2 or obs.shape[1] != 7:
            raise ValueError(f"Expected N x 7 feature matrix, got shape {obs.shape}")
        if np.any(np.isnan(obs)):
            raise ValueError("Feature vector contains NaN values")
        if np.any(np.isinf(obs)):
            raise ValueError("Feature vector contains Infinity values")
        
        # Standardize features using fixed calibration scaler (not moving-window per inference)
        # to ensure absolute market state maps consistently to calibrated emission distributions.
        # Clip z-scores to [-4.0, 4.0] to prevent mathematical Gaussian underflow.
        obs_standardized = (obs - self._calibration_means) / self._calibration_stds
        obs_standardized = np.clip(obs_standardized, -4.0, 4.0)
        
        result = self._impl.classify(obs_standardized)

        # Get dwell time from config
        dwell_times = self._config.get("min_dwell_bars", {})
        dwell_time = dwell_times.get(result.regime, MIN_DWELL_BARS)

        # Convert numpy array to dict
        probs = {}
        for i in range(N_STATES):
            regime_id = f"R{i + 1:02d}"
            probs[regime_id] = float(result.posterior_probabilities[i])

        return RegimeProbability(
            regime=result.regime,
            probabilities=probs,
            entropy=result.entropy,
            normalized_entropy=result.normalized_entropy,
            dwell_time=dwell_time,
            is_confident=result.is_confident,
        )

    def classify_sequence(self, features: List[List[float]]) -> List[RegimeProbability]:
        """Classify a sequence of observations returning per-bar posterior probabilities.

        Unlike classify() which returns a single RegimeProbability for the final bar,
        classify_sequence() performs forward-backward smoothing over the full sequence
        and returns a RegimeProbability for *every* historical bar t, containing bar t's
        true posterior probability distribution P(S_t | O_{1:T}).

        Parameters
        ----------
        features : List[List[float]]
            N x 7 feature matrix.

        Returns
        -------
        List[RegimeProbability]
            List of length N containing per-bar regime probabilities and Viterbi assignments.
        """
        obs = np.array(features, dtype=np.float64)
        if obs.ndim != 2 or obs.shape[1] != 7:
            raise ValueError(f"Expected N x 7 feature matrix, got shape {obs.shape}")

        obs_standardized = (obs - self._calibration_means) / self._calibration_stds
        obs_standardized = np.clip(obs_standardized, -4.0, 4.0)

        gamma, _, _, _ = self._impl._inference.forward_backward(obs_standardized)
        viterbi_path = self._impl._inference.enforce_dwell_time(
            self._impl._inference.viterbi(obs_standardized)
        )

        dwell_times = self._config.get("min_dwell_bars", {})

        sequence: List[RegimeProbability] = []
        for t in range(obs.shape[0]):
            p_t = gamma[t, :].copy()
            p_t_clipped = np.maximum(p_t, 1e-300)
            p_t_clipped /= p_t_clipped.sum()

            entropy = -float(np.sum(p_t_clipped * np.log(p_t_clipped)))
            normalized_entropy = entropy / math.log(N_STATES)
            is_confident = normalized_entropy < 0.5

            probs = {f"R{i + 1:02d}": float(p_t[i]) for i in range(N_STATES)}
            regime_t = viterbi_path[t]
            dwell_time = dwell_times.get(regime_t, MIN_DWELL_BARS)

            sequence.append(
                RegimeProbability(
                    regime=regime_t,
                    probabilities=probs,
                    entropy=entropy,
                    normalized_entropy=normalized_entropy,
                    dwell_time=dwell_time,
                    is_confident=is_confident,
                )
            )

        return sequence

    def get_transition_matrix(self) -> np.ndarray:
        """Return the current HMM transition matrix."""
        return self._impl._params.transition_matrix.copy()

    def get_emission_means(self) -> np.ndarray:
        """Return the current HMM emission means."""
        return self._impl._params.emission_means.copy()

    def get_history(self) -> List[HMMInferenceResult]:
        """Return inference history."""
        return self._impl.get_history()

    def clear_history(self) -> None:
        """Clear inference history."""
        self._impl.clear_history()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_hmm_detector(config_path: Optional[Path] = None) -> HMMRegimeDetector:
    """Get the HMM regime detector.

    Parameters
    ----------
    config_path : Optional[Path]
        Path to regimes.toml configuration file.

    Returns
    -------
    HMMRegimeDetector
        Operational HMM regime detector instance.
    """
    return HMMRegimeDetector(config_path)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "HMMRegimeDetector",
    "RegimeProbability",
    "HMMUnderflowError",
    "get_hmm_detector",
    "_load_regime_config",
]
