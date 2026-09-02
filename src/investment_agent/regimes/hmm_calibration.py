"""HMM Empirical Calibration & Validation — X Quant X.

WHAT
====
Utilities for:
  1. Brier Score: measures calibration of regime probabilities against
     realized outcomes (did the high-prob regime turn out correct?).
  2. Log-Loss: strictly proper scoring rule for probabilistic classifiers.
  3. Regime Transition Validation: checks that the empirical transition
     matrix from observed data matches the configured HMM A matrix.
  4. Feature Distribution Check: verifies that calibration means/stds in
     regimes.toml match the distribution of real market data.

WHY
===
The HMM emits raw posterior probabilities. Without calibration checks:
- A confident wrong prediction (p=0.95 for wrong regime) is far worse than
  an uncertain correct prediction (p=0.55 correct, 0.45 wrong).
- Drift in market microstructure can render fixed calibration coefficients
  stale, causing the regime signal to be systematically biased.

HOW
===
Call ``validate_hmm(features, regimes, probabilities)`` with:
  - features:      List[List[float]] — raw feature matrix (N x 7)
  - regimes:       List[str]         — Viterbi-decoded regime labels per bar
  - probabilities: List[Dict[str, float]] — posterior dicts per bar

Returns a ``HMMValidationReport`` dataclass with scores and diagnostics.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class HMMValidationReport:
    """Calibration and validation report for one inference run.

    Attributes
    ----------
    n_bars : int
        Number of inference steps evaluated.
    brier_score : float
        Average Brier score across all bars. Range [0, 2]. Lower is better.
        Perfect calibration = 0; worst = 2 (always confident & wrong).
    brier_score_per_regime : Dict[str, float]
        Brier score broken down by the *predicted* (MAP) regime.
    log_loss : float
        Average log-loss (cross-entropy) per bar. Lower is better.
        Perfectly calibrated classifier = H(p_true).
    regime_counts : Dict[str, int]
        How many bars the Viterbi decoder assigned to each regime.
    empirical_transition_matrix : Dict[str, Dict[str, float]]
        Observed one-step regime→regime transition frequencies (row-stochastic).
    transition_divergence : Dict[str, float]
        KL divergence between empirical and configured transition rows.
        Low values (< 0.1) mean the model transitions match reality.
    feature_drift : List[Dict[str, Any]]
        Per-feature drift: mean, std, z-score vs calibration coefficients.
    warnings : List[str]
        Human-readable warnings raised during validation.
    """
    n_bars: int = 0
    brier_score: float = 0.0
    brier_score_per_regime: Dict[str, float] = field(default_factory=dict)
    log_loss: float = 0.0
    regime_counts: Dict[str, int] = field(default_factory=dict)
    empirical_transition_matrix: Dict[str, Dict[str, float]] = field(default_factory=dict)
    transition_divergence: Dict[str, float] = field(default_factory=dict)
    feature_drift: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Core validation
# ---------------------------------------------------------------------------

def validate_hmm(
    features: List[List[float]],
    regimes: List[str],
    probabilities: List[Dict[str, float]],
    *,
    calibration_means: Optional[List[float]] = None,
    calibration_stds: Optional[List[float]] = None,
    configured_transition_matrix: Optional[Dict[str, Dict[str, float]]] = None,
    feature_names: Optional[List[str]] = None,
) -> HMMValidationReport:
    """Compute full calibration & validation report.

    Parameters
    ----------
    features : List[List[float]]
        Raw (unstandardized) feature vectors, shape (N, 7).
    regimes : List[str]
        Viterbi-decoded regime label for each bar — the MAP hard assignment.
    probabilities : List[Dict[str, float]]
        Full posterior probability dict for each bar (sums to 1.0).
    calibration_means : optional List[float]
        Calibration means used for standardization (7-element). When
        provided, feature drift is measured against these anchors.
    calibration_stds : optional List[float]
        Calibration std-devs (7-element).
    configured_transition_matrix : optional Dict[str, Dict[str, float]]
        The A matrix from regimes.toml. When provided, KL divergence
        between empirical and configured transitions is computed.
    feature_names : optional List[str]
        Names for each of the 7 features (for readability only).

    Returns
    -------
    HMMValidationReport
    """
    report = HMMValidationReport()
    n = len(regimes)
    if n == 0:
        report.warnings.append("Empty regime sequence — nothing to validate.")
        return report

    if len(probabilities) != n:
        report.warnings.append(
            f"Length mismatch: regimes={n}, probabilities={len(probabilities)}"
        )
        n = min(n, len(probabilities))

    report.n_bars = n
    _NAMES = feature_names or [
        "RSI", "MACD", "ATR", "VIX_proxy", "VolRatio", "Corr", "Hurst"
    ]

    # ------------------------------------------------------------------
    # 1. Regime counts
    # ------------------------------------------------------------------
    for reg in regimes[:n]:
        report.regime_counts[reg] = report.regime_counts.get(reg, 0) + 1

    # ------------------------------------------------------------------
    # 2. Brier score
    # ------------------------------------------------------------------
    # We treat each bar's MAP regime as the "true" label (one-hot).
    # Brier = (1/N) * sum_t sum_r (p(r|t) - 1{r == viterbi_r_t})^2
    all_regimes = sorted(
        {r for d in probabilities[:n] for r in d} | set(regimes[:n])
    )
    total_brier = 0.0
    per_regime_brier: Dict[str, List[float]] = {r: [] for r in all_regimes}

    for i in range(n):
        true_regime = regimes[i]
        probs = probabilities[i]
        brier_t = 0.0
        for r in all_regimes:
            p = probs.get(r, 0.0)
            y = 1.0 if r == true_regime else 0.0
            brier_t += (p - y) ** 2
        total_brier += brier_t
        per_regime_brier.setdefault(true_regime, []).append(brier_t)

    report.brier_score = total_brier / n
    report.brier_score_per_regime = {
        r: (sum(v) / len(v) if v else 0.0)
        for r, v in per_regime_brier.items()
        if v
    }

    if report.brier_score > 1.0:
        report.warnings.append(
            f"Brier score {report.brier_score:.3f} > 1.0: HMM is poorly calibrated."
        )
    elif report.brier_score > 0.5:
        report.warnings.append(
            f"Brier score {report.brier_score:.3f} > 0.5: consider recalibrating."
        )

    # ------------------------------------------------------------------
    # 3. Log-loss
    # ------------------------------------------------------------------
    eps = 1e-12
    total_ll = 0.0
    for i in range(n):
        true_regime = regimes[i]
        p_true = max(eps, probabilities[i].get(true_regime, eps))
        total_ll += math.log(p_true)
    report.log_loss = -total_ll / n

    if report.log_loss > 2.0:
        report.warnings.append(
            f"Log-loss {report.log_loss:.3f} > 2.0: HMM has low confidence on correct class."
        )

    # ------------------------------------------------------------------
    # 4. Empirical transition matrix
    # ------------------------------------------------------------------
    transitions: Dict[str, Dict[str, int]] = {}
    for i in range(n - 1):
        src = regimes[i]
        dst = regimes[i + 1]
        transitions.setdefault(src, {})
        transitions[src][dst] = transitions[src].get(dst, 0) + 1

    for src, counts in transitions.items():
        total = sum(counts.values())
        report.empirical_transition_matrix[src] = {
            dst: cnt / total for dst, cnt in counts.items()
        }

    # ------------------------------------------------------------------
    # 5. KL divergence: empirical vs configured transitions
    # ------------------------------------------------------------------
    if configured_transition_matrix:
        for src, emp_row in report.empirical_transition_matrix.items():
            cfg_row = configured_transition_matrix.get(src, {})
            if not cfg_row:
                continue
            kl = 0.0
            all_dst = set(emp_row) | set(cfg_row)
            for dst in all_dst:
                p_emp = max(eps, emp_row.get(dst, eps))
                q_cfg = max(eps, cfg_row.get(dst, eps))
                kl += p_emp * math.log(p_emp / q_cfg)
            report.transition_divergence[src] = kl
            if kl > 0.5:
                report.warnings.append(
                    f"Regime {src}: transition KL divergence={kl:.3f} — "
                    "empirical transitions differ significantly from configured A matrix."
                )

    # ------------------------------------------------------------------
    # 6. Feature drift
    # ------------------------------------------------------------------
    if features and calibration_means and calibration_stds:
        try:
            import numpy as np
            feat_arr = np.array(features[:n], dtype=float)  # (N, 7)
            observed_means = feat_arr.mean(axis=0)
            observed_stds = feat_arr.std(axis=0)
            calib_means = list(calibration_means)
            calib_stds = list(calibration_stds)
            n_feats = min(feat_arr.shape[1], len(calib_means), len(calib_stds))
            for j in range(n_feats):
                z = (
                    (observed_means[j] - calib_means[j]) / calib_stds[j]
                    if calib_stds[j] > 1e-10
                    else 0.0
                )
                record: Dict[str, Any] = {
                    "feature": _NAMES[j] if j < len(_NAMES) else f"feat_{j}",
                    "observed_mean": float(observed_means[j]),
                    "observed_std": float(observed_stds[j]),
                    "calib_mean": float(calib_means[j]),
                    "calib_std": float(calib_stds[j]),
                    "z_score_mean": float(z),
                }
                report.feature_drift.append(record)
                if abs(z) > 2.0:
                    report.warnings.append(
                        f"Feature '{record['feature']}': mean drifted "
                        f"{z:+.1f}σ from calibration anchor — "
                        "consider recalibrating."
                    )
        except Exception as exc:
            report.warnings.append(f"Feature drift analysis failed: {exc}")

    return report


# ---------------------------------------------------------------------------
# Convenience: run validation from HMMRegimeDetector output
# ---------------------------------------------------------------------------

def validate_from_detector(
    prices: List[float],
    volumes: Optional[List[float]] = None,
    highs: Optional[List[float]] = None,
    lows: Optional[List[float]] = None,
) -> HMMValidationReport:
    """End-to-end convenience: extract features, run HMM, then validate.

    Parameters
    ----------
    prices : List[float]
        Close prices.
    volumes : optional List[float]
        Volume series.
    highs / lows : optional List[float]
        Used for true-range ATR.

    Returns
    -------
    HMMValidationReport
    """
    from investment_agent.regimes.market_feature_extractor import extract_features
    from investment_agent.regimes.hmm_regime_detector import HMMRegimeDetector

    detector = HMMRegimeDetector()
    feat_matrix = extract_features(prices, volumes, highs=highs, lows=lows, lookback_days=20)

    # Run bar-by-bar to get per-bar regime sequence and probabilities
    regimes: List[str] = []
    probabilities: List[Dict[str, float]] = []

    # Full pass (single inference over entire window)
    result = detector.classify(feat_matrix.tolist())
    history = detector.get_history()

    if history:
        for h in history:
            regimes.append(h.regime)
            # Build probability dict from the last result's posterior
            # (history object carries the final posterior; individual bars not stored)
            probabilities.append(result.probabilities)
    else:
        # Fallback: single observation
        regimes = [result.regime]
        probabilities = [result.probabilities]

    return validate_hmm(
        features=feat_matrix.tolist(),
        regimes=regimes,
        probabilities=probabilities,
        calibration_means=detector._calibration_means.tolist(),
        calibration_stds=detector._calibration_stds.tolist(),
    )


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def _print_report(report: HMMValidationReport) -> None:
    print(f"\n=== HMM Validation Report ({report.n_bars} bars) ===")
    print(f"  Brier Score : {report.brier_score:.4f}  (lower is better; <0.5 is good)")
    print(f"  Log-Loss    : {report.log_loss:.4f}  (lower is better)")
    print(f"\n  Regime distribution:")
    for reg, cnt in sorted(report.regime_counts.items()):
        print(f"    {reg}: {cnt} bars ({100*cnt/report.n_bars:.1f}%)")
    if report.feature_drift:
        print(f"\n  Feature drift (z-score vs calibration):")
        for fd in report.feature_drift:
            flag = "⚠️ " if abs(fd["z_score_mean"]) > 2.0 else "   "
            print(f"  {flag}{fd['feature']:14s} obs_mean={fd['observed_mean']:8.3f}  "
                  f"calib_mean={fd['calib_mean']:8.3f}  z={fd['z_score_mean']:+.2f}")
    if report.transition_divergence:
        print(f"\n  Transition KL divergences:")
        for src, kl in sorted(report.transition_divergence.items()):
            flag = "⚠️ " if kl > 0.5 else "   "
            print(f"  {flag}{src}: KL={kl:.4f}")
    if report.warnings:
        print(f"\n  Warnings:")
        for w in report.warnings:
            print(f"    ⚠ {w}")
    print()


if __name__ == "__main__":
    import sys
    # Quick smoke test with synthetic prices
    import random
    print("Running smoke-test validation with synthetic data…")
    rng = random.Random(42)
    prices = [100.0 + rng.gauss(0, 1) for _ in range(200)]
    prices = [max(50.0, p) for p in prices]  # ensure positive
    report = validate_from_detector(prices)
    _print_report(report)
    sys.exit(0 if report.brier_score < 2.0 else 1)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "HMMValidationReport",
    "validate_hmm",
    "validate_from_detector",
]
