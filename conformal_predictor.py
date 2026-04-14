"""
Conformal Prediction Wrapper
===============================
Statistically valid p-values for anomaly scores instead of binary
labels. Provides rigorous false positive rate (FPR) guarantees
under the exchangeability assumption.

Theory (Vovk, Gammerman & Shafer, 2005):
  Given a nonconformity measure α and calibration set Z = {z₁, ..., zₙ},
  the p-value for a new sample x is:

    p(x) = |{i : α(zᵢ) ≥ α(x)}| + 1 / (n + 1)

  This provides a valid p-value under exchangeability:
    P(p(x) ≤ ε) ≤ ε for any significance level ε.

  In words: if we set ε = 0.05, we guarantee ≤ 5% false positives.

For anomaly detection:
  - Nonconformity measure = anomaly score (higher = more anomalous)
  - P-value = fraction of calibration samples at least as anomalous
  - Low p-value → likely anomaly (few calibration samples are this weird)
  - Flag as anomaly if p(x) ≤ ε (significance level)

Key benefit:
  Unlike threshold-based detection, conformal prediction provides
  *statistical guarantees* on the FPR, not just empirical thresholds.

Constraints: NumPy only, no sklearn.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


# ──────────────────────────────────────────────────────────────────────
# Conformal Prediction Result
# ──────────────────────────────────────────────────────────────────────

@dataclass
class ConformalResult:
    """Result of conformal anomaly prediction."""
    p_value: float
    anomaly_score: float
    is_anomaly: bool
    significance_level: float
    confidence: float  # 1 - p_value

    @property
    def summary(self) -> str:
        label = "ANOMALY" if self.is_anomaly else "normal"
        return (
            f"p={self.p_value:.4f}, score={self.anomaly_score:.3f}, "
            f"conf={self.confidence:.4f}, ε={self.significance_level} → {label}"
        )


# ──────────────────────────────────────────────────────────────────────
# Conformal Anomaly Detector
# ──────────────────────────────────────────────────────────────────────

class ConformalAnomalyDetector:
    """Conformal prediction wrapper for anomaly detection.

    Wraps any anomaly scorer and provides statistically valid p-values
    with guaranteed FPR control under exchangeability.

    Two modes:
      1. Offline (batch): calibrate on a held-out set, then predict
      2. Online (streaming): sliding window calibration set that updates

    Args:
        significance_level: ε threshold — flag as anomaly if p ≤ ε.
            ε = 0.015 guarantees ≤ 1.5% FPR (default: 0.015).
        calibration_size: Number of calibration samples to maintain
            in online mode (default: 500).
        online: Whether to use online (sliding window) calibration
            (default: True).
    """

    def __init__(
        self,
        significance_level: float = 0.015,
        calibration_size: int = 500,
        online: bool = True,
    ):
        self.significance_level = significance_level
        self.calibration_size = calibration_size
        self.online = online

        # Calibration scores (nonconformity measures)
        self._calibration_scores: List[float] = []
        self._sorted_scores: Optional[np.ndarray] = None
        self._is_calibrated = False

        # Statistics
        self._total_predictions: int = 0
        self._total_anomalies: int = 0

    def calibrate(self, scores: np.ndarray) -> None:
        """Calibrate with a set of nonconformity scores.

        These should be scores from "normal" or representative data.
        Higher scores = more anomalous.

        Args:
            scores: Nonconformity scores (anomaly scores), shape (n,).
        """
        self._calibration_scores = list(scores.flatten())
        if len(self._calibration_scores) > self.calibration_size:
            # Keep most recent
            self._calibration_scores = self._calibration_scores[-self.calibration_size:]
        self._sorted_scores = np.sort(self._calibration_scores)
        self._is_calibrated = True

    def calibrate_from_scorer(
        self,
        scorer,
        X_cal: np.ndarray,
    ) -> None:
        """Calibrate by scoring calibration data with a given scorer.

        Args:
            scorer: Object with anomaly_scores(X) method.
            X_cal: Calibration data, shape (n_samples, n_features).
        """
        scores = scorer.anomaly_scores(X_cal)
        self.calibrate(scores)

    def predict(self, anomaly_score: float) -> ConformalResult:
        """Compute p-value and make prediction for a single sample.

        p(x) = (|{i : α(zᵢ) ≥ α(x)}| + 1) / (n + 1)

        Args:
            anomaly_score: Nonconformity score for the test sample.

        Returns:
            ConformalResult with p-value and anomaly label.
        """
        if not self._is_calibrated:
            raise RuntimeError(
                "ConformalAnomalyDetector not calibrated. "
                "Call calibrate() or calibrate_from_scorer() first."
            )

        n = len(self._calibration_scores)

        # Count calibration scores >= test score
        # Using sorted array for efficiency
        n_geq = n - int(np.searchsorted(self._sorted_scores, anomaly_score, side='left'))

        # P-value (Vovk formula, including the test sample itself)
        p_value = (n_geq + 1) / (n + 1)

        is_anomaly = p_value <= self.significance_level

        self._total_predictions += 1
        if is_anomaly:
            self._total_anomalies += 1

        # Online update: add this score to calibration set
        if self.online:
            self._calibration_scores.append(anomaly_score)
            if len(self._calibration_scores) > self.calibration_size:
                self._calibration_scores.pop(0)
            self._sorted_scores = np.sort(self._calibration_scores)

        return ConformalResult(
            p_value=p_value,
            anomaly_score=anomaly_score,
            is_anomaly=is_anomaly,
            significance_level=self.significance_level,
            confidence=1.0 - p_value,
        )

    def predict_batch(self, scores: np.ndarray) -> List[ConformalResult]:
        """Predict for multiple samples.

        Note: In online mode, the calibration set is NOT updated during
        batch prediction to maintain exchangeability within the batch.

        Args:
            scores: Anomaly scores, shape (n_samples,).

        Returns:
            List of ConformalResult objects.
        """
        if not self._is_calibrated:
            raise RuntimeError("Not calibrated.")

        # Temporarily disable online updates for batch consistency
        was_online = self.online
        self.online = False

        results = [self.predict(float(s)) for s in scores.flatten()]

        self.online = was_online

        # If online, update calibration with all batch scores at once
        if self.online:
            for s in scores.flatten():
                self._calibration_scores.append(float(s))
            if len(self._calibration_scores) > self.calibration_size:
                self._calibration_scores = self._calibration_scores[-self.calibration_size:]
            self._sorted_scores = np.sort(self._calibration_scores)

        return results

    def predict_scores_to_pvalues(self, scores: np.ndarray) -> np.ndarray:
        """Convert anomaly scores to p-values (batch, no online update).

        Fast vectorized computation for evaluation.

        Args:
            scores: Anomaly scores, shape (n_samples,).

        Returns:
            P-values, shape (n_samples,).
        """
        if not self._is_calibrated:
            raise RuntimeError("Not calibrated.")

        n = len(self._calibration_scores)
        p_values = np.zeros(len(scores))

        for i, s in enumerate(scores):
            n_geq = n - int(np.searchsorted(self._sorted_scores, s, side='left'))
            p_values[i] = (n_geq + 1) / (n + 1)

        return p_values

    def evaluate_coverage(
        self,
        scores_normal: np.ndarray,
        scores_anomaly: np.ndarray,
    ) -> dict:
        """Evaluate coverage guarantee and detection performance.

        Checks that:
          - FPR on normal data ≤ ε (coverage guarantee)
          - Detection on anomaly data is high

        Args:
            scores_normal: Anomaly scores for genuinely normal samples.
            scores_anomaly: Anomaly scores for genuinely anomalous samples.

        Returns:
            Dict with coverage metrics.
        """
        p_normal = self.predict_scores_to_pvalues(scores_normal)
        p_anomaly = self.predict_scores_to_pvalues(scores_anomaly)

        fpr = np.mean(p_normal <= self.significance_level)
        tpr = np.mean(p_anomaly <= self.significance_level)

        return {
            "significance_level": self.significance_level,
            "fpr_actual": float(fpr),
            "fpr_guarantee": self.significance_level,
            "fpr_valid": fpr <= self.significance_level + 0.02,  # Small tolerance
            "tpr_detection": float(tpr),
            "n_normal": len(scores_normal),
            "n_anomaly": len(scores_anomaly),
            "calibration_size": len(self._calibration_scores),
        }

    @property
    def is_calibrated(self) -> bool:
        return self._is_calibrated

    @property
    def total_predictions(self) -> int:
        return self._total_predictions

    @property
    def total_anomalies(self) -> int:
        return self._total_anomalies

    @property
    def anomaly_rate(self) -> float:
        if self._total_predictions == 0:
            return 0.0
        return self._total_anomalies / self._total_predictions

    def get_stats(self) -> dict:
        return {
            "is_calibrated": self._is_calibrated,
            "calibration_size": len(self._calibration_scores),
            "significance_level": self.significance_level,
            "total_predictions": self._total_predictions,
            "total_anomalies": self._total_anomalies,
            "anomaly_rate": self.anomaly_rate,
            "online": self.online,
        }
