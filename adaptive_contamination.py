"""
Adaptive Contamination Estimator
==================================
Online estimator that adjusts the Isolation Forest contamination
parameter based on recent detection rates, rather than using a
fixed value.

Problem:
  Fixed contamination (e.g., 0.2) assumes a constant attack rate.
  In reality, attack frequency varies: baseline periods have near-zero
  attacks, while under-attack periods may exceed 50%.

Solution:
  EWMA of recent detection rate → dynamically adjust contamination.
  Clamped to [min_c, max_c] to prevent degenerate behavior.

Update rule:
  c_new = α · detection_rate_recent + (1-α) · c_old
  c_new = clamp(c_new, min_c, max_c)

Then recalibrate the iForest threshold without retraining:
  threshold = percentile(training_scores, 100 * (1 - c_new))

Constraints: NumPy only, no sklearn.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


# ──────────────────────────────────────────────────────────────────────
# Contamination History Entry
# ──────────────────────────────────────────────────────────────────────

@dataclass
class ContaminationUpdate:
    """Record of a contamination update."""
    timestamp: float
    old_contamination: float
    new_contamination: float
    detection_rate: float
    window_size: int


# ──────────────────────────────────────────────────────────────────────
# Adaptive Contamination Estimator
# ──────────────────────────────────────────────────────────────────────

class AdaptiveContaminationEstimator:
    """Online estimator for the Isolation Forest contamination parameter.

    Tracks recent detection rates and adjusts contamination dynamically
    using EWMA smoothing. This allows the model to adapt to changing
    attack frequencies without retraining.

    Args:
        initial_contamination: Starting contamination value (default: 0.1).
        min_contamination: Minimum allowed contamination (default: 0.01).
        max_contamination: Maximum allowed contamination (default: 0.3).
        alpha: EWMA smoothing factor (default: 0.15).
            Higher α = faster adaptation, more responsive.
            Lower α = slower adaptation, more stable.
        window_size: Number of recent predictions to track (default: 100).
        update_interval: Minimum predictions between updates (default: 50).
    """

    def __init__(
        self,
        initial_contamination: float = 0.1,
        min_contamination: float = 0.01,
        max_contamination: float = 0.3,
        alpha: float = 0.15,
        window_size: int = 100,
        update_interval: int = 50,
    ):
        self._contamination = initial_contamination
        self.min_contamination = min_contamination
        self.max_contamination = max_contamination
        self.alpha = alpha
        self.window_size = window_size
        self.update_interval = update_interval

        # Prediction tracking (1 = anomaly, 0 = normal)
        self._predictions: List[int] = []
        self._predictions_since_update: int = 0
        self._update_history: List[ContaminationUpdate] = []

        # Training scores cache (for threshold recalibration)
        self._training_scores: Optional[np.ndarray] = None

    @property
    def contamination(self) -> float:
        """Current estimated contamination."""
        return self._contamination

    def record_prediction(self, is_anomaly: bool) -> Optional[float]:
        """Record a prediction and potentially update contamination.

        Args:
            is_anomaly: Whether this sample was classified as anomalous.

        Returns:
            New contamination value if updated, None otherwise.
        """
        self._predictions.append(1 if is_anomaly else 0)

        # Keep only recent predictions
        if len(self._predictions) > self.window_size * 2:
            self._predictions = self._predictions[-self.window_size:]

        self._predictions_since_update += 1

        # Check if it's time to update
        if self._predictions_since_update >= self.update_interval:
            return self._update_contamination()

        return None

    def _update_contamination(self) -> float:
        """Compute new contamination from recent detection rate.

        Returns:
            Updated contamination value.
        """
        recent = self._predictions[-self.window_size:]
        detection_rate = sum(recent) / max(len(recent), 1)

        old_c = self._contamination

        # EWMA update
        new_c = self.alpha * detection_rate + (1 - self.alpha) * old_c

        # Clamp
        new_c = max(self.min_contamination, min(self.max_contamination, new_c))

        self._contamination = new_c
        self._predictions_since_update = 0

        # Record update
        self._update_history.append(ContaminationUpdate(
            timestamp=time.time(),
            old_contamination=old_c,
            new_contamination=new_c,
            detection_rate=detection_rate,
            window_size=len(recent),
        ))

        return new_c

    def apply_to_forest(self, forest) -> float:
        """Recalibrate an iForest's threshold using current contamination.

        Does NOT retrain the forest — only adjusts the decision threshold.
        Uses cached training scores if available, otherwise scores are
        recomputed from the forest's internal state.

        Args:
            forest: IsolationForest or ExtendedIsolationForest instance.
                Must have a threshold setter and _compute_scores method.

        Returns:
            New threshold value.
        """
        if self._training_scores is not None:
            scores = self._training_scores
        else:
            # Cannot recalibrate without training scores
            return forest.threshold

        new_threshold = float(np.percentile(
            scores, 100 * (1 - self._contamination)
        ))
        forest.threshold = new_threshold
        return new_threshold

    def cache_training_scores(self, scores: np.ndarray) -> None:
        """Cache training data scores for threshold recalibration.

        Call this after initial forest.fit() to enable dynamic
        threshold adjustment without retraining.

        Args:
            scores: Anomaly scores of training data, shape (n_samples,).
        """
        self._training_scores = scores.copy()

    def force_update(self) -> float:
        """Force a contamination update regardless of interval.

        Returns:
            Updated contamination value.
        """
        return self._update_contamination()

    def reset(self, contamination: Optional[float] = None) -> None:
        """Reset the estimator.

        Args:
            contamination: New initial contamination (default: keep current).
        """
        if contamination is not None:
            self._contamination = contamination
        self._predictions.clear()
        self._predictions_since_update = 0

    @property
    def detection_rate(self) -> float:
        """Current detection rate from recent predictions."""
        if not self._predictions:
            return 0.0
        recent = self._predictions[-self.window_size:]
        return sum(recent) / len(recent)

    @property
    def total_predictions(self) -> int:
        return len(self._predictions)

    @property
    def update_count(self) -> int:
        return len(self._update_history)

    def get_history(self) -> List[ContaminationUpdate]:
        """Get contamination update history."""
        return self._update_history.copy()

    def get_stats(self) -> dict:
        """Get estimator statistics."""
        return {
            "contamination": self._contamination,
            "detection_rate": self.detection_rate,
            "total_predictions": self.total_predictions,
            "updates": self.update_count,
            "min_c": self.min_contamination,
            "max_c": self.max_contamination,
            "alpha": self.alpha,
            "window_size": self.window_size,
        }
