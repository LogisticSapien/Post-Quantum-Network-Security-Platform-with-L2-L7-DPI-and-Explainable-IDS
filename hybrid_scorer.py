"""
Hybrid Anomaly Scorer
=======================
Weighted ensemble fusing Isolation Forest structural anomaly score
with EWMA+z-score volumetric drift into a single combined alert score.

XDR-style correlation: combines two independent detection signals
(isolation-based and statistical) for more robust detection.

Math:
  Combined score: S = w₁·s_iforest + w₂·σ(max_z / z_norm)
  where:
    - s_iforest ∈ [0, 1]: Isolation Forest anomaly score
    - max_z: maximum absolute z-score across all EWMA-tracked metrics
    - z_norm: normalization factor (default: 5.0, maps z=5 → σ ≈ 0.99)
    - σ(x) = 1 / (1 + e^(-x)): sigmoid squashing z-score to [0, 1]
    - w₁, w₂: weights summing to 1.0 (default: 0.6, 0.4)

  Recall-biased threshold calibration:
    Minimize: Σ pinball_loss(yᵢ, ŷᵢ) where:
      pinball_loss = α·|yᵢ - ŷᵢ| if yᵢ > ŷᵢ (false negative)
                   = (1-α)·|yᵢ - ŷᵢ| if yᵢ ≤ ŷᵢ (false positive)
    α > 0.5 penalizes FN more heavily (default: α = 0.8)

Design rationale:
  iForest catches structural anomalies (unusual feature combinations),
  EWMA catches volumetric drift (traffic spikes/drops). Fusing both
  reduces blind spots: iForest may miss gradual volume changes,
  EWMA may miss novel attack patterns at normal volume.

Constraints: NumPy only, no sklearn.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from temporal_scorer import TemporalCorrelationLayer, TemporalAlert


# ──────────────────────────────────────────────────────────────────────
# Utility
# ──────────────────────────────────────────────────────────────────────

def _sigmoid(x: float) -> float:
    """Numerically stable sigmoid function.

    σ(x) = 1 / (1 + e^(-x))

    Clamped to prevent overflow in exp().
    """
    x = max(-500.0, min(500.0, x))
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        ex = math.exp(x)
        return ex / (1.0 + ex)


# ──────────────────────────────────────────────────────────────────────
# Hybrid Score Result
# ──────────────────────────────────────────────────────────────────────

@dataclass
class HybridScore:
    """Result of hybrid scoring for a single sample/window."""
    combined_score: float
    iforest_score: float
    ewma_component: float
    max_z_score: float
    temporal_boost: float
    is_anomaly: bool
    threshold: float

    @property
    def breakdown(self) -> str:
        """Human-readable score breakdown."""
        return (
            f"combined={self.combined_score:.3f} "
            f"(iforest={self.iforest_score:.3f}, "
            f"ewma={self.ewma_component:.3f}, "
            f"z_max={self.max_z_score:.2f}, "
            f"temporal={self.temporal_boost:.3f}) "
            f"threshold={self.threshold:.3f} "
            f"→ {'ANOMALY' if self.is_anomaly else 'normal'}"
        )


# ──────────────────────────────────────────────────────────────────────
# Hybrid Scorer
# ──────────────────────────────────────────────────────────────────────

class HybridScorer:
    """Weighted ensemble fusing iForest + EWMA z-score + temporal trend.

    Produces a single combined anomaly score in [0, 1] that integrates:
      1. Isolation Forest structural anomaly score
      2. EWMA z-score volumetric deviation (sigmoid-normalized)
      3. Optional temporal trend boost from TemporalCorrelationLayer

    The threshold is calibrated to prioritize recall (catching threats)
    over precision (avoiding false positives).

    Args:
        iforest_weight: Weight for iForest score component (default: 0.55).
        ewma_weight: Weight for EWMA z-score component (default: 0.30).
        temporal_weight: Weight for temporal trend component (default: 0.15).
        z_norm: Normalization factor for z-score sigmoid (default: 5.0).
        threshold: Initial decision threshold (default: 0.45).
        recall_bias: Alpha for recall-biased calibration, >0.5 penalizes FN
            (default: 0.8, meaning FN costs 4× more than FP).
        temporal_layer: Optional TemporalCorrelationLayer for trend boost.
    """

    def __init__(
        self,
        iforest_weight: float = 0.55,
        ewma_weight: float = 0.30,
        temporal_weight: float = 0.15,
        z_norm: float = 5.0,
        threshold: float = 0.45,
        recall_bias: float = 0.8,
        temporal_layer: Optional[TemporalCorrelationLayer] = None,
    ):
        # Normalize weights to sum to 1.0
        total_w = iforest_weight + ewma_weight + temporal_weight
        self.iforest_weight = iforest_weight / total_w
        self.ewma_weight = ewma_weight / total_w
        self.temporal_weight = temporal_weight / total_w

        self.z_norm = z_norm
        self._threshold = threshold
        self.recall_bias = recall_bias
        self.temporal_layer = temporal_layer

        # Score history for threshold calibration
        self._score_history: List[float] = []
        self._label_history: List[int] = []  # For supervised calibration if available

    def score(
        self,
        iforest_score: float,
        z_scores: Dict[str, float],
        flow_key: Optional[str] = None,
        timestamp: Optional[float] = None,
    ) -> HybridScore:
        """Compute the combined hybrid anomaly score.

        Args:
            iforest_score: Anomaly score from Isolation Forest ∈ [0, 1].
            z_scores: Dict mapping metric names to their z-scores from EWMA.
            flow_key: Optional flow identifier for temporal tracking.
            timestamp: Optional timestamp for temporal buffer.

        Returns:
            HybridScore with combined score and component breakdown.
        """
        # EWMA component: sigmoid of max absolute z-score
        if z_scores:
            max_z = max(abs(z) for z in z_scores.values())
        else:
            max_z = 0.0

        ewma_component = _sigmoid(max_z / self.z_norm * 5.0 - 2.5)
        # Maps: z=0 → ~0.08, z=2.5 → ~0.5, z=5 → ~0.92

        # Temporal component
        temporal_boost = 0.0
        if self.temporal_layer and flow_key:
            flow_stats = self.temporal_layer.get_flow_stats(flow_key)
            if flow_stats and flow_stats["size"] >= 3:
                # Temporal boost based on trend slope and recent mean
                trend = flow_stats["trend"]
                recent_mean = flow_stats["recent_mean"]
                if trend > 0:
                    temporal_boost = min(1.0, trend * 20.0) * recent_mean

        # Combined score
        combined = (
            self.iforest_weight * iforest_score
            + self.ewma_weight * ewma_component
            + self.temporal_weight * temporal_boost
        )

        # Clamp to [0, 1]
        combined = max(0.0, min(1.0, combined))

        # Record for calibration
        self._score_history.append(combined)

        # Feed to temporal layer if available
        if self.temporal_layer and flow_key:
            self.temporal_layer.record_score(flow_key, combined, timestamp)

        return HybridScore(
            combined_score=combined,
            iforest_score=iforest_score,
            ewma_component=ewma_component,
            max_z_score=max_z,
            temporal_boost=temporal_boost,
            is_anomaly=combined >= self._threshold,
            threshold=self._threshold,
        )

    def score_batch(
        self,
        iforest_scores: np.ndarray,
        z_score_arrays: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Batch scoring for evaluation (no temporal component).

        Args:
            iforest_scores: Array of iForest scores, shape (n_samples,).
            z_score_arrays: Optional array of max z-scores, shape (n_samples,).
                If None, only iForest scores are used.

        Returns:
            Combined scores, shape (n_samples,).
        """
        n = len(iforest_scores)
        combined = self.iforest_weight * iforest_scores

        if z_score_arrays is not None:
            ewma_components = np.array([
                _sigmoid(z / self.z_norm * 5.0 - 2.5) for z in z_score_arrays
            ])
            combined = combined + self.ewma_weight * ewma_components

        return np.clip(combined, 0.0, 1.0)

    def calibrate_threshold(
        self,
        scores: np.ndarray,
        y_true: np.ndarray,
        n_thresholds: int = 200,
    ) -> float:
        """Calibrate threshold to maximize recall with acceptable precision.

        Uses pinball loss / asymmetric cost:
          cost = α · FN + (1-α) · FP

        Where α = recall_bias > 0.5 penalizes false negatives more.

        Equivalent to finding threshold that maximizes:
          recall_bias · recall + (1 - recall_bias) · precision

        Args:
            scores: Anomaly scores, shape (n_samples,).
            y_true: True binary labels, shape (n_samples,). 1 = anomaly.
            n_thresholds: Number of threshold candidates to evaluate.

        Returns:
            Optimal threshold.
        """
        thresholds = np.linspace(scores.min(), scores.max(), n_thresholds)
        best_score = -1.0
        best_thresh = 0.5

        for thresh in thresholds:
            y_pred = (scores >= thresh).astype(int)
            tp = np.sum((y_true == 1) & (y_pred == 1))
            fp = np.sum((y_true == 0) & (y_pred == 1))
            fn = np.sum((y_true == 1) & (y_pred == 0))

            recall = tp / max(tp + fn, 1)
            precision = tp / max(tp + fp, 1)

            # Weighted F-score with recall bias
            weighted = self.recall_bias * recall + (1 - self.recall_bias) * precision
            if weighted > best_score:
                best_score = weighted
                best_thresh = float(thresh)

        self._threshold = best_thresh
        return best_thresh

    def calibrate_from_contamination(
        self,
        scores: np.ndarray,
        contamination: float = 0.2,
    ) -> float:
        """Calibrate threshold from expected contamination ratio.

        Sets threshold at the (1 - contamination) percentile of scores,
        then reduces it by a recall-bias factor to catch more anomalies.

        Args:
            scores: Anomaly scores, shape (n_samples,).
            contamination: Expected proportion of anomalies.

        Returns:
            Calibrated threshold.
        """
        base_threshold = float(np.percentile(scores, 100 * (1 - contamination)))

        # Reduce threshold to bias toward recall
        # recall_bias=0.8 → reduce by 20% → catches more anomalies
        reduction = (1 - self.recall_bias) * 0.5
        self._threshold = base_threshold * (1 - reduction)
        return self._threshold

    @property
    def threshold(self) -> float:
        """Current decision threshold."""
        return self._threshold

    @threshold.setter
    def threshold(self, value: float) -> None:
        self._threshold = value

    def get_params(self) -> dict:
        """Get scorer parameters."""
        return {
            "iforest_weight": self.iforest_weight,
            "ewma_weight": self.ewma_weight,
            "temporal_weight": self.temporal_weight,
            "z_norm": self.z_norm,
            "threshold": self._threshold,
            "recall_bias": self.recall_bias,
            "score_history_size": len(self._score_history),
        }
