"""
Dynamic Conformal Predictor — Production Grade
==================================================
Sliding-window recalibrating conformal anomaly predictor with
statistically rigorous distribution drift detection.

Replaces the static holdout calibration in conformal_predictor.py
with a live rolling calibration buffer that adapts as the network
baseline evolves.

Key capabilities:
  1. Rolling calibration buffer of confirmed-normal anomaly scores
  2. Automatic recalibration every K confirmed-normal observations
  3. BASELINE_DRIFT alert via two-sample Kolmogorov-Smirnov test
     and Page-Hinkley change detection
  4. Thread-safe for concurrent packet processing
  5. Prometheus-metric-ready (qs_conformal_* naming convention)
  6. Batch prediction mode for offline evaluation

Theory:
  P-value computation (Vovk, Gammerman & Shafer, 2005):
    p(x) = (|{i : α(zᵢ) ≥ α(x)}| + 1) / (n + 1)

  The calibration set Z is a sliding window of recent confirmed-normal
  scores. A flow is confirmed normal when its anomaly score remains
  below `low_threshold` for `confirm_streak` consecutive windows.

  Drift detection:
    - Primary: Two-sample KS test on recent vs. historical calibration
      scores. D_KS > critical_value at α=0.01 triggers BASELINE_DRIFT.
    - Secondary: Page-Hinkley test on running mean of calibration scores.
      Detects gradual mean shift that KS may miss at small sample sizes.

Constraints: NumPy only, thread-safe, no sklearn.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Result Dataclass
# ──────────────────────────────────────────────────────────────────────

@dataclass
class DynamicConformalResult:
    """Result from dynamic conformal prediction."""
    p_value: float
    anomaly_score: float
    is_anomaly: bool
    significance_level: float
    confidence: float              # 1 - p_value
    calibration_size: int
    calibration_generation: int    # how many recalibrations have occurred

    @property
    def summary(self) -> str:
        label = "ANOMALY" if self.is_anomaly else "normal"
        return (
            f"p={self.p_value:.4f}, score={self.anomaly_score:.3f}, "
            f"conf={self.confidence:.4f}, ε={self.significance_level}, "
            f"cal_size={self.calibration_size}, gen={self.calibration_generation} "
            f"→ {label}"
        )


# ──────────────────────────────────────────────────────────────────────
# Drift Alert
# ──────────────────────────────────────────────────────────────────────

@dataclass
class BaselineDriftAlert:
    """Alert emitted when calibration distribution has shifted.

    Triggers the engine state machine to reconsider its operating mode:
    BASELINE → WATCH if drift is detected during normal operations.
    """
    drift_type: str            # "KS_TEST" or "PAGE_HINKLEY" or "STDDEV_RATIO"
    statistic: float           # Test statistic value
    critical_value: float      # Threshold exceeded
    p_value: float             # Statistical p-value (for KS test)
    message: str
    timestamp: float
    severity: str = "WARNING"  # "WARNING" or "CRITICAL"

    @property
    def is_critical(self) -> bool:
        return self.severity == "CRITICAL"


# ──────────────────────────────────────────────────────────────────────
# Statistical Tests (pure NumPy)
# ──────────────────────────────────────────────────────────────────────

def _ks_two_sample(sample1: np.ndarray, sample2: np.ndarray) -> Tuple[float, float]:
    """Two-sample Kolmogorov-Smirnov test statistic and approximate p-value.

    Tests H0: sample1 and sample2 are drawn from the same distribution.

    The KS statistic D = max |F1(x) - F2(x)| where F1, F2 are the
    empirical CDFs. Under H0, √(n·m/(n+m)) · D converges to the
    Kolmogorov distribution.

    Args:
        sample1: First sample, shape (n,).
        sample2: Second sample, shape (m,).

    Returns:
        (D_statistic, p_value_approx)
    """
    n1 = len(sample1)
    n2 = len(sample2)
    if n1 == 0 or n2 == 0:
        return 0.0, 1.0

    # Compute empirical CDFs
    all_vals = np.sort(np.concatenate([sample1, sample2]))
    cdf1 = np.searchsorted(np.sort(sample1), all_vals, side='right') / n1
    cdf2 = np.searchsorted(np.sort(sample2), all_vals, side='right') / n2

    D = float(np.max(np.abs(cdf1 - cdf2)))

    # Approximate p-value via asymptotic formula
    # P(D > d) ≈ 2 * exp(-2 * (d * sqrt(n*m/(n+m)))^2)
    en = math.sqrt(n1 * n2 / (n1 + n2))
    lambda_val = (en + 0.12 + 0.11 / en) * D

    if lambda_val <= 0:
        p_val = 1.0
    else:
        # Kolmogorov distribution approximation (first 4 terms)
        p_val = 0.0
        for k in range(1, 5):
            p_val += 2.0 * ((-1.0) ** (k - 1)) * math.exp(-2.0 * k * k * lambda_val * lambda_val)
        p_val = max(0.0, min(1.0, p_val))

    return D, p_val


class _PageHinkleyDetector:
    """Page-Hinkley change detection for gradual mean shift.

    Detects when the running mean of a signal shifts by more than
    `delta` from its historical mean. Uses cumulative sum of
    deviations from expected mean.

    PH(t) = Σᵢ (xᵢ - x̄ₜ - δ)
    m(t)  = min PH(1..t)
    Alert when PH(t) - m(t) > threshold

    Args:
        delta: Minimum detectable mean shift (default: 0.01).
        threshold: Cumulative deviation threshold (default: 10.0).
        alpha: EWMA smoothing for running mean (default: 0.01).
    """

    def __init__(
        self,
        delta: float = 0.01,
        threshold: float = 10.0,
        alpha: float = 0.01,
    ):
        self.delta = delta
        self.threshold = threshold
        self.alpha = alpha

        self._sum: float = 0.0
        self._min_sum: float = float('inf')
        self._mean: float = 0.0
        self._n: int = 0

    def update(self, x: float) -> bool:
        """Feed a new observation and check for change.

        Args:
            x: New observation (anomaly score).

        Returns:
            True if change detected.
        """
        self._n += 1

        if self._n == 1:
            self._mean = x
            self._sum = 0.0
            self._min_sum = 0.0
            return False

        # Update running mean
        self._mean = self.alpha * x + (1 - self.alpha) * self._mean

        # Page-Hinkley statistic
        self._sum += x - self._mean - self.delta
        self._min_sum = min(self._min_sum, self._sum)

        return (self._sum - self._min_sum) > self.threshold

    def reset(self):
        """Reset the detector after a drift event."""
        self._sum = 0.0
        self._min_sum = float('inf')
        self._n = 0
        self._mean = 0.0

    @property
    def deviation(self) -> float:
        """Current PH deviation (distance from minimum)."""
        if self._n < 2:
            return 0.0
        return self._sum - self._min_sum


# ──────────────────────────────────────────────────────────────────────
# Dynamic Conformal Predictor
# ──────────────────────────────────────────────────────────────────────

class DynamicConformalPredictor:
    """Production-grade sliding-window conformal anomaly predictor.

    Maintains a rolling buffer of confirmed-normal anomaly scores
    and recalibrates the p-value computation periodically. Detects
    distribution drift via KS test + Page-Hinkley and emits
    BASELINE_DRIFT alerts.

    Interface:
      result = predictor.score(anomaly_score)
      drift  = predictor.observe_normal(score)
      drift  = predictor.observe_flow_score(flow_key, score)

    Prometheus metrics (qs_conformal_* convention):
      qs_conformal_predictions_total
      qs_conformal_anomalies_total
      qs_conformal_normals_observed_total
      qs_conformal_recalibrations_total
      qs_conformal_drift_alerts_total
      qs_conformal_calibration_size
      qs_conformal_drift_ks_statistic

    Args:
        buffer_size: Maximum calibration buffer size (default: 1000).
        recalibrate_every: Recalibrate after K new normals (default: 50).
        significance_level: ε threshold for anomaly (default: 0.05).
        low_threshold: Score below which flow is potentially normal (default: 0.35).
        confirm_streak: Consecutive low-score windows to confirm normal (default: 3).
        ks_alpha: Significance level for KS drift test (default: 0.01).
        ph_delta: Page-Hinkley minimum mean shift (default: 0.005).
        ph_threshold: Page-Hinkley alert threshold (default: 15.0).
        max_flow_streaks: Max tracked flow streaks before LRU eviction (default: 10000).
    """

    def __init__(
        self,
        buffer_size: int = 1000,
        recalibrate_every: int = 50,
        significance_level: float = 0.05,
        low_threshold: float = 0.35,
        confirm_streak: int = 3,
        ks_alpha: float = 0.01,
        ph_delta: float = 0.005,
        ph_threshold: float = 15.0,
        max_flow_streaks: int = 10_000,
    ):
        self.buffer_size = buffer_size
        self.recalibrate_every = recalibrate_every
        self.significance_level = significance_level
        self.low_threshold = low_threshold
        self.confirm_streak = confirm_streak
        self.ks_alpha = ks_alpha

        # Calibration state (protected by _lock)
        self._calibration_buffer: Deque[float] = deque(maxlen=buffer_size)
        self._sorted_scores: Optional[np.ndarray] = None
        self._lock = threading.RLock()
        self._calibration_generation: int = 0
        self._pending_normals: int = 0

        # Historical calibration snapshot for drift detection
        self._historical_snapshot: Optional[np.ndarray] = None
        self._snapshot_generation: int = 0
        self._snapshot_interval: int = 5  # snapshot every N recalibrations

        # Drift detection
        self._ph_detector = _PageHinkleyDetector(
            delta=ph_delta,
            threshold=ph_threshold,
        )
        self._drift_alerts: List[BaselineDriftAlert] = []
        self._last_drift_time: float = 0.0
        self._drift_cooldown: float = 60.0  # seconds between drift alerts

        # Per-flow streak tracking
        self._flow_streaks: Dict[str, int] = {}
        self._max_flow_streaks = max_flow_streaks

        # Prometheus-style metrics
        self._metrics = {
            "qs_conformal_predictions_total": 0,
            "qs_conformal_anomalies_total": 0,
            "qs_conformal_normals_observed_total": 0,
            "qs_conformal_recalibrations_total": 0,
            "qs_conformal_drift_alerts_total": 0,
            "qs_conformal_calibration_size": 0,
            "qs_conformal_drift_ks_statistic": 0.0,
        }

        # Recalibration daemon
        self._recal_event = threading.Event()
        self._running = True
        self._daemon = threading.Thread(
            target=self._recalibration_daemon,
            daemon=True,
            name="DynConformalRecal",
        )
        self._daemon.start()

    # ──────────────────────────────────────────────────────────────
    # Core: P-value Scoring
    # ──────────────────────────────────────────────────────────────

    def score(self, anomaly_score: float) -> DynamicConformalResult:
        """Compute conformal p-value for an anomaly score.

        p(x) = (|{i : α(zᵢ) ≥ α(x)}| + 1) / (n + 1)

        Thread-safe. Returns conservative (p=1, not anomaly) if
        calibration buffer is empty.

        Args:
            anomaly_score: Nonconformity score (higher = more anomalous).

        Returns:
            DynamicConformalResult with p-value and decision.
        """
        with self._lock:
            n = len(self._calibration_buffer)

            if n == 0:
                self._metrics["qs_conformal_predictions_total"] += 1
                return DynamicConformalResult(
                    p_value=1.0,
                    anomaly_score=anomaly_score,
                    is_anomaly=False,
                    significance_level=self.significance_level,
                    confidence=0.0,
                    calibration_size=0,
                    calibration_generation=self._calibration_generation,
                )

            # Binary search in sorted calibration array
            if self._sorted_scores is not None and len(self._sorted_scores) == n:
                n_geq = n - int(np.searchsorted(
                    self._sorted_scores, anomaly_score, side='left'
                ))
            else:
                # Fallback: linear scan
                n_geq = sum(1 for s in self._calibration_buffer if s >= anomaly_score)

            p_value = (n_geq + 1) / (n + 1)
            is_anomaly = p_value <= self.significance_level
            gen = self._calibration_generation

        self._metrics["qs_conformal_predictions_total"] += 1
        if is_anomaly:
            self._metrics["qs_conformal_anomalies_total"] += 1

        return DynamicConformalResult(
            p_value=p_value,
            anomaly_score=anomaly_score,
            is_anomaly=is_anomaly,
            significance_level=self.significance_level,
            confidence=1.0 - p_value,
            calibration_size=n,
            calibration_generation=gen,
        )

    def score_batch(self, scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Batch p-value computation (no online update, no drift check).

        For offline evaluation. Converts anomaly scores to p-values
        without modifying the calibration buffer.

        Args:
            scores: Anomaly scores, shape (n_samples,).

        Returns:
            (p_values, is_anomaly): both shape (n_samples,).
        """
        with self._lock:
            n = len(self._calibration_buffer)
            if n == 0:
                return np.ones_like(scores), np.zeros_like(scores, dtype=int)
            sorted_cal = self._sorted_scores if self._sorted_scores is not None else np.sort(list(self._calibration_buffer))

        p_values = np.zeros(len(scores))
        for i, s in enumerate(scores):
            n_geq = n - int(np.searchsorted(sorted_cal, s, side='left'))
            p_values[i] = (n_geq + 1) / (n + 1)

        is_anomaly = (p_values <= self.significance_level).astype(int)
        self._metrics["qs_conformal_predictions_total"] += len(scores)
        self._metrics["qs_conformal_anomalies_total"] += int(is_anomaly.sum())

        return p_values, is_anomaly

    # ──────────────────────────────────────────────────────────────
    # Calibration Buffer Management
    # ──────────────────────────────────────────────────────────────

    def observe_normal(self, score: float) -> Optional[BaselineDriftAlert]:
        """Feed a confirmed-normal anomaly score into calibration buffer.

        Triggers recalibration when enough normals have been observed.
        Checks for distribution drift after each addition.

        Args:
            score: Anomaly score confirmed as normal.

        Returns:
            BaselineDriftAlert if drift detected, None otherwise.
        """
        drift_alert = None

        with self._lock:
            self._calibration_buffer.append(score)
            self._pending_normals += 1
            self._metrics["qs_conformal_normals_observed_total"] += 1
            self._metrics["qs_conformal_calibration_size"] = len(self._calibration_buffer)

            # Signal recalibration daemon
            if self._pending_normals >= self.recalibrate_every:
                self._recal_event.set()

            # Page-Hinkley check
            ph_changed = self._ph_detector.update(score)

        # Drift checks (outside lock to avoid blocking)
        if ph_changed:
            drift_alert = self._emit_ph_drift()

        # Periodic KS test (every recalibration generation)
        if self._should_ks_test():
            ks_alert = self._run_ks_test()
            if ks_alert is not None:
                drift_alert = ks_alert  # KS overrides PH

        return drift_alert

    def observe_flow_score(
        self,
        flow_key: str,
        anomaly_score: float,
    ) -> Optional[BaselineDriftAlert]:
        """Track per-flow scores and auto-feed confirmed normals.

        A flow is confirmed normal when it has scored below
        `low_threshold` for `confirm_streak` consecutive windows.

        Args:
            flow_key: Flow identifier string.
            anomaly_score: Current anomaly score for this flow.

        Returns:
            BaselineDriftAlert if drift is detected, None otherwise.
        """
        if anomaly_score < self.low_threshold:
            streak = self._flow_streaks.get(flow_key, 0) + 1
            self._flow_streaks[flow_key] = streak

            if streak >= self.confirm_streak:
                # Confirmed normal — feed to calibration
                return self.observe_normal(anomaly_score)
        else:
            # Reset streak on any elevated score
            self._flow_streaks[flow_key] = 0

        # LRU eviction on flow streaks
        if len(self._flow_streaks) > self._max_flow_streaks:
            # Remove oldest entries (arbitrary, not true LRU, but bounded)
            excess = len(self._flow_streaks) - self._max_flow_streaks
            keys_to_remove = list(self._flow_streaks.keys())[:excess]
            for k in keys_to_remove:
                del self._flow_streaks[k]

        return None

    def seed_calibration(self, scores: np.ndarray) -> None:
        """Seed calibration buffer with initial baseline scores.

        Use during startup with training/baseline data.

        Args:
            scores: Array of normal anomaly scores.
        """
        with self._lock:
            for s in scores.flatten():
                self._calibration_buffer.append(float(s))
            self._sorted_scores = np.sort(list(self._calibration_buffer))
            self._calibration_generation += 1
            self._pending_normals = 0

            # Take initial historical snapshot
            self._historical_snapshot = self._sorted_scores.copy()
            self._snapshot_generation = self._calibration_generation

            self._metrics["qs_conformal_calibration_size"] = len(self._calibration_buffer)
            self._metrics["qs_conformal_recalibrations_total"] += 1

        logger.info(
            f"Calibration seeded with {len(scores)} scores, "
            f"mean={np.mean(scores):.4f}, std={np.std(scores):.4f}"
        )

    # ──────────────────────────────────────────────────────────────
    # Drift Detection
    # ──────────────────────────────────────────────────────────────

    def _should_ks_test(self) -> bool:
        """Check if it's time to run a KS test."""
        with self._lock:
            if self._historical_snapshot is None:
                return False
            return (
                self._calibration_generation > 0
                and self._calibration_generation % self._snapshot_interval == 0
                and self._calibration_generation != self._snapshot_generation
            )

    def _run_ks_test(self) -> Optional[BaselineDriftAlert]:
        """Run two-sample KS test: recent calibration vs. historical snapshot.

        Returns BaselineDriftAlert if distribution shift is significant.
        """
        now = time.time()
        if now - self._last_drift_time < self._drift_cooldown:
            return None

        with self._lock:
            if self._historical_snapshot is None:
                return None
            recent = np.array(list(self._calibration_buffer))
            historical = self._historical_snapshot.copy()

        if len(recent) < 30 or len(historical) < 30:
            return None

        D, ks_pval = _ks_two_sample(recent, historical)
        self._metrics["qs_conformal_drift_ks_statistic"] = D

        # KS critical value at given alpha
        # D_crit ≈ c(α) * sqrt((n+m)/(n*m))
        # c(0.01) ≈ 1.628, c(0.05) ≈ 1.358
        c_alpha = 1.628 if self.ks_alpha <= 0.01 else 1.358
        n1, n2 = len(recent), len(historical)
        D_crit = c_alpha * math.sqrt((n1 + n2) / (n1 * n2))

        if D > D_crit or ks_pval < self.ks_alpha:
            severity = "CRITICAL" if D > 2 * D_crit else "WARNING"

            alert = BaselineDriftAlert(
                drift_type="KS_TEST",
                statistic=D,
                critical_value=D_crit,
                p_value=ks_pval,
                message=(
                    f"BASELINE_DRIFT: Two-sample KS test detected distribution shift. "
                    f"D={D:.4f} > D_crit={D_crit:.4f} (α={self.ks_alpha}), "
                    f"p={ks_pval:.6f}. Recent calibration ({len(recent)} scores) "
                    f"differs from historical baseline ({len(historical)} scores). "
                    f"Network baseline may have shifted — consider retraining."
                ),
                timestamp=now,
                severity=severity,
            )

            self._drift_alerts.append(alert)
            self._metrics["qs_conformal_drift_alerts_total"] += 1
            self._last_drift_time = now
            logger.warning(alert.message)

            # Update historical snapshot to current (so we detect NEXT drift)
            with self._lock:
                self._historical_snapshot = np.sort(list(self._calibration_buffer))
                self._snapshot_generation = self._calibration_generation

            return alert

        return None

    def _emit_ph_drift(self) -> Optional[BaselineDriftAlert]:
        """Emit drift alert from Page-Hinkley detector."""
        now = time.time()
        if now - self._last_drift_time < self._drift_cooldown:
            return None

        deviation = self._ph_detector.deviation

        alert = BaselineDriftAlert(
            drift_type="PAGE_HINKLEY",
            statistic=deviation,
            critical_value=self._ph_detector.threshold,
            p_value=0.0,  # PH doesn't produce a p-value
            message=(
                f"BASELINE_DRIFT: Page-Hinkley detected gradual mean shift "
                f"in calibration scores. Cumulative deviation={deviation:.4f} "
                f"> threshold={self._ph_detector.threshold}. "
                f"Network traffic pattern may be slowly changing."
            ),
            timestamp=now,
            severity="WARNING",
        )

        self._drift_alerts.append(alert)
        self._metrics["qs_conformal_drift_alerts_total"] += 1
        self._last_drift_time = now
        self._ph_detector.reset()

        logger.warning(alert.message)
        return alert

    # ──────────────────────────────────────────────────────────────
    # Recalibration Daemon
    # ──────────────────────────────────────────────────────────────

    def _recalibration_daemon(self):
        """Background thread that re-sorts calibration scores.

        Wakes up when enough new normals have been observed,
        sorts the calibration buffer for O(log n) p-value lookup,
        and takes periodic historical snapshots for drift detection.
        """
        while self._running:
            self._recal_event.wait(timeout=5.0)
            self._recal_event.clear()

            with self._lock:
                if self._pending_normals >= self.recalibrate_every:
                    cal_list = list(self._calibration_buffer)
                    self._sorted_scores = np.sort(cal_list)
                    self._calibration_generation += 1
                    self._pending_normals = 0
                    gen = self._calibration_generation

                    self._metrics["qs_conformal_recalibrations_total"] += 1

                    # Periodic snapshot for KS test
                    if (self._historical_snapshot is None or
                            gen % self._snapshot_interval == 0):
                        if self._historical_snapshot is None:
                            self._historical_snapshot = self._sorted_scores.copy()
                            self._snapshot_generation = gen

                    logger.debug(
                        f"Recalibrated: gen={gen}, "
                        f"size={len(cal_list)}, "
                        f"mean={np.mean(cal_list):.4f}"
                    )

    # ──────────────────────────────────────────────────────────────
    # Properties & Metrics
    # ──────────────────────────────────────────────────────────────

    @property
    def is_calibrated(self) -> bool:
        """Whether the calibration buffer has any scores."""
        with self._lock:
            return len(self._calibration_buffer) > 0

    @property
    def calibration_size(self) -> int:
        with self._lock:
            return len(self._calibration_buffer)

    @property
    def calibration_generation(self) -> int:
        with self._lock:
            return self._calibration_generation

    @property
    def drift_alerts(self) -> List[BaselineDriftAlert]:
        """All drift alerts emitted so far."""
        return list(self._drift_alerts)

    @property
    def drift_count(self) -> int:
        return len(self._drift_alerts)

    @property
    def calibration_statistics(self) -> dict:
        """Current calibration buffer statistics."""
        with self._lock:
            if not self._calibration_buffer:
                return {"mean": 0.0, "std": 0.0, "min": 0.0,
                        "max": 0.0, "size": 0}
            scores = np.array(list(self._calibration_buffer))
        return {
            "mean": float(np.mean(scores)),
            "std": float(np.std(scores)),
            "min": float(np.min(scores)),
            "max": float(np.max(scores)),
            "median": float(np.median(scores)),
            "size": len(scores),
        }

    def get_metrics(self) -> Dict[str, float]:
        """Get Prometheus-ready metric counters.

        All metric names follow qs_<component>_<metric> convention.
        """
        self._metrics["qs_conformal_calibration_size"] = self.calibration_size
        return dict(self._metrics)

    def get_stats(self) -> dict:
        """Get comprehensive predictor statistics."""
        cal_stats = self.calibration_statistics
        return {
            "calibration_size": cal_stats["size"],
            "calibration_mean": cal_stats.get("mean", 0.0),
            "calibration_std": cal_stats.get("std", 0.0),
            "calibration_generation": self._calibration_generation,
            "significance_level": self.significance_level,
            "total_predictions": self._metrics["qs_conformal_predictions_total"],
            "total_anomalies": self._metrics["qs_conformal_anomalies_total"],
            "total_normals_observed": self._metrics["qs_conformal_normals_observed_total"],
            "total_recalibrations": self._metrics["qs_conformal_recalibrations_total"],
            "drift_alerts": len(self._drift_alerts),
            "ks_statistic": self._metrics["qs_conformal_drift_ks_statistic"],
            "ph_deviation": self._ph_detector.deviation,
            "active_flow_streaks": len(self._flow_streaks),
            "buffer_size": self.buffer_size,
        }

    def stop(self):
        """Stop the recalibration daemon thread."""
        self._running = False
        self._recal_event.set()
        if self._daemon.is_alive():
            self._daemon.join(timeout=2.0)

    def reset(self):
        """Full reset of calibration state (keeps configuration)."""
        with self._lock:
            self._calibration_buffer.clear()
            self._sorted_scores = None
            self._calibration_generation = 0
            self._pending_normals = 0
            self._historical_snapshot = None
            self._snapshot_generation = 0
        self._ph_detector.reset()
        self._drift_alerts.clear()
        self._flow_streaks.clear()
        for k in self._metrics:
            self._metrics[k] = 0 if isinstance(self._metrics[k], int) else 0.0
        logger.info("DynamicConformalPredictor reset")

    def __del__(self):
        """Cleanup daemon thread on garbage collection."""
        self.stop()
