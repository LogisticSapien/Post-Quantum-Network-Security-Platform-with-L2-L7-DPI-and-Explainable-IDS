"""
Statistical Anomaly Detection Engine
======================================
Behavior-based traffic anomaly detection using statistical models:
  - EWMA (Exponentially Weighted Moving Average) baseline tracker
  - Z-score deviation detector (flag deviations > 3 sigma)
  - Tracks: packets/sec, bytes/sec, unique IPs/sec, DNS rate, connection rate
  - Not ML — proper statistical models for production reliability

Design rationale:
  Statistical models (z-score, EWMA) over ML because:
  1. Deterministic and explainable
  2. No training data required
  3. Adapt to changing baselines automatically
  4. Zero false positives during learning phase
"""

from __future__ import annotations

import math
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from ids import Severity, ThreatEvent, EvidenceFactor, MITRE


@dataclass
class AnomalyAlert:
    """An anomaly detected by the statistical engine."""
    metric_name: str
    current_value: float
    baseline_mean: float
    baseline_std: float
    z_score: float
    timestamp: float
    severity: Severity


class EWMATracker:
    """
    Exponentially Weighted Moving Average tracker.

    Maintains a smoothed mean and variance for a time-series metric.
    Alpha controls how fast the baseline adapts:
      - alpha=0.1 → slow adaptation, stable baseline (best for detection)
      - alpha=0.5 → fast adaptation, responsive (best for volatile metrics)
    """

    def __init__(self, alpha: float = 0.1, warmup_samples: int = 20):
        self.alpha = alpha
        self.warmup_samples = warmup_samples
        self._mean: Optional[float] = None
        self._variance: float = 0.0
        self._count: int = 0
        self._warmup_values: List[float] = []

    def update(self, value: float) -> Tuple[float, float]:
        """
        Update the EWMA with a new observation.
        Returns (current_mean, current_std).
        """
        self._count += 1

        # Warmup phase: collect initial samples for a stable baseline
        if self._count <= self.warmup_samples:
            self._warmup_values.append(value)
            if self._count == self.warmup_samples:
                self._mean = sum(self._warmup_values) / len(self._warmup_values)
                if len(self._warmup_values) > 1:
                    self._variance = sum(
                        (v - self._mean) ** 2 for v in self._warmup_values
                    ) / (len(self._warmup_values) - 1)
                else:
                    self._variance = 0.0
            return (self._mean or value, math.sqrt(max(self._variance, 0)))

        if self._mean is None:
            self._mean = value
            return (self._mean, 0.0)

        # EWMA update
        diff = value - self._mean
        self._mean = self._mean + self.alpha * diff
        self._variance = (1 - self.alpha) * (self._variance + self.alpha * diff * diff)

        return (self._mean, math.sqrt(max(self._variance, 0)))

    @property
    def mean(self) -> float:
        return self._mean if self._mean is not None else 0.0

    @property
    def std(self) -> float:
        return math.sqrt(max(self._variance, 0))

    @property
    def is_warmed_up(self) -> bool:
        return self._count >= self.warmup_samples

    @property
    def sample_count(self) -> int:
        return self._count


class AnomalyDetector:
    """
    Statistical anomaly detector using EWMA + z-score.

    Tracks multiple metrics and flags deviations exceeding
    the configured z-score threshold (default: 3.0 = 99.7th percentile).
    """

    def __init__(
        self,
        z_threshold: float = 3.0,
        alpha: float = 0.1,
        warmup_samples: int = 20,
    ):
        self.z_threshold = z_threshold
        self.alpha = alpha
        self.warmup_samples = warmup_samples

        # Per-metric EWMA trackers
        self._trackers: Dict[str, EWMATracker] = {}
        self._alert_cooldowns: Dict[str, float] = {}
        self._cooldown_period: float = 10.0  # min seconds between alerts per metric
        self._total_anomalies: int = 0

    def _get_tracker(self, metric: str) -> EWMATracker:
        if metric not in self._trackers:
            self._trackers[metric] = EWMATracker(
                alpha=self.alpha, warmup_samples=self.warmup_samples
            )
        return self._trackers[metric]

    def update(self, metric: str, value: float) -> Optional[ThreatEvent]:
        """
        Feed a new metric observation. Returns a ThreatEvent if anomalous.

        Args:
            metric: Metric name (e.g., "packets_per_sec", "bytes_per_sec")
            value: Current observed value

        Returns:
            ThreatEvent if z-score exceeds threshold, None otherwise.
        """
        tracker = self._get_tracker(metric)
        mean, std = tracker.update(value)

        # Don't alert during warmup
        if not tracker.is_warmed_up:
            return None

        # Calculate z-score
        if std < 1e-10:
            return None  # No variance yet

        z_score = (value - mean) / std

        if abs(z_score) < self.z_threshold:
            return None

        # Cooldown check
        now = time.time()
        last_alert = self._alert_cooldowns.get(metric, 0)
        if now - last_alert < self._cooldown_period:
            return None

        self._alert_cooldowns[metric] = now
        self._total_anomalies += 1

        # Determine severity based on z-score magnitude
        abs_z = abs(z_score)
        if abs_z >= 5.0:
            severity = Severity.CRITICAL
        elif abs_z >= 4.0:
            severity = Severity.HIGH
        elif abs_z >= 3.0:
            severity = Severity.MEDIUM
        else:
            severity = Severity.LOW

        direction = "spike" if z_score > 0 else "drop"

        return ThreatEvent(
            timestamp=now,
            severity=severity,
            category="ANOMALY",
            description=(
                f"Traffic anomaly: {metric} {direction} detected. "
                f"Value={value:.1f}, baseline={mean:.1f} +/- {std:.1f}, "
                f"z-score={z_score:+.2f}"
            ),
            confidence=min(1.0, abs_z / 6.0),
            mitre_ref="T1499 - Endpoint Denial of Service",
            explanation=(
                f"Statistical anomaly detected in '{metric}'. "
                f"Current value ({value:.1f}) deviates {abs_z:.1f} standard deviations "
                f"from the EWMA baseline ({mean:.1f}). "
                f"The z-score threshold is {self.z_threshold} (99.7th percentile). "
                f"This {direction} may indicate a DDoS attack, scanning activity, "
                f"data exfiltration, or infrastructure failure. "
                f"Baseline was established over {tracker.sample_count} observations "
                f"with EWMA alpha={self.alpha}."
            ),
            evidence_factors=[
                EvidenceFactor(
                    "Z-score", f"{z_score:+.2f}",
                    f"threshold: +/-{self.z_threshold}", 0.5
                ),
                EvidenceFactor(
                    "Current value", f"{value:.1f}",
                    f"baseline: {mean:.1f}", 0.3
                ),
                EvidenceFactor(
                    "Baseline std", f"{std:.1f}",
                    f"samples: {tracker.sample_count}", 0.2
                ),
            ],
            detection_logic=(
                f"EWMA(alpha={self.alpha}) + z-score > {self.z_threshold}. "
                f"Warmup: {self.warmup_samples} samples."
            ),
            response_actions=[
                f"Investigate {direction} in {metric}",
                "Check for DDoS, scanning, or exfiltration activity",
                "Review source IPs contributing to the anomaly",
                "Consider temporary rate limiting if sustained",
            ],
        )

    def get_baselines(self) -> Dict[str, dict]:
        """Get current baselines for all tracked metrics."""
        return {
            name: {
                "mean": t.mean,
                "std": t.std,
                "samples": t.sample_count,
                "warmed_up": t.is_warmed_up,
            }
            for name, t in self._trackers.items()
        }

    @property
    def total_anomalies(self) -> int:
        return self._total_anomalies

    @property
    def tracked_metrics(self) -> List[str]:
        return list(self._trackers.keys())
