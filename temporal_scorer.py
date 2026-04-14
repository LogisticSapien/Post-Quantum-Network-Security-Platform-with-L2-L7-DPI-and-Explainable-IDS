"""
Temporal Correlation Layer
============================
Rolling buffer of anomaly scores per flow with trend detection.
Catches slow-building threats (stealthy exfiltration) that never
spike high enough to trigger single-window alerts.

Math:
  Trend β via ordinary least-squares on score buffer:
    β = Σ(tᵢ − t̄)(sᵢ − s̄) / Σ(tᵢ − t̄)²
  where tᵢ = relative timestamp, sᵢ = anomaly score at window i.

  Alert when:
    1. β > trend_threshold (scores are rising), AND
    2. mean(recent_scores) > baseline_factor (not trivially low)

Design rationale:
  Single-window iForest scoring misses slow-drip attacks where each
  window looks borderline-normal. By tracking the *trend* of scores
  over consecutive windows, we detect the gradual escalation pattern
  characteristic of stealthy data exfil, slow scans, and APT lateral
  movement.

Constraints: NumPy only, no sklearn.
"""

from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# ──────────────────────────────────────────────────────────────────────
# Score Entry
# ──────────────────────────────────────────────────────────────────────

@dataclass
class ScoreEntry:
    """A single timestamped anomaly score."""
    timestamp: float
    score: float


# ──────────────────────────────────────────────────────────────────────
# Per-Flow Score Buffer
# ──────────────────────────────────────────────────────────────────────

class TemporalScoreBuffer:
    """Fixed-size rolling buffer of anomaly scores for a single flow.

    Stores the last `max_size` (timestamp, score) pairs and computes
    temporal statistics: trend (OLS slope), mean, variance, max.

    Args:
        max_size: Maximum number of scores to retain (default: 10).
    """

    def __init__(self, max_size: int = 10):
        self.max_size = max_size
        self._buffer: List[ScoreEntry] = []
        self._last_access: float = time.time()

    def add(self, timestamp: float, score: float) -> None:
        """Add a new score to the buffer.

        Args:
            timestamp: Unix timestamp of the window.
            score: Anomaly score ∈ [0, 1].
        """
        self._buffer.append(ScoreEntry(timestamp=timestamp, score=score))
        if len(self._buffer) > self.max_size:
            self._buffer.pop(0)
        self._last_access = timestamp

    def trend(self) -> float:
        """Compute the OLS slope of scores over time.

        β = Σ(tᵢ − t̄)(sᵢ − s̄) / Σ(tᵢ − t̄)²

        Returns:
            Slope β. Positive = scores increasing over time.
            Returns 0.0 if fewer than 2 data points.
        """
        n = len(self._buffer)
        if n < 2:
            return 0.0

        # Use relative timestamps for numerical stability
        t0 = self._buffer[0].timestamp
        times = np.array([e.timestamp - t0 for e in self._buffer])
        scores = np.array([e.score for e in self._buffer])

        t_mean = times.mean()
        s_mean = scores.mean()

        numerator = np.sum((times - t_mean) * (scores - s_mean))
        denominator = np.sum((times - t_mean) ** 2)

        if abs(denominator) < 1e-12:
            return 0.0

        return float(numerator / denominator)

    def mean_score(self) -> float:
        """Average anomaly score in the buffer."""
        if not self._buffer:
            return 0.0
        return float(np.mean([e.score for e in self._buffer]))

    def max_score(self) -> float:
        """Maximum anomaly score in the buffer."""
        if not self._buffer:
            return 0.0
        return float(max(e.score for e in self._buffer))

    def variance(self) -> float:
        """Variance of scores in the buffer."""
        if len(self._buffer) < 2:
            return 0.0
        return float(np.var([e.score for e in self._buffer]))

    def recent_mean(self, n: int = 3) -> float:
        """Mean of the last n scores.

        Args:
            n: Number of recent scores to average.
        """
        if not self._buffer:
            return 0.0
        recent = self._buffer[-n:]
        return float(np.mean([e.score for e in recent]))

    @property
    def size(self) -> int:
        """Current number of scores in the buffer."""
        return len(self._buffer)

    @property
    def last_access(self) -> float:
        """Timestamp of last score addition."""
        return self._last_access

    @property
    def is_full(self) -> bool:
        """Whether the buffer has reached max_size."""
        return len(self._buffer) >= self.max_size


# ──────────────────────────────────────────────────────────────────────
# Temporal Correlation Result
# ──────────────────────────────────────────────────────────────────────

@dataclass
class TemporalAlert:
    """Alert from the temporal correlation layer."""
    flow_key: str
    trend_slope: float
    mean_score: float
    max_score: float
    recent_mean: float
    buffer_size: int
    message: str


# ──────────────────────────────────────────────────────────────────────
# Temporal Correlation Layer
# ──────────────────────────────────────────────────────────────────────

class TemporalCorrelationLayer:
    """Manages per-flow score buffers and detects rising threat trends.

    Tracks anomaly scores across consecutive time windows for each
    flow (keyed by string identifier). Flags flows where:
      1. Trend slope β > trend_threshold (scores are rising)
      2. Mean recent score > baseline_factor (not trivially low)

    This catches stealthy attacks that individually score below the
    single-window threshold but show a consistent upward pattern.

    Args:
        buffer_size: Max scores per flow buffer (default: 10).
        trend_threshold: Minimum OLS slope to flag as trending (default: 0.015).
            Units: score-per-second. For 30s windows, β=0.015 means
            score rises ~0.45 points over 30 seconds — significant.
        baseline_factor: Minimum mean score to be worth flagging (default: 0.35).
            Prevents alerting on trivially low scores that happen to trend up.
        max_flows: Maximum number of tracked flows before LRU eviction (default: 5000).
        stale_timeout: Seconds before a flow buffer is considered stale (default: 300).
        min_buffer_fill: Minimum buffer entries before trend evaluation (default: 3).
    """

    def __init__(
        self,
        buffer_size: int = 10,
        trend_threshold: float = 0.015,
        baseline_factor: float = 0.35,
        max_flows: int = 5000,
        stale_timeout: float = 300.0,
        min_buffer_fill: int = 3,
    ):
        self.buffer_size = buffer_size
        self.trend_threshold = trend_threshold
        self.baseline_factor = baseline_factor
        self.max_flows = max_flows
        self.stale_timeout = stale_timeout
        self.min_buffer_fill = min_buffer_fill

        # OrderedDict for LRU eviction (most recently accessed at end)
        self._flows: OrderedDict[str, TemporalScoreBuffer] = OrderedDict()
        self._total_alerts: int = 0

    def record_score(
        self,
        flow_key: str,
        score: float,
        timestamp: Optional[float] = None,
    ) -> Optional[TemporalAlert]:
        """Record an anomaly score for a flow and check for trends.

        Args:
            flow_key: Flow identifier string (e.g., "10.0.0.1->10.0.0.2:443").
            score: Anomaly score ∈ [0, 1] from iForest or hybrid scorer.
            timestamp: Unix timestamp (default: now).

        Returns:
            TemporalAlert if a rising trend is detected, None otherwise.
        """
        if timestamp is None:
            timestamp = time.time()

        # Get or create buffer
        if flow_key in self._flows:
            # Move to end (most recently used)
            self._flows.move_to_end(flow_key)
            buffer = self._flows[flow_key]
        else:
            # Evict LRU if at capacity
            self._evict_if_needed()
            buffer = TemporalScoreBuffer(max_size=self.buffer_size)
            self._flows[flow_key] = buffer

        buffer.add(timestamp, score)

        # Check for trend (only if buffer has enough data)
        if buffer.size < self.min_buffer_fill:
            return None

        return self._evaluate_trend(flow_key, buffer)

    def _evaluate_trend(
        self,
        flow_key: str,
        buffer: TemporalScoreBuffer,
    ) -> Optional[TemporalAlert]:
        """Evaluate whether a flow shows a rising anomaly trend.

        Args:
            flow_key: Flow identifier.
            buffer: Score buffer for this flow.

        Returns:
            TemporalAlert if conditions met, None otherwise.
        """
        trend = buffer.trend()
        mean = buffer.mean_score()
        recent = buffer.recent_mean(n=3)

        # Both conditions must be met:
        # 1. Positive trend exceeding threshold
        # 2. Mean score above baseline (not trivially low)
        if trend > self.trend_threshold and mean > self.baseline_factor:
            self._total_alerts += 1
            return TemporalAlert(
                flow_key=flow_key,
                trend_slope=trend,
                mean_score=mean,
                max_score=buffer.max_score(),
                recent_mean=recent,
                buffer_size=buffer.size,
                message=(
                    f"Rising anomaly trend for flow {flow_key}: "
                    f"slope={trend:.4f}/s, mean={mean:.3f}, "
                    f"recent={recent:.3f}, max={buffer.max_score():.3f} "
                    f"(buffer: {buffer.size}/{buffer.max_size})"
                ),
            )

        return None

    def _evict_if_needed(self) -> None:
        """Evict oldest/stale flows if at capacity."""
        now = time.time()

        # First pass: evict stale flows
        stale_keys = [
            key for key, buf in self._flows.items()
            if now - buf.last_access > self.stale_timeout
        ]
        for key in stale_keys:
            del self._flows[key]

        # LRU eviction if still over capacity
        while len(self._flows) >= self.max_flows:
            self._flows.popitem(last=False)  # Remove oldest

    def get_flow_stats(self, flow_key: str) -> Optional[dict]:
        """Get statistics for a specific flow.

        Args:
            flow_key: Flow identifier.

        Returns:
            Dict with trend, mean, max, size, or None if flow not tracked.
        """
        buffer = self._flows.get(flow_key)
        if buffer is None:
            return None

        return {
            "trend": buffer.trend(),
            "mean": buffer.mean_score(),
            "max": buffer.max_score(),
            "variance": buffer.variance(),
            "recent_mean": buffer.recent_mean(),
            "size": buffer.size,
            "last_access": buffer.last_access,
        }

    def get_top_trending_flows(self, n: int = 10) -> List[Tuple[str, float]]:
        """Get the top N flows by trend slope.

        Args:
            n: Number of top flows to return.

        Returns:
            List of (flow_key, trend_slope) tuples, descending by slope.
        """
        trends = [
            (key, buf.trend())
            for key, buf in self._flows.items()
            if buf.size >= self.min_buffer_fill
        ]
        trends.sort(key=lambda x: x[1], reverse=True)
        return trends[:n]

    def cleanup_stale(self, now: Optional[float] = None) -> int:
        """Remove stale flow buffers.

        Args:
            now: Current timestamp (default: time.time()).

        Returns:
            Number of flows evicted.
        """
        if now is None:
            now = time.time()

        stale_keys = [
            key for key, buf in self._flows.items()
            if now - buf.last_access > self.stale_timeout
        ]
        for key in stale_keys:
            del self._flows[key]

        return len(stale_keys)

    @property
    def active_flows(self) -> int:
        """Number of currently tracked flows."""
        return len(self._flows)

    @property
    def total_alerts(self) -> int:
        """Total temporal alerts emitted."""
        return self._total_alerts

    def get_stats(self) -> dict:
        """Get layer statistics."""
        return {
            "active_flows": self.active_flows,
            "total_alerts": self._total_alerts,
            "buffer_size": self.buffer_size,
            "trend_threshold": self.trend_threshold,
            "baseline_factor": self.baseline_factor,
            "max_flows": self.max_flows,
        }
