"""
Per-Flow Feature Tracking
===========================
Tracks per-flow feature vectors keyed by (src_ip, dst_ip, dst_port)
tuples over time. Catches slow exfiltration that looks normal
window-by-window but is anomalous at the flow level.

Replaces per-window aggregation with per-flow vectors so the
iForest can score individual flows rather than aggregate windows.

Key flow-level features tracked:
  - Total bytes transferred
  - Packet count
  - Duration (first packet → most recent)
  - Average inter-arrival time
  - Bytes per packet (average packet size)
  - Packets per second (flow-level rate)
  - Bytes per second (flow-level throughput)
  - Byte ratio (outbound / total)

Uses LRU eviction with configurable max_flows (default 10,000)
to bound memory usage in long-running captures.

Constraints: NumPy only, no sklearn.
"""

from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, NamedTuple

import numpy as np


# ──────────────────────────────────────────────────────────────────────
# Flow Key
# ──────────────────────────────────────────────────────────────────────

class FlowKey(NamedTuple):
    """Identifies a network flow by source, destination, and port."""
    src_ip: str
    dst_ip: str
    dst_port: int

    def __str__(self) -> str:
        return f"{self.src_ip}->{self.dst_ip}:{self.dst_port}"


# ──────────────────────────────────────────────────────────────────────
# Per-Flow State
# ──────────────────────────────────────────────────────────────────────

FLOW_FEATURE_NAMES = [
    "total_bytes",
    "packet_count",
    "duration",
    "avg_inter_arrival",
    "bytes_per_packet",
    "packets_per_second",
    "bytes_per_second",
    "byte_ratio_out",
]

NUM_FLOW_FEATURES = len(FLOW_FEATURE_NAMES)


@dataclass
class FlowState:
    """Tracks per-flow statistics over time.

    Updated with each packet belonging to this flow.
    Can produce a feature vector for iForest scoring.
    """
    first_seen: float = 0.0
    last_seen: float = 0.0
    total_bytes: int = 0
    total_bytes_out: int = 0  # Bytes from src to dst
    packet_count: int = 0
    packet_count_out: int = 0  # Packets from src to dst

    # Inter-arrival tracking
    _last_packet_time: float = 0.0
    _inter_arrival_sum: float = 0.0
    _inter_arrival_count: int = 0

    # Protocol tracking
    syn_count: int = 0
    protocols_seen: set = field(default_factory=set)

    def record_packet(
        self,
        timestamp: float,
        size: int,
        is_outbound: bool = True,
        protocol: str = "",
        is_syn: bool = False,
    ) -> None:
        """Record a packet belonging to this flow.

        Args:
            timestamp: Unix timestamp.
            size: Packet size in bytes.
            is_outbound: True if packet goes from src to dst.
            protocol: Protocol name.
            is_syn: Whether this is a SYN packet.
        """
        if self.packet_count == 0:
            self.first_seen = timestamp
            self._last_packet_time = timestamp

        # Inter-arrival time
        if self.packet_count > 0:
            iat = timestamp - self._last_packet_time
            if iat > 0:
                self._inter_arrival_sum += iat
                self._inter_arrival_count += 1

        self.last_seen = timestamp
        self._last_packet_time = timestamp
        self.total_bytes += size
        self.packet_count += 1

        if is_outbound:
            self.total_bytes_out += size
            self.packet_count_out += 1

        if is_syn:
            self.syn_count += 1

        if protocol:
            self.protocols_seen.add(protocol)

    @property
    def duration(self) -> float:
        """Flow duration in seconds."""
        return max(self.last_seen - self.first_seen, 0.001)

    @property
    def avg_inter_arrival(self) -> float:
        """Average inter-arrival time in seconds."""
        if self._inter_arrival_count == 0:
            return 0.0
        return self._inter_arrival_sum / self._inter_arrival_count

    @property
    def bytes_per_packet(self) -> float:
        """Average bytes per packet."""
        if self.packet_count == 0:
            return 0.0
        return self.total_bytes / self.packet_count

    @property
    def packets_per_second(self) -> float:
        """Packet rate."""
        return self.packet_count / self.duration

    @property
    def bytes_per_second(self) -> float:
        """Byte rate / throughput."""
        return self.total_bytes / self.duration

    @property
    def byte_ratio_out(self) -> float:
        """Fraction of bytes that are outbound (src→dst)."""
        if self.total_bytes == 0:
            return 0.5
        return self.total_bytes_out / self.total_bytes

    def to_feature_vector(self) -> np.ndarray:
        """Generate a feature vector for iForest scoring.

        Returns:
            Feature vector of shape (NUM_FLOW_FEATURES,).
        """
        return np.array([
            self.total_bytes,
            self.packet_count,
            self.duration,
            self.avg_inter_arrival,
            self.bytes_per_packet,
            self.packets_per_second,
            self.bytes_per_second,
            self.byte_ratio_out,
        ], dtype=np.float64)

    def to_dict(self) -> dict:
        """Summary dict for inspection."""
        return {
            "total_bytes": self.total_bytes,
            "packet_count": self.packet_count,
            "duration": round(self.duration, 2),
            "avg_iat": round(self.avg_inter_arrival, 4),
            "bpp": round(self.bytes_per_packet, 1),
            "pps": round(self.packets_per_second, 1),
            "bps": round(self.bytes_per_second, 1),
            "byte_ratio_out": round(self.byte_ratio_out, 3),
            "protocols": list(self.protocols_seen),
        }


# ──────────────────────────────────────────────────────────────────────
# Flow Feature Tracker
# ──────────────────────────────────────────────────────────────────────

@dataclass
class FlowAlert:
    """Alert from flow-level anomaly detection."""
    flow_key: FlowKey
    anomaly_score: float
    flow_features: dict
    message: str


class FlowFeatureTracker:
    """Manages per-flow state with LRU eviction.

    Tracks individual network flows and can score them with
    an iForest model for flow-level anomaly detection.

    Args:
        max_flows: Maximum tracked flows before LRU eviction (default: 10000).
        stale_timeout: Seconds before a flow is considered stale (default: 600).
        min_packets: Minimum packets before a flow is scored (default: 5).
    """

    def __init__(
        self,
        max_flows: int = 10000,
        stale_timeout: float = 600.0,
        min_packets: int = 5,
    ):
        self.max_flows = max_flows
        self.stale_timeout = stale_timeout
        self.min_packets = min_packets

        # LRU ordered dict: most recent at end
        self._flows: OrderedDict[FlowKey, FlowState] = OrderedDict()
        self._total_evicted: int = 0
        self._total_scored: int = 0

        # Normalization stats (set during calibrate or first batch)
        self._feature_means: Optional[np.ndarray] = None
        self._feature_stds: Optional[np.ndarray] = None

    def record_packet(
        self,
        src_ip: str,
        dst_ip: str,
        dst_port: int,
        size: int,
        timestamp: Optional[float] = None,
        is_outbound: bool = True,
        protocol: str = "",
        is_syn: bool = False,
    ) -> FlowKey:
        """Record a packet and update the corresponding flow.

        Args:
            src_ip: Source IP address.
            dst_ip: Destination IP address.
            dst_port: Destination port.
            size: Packet size in bytes.
            timestamp: Unix timestamp (default: now).
            is_outbound: Whether packet flows from src to dst.
            protocol: Protocol name.
            is_syn: Whether this is a SYN packet.

        Returns:
            FlowKey of the recorded flow.
        """
        if timestamp is None:
            timestamp = time.time()

        key = FlowKey(src_ip, dst_ip, dst_port)

        if key in self._flows:
            self._flows.move_to_end(key)
            state = self._flows[key]
        else:
            self._evict_if_needed(timestamp)
            state = FlowState()
            self._flows[key] = state

        state.record_packet(timestamp, size, is_outbound, protocol, is_syn)
        return key

    def get_flow_features(self, key: FlowKey) -> Optional[np.ndarray]:
        """Get the feature vector for a specific flow.

        Args:
            key: FlowKey.

        Returns:
            Feature vector or None if flow not found.
        """
        state = self._flows.get(key)
        if state is None:
            return None
        return state.to_feature_vector()

    def get_scoreable_flows(self) -> List[Tuple[FlowKey, np.ndarray]]:
        """Get all flows with enough packets for scoring.

        Returns:
            List of (FlowKey, feature_vector) tuples.
        """
        result = []
        for key, state in self._flows.items():
            if state.packet_count >= self.min_packets:
                result.append((key, state.to_feature_vector()))
        return result

    def calibrate(self, feature_matrix: np.ndarray) -> None:
        """Set normalization statistics from training data.

        Args:
            feature_matrix: Training flow features, shape (n_flows, NUM_FLOW_FEATURES).
        """
        self._feature_means = feature_matrix.mean(axis=0)
        self._feature_stds = feature_matrix.std(axis=0)
        self._feature_stds[self._feature_stds < 1e-10] = 1.0

    def normalize_features(self, features: np.ndarray) -> np.ndarray:
        """Normalize features using stored statistics.

        Args:
            features: Raw features, shape (n, NUM_FLOW_FEATURES) or (NUM_FLOW_FEATURES,).

        Returns:
            Normalized features.
        """
        if self._feature_means is None:
            return features

        return (features - self._feature_means) / self._feature_stds

    def score_flow(
        self,
        key: FlowKey,
        forest,
    ) -> Optional[float]:
        """Score a single flow using an iForest model.

        Args:
            key: FlowKey to score.
            forest: IsolationForest or ExtendedIsolationForest instance.

        Returns:
            Anomaly score or None if flow not found / insufficient data.
        """
        state = self._flows.get(key)
        if state is None or state.packet_count < self.min_packets:
            return None

        features = state.to_feature_vector().reshape(1, -1)
        features = self.normalize_features(features)

        score = float(forest.anomaly_scores(features)[0])
        self._total_scored += 1
        return score

    def score_all_flows(self, forest) -> List[Tuple[FlowKey, float]]:
        """Score all eligible flows.

        Args:
            forest: IsolationForest or ExtendedIsolationForest instance.

        Returns:
            List of (FlowKey, anomaly_score) sorted by score descending.
        """
        scoreable = self.get_scoreable_flows()
        if not scoreable:
            return []

        keys = [k for k, _ in scoreable]
        features = np.vstack([f for _, f in scoreable])
        features = self.normalize_features(features)

        scores = forest.anomaly_scores(features)
        self._total_scored += len(scores)

        results = list(zip(keys, scores.tolist()))
        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def _evict_if_needed(self, now: float) -> None:
        """Evict stale/LRU flows if at capacity."""
        # Evict stale
        stale_keys = [
            key for key, state in self._flows.items()
            if now - state.last_seen > self.stale_timeout
        ]
        for key in stale_keys:
            del self._flows[key]
            self._total_evicted += 1

        # LRU eviction
        while len(self._flows) >= self.max_flows:
            self._flows.popitem(last=False)
            self._total_evicted += 1

    def cleanup_stale(self, now: Optional[float] = None) -> int:
        """Remove stale flows.

        Args:
            now: Current timestamp.

        Returns:
            Number of flows evicted.
        """
        if now is None:
            now = time.time()
        before = len(self._flows)
        stale_keys = [
            key for key, state in self._flows.items()
            if now - state.last_seen > self.stale_timeout
        ]
        for key in stale_keys:
            del self._flows[key]
            self._total_evicted += 1
        return before - len(self._flows)

    @property
    def active_flows(self) -> int:
        return len(self._flows)

    @property
    def total_evicted(self) -> int:
        return self._total_evicted

    @property
    def total_scored(self) -> int:
        return self._total_scored

    def get_stats(self) -> dict:
        return {
            "active_flows": self.active_flows,
            "total_evicted": self._total_evicted,
            "total_scored": self._total_scored,
            "max_flows": self.max_flows,
            "stale_timeout": self.stale_timeout,
        }

    def get_top_flows(self, n: int = 10) -> List[Tuple[FlowKey, dict]]:
        """Get top N flows by byte count.

        Args:
            n: Number of flows.

        Returns:
            List of (FlowKey, stats_dict).
        """
        flows = [
            (key, state.to_dict())
            for key, state in self._flows.items()
        ]
        flows.sort(key=lambda x: x[1]["total_bytes"], reverse=True)
        return flows[:n]
