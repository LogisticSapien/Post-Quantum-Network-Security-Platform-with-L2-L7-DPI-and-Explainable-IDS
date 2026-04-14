"""
Isolation Forest Network Attack Detector
==========================================
Network-aware wrapper around the pure-Python Isolation Forest.
Extracts traffic features from packet metadata, builds baselines,
and detects zero-day attack patterns in real-time.

Feature extraction pipeline:
  1. Collect per-packet metadata (protocol, size, ports, IPs)
  2. Aggregate into time-window features (rates, distributions, entropy)
  3. Feed feature vectors to Isolation Forest for anomaly scoring
  4. Emit ThreatEvents with full explainability on detection

Integrates with quantum_sniffer's existing IDS/alert infrastructure.
"""

from __future__ import annotations

import math
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from isolation_forest import IsolationForest
from ids import Severity, ThreatEvent, EvidenceFactor, MITRE
from temporal_scorer import TemporalCorrelationLayer
from hybrid_scorer import HybridScorer


# ──────────────────────────────────────────────────────────────────────
# Feature Names (for explainability & visualization)
# ──────────────────────────────────────────────────────────────────────

FEATURE_NAMES = [
    "packet_rate",          # packets per second
    "byte_rate",            # bytes per second
    "avg_packet_size",      # average packet size in bytes
    "unique_src_ips",       # unique source IPs in window
    "unique_dst_ips",       # unique destination IPs in window
    "unique_dst_ports",     # unique destination ports in window
    "tcp_ratio",            # TCP packets as fraction of total
    "udp_ratio",            # UDP packets as fraction of total
    "dns_ratio",            # DNS packets as fraction of total
    "icmp_ratio",           # ICMP packets as fraction of total
    "syn_ratio",            # SYN-flagged packets as fraction of TCP
    "connection_rate",      # new connections per second
    "port_entropy",         # Shannon entropy of destination port distribution
    "ip_entropy",           # Shannon entropy of source IP distribution
]

NUM_FEATURES = len(FEATURE_NAMES)


# ──────────────────────────────────────────────────────────────────────
# Utility
# ──────────────────────────────────────────────────────────────────────

def _shannon_entropy(counts: Dict) -> float:
    """Compute Shannon entropy from a frequency distribution.

    H = -Σ p(x) · log2(p(x))

    Args:
        counts: Dictionary mapping items to their counts.

    Returns:
        Shannon entropy in bits.
    """
    total = sum(counts.values())
    if total == 0:
        return 0.0
    entropy = 0.0
    for count in counts.values():
        if count > 0:
            p = count / total
            entropy -= p * math.log2(p)
    return entropy


# ──────────────────────────────────────────────────────────────────────
# Per-Window Packet Accumulator
# ──────────────────────────────────────────────────────────────────────

@dataclass
class WindowAccumulator:
    """Accumulates packet metadata within a single time window.

    Produces a feature vector summarizing the traffic pattern
    observed during the window.
    """
    start_time: float = 0.0
    packet_count: int = 0
    total_bytes: int = 0

    # Protocol counts
    tcp_count: int = 0
    udp_count: int = 0
    dns_count: int = 0
    icmp_count: int = 0
    syn_count: int = 0

    # Unique tracking
    src_ips: Set[str] = field(default_factory=set)
    dst_ips: Set[str] = field(default_factory=set)
    dst_ports: Set[int] = field(default_factory=set)
    connections: Set[Tuple[str, int]] = field(default_factory=set)

    # Distributions for entropy
    dst_port_counts: Dict[int, int] = field(default_factory=lambda: defaultdict(int))
    src_ip_counts: Dict[str, int] = field(default_factory=lambda: defaultdict(int))

    def record(
        self,
        protocol: str,
        src_ip: str,
        dst_ip: str,
        size: int,
        src_port: int = 0,
        dst_port: int = 0,
        is_syn: bool = False,
    ):
        """Record a single packet."""
        self.packet_count += 1
        self.total_bytes += size

        proto_upper = protocol.upper()
        if proto_upper in ("TCP", "TLS", "HTTP", "SSH"):
            self.tcp_count += 1
        elif proto_upper == "UDP":
            self.udp_count += 1
        elif proto_upper == "DNS":
            self.dns_count += 1
        elif proto_upper == "ICMP":
            self.icmp_count += 1

        if is_syn:
            self.syn_count += 1

        self.src_ips.add(src_ip)
        self.dst_ips.add(dst_ip)
        if dst_port > 0:
            self.dst_ports.add(dst_port)
            self.dst_port_counts[dst_port] += 1
        self.src_ip_counts[src_ip] += 1

        if src_ip and dst_port > 0:
            self.connections.add((src_ip, dst_port))

    def to_feature_vector(self, window_duration: float) -> np.ndarray:
        """Convert accumulated stats to a feature vector.

        Args:
            window_duration: Duration of the window in seconds.

        Returns:
            Feature vector of shape (NUM_FEATURES,).
        """
        duration = max(window_duration, 0.1)  # avoid division by zero
        total = max(self.packet_count, 1)
        tcp_total = max(self.tcp_count, 1)

        features = np.array([
            self.packet_count / duration,                  # packet_rate
            self.total_bytes / duration,                   # byte_rate
            self.total_bytes / total,                      # avg_packet_size
            len(self.src_ips),                             # unique_src_ips
            len(self.dst_ips),                             # unique_dst_ips
            len(self.dst_ports),                           # unique_dst_ports
            self.tcp_count / total,                        # tcp_ratio
            self.udp_count / total,                        # udp_ratio
            self.dns_count / total,                        # dns_ratio
            self.icmp_count / total,                       # icmp_ratio
            self.syn_count / tcp_total if self.tcp_count > 0 else 0,  # syn_ratio
            len(self.connections) / duration,               # connection_rate
            _shannon_entropy(self.dst_port_counts),        # port_entropy
            _shannon_entropy(self.src_ip_counts),          # ip_entropy
        ], dtype=np.float64)

        return features

    def reset(self):
        """Reset all accumulators for a new window."""
        self.start_time = time.time()
        self.packet_count = 0
        self.total_bytes = 0
        self.tcp_count = 0
        self.udp_count = 0
        self.dns_count = 0
        self.icmp_count = 0
        self.syn_count = 0
        self.src_ips.clear()
        self.dst_ips.clear()
        self.dst_ports.clear()
        self.connections.clear()
        self.dst_port_counts.clear()
        self.src_ip_counts.clear()


# ──────────────────────────────────────────────────────────────────────
# iForest Network Detector
# ──────────────────────────────────────────────────────────────────────

class IForestNetworkDetector:
    """Network attack detector using Isolation Forest.

    Operates in two phases:
      1. BASELINE (training): Collects feature vectors from normal traffic
         across `min_training_windows` time windows.
      2. DETECTION: Runs Isolation Forest on each new window's features.
         Emits ThreatEvent alerts for anomalous windows.

    Parameters:
        window_seconds: Duration of each feature-extraction window (default: 30).
        min_training_windows: Minimum windows before training (default: 20).
        n_estimators: Number of Isolation Trees (default: 100).
        max_samples: Subsample size per tree (default: 256).
        contamination: Expected anomaly ratio (default: 0.05).
        cooldown_seconds: Minimum time between alerts (default: 60).
    """

    def __init__(
        self,
        window_seconds: float = 30.0,
        min_training_windows: int = 20,
        n_estimators: int = 100,
        max_samples: int = 256,
        contamination: float = 0.015,
        cooldown_seconds: float = 60.0,
    ):
        self.window_seconds = window_seconds
        self.min_training_windows = min_training_windows
        self.cooldown_seconds = cooldown_seconds

        # Isolation Forest
        self._forest = IsolationForest(
            n_estimators=n_estimators,
            max_samples=max_samples,
            contamination=contamination,
        )

        # Temporal Correlation Layer (Tier 1) — catches slow-building threats
        self._temporal_layer = TemporalCorrelationLayer(
            buffer_size=10,
            trend_threshold=0.015,
            baseline_factor=0.35,
        )

        # Hybrid Scorer (Tier 1) — fuses iForest + EWMA z-score
        self._hybrid_scorer = HybridScorer(
            iforest_weight=0.55,
            ewma_weight=0.30,
            temporal_weight=0.15,
            recall_bias=0.8,
            temporal_layer=self._temporal_layer,
        )

        # State
        self._current_window = WindowAccumulator()
        self._current_window.start_time = time.time()
        self._training_data: List[np.ndarray] = []
        self._is_trained = False
        self._last_alert_time: float = 0.0
        self._total_windows_processed: int = 0
        self._total_anomalies_detected: int = 0

        # Feature statistics (for normalization & explainability)
        self._feature_means: Optional[np.ndarray] = None
        self._feature_stds: Optional[np.ndarray] = None

    @property
    def is_trained(self) -> bool:
        return self._is_trained

    @property
    def status(self) -> str:
        if not self._is_trained:
            collected = len(self._training_data)
            return f"BASELINE ({collected}/{self.min_training_windows} windows)"
        return f"ACTIVE (processed: {self._total_windows_processed}, anomalies: {self._total_anomalies_detected})"

    def record_packet(
        self,
        protocol: str,
        src_ip: str,
        dst_ip: str,
        size: int,
        src_port: int = 0,
        dst_port: int = 0,
        is_syn: bool = False,
    ) -> Optional[ThreatEvent]:
        """Feed a packet to the detector.

        If the current window has elapsed, processes it and
        either adds to training data or runs detection.

        Returns:
            ThreatEvent if an anomaly is detected, None otherwise.
        """
        now = time.time()
        self._current_window.record(
            protocol, src_ip, dst_ip, size, src_port, dst_port, is_syn
        )

        # Check if window has elapsed
        elapsed = now - self._current_window.start_time
        if elapsed < self.window_seconds:
            return None

        # Window complete → extract features
        features = self._current_window.to_feature_vector(elapsed)
        self._total_windows_processed += 1

        result = None

        if not self._is_trained:
            # Baseline phase: collect training data
            self._training_data.append(features)
            if len(self._training_data) >= self.min_training_windows:
                self._train()
        else:
            # Detection phase
            result = self._detect(features, now)

        # Reset window
        self._current_window.reset()

        return result

    def _train(self):
        """Train the Isolation Forest on collected baseline data."""
        X_train = np.vstack(self._training_data)

        # Store feature statistics for normalization
        self._feature_means = X_train.mean(axis=0)
        self._feature_stds = X_train.std(axis=0)
        self._feature_stds[self._feature_stds < 1e-10] = 1.0  # avoid div by zero

        # Normalize training data
        X_normalized = (X_train - self._feature_means) / self._feature_stds

        # Fit the forest
        self._forest.fit(X_normalized)
        self._is_trained = True

    def _detect(self, features: np.ndarray, now: float) -> Optional[ThreatEvent]:
        """Run detection on a feature vector.

        Uses the hybrid scoring pipeline:
          1. iForest raw score
          2. EWMA z-scores from feature deviations
          3. Temporal trend from TemporalCorrelationLayer
          4. Combined via HybridScorer

        Args:
            features: Raw feature vector from the current window.
            now: Current timestamp.

        Returns:
            ThreatEvent if anomalous, None otherwise.
        """
        # Normalize using training statistics
        normalized = (features - self._feature_means) / self._feature_stds
        normalized = normalized.reshape(1, -1)

        # Get iForest anomaly score
        iforest_score = float(self._forest.anomaly_scores(normalized)[0])

        # Compute feature-level z-scores for EWMA component
        deviations = (features - self._feature_means) / self._feature_stds
        z_scores = {FEATURE_NAMES[i]: float(deviations[i]) for i in range(len(FEATURE_NAMES))}

        # Generate flow key from window features (use top IPs as proxy)
        flow_key = f"window_{self._total_windows_processed}"

        # Hybrid score: fuses iForest + EWMA + temporal trend
        hybrid_result = self._hybrid_scorer.score(
            iforest_score=iforest_score,
            z_scores=z_scores,
            flow_key=flow_key,
            timestamp=now,
        )

        score = hybrid_result.combined_score

        if not hybrid_result.is_anomaly:
            return None

        # Cooldown check
        if now - self._last_alert_time < self.cooldown_seconds:
            return None

        self._last_alert_time = now
        self._total_anomalies_detected += 1

        # Determine severity based on combined score
        if score >= 0.75:
            severity = Severity.CRITICAL
        elif score >= 0.65:
            severity = Severity.HIGH
        elif score >= 0.55:
            severity = Severity.MEDIUM
        else:
            severity = Severity.LOW

        # Identify top contributing features
        deviations = np.abs(
            (features - self._feature_means) / self._feature_stds
        )
        top_features = np.argsort(deviations)[::-1][:5]

        evidence = []
        anomaly_details = []
        for idx in top_features:
            fname = FEATURE_NAMES[idx]
            val = features[idx]
            baseline = self._feature_means[idx]
            std = self._feature_stds[idx]
            z = deviations[idx]
            direction = "above" if val > baseline else "below"
            evidence.append(EvidenceFactor(
                fname,
                f"{val:.2f}",
                f"baseline: {baseline:.2f} ± {std:.2f}",
                min(1.0, z / 5.0),
            ))
            anomaly_details.append(
                f"{fname}={val:.1f} ({z:.1f}σ {direction} baseline)"
            )

        return ThreatEvent(
            timestamp=now,
            severity=severity,
            category="IFOREST_ANOMALY",
            description=(
                f"Hybrid scorer detected anomalous network pattern "
                f"(combined={score:.3f}, iforest={iforest_score:.3f}, "
                f"threshold={self._hybrid_scorer.threshold:.3f})"
            ),
            confidence=score,
            mitre_ref="T1595 - Active Scanning / T1499 - Endpoint DoS",
            explanation=(
                f"The hybrid detection pipeline (iForest + EWMA + temporal trend) "
                f"detected an anomalous traffic pattern "
                f"in the last {self.window_seconds:.0f}s window. "
                f"Combined score: {score:.3f} "
                f"(iForest={iforest_score:.3f}, "
                f"EWMA={hybrid_result.ewma_component:.3f}, "
                f"temporal={hybrid_result.temporal_boost:.3f}). "
                f"Top anomalous features: {'; '.join(anomaly_details[:3])}. "
                f"This pattern deviates significantly from the learned baseline "
                f"of {len(self._training_data)} training windows. "
                f"Potential zero-day attack or unusual network behavior."
            ),
            evidence_factors=evidence,
            detection_logic=(
                f"Hybrid scorer: iForest (trees={self._forest.n_estimators}, "
                f"ψ={self._forest._psi}) + EWMA z-score + temporal trend. "
                f"Weights: iF={self._hybrid_scorer.iforest_weight:.2f}, "
                f"ewma={self._hybrid_scorer.ewma_weight:.2f}, "
                f"temp={self._hybrid_scorer.temporal_weight:.2f}. "
                f"Window: {self.window_seconds}s. "
                f"Combined score > threshold → anomaly."
            ),
            response_actions=[
                "Investigate traffic pattern in the flagged time window",
                "Check top anomalous features for attack indicators",
                "Cross-reference with IDS rule-based alerts",
                "Consider blocking suspicious source IPs if confirmed",
                "Review network logs for correlated events",
            ],
        )

    def get_stats(self) -> dict:
        """Get detector statistics."""
        return {
            "status": self.status,
            "is_trained": self._is_trained,
            "windows_processed": self._total_windows_processed,
            "anomalies_detected": self._total_anomalies_detected,
            "training_windows": len(self._training_data),
            "model_params": self._forest.get_params() if self._is_trained else None,
        }

    def force_train(self, X: np.ndarray):
        """Force-train the model on external data.

        Useful for pre-training on historical data or for the demo.

        Args:
            X: Training data of shape (n_samples, n_features).
        """
        self._feature_means = X.mean(axis=0)
        self._feature_stds = X.std(axis=0)
        self._feature_stds[self._feature_stds < 1e-10] = 1.0

        X_normalized = (X - self._feature_means) / self._feature_stds
        self._forest.fit(X_normalized)
        self._is_trained = True
        self._training_data = list(X)
