"""
Flow-Level Feature Extractor
================================
Extracts per-flow statistical features from Scapy packet objects
for ML model input. Replaces raw-byte feeding that caused iForest
to perform at random-classifier level.

Feature vector (14 features, CICIDS2017-aligned):
  0  flow_duration         — seconds from first to last packet
  1  fwd_packet_count      — packets src→dst
  2  bwd_packet_count      — packets dst→src
  3  fwd_bytes_total       — total bytes src→dst
  4  bwd_bytes_total       — total bytes dst→src
  5  fwd_pkt_len_mean      — mean forward packet length
  6  fwd_pkt_len_std       — std of forward packet lengths
  7  bwd_pkt_len_mean      — mean backward packet length
  8  bwd_pkt_len_std       — std of backward packet lengths
  9  flow_iat_mean         — mean inter-arrival time (all packets)
  10 flow_iat_std          — std of inter-arrival times
  11 syn_flag_count        — TCP SYN flag count
  12 ack_flag_count        — TCP ACK flag count
  13 packets_per_second    — total packets / duration

Bidirectional flow tracking:
  A flow is identified by a canonical 5-tuple: (min_ip, max_ip,
  min_port, max_port, protocol). This ensures src→dst and dst→src
  packets are merged into a single flow, with fwd/bwd direction
  determined by the original initiator (first packet seen).

Input:  List[scapy.packet.Packet] (live sniffer output)
Output: np.ndarray shape (n_flows, 14), z-score normalized

Constraints: Pure Python + NumPy, no sklearn.
"""

from __future__ import annotations

import logging
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, NamedTuple

import numpy as np

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────

FLOW_FEATURE_NAMES = [
    "flow_duration",
    "fwd_packet_count",
    "bwd_packet_count",
    "fwd_bytes_total",
    "bwd_bytes_total",
    "fwd_pkt_len_mean",
    "fwd_pkt_len_std",
    "bwd_pkt_len_mean",
    "bwd_pkt_len_std",
    "flow_iat_mean",
    "flow_iat_std",
    "syn_flag_count",
    "ack_flag_count",
    "packets_per_second",
]

NUM_FEATURES = len(FLOW_FEATURE_NAMES)  # 14

# TCP flag bitmasks
_TCP_SYN = 0x02
_TCP_ACK = 0x10

# Minimum packets to consider a flow valid for scoring
_MIN_FLOW_PACKETS = 2

# Epsilon for numerical stability
_EPS = 1e-10


# ──────────────────────────────────────────────────────────────────────
# Canonical Flow Key (bidirectional)
# ──────────────────────────────────────────────────────────────────────

class FlowKey(NamedTuple):
    """Canonical bidirectional flow identifier.

    Constructed so that (A→B) and (B→A) map to the same key.
    The 'initiator' (first packet's src) determines forward direction.
    """
    ip_low: str
    ip_high: str
    port_low: int
    port_high: int
    protocol: int  # 6=TCP, 17=UDP

    def __str__(self) -> str:
        proto = "TCP" if self.protocol == 6 else "UDP" if self.protocol == 17 else str(self.protocol)
        return f"{self.ip_low}:{self.port_low}<->{self.ip_high}:{self.port_high}/{proto}"


def _make_flow_key(
    src_ip: str, dst_ip: str,
    src_port: int, dst_port: int,
    protocol: int,
) -> Tuple[FlowKey, bool]:
    """Create a canonical flow key and determine direction.

    Returns:
        (FlowKey, is_forward): is_forward=True if src is the
        lexicographically/numerically smaller endpoint.
    """
    # Canonical ordering: sort by (ip, port)
    src_tuple = (src_ip, src_port)
    dst_tuple = (dst_ip, dst_port)

    if src_tuple <= dst_tuple:
        key = FlowKey(src_ip, dst_ip, src_port, dst_port, protocol)
        is_forward = True
    else:
        key = FlowKey(dst_ip, src_ip, dst_port, src_port, protocol)
        is_forward = False

    return key, is_forward


# ──────────────────────────────────────────────────────────────────────
# Per-Flow Accumulator
# ──────────────────────────────────────────────────────────────────────

@dataclass
class FlowAccumulator:
    """Accumulates per-flow statistics for feature extraction.

    Tracks forward (initiator→responder) and backward (responder→initiator)
    packet statistics independently. Produces the 14-feature vector
    matching the CICIDS2017 feature subset used by the ML models.
    """
    # Who initiated: first packet's src determines forward
    initiator_ip: str = ""
    initiator_port: int = 0

    # Timestamps
    first_seen: float = 0.0
    last_seen: float = 0.0

    # Forward direction (initiator → responder)
    fwd_packet_count: int = 0
    fwd_bytes_total: int = 0
    fwd_pkt_lengths: List[int] = field(default_factory=list)

    # Backward direction (responder → initiator)
    bwd_packet_count: int = 0
    bwd_bytes_total: int = 0
    bwd_pkt_lengths: List[int] = field(default_factory=list)

    # Inter-arrival times (all directions)
    _prev_timestamp: float = 0.0
    _iat_values: List[float] = field(default_factory=list)

    # TCP flags
    syn_flag_count: int = 0
    ack_flag_count: int = 0

    @property
    def total_packets(self) -> int:
        return self.fwd_packet_count + self.bwd_packet_count

    @property
    def duration(self) -> float:
        """Flow duration in seconds. Minimum 1µs for zero-duration flows."""
        d = self.last_seen - self.first_seen
        return max(d, 1e-6)

    def record_packet(
        self,
        timestamp: float,
        pkt_len: int,
        is_forward: bool,
        tcp_flags: int = 0,
    ) -> None:
        """Record a single packet into this flow.

        Args:
            timestamp: Packet arrival time (float, Unix epoch or relative).
            pkt_len: IP-layer payload length in bytes.
            is_forward: True if packet goes initiator→responder.
            tcp_flags: Raw TCP flag byte (0 for non-TCP).
        """
        # First packet → set initiator timing
        if self.total_packets == 0:
            self.first_seen = timestamp
            self._prev_timestamp = timestamp

        # Inter-arrival time (skip first packet)
        if self.total_packets > 0:
            iat = timestamp - self._prev_timestamp
            # Guard against negative IAT from re-ordered pcap
            if iat >= 0:
                self._iat_values.append(iat)

        self._prev_timestamp = timestamp
        self.last_seen = timestamp

        # Direction-specific stats
        if is_forward:
            self.fwd_packet_count += 1
            self.fwd_bytes_total += pkt_len
            self.fwd_pkt_lengths.append(pkt_len)
        else:
            self.bwd_packet_count += 1
            self.bwd_bytes_total += pkt_len
            self.bwd_pkt_lengths.append(pkt_len)

        # TCP flag counting
        if tcp_flags & _TCP_SYN:
            self.syn_flag_count += 1
        if tcp_flags & _TCP_ACK:
            self.ack_flag_count += 1

    def to_feature_vector(self) -> np.ndarray:
        """Extract the 14-dimensional feature vector.

        Returns:
            np.ndarray of shape (14,), dtype float64.
            Features are in the canonical order defined by FLOW_FEATURE_NAMES.
        """
        dur = self.duration

        # Forward packet length stats
        if self.fwd_pkt_lengths:
            fwd_arr = np.array(self.fwd_pkt_lengths, dtype=np.float64)
            fwd_mean = float(np.mean(fwd_arr))
            fwd_std = float(np.std(fwd_arr, ddof=0))
        else:
            fwd_mean = 0.0
            fwd_std = 0.0

        # Backward packet length stats
        if self.bwd_pkt_lengths:
            bwd_arr = np.array(self.bwd_pkt_lengths, dtype=np.float64)
            bwd_mean = float(np.mean(bwd_arr))
            bwd_std = float(np.std(bwd_arr, ddof=0))
        else:
            bwd_mean = 0.0
            bwd_std = 0.0

        # Inter-arrival time stats
        if self._iat_values:
            iat_arr = np.array(self._iat_values, dtype=np.float64)
            iat_mean = float(np.mean(iat_arr))
            iat_std = float(np.std(iat_arr, ddof=0))
        else:
            iat_mean = 0.0
            iat_std = 0.0

        # Packets per second
        pps = self.total_packets / dur

        return np.array([
            dur,                      # 0:  flow_duration
            self.fwd_packet_count,    # 1:  fwd_packet_count
            self.bwd_packet_count,    # 2:  bwd_packet_count
            self.fwd_bytes_total,     # 3:  fwd_bytes_total
            self.bwd_bytes_total,     # 4:  bwd_bytes_total
            fwd_mean,                 # 5:  fwd_pkt_len_mean
            fwd_std,                  # 6:  fwd_pkt_len_std
            bwd_mean,                 # 7:  bwd_pkt_len_mean
            bwd_std,                  # 8:  bwd_pkt_len_std
            iat_mean,                 # 9:  flow_iat_mean
            iat_std,                  # 10: flow_iat_std
            self.syn_flag_count,      # 11: syn_flag_count
            self.ack_flag_count,      # 12: ack_flag_count
            pps,                      # 13: packets_per_second
        ], dtype=np.float64)


# ──────────────────────────────────────────────────────────────────────
# Flow Feature Extractor (production interface)
# ──────────────────────────────────────────────────────────────────────

class FlowFeatureExtractor:
    """Extracts normalized flow-level features from Scapy packets.

    Dual-mode:
      1. LIVE: Call ingest_packet() per-packet, then flush_window()
         at 30s intervals to get feature matrix for ML models.
      2. OFFLINE: Call extract_from_packets() with a batch of Scapy
         packets (e.g., from rdpcap or PcapReader).

    Both modes output np.ndarray of shape (n_flows, 14), z-score
    normalized and ready for iForest/EIF/Autoencoder input.

    Args:
        window_seconds: Duration of each extraction window (default: 30.0).
        max_flows: LRU eviction threshold (default: 50000).
        min_packets_per_flow: Minimum packets to include flow (default: 2).
        normalize: Whether to z-score normalize output (default: True).
    """

    def __init__(
        self,
        window_seconds: float = 30.0,
        max_flows: int = 50_000,
        min_packets_per_flow: int = _MIN_FLOW_PACKETS,
        normalize: bool = True,
    ):
        self.window_seconds = window_seconds
        self.max_flows = max_flows
        self.min_packets_per_flow = min_packets_per_flow
        self.normalize = normalize

        # Active flows in current window
        self._flows: OrderedDict[FlowKey, FlowAccumulator] = OrderedDict()

        # Normalization stats (fitted on training/baseline data)
        self._feature_means: Optional[np.ndarray] = None
        self._feature_stds: Optional[np.ndarray] = None
        self._is_fitted: bool = False

        # Window timing
        self._window_start: float = 0.0

        # Prometheus-style counters
        self._metrics = {
            "qs_flow_extractor_packets_total": 0,
            "qs_flow_extractor_flows_total": 0,
            "qs_flow_extractor_flows_evicted": 0,
            "qs_flow_extractor_windows_total": 0,
            "qs_flow_extractor_malformed_packets": 0,
        }

    # ──────────────────────────────────────────────────────────────
    # Core: Packet Ingestion
    # ──────────────────────────────────────────────────────────────

    def ingest_packet(self, pkt) -> Optional[FlowKey]:
        """Ingest a single Scapy packet into the flow table.

        Safe against malformed packets — returns None on failure.

        Args:
            pkt: Scapy packet object (must have IP layer).

        Returns:
            FlowKey if packet was successfully recorded, None otherwise.
        """
        try:
            return self._parse_and_record(pkt)
        except Exception as e:
            self._metrics["qs_flow_extractor_malformed_packets"] += 1
            logger.debug(f"Malformed packet skipped: {e}")
            return None

    def _parse_and_record(self, pkt) -> Optional[FlowKey]:
        """Parse Scapy packet and record into flow table.

        Supports both Scapy packet objects (live sniffer) and
        pre-extracted tuples for testing.
        """
        # Lazy import — Scapy is only needed at runtime
        try:
            from scapy.layers.inet import IP, TCP, UDP
        except ImportError:
            raise ImportError(
                "Scapy is required for live packet ingestion. "
                "Install with: pip install scapy"
            )

        # Must have IP layer
        if not pkt.haslayer(IP):
            return None

        ip_layer = pkt[IP]
        src_ip = ip_layer.src
        dst_ip = ip_layer.dst
        proto = ip_layer.proto
        pkt_len = len(ip_layer)

        # Extract L4 info
        src_port = 0
        dst_port = 0
        tcp_flags = 0

        if pkt.haslayer(TCP):
            tcp_layer = pkt[TCP]
            src_port = tcp_layer.sport
            dst_port = tcp_layer.dport
            # Scapy stores flags as a FlagValue; convert to int
            tcp_flags = int(tcp_layer.flags)
            proto = 6
        elif pkt.haslayer(UDP):
            udp_layer = pkt[UDP]
            src_port = udp_layer.sport
            dst_port = udp_layer.dport
            proto = 17
        else:
            # Non-TCP/UDP (ICMP, etc.) — use proto=ip.proto, ports=0
            pass

        # Get timestamp
        if hasattr(pkt, 'time'):
            timestamp = float(pkt.time)
        else:
            timestamp = time.time()

        # Create canonical flow key
        flow_key, is_canonical_forward = _make_flow_key(
            src_ip, dst_ip, src_port, dst_port, proto
        )

        # Get or create flow accumulator
        if flow_key in self._flows:
            self._flows.move_to_end(flow_key)  # LRU touch
            acc = self._flows[flow_key]
        else:
            self._evict_if_needed()
            acc = FlowAccumulator(
                initiator_ip=src_ip,
                initiator_port=src_port,
            )
            self._flows[flow_key] = acc
            self._metrics["qs_flow_extractor_flows_total"] += 1

        # Determine actual forward/backward direction
        # Forward = packet goes from initiator to responder
        # First packet always sets the initiator
        if acc.total_packets == 0:
            is_forward = True  # First packet defines forward
        else:
            # Forward if src matches the initiator
            is_forward = (src_ip == acc.initiator_ip and
                          src_port == acc.initiator_port)

        acc.record_packet(timestamp, pkt_len, is_forward, tcp_flags)
        self._metrics["qs_flow_extractor_packets_total"] += 1

        return flow_key

    def ingest_raw_tuple(
        self,
        src_ip: str,
        dst_ip: str,
        src_port: int,
        dst_port: int,
        protocol: int,
        pkt_len: int,
        timestamp: float,
        tcp_flags: int = 0,
    ) -> FlowKey:
        """Ingest packet from pre-parsed fields (for testing / CSV replay).

        Args:
            src_ip: Source IP address.
            dst_ip: Destination IP address.
            src_port: Source port.
            dst_port: Destination port.
            protocol: IP protocol number (6=TCP, 17=UDP).
            pkt_len: Packet length in bytes.
            timestamp: Unix timestamp.
            tcp_flags: Raw TCP flags byte.

        Returns:
            FlowKey of the recorded flow.
        """
        flow_key, is_canonical_forward = _make_flow_key(
            src_ip, dst_ip, src_port, dst_port, protocol
        )

        if flow_key in self._flows:
            self._flows.move_to_end(flow_key)
            acc = self._flows[flow_key]
        else:
            self._evict_if_needed()
            acc = FlowAccumulator(
                initiator_ip=src_ip,
                initiator_port=src_port,
            )
            self._flows[flow_key] = acc
            self._metrics["qs_flow_extractor_flows_total"] += 1

        if acc.total_packets == 0:
            is_forward = True
        else:
            is_forward = (src_ip == acc.initiator_ip and
                          src_port == acc.initiator_port)

        acc.record_packet(timestamp, pkt_len, is_forward, tcp_flags)
        self._metrics["qs_flow_extractor_packets_total"] += 1

        return flow_key

    # ──────────────────────────────────────────────────────────────
    # Window Flush: Extract Feature Matrix
    # ──────────────────────────────────────────────────────────────

    def flush_window(self) -> Tuple[np.ndarray, List[FlowKey]]:
        """Flush current window: extract features and reset flows.

        Returns:
            (features, keys) where:
              features: np.ndarray of shape (n_flows, 14), normalized
                        if self.normalize and self._is_fitted.
              keys: List of FlowKey objects corresponding to each row.

            If no valid flows exist, returns (empty (0,14) array, []).
        """
        self._metrics["qs_flow_extractor_windows_total"] += 1

        keys = []
        vectors = []

        for flow_key, acc in self._flows.items():
            if acc.total_packets >= self.min_packets_per_flow:
                keys.append(flow_key)
                vectors.append(acc.to_feature_vector())

        # Reset flow table for next window
        self._flows.clear()

        if not vectors:
            return np.empty((0, NUM_FEATURES), dtype=np.float64), []

        features = np.vstack(vectors)

        # Normalize if fitted
        if self.normalize and self._is_fitted:
            features = self._normalize(features)

        return features, keys

    def peek_features(self) -> Tuple[np.ndarray, List[FlowKey]]:
        """Extract features WITHOUT resetting flows (non-destructive).

        Useful for mid-window scoring or monitoring.

        Returns:
            Same format as flush_window().
        """
        keys = []
        vectors = []

        for flow_key, acc in self._flows.items():
            if acc.total_packets >= self.min_packets_per_flow:
                keys.append(flow_key)
                vectors.append(acc.to_feature_vector())

        if not vectors:
            return np.empty((0, NUM_FEATURES), dtype=np.float64), []

        features = np.vstack(vectors)

        if self.normalize and self._is_fitted:
            features = self._normalize(features)

        return features, keys

    # ──────────────────────────────────────────────────────────────
    # Batch: Extract from PCAP / Packet List
    # ──────────────────────────────────────────────────────────────

    def extract_from_packets(
        self,
        packets: list,
        window_seconds: Optional[float] = None,
    ) -> Tuple[np.ndarray, List[FlowKey]]:
        """Extract flow features from a list of Scapy packets.

        Groups packets into bidirectional flows and extracts the
        14-dimensional feature vector per flow. This is the offline
        batch interface for PCAP replay / evaluation.

        If window_seconds is specified, splits the packet trace into
        non-overlapping windows and returns flows from ALL windows
        concatenated. Otherwise treats the entire list as one window.

        Args:
            packets: List of Scapy packet objects.
            window_seconds: Optional window duration for splitting.
                If None, uses self.window_seconds. Set to 0 or
                float('inf') to treat all packets as one window.

        Returns:
            (features, keys): features shape (n_flows, 14).
        """
        if window_seconds is None:
            window_seconds = self.window_seconds

        # Reset state
        self._flows.clear()

        all_features = []
        all_keys = []

        # If no windowing requested, process all at once
        if window_seconds <= 0 or window_seconds == float('inf'):
            for pkt in packets:
                self.ingest_packet(pkt)
            features, keys = self.flush_window()
            return features, keys

        # Windowed extraction
        window_start = None

        for pkt in packets:
            # Get packet timestamp
            try:
                from scapy.layers.inet import IP
                if not pkt.haslayer(IP):
                    continue
                ts = float(pkt.time) if hasattr(pkt, 'time') else time.time()
            except Exception:
                continue

            if window_start is None:
                window_start = ts

            # Check if packet falls outside current window
            if ts - window_start >= window_seconds:
                # Flush current window
                features, keys = self.flush_window()
                if len(features) > 0:
                    all_features.append(features)
                    all_keys.extend(keys)
                window_start = ts

            self.ingest_packet(pkt)

        # Flush final window
        features, keys = self.flush_window()
        if len(features) > 0:
            all_features.append(features)
            all_keys.extend(keys)

        if not all_features:
            return np.empty((0, NUM_FEATURES), dtype=np.float64), []

        return np.vstack(all_features), all_keys

    # ──────────────────────────────────────────────────────────────
    # Normalization (z-score)
    # ──────────────────────────────────────────────────────────────

    def fit_normalization(self, X: np.ndarray) -> None:
        """Fit z-score normalization statistics on training data.

        Call this on benign baseline flow features BEFORE detection.
        The same stats will be used to normalize all subsequent
        feature vectors, ensuring train-test consistency.

        Args:
            X: Training feature matrix, shape (n_samples, 14).
        """
        if X.ndim != 2 or X.shape[1] != NUM_FEATURES:
            raise ValueError(
                f"Expected shape (n, {NUM_FEATURES}), got {X.shape}"
            )

        self._feature_means = X.mean(axis=0).astype(np.float64)
        self._feature_stds = X.std(axis=0).astype(np.float64)
        # Prevent division by zero on constant features
        self._feature_stds[self._feature_stds < _EPS] = 1.0
        self._is_fitted = True

        logger.info(
            f"Normalization fitted on {X.shape[0]} samples. "
            f"Feature means: [{', '.join(f'{m:.2f}' for m in self._feature_means)}]"
        )

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply z-score normalization to feature matrix.

        Args:
            X: Feature matrix, shape (n, 14).

        Returns:
            Normalized features, same shape.

        Raises:
            RuntimeError: If fit_normalization has not been called.
        """
        if not self._is_fitted:
            raise RuntimeError(
                "FlowFeatureExtractor not fitted. "
                "Call fit_normalization() first."
            )
        return self._normalize(X)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit normalization and transform in one step.

        Args:
            X: Training feature matrix, shape (n, 14).

        Returns:
            Normalized features.
        """
        self.fit_normalization(X)
        return self._normalize(X)

    def _normalize(self, X: np.ndarray) -> np.ndarray:
        """Internal z-score normalization."""
        return (X - self._feature_means) / self._feature_stds

    # ──────────────────────────────────────────────────────────────
    # CSV Feature Mapping (CICIDS2017 compatibility)
    # ──────────────────────────────────────────────────────────────

    @staticmethod
    def extract_from_cicids_csv(
        df,
        feature_columns: Optional[List[str]] = None,
    ) -> np.ndarray:
        """Extract the 14 target features from a CICIDS2017 DataFrame.

        Maps CICIDS2017's 80+ features down to the 14 flow features
        used by the live pipeline, ensuring offline evaluation uses
        the same feature space as production.

        Args:
            df: pandas DataFrame loaded from CICIDS2017 CSV.
            feature_columns: Optional override for column name mapping.

        Returns:
            np.ndarray of shape (n_rows, 14).
        """
        # Default CICIDS2017 → our 14-feature mapping
        # Column names may have leading spaces in CICIDS2017 CSVs
        if feature_columns is None:
            feature_columns = _CICIDS_COLUMN_MAP

        features = np.zeros((len(df), NUM_FEATURES), dtype=np.float64)

        for i, (our_name, cicids_candidates) in enumerate(feature_columns):
            col_found = False
            for col_name in cicids_candidates:
                # Try with/without leading space
                for variant in [col_name, f" {col_name}", col_name.strip()]:
                    if variant in df.columns:
                        vals = df[variant].values.astype(np.float64)
                        # Handle inf/nan
                        vals = np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0)
                        features[:, i] = vals
                        col_found = True
                        break
                if col_found:
                    break
            if not col_found:
                logger.warning(
                    f"CICIDS column not found for '{our_name}', "
                    f"tried: {cicids_candidates}. Using zeros."
                )

        return features

    # ──────────────────────────────────────────────────────────────
    # LRU Eviction
    # ──────────────────────────────────────────────────────────────

    def _evict_if_needed(self) -> None:
        """Evict oldest flows if at capacity."""
        evicted = 0
        while len(self._flows) >= self.max_flows:
            self._flows.popitem(last=False)
            evicted += 1
        if evicted > 0:
            self._metrics["qs_flow_extractor_flows_evicted"] += evicted
            logger.debug(f"Evicted {evicted} flows (at capacity {self.max_flows})")

    # ──────────────────────────────────────────────────────────────
    # Properties & Metrics
    # ──────────────────────────────────────────────────────────────

    @property
    def active_flows(self) -> int:
        """Number of flows in the current window."""
        return len(self._flows)

    @property
    def is_fitted(self) -> bool:
        """Whether normalization statistics have been set."""
        return self._is_fitted

    @property
    def feature_names(self) -> List[str]:
        """Feature names in output column order."""
        return list(FLOW_FEATURE_NAMES)

    def get_metrics(self) -> Dict[str, int]:
        """Get Prometheus-ready metric counters.

        All metric names follow qs_<component>_<metric> convention.

        Returns:
            Dict of metric_name → value.
        """
        return dict(self._metrics)

    def get_params(self) -> dict:
        """Get extractor configuration."""
        return {
            "window_seconds": self.window_seconds,
            "max_flows": self.max_flows,
            "min_packets_per_flow": self.min_packets_per_flow,
            "normalize": self.normalize,
            "is_fitted": self._is_fitted,
            "active_flows": self.active_flows,
        }

    def reset(self) -> None:
        """Reset all internal state (flows + metrics, keeps normalization)."""
        self._flows.clear()
        for k in self._metrics:
            self._metrics[k] = 0


# ──────────────────────────────────────────────────────────────────────
# CICIDS2017 Column Mapping
# ──────────────────────────────────────────────────────────────────────

# Maps our 14 features to CICIDS2017 CSV column name candidates.
# Each entry: (our_feature_name, [candidate_column_names])
# The list is searched in order; first match wins.

_CICIDS_COLUMN_MAP = [
    ("flow_duration", [
        "Flow Duration",
    ]),
    ("fwd_packet_count", [
        "Total Fwd Packets",
        "Fwd Packet Count",
        "Total Fwd Packet",
    ]),
    ("bwd_packet_count", [
        "Total Backward Packets",
        "Total Bwd packets",
        "Bwd Packet Count",
    ]),
    ("fwd_bytes_total", [
        "Total Length of Fwd Packets",
        "Total Length of Fwd Packet",
        "Fwd Packets Total Length",
        "TotalLenofFwdPkts",
    ]),
    ("bwd_bytes_total", [
        "Total Length of Bwd Packets",
        "Total Length of Bwd Packet",
        "Bwd Packets Total Length",
        "TotalLenofBwdPkts",
    ]),
    ("fwd_pkt_len_mean", [
        "Fwd Packet Length Mean",
        "Fwd Pkt Len Mean",
    ]),
    ("fwd_pkt_len_std", [
        "Fwd Packet Length Std",
        "Fwd Pkt Len Std",
    ]),
    ("bwd_pkt_len_mean", [
        "Bwd Packet Length Mean",
        "Bwd Pkt Len Mean",
    ]),
    ("bwd_pkt_len_std", [
        "Bwd Packet Length Std",
        "Bwd Pkt Len Std",
    ]),
    ("flow_iat_mean", [
        "Flow IAT Mean",
    ]),
    ("flow_iat_std", [
        "Flow IAT Std",
    ]),
    ("syn_flag_count", [
        "SYN Flag Count",
        "SYN Flag Cnt",
    ]),
    ("ack_flag_count", [
        "ACK Flag Count",
        "ACK Flag Cnt",
    ]),
    ("packets_per_second", [
        "Flow Packets/s",
        "Flow Pkts/s",
    ]),
]
