"""
ML-DPI Feedback Controller
============================
Bidirectional ML ↔ DPI controller that makes the ML layer the
instruction-giver for the packet inspector's parse depth.

Architecture:
  1. THREAT STATE MACHINE: per-flow 4-level state (BASELINE→WATCH→SUSPECT→HOSTILE)
  2. DPI PROFILES: each state produces a DPIProfile that tells the parser which
     layers to inspect and how many features to extract
  3. FEEDBACK LOOP: richer data from elevated states feeds back into the iForest
     for online learning; false-positives recalibrate the conformal predictor
  4. DATA PIPELINE: all ML input comes from the packet inspector — the controller
     observes parsed packets, accumulates per-flow features, triggers scoring,
     and updates state based on scores

Data flow:
  engine.py → controller.get_parse_profile(flow_id) → which parsers to run
  engine.py → controller.observe(flow_id, packet_meta) → accumulate features
  [window elapsed] → controller scores features via iforest → update_state()
  state transition → new DPIProfile for next packets on this flow

Thread-safe. Memory-bounded PCAP ring buffers.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Threat Levels
# ──────────────────────────────────────────────────────────────────────

class ThreatLevel(IntEnum):
    """Per-flow threat level — drives DPI depth."""
    BASELINE = 0   # Normal traffic, minimal parsing
    WATCH = 1      # Elevated anomaly, increased extraction
    SUSPECT = 2    # High anomaly, full extraction + PCAP ring
    HOSTILE = 3    # Confirmed threat, forensic mode


# ──────────────────────────────────────────────────────────────────────
# DPI Instruction Profile
# ──────────────────────────────────────────────────────────────────────

class ParseDepth(IntEnum):
    """Parser depth levels."""
    HEADERS_ONLY = 0     # L2-L4 headers
    STANDARD = 1         # + DNS labels, TLS SNI, JA3
    FULL = 2             # + HTTP body, TLS handshake, QUIC DCID
    FORENSIC = 3         # Everything, per-packet scoring


@dataclass
class DPIProfile:
    """Instruction set for the DPI layer — what to parse and extract.

    Returned by controller.get_parse_profile(flow_id) and consumed
    by engine.py to decide which parsers to invoke.
    """
    parse_depth: ParseDepth
    feature_count: int          # 5, 14, or 18
    window_seconds: float       # 30, 15, 10, or 5
    pcap_enabled: bool          # Whether to buffer raw packets
    pcap_to_disk: bool          # Write PCAP to disk immediately
    per_packet_scoring: bool    # Score every packet, not just windows
    threat_level: ThreatLevel

    # Which parser categories to enable
    parse_ethernet: bool = True
    parse_ip: bool = True
    parse_tcp_udp: bool = True
    parse_dns_labels: bool = False
    parse_tls_handshake: bool = False
    parse_http_body: bool = False
    parse_quic: bool = False
    parse_ssh: bool = False


# Pre-built profiles for each threat level
PROFILES = {
    ThreatLevel.BASELINE: DPIProfile(
        parse_depth=ParseDepth.HEADERS_ONLY,
        feature_count=5,
        window_seconds=30.0,
        pcap_enabled=False,
        pcap_to_disk=False,
        per_packet_scoring=False,
        threat_level=ThreatLevel.BASELINE,
        parse_ethernet=True,
        parse_ip=True,
        parse_tcp_udp=True,
        parse_dns_labels=False,
        parse_tls_handshake=False,
        parse_http_body=False,
        parse_quic=False,
        parse_ssh=False,
    ),
    ThreatLevel.WATCH: DPIProfile(
        parse_depth=ParseDepth.STANDARD,
        feature_count=14,
        window_seconds=15.0,
        pcap_enabled=False,
        pcap_to_disk=False,
        per_packet_scoring=False,
        threat_level=ThreatLevel.WATCH,
        parse_ethernet=True,
        parse_ip=True,
        parse_tcp_udp=True,
        parse_dns_labels=True,
        parse_tls_handshake=True,
        parse_http_body=False,
        parse_quic=False,
        parse_ssh=True,
    ),
    ThreatLevel.SUSPECT: DPIProfile(
        parse_depth=ParseDepth.FULL,
        feature_count=18,
        window_seconds=10.0,
        pcap_enabled=True,
        pcap_to_disk=False,
        per_packet_scoring=False,
        threat_level=ThreatLevel.SUSPECT,
        parse_ethernet=True,
        parse_ip=True,
        parse_tcp_udp=True,
        parse_dns_labels=True,
        parse_tls_handshake=True,
        parse_http_body=True,
        parse_quic=True,
        parse_ssh=True,
    ),
    ThreatLevel.HOSTILE: DPIProfile(
        parse_depth=ParseDepth.FORENSIC,
        feature_count=18,
        window_seconds=5.0,
        pcap_enabled=True,
        pcap_to_disk=True,
        per_packet_scoring=True,
        threat_level=ThreatLevel.HOSTILE,
        parse_ethernet=True,
        parse_ip=True,
        parse_tcp_udp=True,
        parse_dns_labels=True,
        parse_tls_handshake=True,
        parse_http_body=True,
        parse_quic=True,
        parse_ssh=True,
    ),
}


# ──────────────────────────────────────────────────────────────────────
# Extended Feature Names (18 features for SUSPECT/HOSTILE)
# ──────────────────────────────────────────────────────────────────────

BASELINE_FEATURES = [
    "packet_rate", "byte_rate", "tcp_ratio", "udp_ratio", "dns_ratio",
]

STANDARD_FEATURES = [
    "packet_rate", "byte_rate", "avg_packet_size",
    "unique_src_ips", "unique_dst_ips", "unique_dst_ports",
    "tcp_ratio", "udp_ratio", "dns_ratio", "icmp_ratio",
    "syn_ratio", "connection_rate", "port_entropy", "ip_entropy",
]

ENRICHED_FEATURES = STANDARD_FEATURES + [
    "tls_version_ratio",        # TLS 1.3 sessions / total TLS sessions
    "avg_dns_label_entropy",    # Mean Shannon entropy of DNS labels
    "pq_kem_ratio",             # PQ key share sessions / total TLS
    "retransmit_ratio",         # TCP retransmits / total TCP packets
]


# ──────────────────────────────────────────────────────────────────────
# Per-Flow Feature Accumulator
# ──────────────────────────────────────────────────────────────────────

@dataclass
class FlowAccumulator:
    """Accumulates packet metadata per flow within a feature window.

    Supports both the standard 14-feature set and the enriched 18-feature
    set. The packet inspector feeds data here via controller.observe().
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

    # Enriched features (SUSPECT/HOSTILE only)
    tls_sessions: int = 0
    tls13_sessions: int = 0
    pq_kem_sessions: int = 0
    dns_label_entropies: List[float] = field(default_factory=list)
    tcp_retransmits: int = 0

    def record(
        self,
        protocol: str,
        src_ip: str,
        dst_ip: str,
        size: int,
        src_port: int = 0,
        dst_port: int = 0,
        is_syn: bool = False,
        tls_version: Optional[str] = None,
        has_pq_kem: bool = False,
        dns_label_entropy: Optional[float] = None,
        is_retransmit: bool = False,
    ):
        """Record a single packet's metadata from the packet inspector."""
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

        # Enriched data (only populated when DPI runs at SUSPECT+ depth)
        if tls_version is not None:
            self.tls_sessions += 1
            if "1.3" in tls_version or tls_version in ("0x0304",):
                self.tls13_sessions += 1
        if has_pq_kem:
            self.pq_kem_sessions += 1
        if dns_label_entropy is not None:
            self.dns_label_entropies.append(dns_label_entropy)
        if is_retransmit:
            self.tcp_retransmits += 1

    def to_feature_vector(self, window_duration: float, feature_count: int) -> np.ndarray:
        """Extract feature vector from accumulated data.

        Args:
            window_duration: Window duration in seconds.
            feature_count: 5 (BASELINE), 14 (WATCH), or 18 (SUSPECT/HOSTILE).

        Returns:
            Feature vector of the requested dimensionality.
        """
        duration = max(window_duration, 0.1)
        total = max(self.packet_count, 1)
        tcp_total = max(self.tcp_count, 1)

        if feature_count <= 5:
            # BASELINE: minimal features
            return np.array([
                self.packet_count / duration,            # packet_rate
                self.total_bytes / duration,             # byte_rate
                self.tcp_count / total,                  # tcp_ratio
                self.udp_count / total,                  # udp_ratio
                self.dns_count / total,                  # dns_ratio
            ], dtype=np.float64)

        # Standard 14 features (WATCH and above)
        features_14 = np.array([
            self.packet_count / duration,                # packet_rate
            self.total_bytes / duration,                 # byte_rate
            self.total_bytes / total,                    # avg_packet_size
            len(self.src_ips),                           # unique_src_ips
            len(self.dst_ips),                           # unique_dst_ips
            len(self.dst_ports),                         # unique_dst_ports
            self.tcp_count / total,                      # tcp_ratio
            self.udp_count / total,                      # udp_ratio
            self.dns_count / total,                      # dns_ratio
            self.icmp_count / total,                     # icmp_ratio
            self.syn_count / tcp_total if self.tcp_count > 0 else 0,  # syn_ratio
            len(self.connections) / duration,             # connection_rate
            self._shannon_entropy(self.dst_port_counts), # port_entropy
            self._shannon_entropy(self.src_ip_counts),   # ip_entropy
        ], dtype=np.float64)

        if feature_count <= 14:
            return features_14

        # Enriched 18 features (SUSPECT/HOSTILE)
        tls_total = max(self.tls_sessions, 1)
        avg_dns_entropy = (
            float(np.mean(self.dns_label_entropies))
            if self.dns_label_entropies else 0.0
        )

        enriched = np.array([
            self.tls13_sessions / tls_total,             # tls_version_ratio
            avg_dns_entropy,                             # avg_dns_label_entropy
            self.pq_kem_sessions / tls_total,            # pq_kem_ratio
            self.tcp_retransmits / tcp_total if self.tcp_count > 0 else 0,  # retransmit_ratio
        ], dtype=np.float64)

        return np.concatenate([features_14, enriched])

    @staticmethod
    def _shannon_entropy(counts: dict) -> float:
        total = sum(counts.values())
        if total == 0:
            return 0.0
        entropy = 0.0
        for count in counts.values():
            if count > 0:
                p = count / total
                entropy -= p * math.log2(p)
        return entropy

    def reset(self):
        """Reset for next window."""
        self.start_time = time.time()
        self.packet_count = 0
        self.total_bytes = 0
        self.tcp_count = self.udp_count = self.dns_count = 0
        self.icmp_count = self.syn_count = 0
        self.src_ips.clear()
        self.dst_ips.clear()
        self.dst_ports.clear()
        self.connections.clear()
        self.dst_port_counts.clear()
        self.src_ip_counts.clear()
        self.tls_sessions = self.tls13_sessions = self.pq_kem_sessions = 0
        self.dns_label_entropies.clear()
        self.tcp_retransmits = 0


# ──────────────────────────────────────────────────────────────────────
# Per-Flow State
# ──────────────────────────────────────────────────────────────────────

@dataclass
class FlowState:
    """Complete per-flow tracking state."""
    flow_id: str
    level: ThreatLevel = ThreatLevel.BASELINE
    accumulator: FlowAccumulator = field(default_factory=FlowAccumulator)

    # State machine bookkeeping
    consecutive_normal: int = 0          # windows with score < 0.25
    consecutive_elevated: int = 0        # windows with score ≥ 0.35
    last_score: float = 0.0
    last_update: float = field(default_factory=time.time)
    score_history: deque = field(default_factory=lambda: deque(maxlen=20))

    # PCAP ring buffer (bytes) — used in SUSPECT/HOSTILE
    pcap_buffer: deque = field(default_factory=lambda: deque(maxlen=1000))
    pcap_buffer_bytes: int = 0

    # Feature vectors from elevated states (for feedback loop)
    elevated_features: List[np.ndarray] = field(default_factory=list)

    @property
    def profile(self) -> DPIProfile:
        return PROFILES[self.level]


# ──────────────────────────────────────────────────────────────────────
# PCAP Ring Buffer Manager
# ──────────────────────────────────────────────────────────────────────

class PCAPManager:
    """Memory-bounded PCAP ring buffer across all flows.

    Enforces max_total_bytes across all SUSPECT+ flows.
    """
    def __init__(self, max_total_bytes: int = 100 * 1024 * 1024):
        self.max_total_bytes = max_total_bytes
        self._total_bytes = 0
        self._lock = threading.Lock()

    @property
    def total_bytes(self) -> int:
        return self._total_bytes

    def can_buffer(self, size: int) -> bool:
        return self._total_bytes + size <= self.max_total_bytes

    def add(self, flow: FlowState, raw_bytes: bytes) -> bool:
        """Add raw packet bytes to a flow's PCAP buffer."""
        size = len(raw_bytes)
        with self._lock:
            if self._total_bytes + size > self.max_total_bytes:
                # Evict oldest from this flow
                if flow.pcap_buffer:
                    evicted = flow.pcap_buffer.popleft()
                    evicted_size = len(evicted)
                    flow.pcap_buffer_bytes -= evicted_size
                    self._total_bytes -= evicted_size
                else:
                    return False

            flow.pcap_buffer.append(raw_bytes)
            flow.pcap_buffer_bytes += size
            self._total_bytes += size
            return True

    def clear_flow(self, flow: FlowState):
        """Remove all PCAP data for a flow."""
        with self._lock:
            self._total_bytes -= flow.pcap_buffer_bytes
            flow.pcap_buffer.clear()
            flow.pcap_buffer_bytes = 0


# ──────────────────────────────────────────────────────────────────────
# State Transition Event
# ──────────────────────────────────────────────────────────────────────

@dataclass
class StateTransition:
    """Record of a flow's threat level change."""
    flow_id: str
    from_level: ThreatLevel
    to_level: ThreatLevel
    anomaly_score: float
    reason: str
    timestamp: float


# ──────────────────────────────────────────────────────────────────────
# ML-DPI Controller
# ──────────────────────────────────────────────────────────────────────

# Transition thresholds
WATCH_THRESHOLD = 0.35
SUSPECT_THRESHOLD = 0.55
HOSTILE_THRESHOLD = 0.75
NORMAL_THRESHOLD = 0.25
NORMAL_STREAK_REQUIRED = 3
ELEVATED_STREAK_FOR_SUSPECT = 2

# Feedback loop
HOSTILE_RETRAIN_BATCH = 100
FLOW_TTL_SECONDS = 300  # 5 min inactive TTL


class MLDPIController:
    """Unified ML ↔ DPI feedback controller.

    Sits between engine.py and the ML pipeline. For each flow:
      1. engine.py calls get_parse_profile(flow_id) → gets a DPIProfile
      2. engine.py parses the packet using that profile's depth
      3. engine.py calls observe(flow_id, packet_metadata) → accumulates features
      4. When the flow's window elapses, controller scores features via iForest
      5. Score updates the state machine → new profile for next window

    The ML model learns from the packet inspector's data:
      - Baseline training: accumulates from normal traffic windows
      - Online updates: HOSTILE-confirmed vectors retrain the iForest
      - False-positive feedback: recalibrates the conformal predictor

    Args:
        iforest_detector: The IForestNetworkDetector instance (or None for standalone).
        conformal_predictor: The DynamicConformalPredictor instance (or None).
        max_flows: Maximum tracked flows before TTL eviction.
        pcap_max_bytes: Max PCAP memory across all SUSPECT flows.
    """

    def __init__(
        self,
        iforest_detector: Any = None,
        conformal_predictor: Any = None,
        max_flows: int = 10000,
        pcap_max_bytes: int = 100 * 1024 * 1024,
    ):
        self._iforest = iforest_detector
        self._conformal = conformal_predictor
        self._max_flows = max_flows

        self._lock = threading.Lock()
        self._flows: Dict[str, FlowState] = {}
        self._pcap_mgr = PCAPManager(max_total_bytes=pcap_max_bytes)

        # Feedback buffers
        self._hostile_training_buffer: List[np.ndarray] = []
        self._false_positive_buffer: List[np.ndarray] = []

        # Transition history
        self._transitions: deque = deque(maxlen=1000)

        # Response action hooks
        self._response_hooks: List[Callable] = []

        # Statistics (Prometheus-ready)
        self._stats = {
            "flows_in_baseline": 0,
            "flows_in_watch": 0,
            "flows_in_suspect": 0,
            "flows_in_hostile": 0,
            "dpi_depth_upgrades_total": 0,
            "dpi_depth_downgrades_total": 0,
            "hostile_flows_total": 0,
            "false_positive_feedback_total": 0,
            "online_retrains_total": 0,
            "total_windows_scored": 0,
            "pcap_bytes_total": 0,
        }

    # ── Profile Dispatch ──

    def get_parse_profile(self, flow_id: str) -> DPIProfile:
        """Get the DPI instruction profile for a flow.

        Called by engine.py BEFORE parsing each packet to determine
        which parsers to invoke and at what depth.

        Args:
            flow_id: Flow identifier (e.g. "10.0.0.1:443->10.0.0.2:50000").

        Returns:
            DPIProfile with parser instructions for this flow.
        """
        with self._lock:
            flow = self._flows.get(flow_id)
            if flow is None:
                return PROFILES[ThreatLevel.BASELINE]
            return flow.profile

    # ── Packet Observation (from Packet Inspector) ──

    def observe(
        self,
        flow_id: str,
        protocol: str,
        src_ip: str,
        dst_ip: str,
        size: int,
        src_port: int = 0,
        dst_port: int = 0,
        is_syn: bool = False,
        tls_version: Optional[str] = None,
        has_pq_kem: bool = False,
        dns_label_entropy: Optional[float] = None,
        is_retransmit: bool = False,
        raw_bytes: Optional[bytes] = None,
    ) -> Optional[List['StateTransition']]:
        """Observe a parsed packet from the packet inspector.

        Accumulates per-flow features. When the flow's window elapses,
        extracts features, scores them, and updates the state machine.

        This is the primary data ingestion point — ALL ML model input
        originates from here via the packet inspector.

        Args:
            flow_id: Flow identifier.
            protocol: Protocol string from DPI.
            src_ip, dst_ip: IP addresses.
            size: Packet size in bytes.
            src_port, dst_port: Port numbers.
            is_syn: Whether this is a SYN packet.
            tls_version: TLS version if parsed (WATCH+).
            has_pq_kem: Whether PQ KEM key share detected (SUSPECT+).
            dns_label_entropy: DNS label Shannon entropy (SUSPECT+).
            is_retransmit: TCP retransmit flag (SUSPECT+).
            raw_bytes: Raw packet bytes for PCAP capture (SUSPECT+).

        Returns:
            List of StateTransitions if any occurred, None otherwise.
        """
        now = time.time()

        with self._lock:
            flow = self._get_or_create_flow(flow_id, now)
            profile = flow.profile

            # Record packet in the flow's accumulator
            flow.accumulator.record(
                protocol=protocol,
                src_ip=src_ip,
                dst_ip=dst_ip,
                size=size,
                src_port=src_port,
                dst_port=dst_port,
                is_syn=is_syn,
                tls_version=tls_version,
                has_pq_kem=has_pq_kem,
                dns_label_entropy=dns_label_entropy,
                is_retransmit=is_retransmit,
            )

            # PCAP capture for SUSPECT+ flows
            if profile.pcap_enabled and raw_bytes is not None:
                self._pcap_mgr.add(flow, raw_bytes)

            # Check if window has elapsed
            elapsed = now - flow.accumulator.start_time
            if elapsed < profile.window_seconds:
                return None

            # ── Window complete: extract features, score, update state ──
            features = flow.accumulator.to_feature_vector(
                elapsed, profile.feature_count
            )
            flow.accumulator.reset()

            transitions = self._score_and_update(flow, features, now)
            flow.last_update = now

            return transitions if transitions else None

    def _get_or_create_flow(self, flow_id: str, now: float) -> FlowState:
        """Get or create a flow state (must hold _lock)."""
        if flow_id not in self._flows:
            # TTL eviction if at capacity
            if len(self._flows) >= self._max_flows:
                self._evict_oldest()

            flow = FlowState(flow_id=flow_id)
            flow.accumulator.start_time = now
            self._flows[flow_id] = flow
            self._stats["flows_in_baseline"] += 1

        return self._flows[flow_id]

    # ── Scoring and State Machine ──

    def _score_and_update(
        self,
        flow: FlowState,
        features: np.ndarray,
        now: float,
    ) -> List[StateTransition]:
        """Score a feature vector and update the threat state machine.

        The anomaly score comes from the iForest detector (trained on
        data from the packet inspector). If no detector is available,
        a synthetic score is computed from feature deviation.

        Must hold _lock.
        """
        self._stats["total_windows_scored"] += 1

        # Get anomaly score from the ML pipeline
        score = self._compute_anomaly_score(features)
        flow.last_score = score
        flow.score_history.append(score)

        # Store features for potential feedback
        if flow.level >= ThreatLevel.SUSPECT:
            flow.elevated_features.append(features)

        # Feed to conformal predictor for dynamic p-value
        if self._conformal is not None:
            try:
                self._conformal.observe_flow_score(flow.flow_id, score)
            except Exception:
                pass

        transitions = []

        # ── State machine transitions ──

        old_level = flow.level

        if score < NORMAL_THRESHOLD:
            flow.consecutive_normal += 1
            flow.consecutive_elevated = 0
        else:
            flow.consecutive_normal = 0
            if score >= WATCH_THRESHOLD:
                flow.consecutive_elevated += 1
            else:
                flow.consecutive_elevated = 0

        # Downgrade: 3 consecutive normal windows → BASELINE
        if flow.consecutive_normal >= NORMAL_STREAK_REQUIRED and flow.level != ThreatLevel.BASELINE:
            # Check for false-positive feedback (HOSTILE → BASELINE)
            if flow.level == ThreatLevel.HOSTILE:
                self._handle_false_positive(flow)

            self._set_level(flow, ThreatLevel.BASELINE,
                            f"3 consecutive windows below {NORMAL_THRESHOLD}")
            transitions.append(StateTransition(
                flow.flow_id, old_level, ThreatLevel.BASELINE,
                score, f"Consecutive normal: {flow.consecutive_normal}", now,
            ))
            old_level = ThreatLevel.BASELINE

        # Upgrade: BASELINE → WATCH
        elif flow.level == ThreatLevel.BASELINE and score >= WATCH_THRESHOLD:
            self._set_level(flow, ThreatLevel.WATCH,
                            f"Score {score:.3f} >= {WATCH_THRESHOLD}")
            transitions.append(StateTransition(
                flow.flow_id, old_level, ThreatLevel.WATCH,
                score, f"Score crossed WATCH threshold", now,
            ))

        # Upgrade: WATCH → SUSPECT
        elif flow.level == ThreatLevel.WATCH:
            if score >= SUSPECT_THRESHOLD or flow.consecutive_elevated >= ELEVATED_STREAK_FOR_SUSPECT:
                reason = (f"Score {score:.3f} >= {SUSPECT_THRESHOLD}"
                          if score >= SUSPECT_THRESHOLD
                          else f"{flow.consecutive_elevated} consecutive elevated windows")
                self._set_level(flow, ThreatLevel.SUSPECT, reason)
                transitions.append(StateTransition(
                    flow.flow_id, old_level, ThreatLevel.SUSPECT,
                    score, reason, now,
                ))

        # Upgrade: SUSPECT → HOSTILE
        elif flow.level == ThreatLevel.SUSPECT:
            if score >= HOSTILE_THRESHOLD:
                self._set_level(flow, ThreatLevel.HOSTILE,
                                f"Score {score:.3f} >= {HOSTILE_THRESHOLD}")
                transitions.append(StateTransition(
                    flow.flow_id, old_level, ThreatLevel.HOSTILE,
                    score, f"Score crossed HOSTILE threshold", now,
                ))
                self._handle_hostile_confirmed(flow)

        # Upgrade: WATCH straight to HOSTILE if score is extreme
        elif flow.level == ThreatLevel.WATCH and score >= HOSTILE_THRESHOLD:
            self._set_level(flow, ThreatLevel.HOSTILE,
                            f"Score {score:.3f} >= {HOSTILE_THRESHOLD} (skip SUSPECT)")
            transitions.append(StateTransition(
                flow.flow_id, old_level, ThreatLevel.HOSTILE,
                score, f"Direct escalation to HOSTILE", now,
            ))
            self._handle_hostile_confirmed(flow)

        # Log transitions
        for t in transitions:
            self._transitions.append(t)
            logger.info(
                f"STATE_TRANSITION: {t.flow_id} "
                f"{t.from_level.name} → {t.to_level.name} "
                f"(score={t.anomaly_score:.3f}, reason={t.reason})"
            )

        return transitions

    def _compute_anomaly_score(self, features: np.ndarray) -> float:
        """Compute anomaly score from features.

        Uses the iForest detector if available (trained on packet inspector
        data). Falls back to a simple z-score based heuristic.
        """
        if self._iforest is not None and hasattr(self._iforest, '_is_trained') and self._iforest._is_trained:
            try:
                # Use only the first 14 features for the iForest
                # (it was trained on 14-dim vectors)
                feat_14 = features[:14] if len(features) > 14 else features
                if len(feat_14) < 14:
                    # Pad with zeros if we only have 5 BASELINE features
                    feat_14 = np.pad(feat_14, (0, 14 - len(feat_14)))

                means = self._iforest._feature_means
                stds = self._iforest._feature_stds
                normalized = ((feat_14 - means) / stds).reshape(1, -1)
                score = float(self._iforest._forest.anomaly_scores(normalized)[0])
                return max(0.0, min(1.0, score))
            except Exception:
                pass

        # Fallback: simple deviation-based score
        return self._heuristic_score(features)

    @staticmethod
    def _heuristic_score(features: np.ndarray) -> float:
        """Simple heuristic anomaly score when iForest is unavailable."""
        if len(features) == 0:
            return 0.0
        # Use coefficient of variation as a rough proxy
        mean = float(np.mean(np.abs(features)))
        if mean < 1e-10:
            return 0.0
        cv = float(np.std(features) / mean)
        return max(0.0, min(1.0, cv / 5.0))

    # ── State Level Management ──

    def _set_level(self, flow: FlowState, new_level: ThreatLevel, reason: str):
        """Set a flow's threat level (must hold _lock)."""
        old_level = flow.level

        # Update counters
        level_key = {
            ThreatLevel.BASELINE: "flows_in_baseline",
            ThreatLevel.WATCH: "flows_in_watch",
            ThreatLevel.SUSPECT: "flows_in_suspect",
            ThreatLevel.HOSTILE: "flows_in_hostile",
        }
        self._stats[level_key[old_level]] = max(0, self._stats[level_key[old_level]] - 1)
        self._stats[level_key[new_level]] += 1

        if new_level > old_level:
            self._stats["dpi_depth_upgrades_total"] += 1
        elif new_level < old_level:
            self._stats["dpi_depth_downgrades_total"] += 1

        flow.level = new_level
        flow.consecutive_normal = 0
        flow.consecutive_elevated = 0

        # Clear PCAP if downgrading below SUSPECT
        if new_level < ThreatLevel.SUSPECT and old_level >= ThreatLevel.SUSPECT:
            self._pcap_mgr.clear_flow(flow)

    # ── Correlated Alert Integration ──

    def on_correlated_alert(self, flow_id: str):
        """Called by unified_explainer when it fires a CORRELATED_ALERT.

        Immediately escalates the flow to HOSTILE regardless of score.
        """
        now = time.time()
        with self._lock:
            flow = self._flows.get(flow_id)
            if flow is None:
                return

            if flow.level < ThreatLevel.HOSTILE:
                old_level = flow.level
                self._set_level(flow, ThreatLevel.HOSTILE,
                                "CORRELATED_ALERT from unified_explainer")
                transition = StateTransition(
                    flow_id, old_level, ThreatLevel.HOSTILE,
                    flow.last_score,
                    "CORRELATED_ALERT: rule+ML co-detection", now,
                )
                self._transitions.append(transition)
                self._handle_hostile_confirmed(flow)
                logger.warning(
                    f"CORRELATED_ALERT escalation: {flow_id} → HOSTILE"
                )

    # ── Feedback Loop ──

    def _handle_hostile_confirmed(self, flow: FlowState):
        """Handle transition to HOSTILE — feed enriched data back to ML.

        The enriched 18-feature vectors from SUSPECT mode are added to
        the hostile training buffer. When enough accumulate, the iForest
        is retrained online to improve detection of this attack pattern.

        Must hold _lock.
        """
        self._stats["hostile_flows_total"] += 1

        # Stash elevated feature vectors for online retraining
        for fv in flow.elevated_features:
            self._hostile_training_buffer.append(fv)
        flow.elevated_features.clear()

        # Attempt online retrain
        if len(self._hostile_training_buffer) >= HOSTILE_RETRAIN_BATCH:
            self._online_retrain()

        # Fire response hooks
        for hook in self._response_hooks:
            try:
                hook(flow.flow_id, flow.level, flow.last_score)
            except Exception:
                pass

    def _handle_false_positive(self, flow: FlowState):
        """Handle HOSTILE → BASELINE transition (false alarm).

        The feature vectors that triggered HOSTILE are fed to the
        false_positive_buffer within the conformal predictor to
        recalibrate thresholds downward.

        Must hold _lock.
        """
        self._stats["false_positive_feedback_total"] += 1

        for fv in flow.elevated_features:
            self._false_positive_buffer.append(fv)

            # Feed to conformal predictor
            if self._conformal is not None:
                try:
                    # These were flagged as anomalous but turned out normal
                    # → adjust calibration downward
                    score = flow.last_score
                    self._conformal.observe_normal(score)
                except Exception:
                    pass

        flow.elevated_features.clear()

        logger.info(
            f"FALSE_POSITIVE_FEEDBACK: {flow.flow_id}, "
            f"buffered {len(self._false_positive_buffer)} FP vectors"
        )

    def _online_retrain(self):
        """Retrain the iForest with hostile-confirmed feature vectors.

        Appends hostile vectors to the training set and refits.
        Only uses the first 14 features to match the iForest's
        expected dimensionality.

        Must hold _lock.
        """
        if self._iforest is None or not hasattr(self._iforest, '_training_data'):
            return

        try:
            # Truncate to 14 features (iForest dimensionality)
            new_vectors = [fv[:14] if len(fv) > 14 else fv
                           for fv in self._hostile_training_buffer]
            self._hostile_training_buffer.clear()

            # Append to existing training data and retrain
            for v in new_vectors:
                if len(v) == 14:
                    self._iforest._training_data.append(v)

            # Retrain with augmented dataset
            X = np.vstack(self._iforest._training_data)
            self._iforest._feature_means = X.mean(axis=0)
            self._iforest._feature_stds = X.std(axis=0)
            self._iforest._feature_stds[self._iforest._feature_stds < 1e-10] = 1.0

            X_norm = (X - self._iforest._feature_means) / self._iforest._feature_stds
            self._iforest._forest.fit(X_norm)

            self._stats["online_retrains_total"] += 1
            logger.info(
                f"ONLINE_RETRAIN: iForest retrained with {len(X)} total samples "
                f"(+{len(new_vectors)} hostile-confirmed)"
            )
        except Exception as exc:
            logger.error(f"Online retrain failed: {exc}")

    # ── TTL Eviction ──

    def _evict_oldest(self):
        """Evict the oldest inactive flow (must hold _lock)."""
        if not self._flows:
            return
        oldest_id = min(self._flows, key=lambda k: self._flows[k].last_update)
        old_flow = self._flows.pop(oldest_id)
        level_key = f"flows_in_{old_flow.level.name.lower()}"
        self._stats[level_key] = max(0, self._stats[level_key] - 1)
        self._pcap_mgr.clear_flow(old_flow)

    # ── Response Hook Registration ──

    def register_response_hook(self, hook: Callable):
        """Register a callback for HOSTILE state transitions.

        Hook signature: hook(flow_id: str, level: ThreatLevel, score: float)
        """
        self._response_hooks.append(hook)

    # ── Prometheus Metrics ──

    def get_prometheus_metrics(self) -> str:
        """Generate Prometheus-format metrics."""
        lines = []
        for key, val in self._stats.items():
            prom_key = f"ml_dpi_{key}"
            lines.append(f"# TYPE {prom_key} gauge")
            lines.append(f"{prom_key} {val}")

        # Add PCAP usage
        lines.append("# TYPE ml_dpi_pcap_bytes_used gauge")
        lines.append(f"ml_dpi_pcap_bytes_used {self._pcap_mgr.total_bytes}")

        return "\n".join(lines) + "\n"

    def get_stats(self) -> dict:
        """Get controller statistics."""
        with self._lock:
            return {
                **self._stats,
                "total_flows": len(self._flows),
                "pcap_bytes_used": self._pcap_mgr.total_bytes,
                "hostile_training_buffer_size": len(self._hostile_training_buffer),
                "false_positive_buffer_size": len(self._false_positive_buffer),
                "transition_history_size": len(self._transitions),
            }

    def get_flow_summary(self) -> dict:
        """Per-level flow summary for dashboard."""
        with self._lock:
            summary = {level.name: [] for level in ThreatLevel}
            for flow in self._flows.values():
                summary[flow.level.name].append({
                    "flow_id": flow.flow_id,
                    "score": flow.last_score,
                    "window": flow.profile.window_seconds,
                })
            return summary

    def get_recent_transitions(self, n: int = 20) -> List[dict]:
        """Get recent state transitions."""
        with self._lock:
            return [
                {
                    "flow_id": t.flow_id,
                    "from": t.from_level.name,
                    "to": t.to_level.name,
                    "score": round(t.anomaly_score, 4),
                    "reason": t.reason,
                    "timestamp": t.timestamp,
                }
                for t in list(self._transitions)[-n:]
            ]

    # ── Direct State Update (for external callers) ──

    def update_state(
        self, flow_id: str, anomaly_score: float,
    ) -> List[StateTransition]:
        """Directly update a flow's state with an external anomaly score.

        Used when the score comes from outside the controller's own
        feature pipeline (e.g., from the global iForest detector).
        """
        with self._lock:
            flow = self._get_or_create_flow(flow_id, time.time())
            # Build a dummy feature vector from the score
            dummy_features = np.array([anomaly_score])
            flow.last_score = anomaly_score
            flow.score_history.append(anomaly_score)

            # Run state machine with the provided score
            transitions = []
            old_level = flow.level
            now = time.time()

            if anomaly_score < NORMAL_THRESHOLD:
                flow.consecutive_normal += 1
                flow.consecutive_elevated = 0
            else:
                flow.consecutive_normal = 0
                if anomaly_score >= WATCH_THRESHOLD:
                    flow.consecutive_elevated += 1

            # Downgrade
            if (flow.consecutive_normal >= NORMAL_STREAK_REQUIRED
                    and flow.level != ThreatLevel.BASELINE):
                if flow.level == ThreatLevel.HOSTILE:
                    self._handle_false_positive(flow)
                self._set_level(flow, ThreatLevel.BASELINE,
                                f"{NORMAL_STREAK_REQUIRED} consecutive normals")
                transitions.append(StateTransition(
                    flow_id, old_level, ThreatLevel.BASELINE,
                    anomaly_score, "Consecutive normal windows", now,
                ))

            # Upgrade: BASELINE → WATCH
            elif flow.level == ThreatLevel.BASELINE and anomaly_score >= WATCH_THRESHOLD:
                self._set_level(flow, ThreatLevel.WATCH, f"Score {anomaly_score:.3f}")
                transitions.append(StateTransition(
                    flow_id, old_level, ThreatLevel.WATCH,
                    anomaly_score, "Score crossed WATCH", now,
                ))

            # WATCH → SUSPECT
            elif flow.level == ThreatLevel.WATCH and (
                anomaly_score >= SUSPECT_THRESHOLD
                or flow.consecutive_elevated >= ELEVATED_STREAK_FOR_SUSPECT
            ):
                self._set_level(flow, ThreatLevel.SUSPECT, f"Score {anomaly_score:.3f}")
                transitions.append(StateTransition(
                    flow_id, old_level, ThreatLevel.SUSPECT,
                    anomaly_score, "Score crossed SUSPECT", now,
                ))

            # SUSPECT → HOSTILE
            elif flow.level == ThreatLevel.SUSPECT and anomaly_score >= HOSTILE_THRESHOLD:
                self._set_level(flow, ThreatLevel.HOSTILE, f"Score {anomaly_score:.3f}")
                transitions.append(StateTransition(
                    flow_id, old_level, ThreatLevel.HOSTILE,
                    anomaly_score, "Score crossed HOSTILE", now,
                ))
                self._handle_hostile_confirmed(flow)

            for t in transitions:
                self._transitions.append(t)

            return transitions
