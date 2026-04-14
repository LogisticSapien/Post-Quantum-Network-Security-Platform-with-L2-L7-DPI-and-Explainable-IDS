"""
PQC Migration Readiness Scorer
=================================
Per-host tracker for Post-Quantum Cryptography migration readiness.

Monitors TLS ClientHello key shares across the network and computes:
  - Per-host migration scores (pq_capable / total_tls_sessions)
  - Trend detection (IMPROVING / STABLE / DEGRADING)
  - Network-wide migration readiness score (mean of all hosts)

Integration:
  - Hooks into protocols.py TLSClientHello.has_post_quantum
  - Exposes Prometheus metrics: pqc_migration_score{host}, pqc_network_migration_score
  - Provides /api/pqc/migration endpoint data for Flask dashboard

PQ key share groups detected:
  - 0x6399: X25519Kyber768Draft00
  - 0x639A: SecP256r1Kyber768Draft00
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple


# ──────────────────────────────────────────────────────────────────────
# Trend Classification
# ──────────────────────────────────────────────────────────────────────

class MigrationTrend(Enum):
    IMPROVING = "IMPROVING"
    STABLE = "STABLE"
    DEGRADING = "DEGRADING"
    UNKNOWN = "UNKNOWN"


# ──────────────────────────────────────────────────────────────────────
# Per-Host Record
# ──────────────────────────────────────────────────────────────────────

@dataclass
class HostMigrationRecord:
    """Migration readiness data for a single host."""
    host_ip: str
    total_tls_sessions: int = 0
    pq_capable_sessions: int = 0
    classical_only_sessions: int = 0
    last_seen: float = 0.0
    # Session history: True = PQ capable, False = classical only
    _session_history: deque = field(default_factory=lambda: deque(maxlen=100))

    @property
    def migration_score(self) -> float:
        """PQ readiness score: 0.0 (no PQ) to 1.0 (all PQ)."""
        if self.total_tls_sessions == 0:
            return 0.0
        return self.pq_capable_sessions / self.total_tls_sessions

    @property
    def migration_score_pct(self) -> float:
        """Migration score as percentage."""
        return self.migration_score * 100.0

    @property
    def trend(self) -> MigrationTrend:
        """Compare last 10 vs previous 10 sessions to determine trend."""
        history = list(self._session_history)
        if len(history) < 10:
            return MigrationTrend.UNKNOWN

        recent_10 = history[-10:]
        recent_pq = sum(1 for s in recent_10 if s)

        if len(history) < 20:
            # Not enough data for comparison — use absolute threshold
            ratio = recent_pq / 10.0
            if ratio >= 0.5:
                return MigrationTrend.IMPROVING
            return MigrationTrend.STABLE

        previous_10 = history[-20:-10]
        prev_pq = sum(1 for s in previous_10 if s)

        diff = recent_pq - prev_pq
        if diff >= 2:
            return MigrationTrend.IMPROVING
        elif diff <= -2:
            return MigrationTrend.DEGRADING
        return MigrationTrend.STABLE

    def record_session(self, has_pq: bool) -> None:
        """Record a TLS session observation."""
        self.total_tls_sessions += 1
        if has_pq:
            self.pq_capable_sessions += 1
        else:
            self.classical_only_sessions += 1
        self.last_seen = time.time()
        self._session_history.append(has_pq)

    def to_dict(self) -> dict:
        """Serialize to dict for JSON API."""
        return {
            "host_ip": self.host_ip,
            "total_tls_sessions": self.total_tls_sessions,
            "pq_capable_sessions": self.pq_capable_sessions,
            "classical_only_sessions": self.classical_only_sessions,
            "migration_score": round(self.migration_score, 4),
            "migration_score_pct": round(self.migration_score_pct, 1),
            "trend": self.trend.value,
            "last_seen": self.last_seen,
        }


# ──────────────────────────────────────────────────────────────────────
# PQC Migration Scorer
# ──────────────────────────────────────────────────────────────────────

class PQCMigrationScorer:
    """Network-wide PQC migration readiness tracker.

    Records TLS sessions per host and computes migration scores,
    trends, and network-wide readiness.

    Thread-safe for concurrent packet processing.

    Args:
        max_hosts: Maximum tracked hosts before LRU eviction (default: 10000).
        stale_timeout: Seconds before a host is considered stale (default: 3600).
    """

    def __init__(
        self,
        max_hosts: int = 10000,
        stale_timeout: float = 3600.0,
    ):
        self.max_hosts = max_hosts
        self.stale_timeout = stale_timeout
        self._hosts: Dict[str, HostMigrationRecord] = {}
        self._lock = threading.Lock()
        self._total_sessions: int = 0
        self._total_pq_sessions: int = 0

    def record_tls_session(self, host_ip: str, has_pq_key_share: bool) -> None:
        """Record a TLS session for a host.

        Call this for every parsed TLS ClientHello.

        Args:
            host_ip: Source IP of the TLS client.
            has_pq_key_share: True if ClientHello contains PQ key shares
                             (X25519Kyber768Draft00 or SecP256r1Kyber768Draft00).
        """
        with self._lock:
            self._total_sessions += 1
            if has_pq_key_share:
                self._total_pq_sessions += 1

            if host_ip not in self._hosts:
                self._evict_if_needed()
                self._hosts[host_ip] = HostMigrationRecord(host_ip=host_ip)

            self._hosts[host_ip].record_session(has_pq_key_share)

    def _evict_if_needed(self) -> None:
        """Evict stale or LRU hosts if at capacity."""
        if len(self._hosts) < self.max_hosts:
            return

        now = time.time()
        # Remove stale hosts first
        stale = [
            ip for ip, rec in self._hosts.items()
            if now - rec.last_seen > self.stale_timeout
        ]
        for ip in stale:
            del self._hosts[ip]

        # LRU eviction if still over capacity
        while len(self._hosts) >= self.max_hosts:
            oldest_ip = min(self._hosts, key=lambda ip: self._hosts[ip].last_seen)
            del self._hosts[oldest_ip]

    def get_host_report(self, host_ip: str) -> Optional[dict]:
        """Get migration report for a single host."""
        with self._lock:
            rec = self._hosts.get(host_ip)
            if rec is None:
                return None
            return rec.to_dict()

    def get_all_hosts(self) -> List[dict]:
        """Get all host reports sorted by total sessions (descending)."""
        with self._lock:
            records = [rec.to_dict() for rec in self._hosts.values()]
        records.sort(key=lambda r: r["total_tls_sessions"], reverse=True)
        return records

    @property
    def network_readiness_score(self) -> float:
        """Network-wide migration readiness = mean of all host scores."""
        with self._lock:
            if not self._hosts:
                return 0.0
            scores = [rec.migration_score for rec in self._hosts.values()]
        return sum(scores) / len(scores)

    @property
    def tracked_hosts(self) -> int:
        """Number of currently tracked hosts."""
        with self._lock:
            return len(self._hosts)

    def get_dashboard_table(self) -> List[dict]:
        """Get formatted table data for the analytics dashboard.

        Returns list of dicts with keys: HOST, TOTAL, PQ, CLASSICAL, SCORE, TREND
        """
        hosts = self.get_all_hosts()
        return [
            {
                "HOST": h["host_ip"],
                "TOTAL": h["total_tls_sessions"],
                "PQ": h["pq_capable_sessions"],
                "CLASSICAL": h["classical_only_sessions"],
                "SCORE": f"{h['migration_score_pct']:.1f}%",
                "TREND": h["trend"],
            }
            for h in hosts
        ]

    def get_prometheus_metrics(self) -> str:
        """Generate Prometheus-format metrics.

        Outputs:
          pqc_migration_score{host="x.x.x.x"} 0.064
          pqc_network_migration_score 0.432
        """
        lines = [
            "# HELP pqc_migration_score Per-host PQC migration readiness score (0-1)",
            "# TYPE pqc_migration_score gauge",
        ]

        with self._lock:
            for ip, rec in self._hosts.items():
                lines.append(
                    f'pqc_migration_score{{host="{ip}"}} {rec.migration_score:.4f}'
                )

        net_score = self.network_readiness_score
        lines.extend([
            "# HELP pqc_network_migration_score Network-wide PQC migration readiness (0-1)",
            "# TYPE pqc_network_migration_score gauge",
            f"pqc_network_migration_score {net_score:.4f}",
        ])

        return "\n".join(lines) + "\n"

    def get_api_response(self) -> dict:
        """Generate JSON response for /api/pqc/migration endpoint."""
        return {
            "network_readiness_score": round(self.network_readiness_score, 4),
            "network_readiness_pct": round(self.network_readiness_score * 100, 1),
            "tracked_hosts": self.tracked_hosts,
            "total_sessions": self._total_sessions,
            "total_pq_sessions": self._total_pq_sessions,
            "hosts": self.get_all_hosts(),
        }

    def get_stats(self) -> dict:
        """Get scorer statistics."""
        return {
            "tracked_hosts": self.tracked_hosts,
            "total_sessions": self._total_sessions,
            "total_pq_sessions": self._total_pq_sessions,
            "network_readiness": round(self.network_readiness_score, 4),
        }
