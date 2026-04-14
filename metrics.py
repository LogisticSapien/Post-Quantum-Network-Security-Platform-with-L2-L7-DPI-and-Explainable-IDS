"""
Prometheus Metrics Exporter
============================
Production-grade observability for Quantum Sniffer:
  - Counter: packets, alerts (by severity), bytes
  - Gauge: packets/sec, active flows, PQC entries
  - Histogram: packet processing latency
  - WSGI app for /metrics endpoint
"""

from __future__ import annotations

import threading
import time
from typing import Optional

try:
    from prometheus_client import (
        Counter, Gauge, Histogram, Summary,
        CollectorRegistry, generate_latest, CONTENT_TYPE_LATEST,
    )
    HAS_PROMETHEUS = True
except ImportError:
    HAS_PROMETHEUS = False


class MetricsCollector:
    """Prometheus metrics collector for the capture pipeline."""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled and HAS_PROMETHEUS
        if not self.enabled:
            return

        self.registry = CollectorRegistry()

        # ── Counters ──
        self.packets_total = Counter(
            'qs_packets_total',
            'Total packets processed',
            ['protocol'],
            registry=self.registry,
        )
        self.alerts_total = Counter(
            'qs_alerts_total',
            'Total alerts generated',
            ['severity', 'category'],
            registry=self.registry,
        )
        self.bytes_total = Counter(
            'qs_bytes_total',
            'Total bytes processed',
            registry=self.registry,
        )
        self.drops_total = Counter(
            'qs_drops_total',
            'Total packets dropped (queue full)',
            registry=self.registry,
        )

        # ── Gauges ──
        self.packets_per_sec = Gauge(
            'qs_packets_per_sec',
            'Current packets per second',
            registry=self.registry,
        )
        self.bytes_per_sec = Gauge(
            'qs_bytes_per_sec',
            'Current bytes per second',
            registry=self.registry,
        )
        self.active_flows = Gauge(
            'qs_active_flows',
            'Number of active TCP flows',
            registry=self.registry,
        )
        self.pqc_encrypted_entries = Gauge(
            'qs_pqc_encrypted_entries',
            'PQC encrypted log entries',
            registry=self.registry,
        )
        self.ids_tracked_sources = Gauge(
            'qs_ids_tracked_sources',
            'Number of source IPs tracked by IDS',
            registry=self.registry,
        )
        self.queue_size = Gauge(
            'qs_queue_size',
            'Current packet processing queue size',
            registry=self.registry,
        )

        # ── Histograms ──
        self.processing_latency = Histogram(
            'qs_packet_processing_seconds',
            'Packet processing latency in seconds',
            buckets=[0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1],
            registry=self.registry,
        )

        # ── Summary ──
        self.alert_rate = Summary(
            'qs_alert_processing_seconds',
            'Alert processing time',
            registry=self.registry,
        )

        # ── PQC Transport Counters ──
        self.pqc_alerts_transmitted = Counter(
            'qs_pqc_alerts_transmitted',
            'Total alerts PQC-encrypted and sent by sensor',
            registry=self.registry,
        )
        self.pqc_alerts_received = Counter(
            'qs_pqc_alerts_received',
            'Total alerts PQC-decrypted successfully by aggregator',
            registry=self.registry,
        )

        # Uptime
        self._start_time = time.time()
        self.uptime = Gauge(
            'qs_uptime_seconds',
            'Process uptime in seconds',
            registry=self.registry,
        )

    def record_packet(self, protocol: str, size: int):
        """Record a processed packet."""
        if not self.enabled:
            return
        self.packets_total.labels(protocol=protocol).inc()
        self.bytes_total.inc(size)

    def record_alert(self, severity: str, category: str):
        """Record an IDS alert."""
        if not self.enabled:
            return
        self.alerts_total.labels(severity=severity, category=category).inc()

    def record_drop(self):
        """Record a dropped packet."""
        if not self.enabled:
            return
        self.drops_total.inc()

    def observe_latency(self, seconds: float):
        """Record packet processing latency."""
        if not self.enabled:
            return
        self.processing_latency.observe(seconds)

    def record_pqc_transmitted(self):
        """Record a PQC-encrypted alert sent by a sensor."""
        if not self.enabled:
            return
        self.pqc_alerts_transmitted.inc()

    def record_pqc_received(self):
        """Record a PQC-decrypted alert received by the aggregator."""
        if not self.enabled:
            return
        self.pqc_alerts_received.inc()

    def update_gauges(
        self,
        pps: float = 0,
        bps: float = 0,
        flows: int = 0,
        pqc_entries: int = 0,
        tracked_sources: int = 0,
        q_size: int = 0,
    ):
        """Update gauge values (call periodically from stats thread)."""
        if not self.enabled:
            return
        self.packets_per_sec.set(pps)
        self.bytes_per_sec.set(bps)
        self.active_flows.set(flows)
        self.pqc_encrypted_entries.set(pqc_entries)
        self.ids_tracked_sources.set(tracked_sources)
        self.queue_size.set(q_size)
        self.uptime.set(time.time() - self._start_time)

    def generate_metrics(self) -> bytes:
        """Generate Prometheus text-format metrics output."""
        if not self.enabled:
            return b"# Prometheus client not installed\n"
        self.uptime.set(time.time() - self._start_time)
        return generate_latest(self.registry)

    @property
    def content_type(self) -> str:
        """Content-Type header for metrics endpoint."""
        if not self.enabled:
            return "text/plain"
        return CONTENT_TYPE_LATEST


# ── Singleton for easy import ──
_global_metrics: Optional[MetricsCollector] = None


def get_metrics(enabled: bool = True) -> MetricsCollector:
    """Get or create the global metrics collector."""
    global _global_metrics
    if _global_metrics is None:
        _global_metrics = MetricsCollector(enabled=enabled)
    return _global_metrics


def reset_metrics():
    """Reset global metrics (for testing)."""
    global _global_metrics
    _global_metrics = None
