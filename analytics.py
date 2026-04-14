"""
Analytics Engine
=================
Real-time network traffic analytics:
  • Bandwidth monitoring (per-connection & aggregate)
  • Protocol distribution statistics
  • Top talkers analysis
  • TCP flow tracking with state machine
  • GeoIP lookup (async, cached)
  • Session reconstruction
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict, deque, Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


# ──────────────────────────────────────────────────────────────────────
# Bandwidth Monitor
# ──────────────────────────────────────────────────────────────────────

class BandwidthMonitor:
    """Track bytes/sec with rolling windows."""

    def __init__(self, window: float = 5.0):
        self.window = window
        self._samples: deque = deque(maxlen=10000)
        self._per_conn: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.total_bytes = 0
        self.total_packets = 0

    def record(self, size: int, conn_key: Optional[str] = None):
        now = time.time()
        self._samples.append((now, size))
        self.total_bytes += size
        self.total_packets += 1
        if conn_key:
            self._per_conn[conn_key].append((now, size))

    @property
    def bytes_per_second(self) -> float:
        now = time.time()
        cutoff = now - self.window
        total = sum(s for t, s in self._samples if t > cutoff)
        return total / self.window

    @property
    def packets_per_second(self) -> float:
        now = time.time()
        cutoff = now - self.window
        count = sum(1 for t, _ in self._samples if t > cutoff)
        return count / self.window

    def conn_bps(self, conn_key: str) -> float:
        now = time.time()
        cutoff = now - self.window
        samples = self._per_conn.get(conn_key, [])
        total = sum(s for t, s in samples if t > cutoff)
        return total / self.window

    @property
    def top_connections(self) -> List[Tuple[str, float]]:
        """Top 10 connections by bandwidth."""
        results = []
        for key in list(self._per_conn.keys()):
            bps = self.conn_bps(key)
            if bps > 0:
                results.append((key, bps))
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:10]

    def format_bytes(self, n: float) -> str:
        for unit in ('B', 'KB', 'MB', 'GB', 'TB'):
            if abs(n) < 1024:
                return f"{n:.1f} {unit}"
            n /= 1024
        return f"{n:.1f} PB"


# ──────────────────────────────────────────────────────────────────────
# Protocol Distribution
# ──────────────────────────────────────────────────────────────────────

class ProtocolStats:
    """Track protocol distribution."""

    def __init__(self):
        self.counts: Counter = Counter()
        self.bytes: Counter = Counter()
        self._first_seen: Dict[str, float] = {}
        self._last_seen: Dict[str, float] = {}

    def record(self, protocol: str, size: int = 0):
        now = time.time()
        self.counts[protocol] += 1
        self.bytes[protocol] += size
        if protocol not in self._first_seen:
            self._first_seen[protocol] = now
        self._last_seen[protocol] = now

    @property
    def distribution(self) -> Dict[str, float]:
        """Protocol distribution as percentages."""
        total = sum(self.counts.values())
        if total == 0:
            return {}
        return {p: (c / total) * 100 for p, c in self.counts.most_common()}

    @property
    def top_protocols(self) -> List[Tuple[str, int]]:
        return self.counts.most_common(15)


# ──────────────────────────────────────────────────────────────────────
# Top Talkers
# ──────────────────────────────────────────────────────────────────────

class TopTalkers:
    """Track most active IP addresses with bounded memory."""

    def __init__(self, max_tracked: int = 1000):
        self.tx_bytes: Counter = Counter()
        self.rx_bytes: Counter = Counter()
        self.tx_packets: Counter = Counter()
        self.rx_packets: Counter = Counter()
        self._connections: Dict[str, Set[str]] = defaultdict(set)
        self.max_tracked = max_tracked
        self.evictions: int = 0

    def record(self, src_ip: str, dst_ip: str, size: int):
        self.tx_bytes[src_ip] += size
        self.rx_bytes[dst_ip] += size
        self.tx_packets[src_ip] += 1
        self.rx_packets[dst_ip] += 1
        self._connections[src_ip].add(dst_ip)
        # Evict lowest-count entries when over capacity
        if len(self.tx_bytes) > self.max_tracked:
            self._evict()

    def _evict(self):
        """Prune lowest-count entries down to max_tracked."""
        excess = len(self.tx_bytes) - self.max_tracked
        if excess <= 0:
            return
        # Keep top entries, remove the rest
        keep = {ip for ip, _ in self.tx_bytes.most_common(self.max_tracked)}
        for ip in list(self.tx_bytes.keys()):
            if ip not in keep:
                del self.tx_bytes[ip]
                self.rx_bytes.pop(ip, None)
                self.tx_packets.pop(ip, None)
                self.rx_packets.pop(ip, None)
                self._connections.pop(ip, None)
                self.evictions += 1

    @property
    def top_senders(self) -> List[Tuple[str, int]]:
        return self.tx_bytes.most_common(10)

    @property
    def top_receivers(self) -> List[Tuple[str, int]]:
        return self.rx_bytes.most_common(10)

    @property
    def most_connected(self) -> List[Tuple[str, int]]:
        """IPs with most unique connections."""
        items = [(ip, len(conns)) for ip, conns in self._connections.items()]
        items.sort(key=lambda x: x[1], reverse=True)
        return items[:10]


# ──────────────────────────────────────────────────────────────────────
# TCP Flow Tracker (state machine)
# ──────────────────────────────────────────────────────────────────────

class FlowState(Enum):
    NEW          = "NEW"
    SYN_SENT     = "SYN_SENT"
    SYN_RECEIVED = "SYN_RECEIVED"
    ESTABLISHED  = "ESTABLISHED"
    FIN_WAIT     = "FIN_WAIT"
    CLOSING      = "CLOSING"
    CLOSED       = "CLOSED"
    RESET        = "RESET"


@dataclass
class TCPFlow:
    """Tracked TCP connection."""
    src_ip: str
    src_port: int
    dst_ip: str
    dst_port: int
    state: FlowState = FlowState.NEW
    start_time: float = 0.0
    last_seen: float = 0.0
    packets: int = 0
    bytes_sent: int = 0
    bytes_recv: int = 0
    sni: Optional[str] = None
    ja3: Optional[str] = None

    @property
    def key(self) -> str:
        return f"{self.src_ip}:{self.src_port}->{self.dst_ip}:{self.dst_port}"

    @property
    def duration(self) -> float:
        return self.last_seen - self.start_time if self.start_time > 0 else 0.0

    @property
    def is_active(self) -> bool:
        return self.state in (FlowState.NEW, FlowState.SYN_SENT,
                              FlowState.SYN_RECEIVED, FlowState.ESTABLISHED)


class FlowTracker:
    """Track TCP connections with a simplified state machine and TTL eviction."""

    def __init__(self, timeout: float = 120.0, eviction_ttl: float = 300.0):
        self.flows: Dict[str, TCPFlow] = {}
        self.completed_flows: deque = deque(maxlen=1000)
        self.timeout = timeout
        self.eviction_ttl = eviction_ttl  # 5 minutes hard TTL
        self._lock = threading.Lock()
        self.flows_evicted_total: int = 0
        # Start background eviction thread
        self._eviction_running = True
        self._eviction_thread = threading.Thread(
            target=self._eviction_loop, daemon=True, name="flow-evictor"
        )
        self._eviction_thread.start()

    def _eviction_loop(self):
        """Background thread: evict stale flows every 60s."""
        while self._eviction_running:
            time.sleep(60)
            self._evict_stale()

    def _evict_stale(self) -> int:
        """Evict flows with last_seen > eviction_ttl. Returns count evicted."""
        now = time.time()
        evicted = 0
        with self._lock:
            stale = [k for k, f in self.flows.items()
                     if now - f.last_seen > self.eviction_ttl]
            for k in stale:
                flow = self.flows.pop(k, None)
                if flow:
                    flow.state = FlowState.CLOSED
                    self.completed_flows.append(flow)
                    evicted += 1
        self.flows_evicted_total += evicted
        return evicted

    def stop(self):
        """Stop the eviction thread."""
        self._eviction_running = False

    def _flow_key(self, src_ip: str, src_port: int, dst_ip: str, dst_port: int) -> str:
        # Normalize direction
        if (src_ip, src_port) < (dst_ip, dst_port):
            return f"{src_ip}:{src_port}<->{dst_ip}:{dst_port}"
        return f"{dst_ip}:{dst_port}<->{src_ip}:{src_port}"

    def update(
        self, src_ip: str, src_port: int, dst_ip: str, dst_port: int,
        flags: int, payload_len: int = 0,
        sni: Optional[str] = None, ja3: Optional[str] = None,
    ):
        """Update flow state based on TCP flags."""
        now = time.time()
        key = self._flow_key(src_ip, src_port, dst_ip, dst_port)

        with self._lock:
            flow = self.flows.get(key)

            if flow is None:
                flow = TCPFlow(
                    src_ip=src_ip, src_port=src_port,
                    dst_ip=dst_ip, dst_port=dst_port,
                    start_time=now,
                )
                self.flows[key] = flow

            flow.last_seen = now
            flow.packets += 1
            flow.bytes_sent += payload_len

            if sni:
                flow.sni = sni
            if ja3:
                flow.ja3 = ja3

            from protocols import TCPFlags

            # State transitions
            if flags & TCPFlags.RST:
                flow.state = FlowState.RESET
                self._complete_flow(key)
            elif flags & TCPFlags.SYN and not (flags & TCPFlags.ACK):
                flow.state = FlowState.SYN_SENT
            elif flags & TCPFlags.SYN and flags & TCPFlags.ACK:
                flow.state = FlowState.SYN_RECEIVED
            elif flags & TCPFlags.ACK and flow.state == FlowState.SYN_RECEIVED:
                flow.state = FlowState.ESTABLISHED
            elif flags & TCPFlags.FIN:
                if flow.state == FlowState.FIN_WAIT:
                    flow.state = FlowState.CLOSED
                    self._complete_flow(key)
                else:
                    flow.state = FlowState.FIN_WAIT

    def _complete_flow(self, key: str):
        """Move flow to completed."""
        flow = self.flows.pop(key, None)
        if flow:
            self.completed_flows.append(flow)

    def cleanup_stale(self):
        """Remove stale flows (uses soft timeout)."""
        now = time.time()
        with self._lock:
            stale = [k for k, f in self.flows.items()
                     if now - f.last_seen > self.timeout]
            for k in stale:
                self._complete_flow(k)

    @property
    def active_flows(self) -> List[TCPFlow]:
        with self._lock:
            return sorted(
                [f for f in self.flows.values() if f.is_active],
                key=lambda f: f.last_seen,
                reverse=True,
            )[:50]

    @property
    def flow_count(self) -> dict:
        with self._lock:
            counts = Counter(f.state.value for f in self.flows.values())
        return dict(counts)


# ──────────────────────────────────────────────────────────────────────
# GeoIP Lookup (cached, rate-limited)
# ──────────────────────────────────────────────────────────────────────

class GeoIPLookup:
    """GeoIP lookup via ip-api.com with caching."""

    API_URL = "http://ip-api.com/json/{ip}?fields=status,country,regionName,city,isp,org,as,lat,lon"

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self._cache: Dict[str, dict] = {}
        self._lock = threading.Lock()
        self._rate_limit = 0.7  # seconds between requests
        self._last_request = 0.0

    def _is_private(self, ip: str) -> bool:
        """Check if IP is RFC1918 private."""
        parts = ip.split('.')
        if len(parts) != 4:
            return True
        try:
            first, second = int(parts[0]), int(parts[1])
        except ValueError:
            return True
        if first == 10:
            return True
        if first == 172 and 16 <= second <= 31:
            return True
        if first == 192 and second == 168:
            return True
        if first == 127:
            return True
        return False

    def lookup(self, ip: str) -> Optional[dict]:
        """Look up geolocation for an IP address."""
        if not self.enabled or not HAS_REQUESTS:
            return None

        if self._is_private(ip):
            return {"status": "private", "country": "Private", "city": "LAN"}

        with self._lock:
            if ip in self._cache:
                return self._cache[ip]

        now = time.time()
        if now - self._last_request < self._rate_limit:
            return None

        try:
            self._last_request = now
            resp = requests.get(
                self.API_URL.format(ip=ip),
                timeout=2,
            )
            data = resp.json()
            with self._lock:
                self._cache[ip] = data
            return data
        except Exception:
            return None

    def get_cached(self, ip: str) -> Optional[dict]:
        """Get cached result only (no API call)."""
        with self._lock:
            return self._cache.get(ip)

    @property
    def cache_size(self) -> int:
        return len(self._cache)


# ──────────────────────────────────────────────────────────────────────
# Unified Analytics Manager
# ──────────────────────────────────────────────────────────────────────

class AnalyticsManager:
    """Central analytics coordinator."""

    def __init__(self, geo_enabled: bool = False):
        self.bandwidth = BandwidthMonitor()
        self.protocols = ProtocolStats()
        self.talkers = TopTalkers()
        self.flows = FlowTracker()
        self.geo = GeoIPLookup(enabled=geo_enabled)
        self.start_time = time.time()

        # DNS query log
        self.dns_queries: deque = deque(maxlen=500)
        # TLS SNI log
        self.tls_snis: deque = deque(maxlen=500)
        # HTTP requests log
        self.http_requests: deque = deque(maxlen=500)

    def record_packet(
        self, protocol: str, src_ip: str, dst_ip: str,
        size: int, src_port: int = 0, dst_port: int = 0,
    ):
        """Record basic packet stats."""
        conn_key = f"{src_ip}:{src_port}<->{dst_ip}:{dst_port}"
        self.bandwidth.record(size, conn_key)
        self.protocols.record(protocol, size)
        self.talkers.record(src_ip, dst_ip, size)

    def record_tcp_flow(
        self, src_ip: str, src_port: int, dst_ip: str, dst_port: int,
        flags: int, payload_len: int = 0,
        sni: Optional[str] = None, ja3: Optional[str] = None,
    ):
        """Update TCP flow state."""
        self.flows.update(
            src_ip, src_port, dst_ip, dst_port,
            flags, payload_len, sni, ja3,
        )

    def record_dns(self, query: str, qtype: str, src_ip: str):
        self.dns_queries.append({
            "time": time.time(), "query": query,
            "type": qtype, "src": src_ip,
        })

    def record_tls_sni(self, sni: str, ja3: Optional[str], src_ip: str):
        self.tls_snis.append({
            "time": time.time(), "sni": sni,
            "ja3": ja3, "src": src_ip,
        })

    def record_http(self, method: str, uri: str, host: str, src_ip: str):
        self.http_requests.append({
            "time": time.time(), "method": method,
            "uri": uri, "host": host, "src": src_ip,
        })

    @property
    def uptime(self) -> float:
        return time.time() - self.start_time

    @property
    def summary(self) -> dict:
        return {
            "uptime_seconds": self.uptime,
            "total_packets": self.bandwidth.total_packets,
            "total_bytes": self.bandwidth.total_bytes,
            "bytes_per_second": self.bandwidth.bytes_per_second,
            "packets_per_second": self.bandwidth.packets_per_second,
            "protocols": dict(self.protocols.counts),
            "active_flows": len(self.flows.active_flows),
            "flow_states": self.flows.flow_count,
            "geo_cache_size": self.geo.cache_size,
            "dns_queries_logged": len(self.dns_queries),
            "tls_snis_logged": len(self.tls_snis),
            "http_requests_logged": len(self.http_requests),
        }

    def export_json(self) -> dict:
        """Export all analytics as JSON-serializable dict."""
        return {
            "summary": self.summary,
            "top_senders": self.talkers.top_senders,
            "top_receivers": self.talkers.top_receivers,
            "most_connected": self.talkers.most_connected,
            "protocol_distribution": self.protocols.distribution,
            "active_flows": [
                {
                    "key": f.key,
                    "state": f.state.value,
                    "duration": f.duration,
                    "packets": f.packets,
                    "sni": f.sni,
                    "ja3": f.ja3,
                }
                for f in self.flows.active_flows
            ],
            "recent_dns": list(self.dns_queries)[-20:],
            "recent_tls": list(self.tls_snis)[-20:],
            "recent_http": list(self.http_requests)[-20:],
        }
