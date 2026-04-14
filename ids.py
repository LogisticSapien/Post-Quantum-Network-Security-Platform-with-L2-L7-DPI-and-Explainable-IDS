"""
Intrusion Detection System (IDS) v2
=====================================
Real-time threat detection engine with EXPLAINABILITY:
  • Port scan detection (SYN, FIN, XMAS, NULL)
  • SYN flood detection
  • DNS tunneling detection via Shannon entropy
  • ARP spoofing detection
  • Brute-force detection on auth ports
  • Anomalous TTL detection
  • Protocol anomaly detection
  • ICMP tunneling detection
  • MITRE ATT&CK-referenced threat scoring
  • Rich evidence chains with natural-language explanations
  • Structured contributing factors with confidence weights
  • Recommended response actions
"""

from __future__ import annotations

import ipaddress
import math
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, List, Optional, Set, Tuple

from protocols import (
    ARPPacket, DNSMessage, ICMPPacket, IPv4Packet, TCPSegment, UDPDatagram,
    TLSClientHello, TCPFlags,
)


# ──────────────────────────────────────────────────────────────────────
# Severity & Threat Classification
# ──────────────────────────────────────────────────────────────────────

class Severity(IntEnum):
    INFO     = 1
    LOW      = 2
    MEDIUM   = 3
    HIGH     = 4
    CRITICAL = 5


SEVERITY_LABELS = {
    Severity.INFO:     "INFO",
    Severity.LOW:      "LOW",
    Severity.MEDIUM:   "MEDIUM",
    Severity.HIGH:     "HIGH",
    Severity.CRITICAL: "CRITICAL",
}

SEVERITY_ICONS = {
    Severity.INFO:     "i",
    Severity.LOW:      "!",
    Severity.MEDIUM:   "!!",
    Severity.HIGH:     "!!!",
    Severity.CRITICAL: "!!!!",
}

# MITRE ATT&CK Technique references
MITRE = {
    "port_scan":      "T1046 - Network Service Discovery",
    "syn_flood":      "T1498.001 - DoS: Direct Flood",
    "dns_tunnel":     "T1071.004 - App Layer Protocol: DNS",
    "arp_spoof":      "T1557.002 - ARP Cache Poisoning",
    "brute_force":    "T1110 - Brute Force",
    "ttl_anomaly":    "T1090.003 - Proxy: Multi-hop Proxy",
    "proto_anomaly":  "T1036 - Masquerading",
    "quant_vuln":     "T1600 - Weaken Encryption",
    "icmp_tunnel":    "T1095 - Non-Application Layer Protocol",
    "dns_exfil":      "T1048.003 - Exfiltration Over Alternative Protocol: DNS",
}

# Recommended response actions
RESPONSE_ACTIONS = {
    "PORT_SCAN": [
        "Block source IP at firewall",
        "Enable rate limiting on target host",
        "Monitor for follow-up exploitation attempts",
    ],
    "SYN_FLOOD": [
        "Enable SYN cookies on target",
        "Activate DDoS mitigation / rate limiting",
        "Block source IPs or enable upstream filtering",
        "Alert network operations team",
    ],
    "DNS_TUNNEL": [
        "Block DNS queries to suspicious domains",
        "Inspect endpoint for malware / C2 implants",
        "Enable DNS query logging and deep inspection",
    ],
    "DNS_EXFIL": [
        "Block DNS queries with abnormally long labels",
        "Inspect source host for data exfiltration tools",
    ],
    "DNS_FLOOD": [
        "Rate-limit DNS queries from source",
        "Check for DNS amplification abuse",
    ],
    "ARP_SPOOF": [
        "Enable Dynamic ARP Inspection (DAI)",
        "Verify gateway MAC address manually",
        "Isolate suspected attacker's switch port",
        "Alert security team immediately",
    ],
    "BRUTE_FORCE": [
        "Temporarily block source IP",
        "Enable account lockout policies",
        "Require MFA on target service",
        "Check for compromised credentials",
    ],
    "TTL_ANOMALY": [
        "Investigate routing path for anomalies",
        "Check for traceroute or proxy activity",
    ],
    "PROTO_ANOMALY": [
        "Block source IP — likely malicious scan tool",
        "Enable strict TCP validation on firewall",
    ],
    "ICMP_TUNNEL": [
        "Block ICMP payloads > 64 bytes at firewall",
        "Inspect endpoint for tunneling tools (e.g., ptunnel)",
        "Rate-limit ICMP traffic from source",
    ],
    "ICMP_FLOOD": [
        "Rate-limit ICMP from source",
        "Check for ping flood or Smurf attack",
    ],
}


# ──────────────────────────────────────────────────────────────────────
# Evidence & Explainability
# ──────────────────────────────────────────────────────────────────────

@dataclass
class EvidenceFactor:
    """A single contributing factor to a detection."""
    metric: str         # e.g., "SYN packet rate"
    observed: str       # e.g., "142 SYN/5s"
    threshold: str      # e.g., "threshold: 100 SYN/5s"
    weight: float       # 0.0 - 1.0 contribution to confidence


@dataclass
class ThreatEvent:
    """A detected threat/anomaly with full explainability."""
    timestamp: float
    severity: Severity
    category: str
    description: str
    source_ip: Optional[str] = None
    dest_ip: Optional[str] = None
    source_port: Optional[int] = None
    dest_port: Optional[int] = None
    confidence: float = 0.0
    mitre_ref: Optional[str] = None
    raw_evidence: Optional[str] = None

    # ── Explainability fields ──
    explanation: str = ""               # Natural-language WHY
    evidence_factors: List[EvidenceFactor] = field(default_factory=list)
    response_actions: List[str] = field(default_factory=list)
    detection_logic: str = ""           # How the detector works

    @property
    def severity_label(self) -> str:
        return SEVERITY_LABELS.get(self.severity, str(self.severity))

    @property
    def full_explanation(self) -> str:
        """Rich, multi-line explanation for display."""
        lines = [f"[{self.severity_label}] {self.category}: {self.description}"]
        if self.explanation:
            lines.append(f"  WHY: {self.explanation}")
        if self.evidence_factors:
            lines.append("  EVIDENCE:")
            for ef in self.evidence_factors:
                lines.append(f"    - {ef.metric}: {ef.observed} ({ef.threshold}) [weight={ef.weight:.1f}]")
        if self.mitre_ref:
            lines.append(f"  MITRE: {self.mitre_ref}")
        if self.detection_logic:
            lines.append(f"  LOGIC: {self.detection_logic}")
        if self.response_actions:
            lines.append("  ACTIONS:")
            for a in self.response_actions:
                lines.append(f"    -> {a}")
        return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────
# Detection Configuration
# ──────────────────────────────────────────────────────────────────────

@dataclass
class IDSConfig:
    """Tuneable detection parameters."""
    port_scan_threshold: int = 15
    port_scan_window: float = 10.0
    syn_flood_threshold: int = 100
    syn_flood_window: float = 5.0
    dns_entropy_threshold: float = 3.8
    dns_label_len_threshold: int = 40
    dns_query_rate_threshold: int = 50
    dns_query_window: float = 10.0
    arp_change_alert: bool = True
    brute_force_threshold: int = 10
    brute_force_window: float = 30.0
    brute_force_ports: Set[int] = field(
        default_factory=lambda: {22, 23, 3389, 5900, 21, 110, 143, 993, 995, 3306, 5432, 1433, 27017}
    )
    ttl_min_suspicious: int = 1
    ttl_max_suspicious: int = 5
    ttl_known_hops: Dict[str, int] = field(default_factory=dict)
    icmp_payload_threshold: int = 64
    icmp_rate_threshold: int = 30
    icmp_rate_window: float = 10.0

    # ── Whitelist (false-positive suppression) ──
    whitelist_ip_cidrs: List[str] = field(default_factory=lambda: [
        # Cloudflare
        "104.16.0.0/12", "162.158.0.0/15", "173.245.48.0/20",
        "103.21.244.0/22", "103.22.200.0/22", "103.31.4.0/22",
        "141.101.64.0/18", "108.162.192.0/18", "190.93.240.0/20",
        "188.114.96.0/20", "197.234.240.0/22", "198.41.128.0/17",
        # Akamai (some common ranges)
        "23.0.0.0/12", "104.64.0.0/10",
        # Fastly
        "151.101.0.0/16",
    ])
    whitelist_multicast_ips: Set[str] = field(default_factory=lambda: {
        "239.255.255.250",  # SSDP
        "224.0.0.251",      # mDNS
        "224.0.0.252",      # LLMNR
        "224.0.0.1",        # All Hosts
        "224.0.0.2",        # All Routers
    })
    whitelist_dns_domains: Set[str] = field(default_factory=lambda: {
        "windowsupdate.com", "microsoft.com", "googleapis.com",
        "gstatic.com", "google.com", "apple.com", "icloud.com",
        "akamaized.net", "cloudflare.com", "amazonaws.com",
        "azure.com", "live.com", "office.com", "office365.com",
        "msftconnecttest.com", "digicert.com", "verisign.com",
        "github.com", "npmjs.org", "pypi.org",
    })


# ──────────────────────────────────────────────────────────────────────
# IDS Engine
# ──────────────────────────────────────────────────────────────────────

class IDSEngine:
    """Real-time Intrusion Detection System with Explainability."""

    def __init__(self, config: Optional[IDSConfig] = None, sensitivity: str = "medium"):
        self.config = config or IDSConfig()
        self._apply_sensitivity(sensitivity)

        self._port_hits: Dict[str, deque] = defaultdict(lambda: deque(maxlen=500))
        self._syn_hits: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self._syn_ack_count: Dict[str, int] = defaultdict(int)  # track responses
        self._dns_queries: Dict[str, deque] = defaultdict(lambda: deque(maxlen=500))
        self._arp_table: Dict[str, str] = {}
        self._auth_hits: Dict[Tuple[str, int], deque] = defaultdict(lambda: deque(maxlen=200))
        self._icmp_hits: Dict[str, deque] = defaultdict(lambda: deque(maxlen=300))
        self._scan_types: Dict[str, Set[str]] = defaultdict(set)

        self.alerts: List[ThreatEvent] = []
        self.stats = {
            "total_packets_analyzed": 0,
            "threats_detected": 0,
            "severity_counts": defaultdict(int),
            "whitelist_suppressed": 0,
        }

        # Pre-parse CIDR whitelist into network objects
        self._whitelist_nets = []
        for cidr in self.config.whitelist_ip_cidrs:
            try:
                self._whitelist_nets.append(ipaddress.ip_network(cidr, strict=False))
            except ValueError:
                pass

    def _is_whitelisted_ip(self, ip_str: Optional[str]) -> bool:
        """Check if an IP falls within any whitelisted CIDR range."""
        if not ip_str:
            return False
        try:
            addr = ipaddress.ip_address(ip_str)
        except ValueError:
            return False
        return any(addr in net for net in self._whitelist_nets)

    def _is_whitelisted_domain(self, domain: str) -> bool:
        """Check if a domain matches any whitelisted suffix."""
        domain_lower = domain.lower()
        return any(domain_lower.endswith(d) for d in self.config.whitelist_dns_domains)

    def _apply_sensitivity(self, level: str):
        if level == "low":
            self.config.port_scan_threshold = 25
            self.config.syn_flood_threshold = 200
            self.config.dns_entropy_threshold = 4.0
            self.config.brute_force_threshold = 20
        elif level == "high":
            self.config.port_scan_threshold = 8
            self.config.syn_flood_threshold = 50
            self.config.dns_entropy_threshold = 3.5
            self.config.brute_force_threshold = 5

    def analyze_packet(
        self,
        ip: Optional[IPv4Packet] = None,
        tcp: Optional[TCPSegment] = None,
        udp: Optional[UDPDatagram] = None,
        dns: Optional[DNSMessage] = None,
        arp: Optional[ARPPacket] = None,
        icmp: Optional[ICMPPacket] = None,
        tls: Optional[TLSClientHello] = None,
    ) -> List[ThreatEvent]:
        """Analyze a parsed packet for threats. Returns list of new alerts."""
        self.stats["total_packets_analyzed"] += 1
        new_alerts = []
        now = time.time()

        src_ip = ip.src_ip if ip else None
        dst_ip = ip.dst_ip if ip else None

        # Track SYN-ACK responses for SYN flood explainability
        if tcp and tcp.is_syn_ack and src_ip:
            self._syn_ack_count[src_ip] = self._syn_ack_count.get(src_ip, 0) + 1

        if tcp and src_ip:
            scan_type = None
            if tcp.is_syn:
                scan_type = "SYN"
            elif tcp.flags == TCPFlags.FIN:
                scan_type = "FIN"
            elif tcp.is_xmas:
                scan_type = "XMAS"
            elif tcp.is_null:
                scan_type = "NULL"

            if scan_type:
                self._scan_types[src_ip].add(scan_type)
                self._port_hits[src_ip].append((now, tcp.dst_port))
                alert = self._check_port_scan(src_ip, dst_ip, now)
                if alert:
                    new_alerts.append(alert)

        if tcp and tcp.is_syn and dst_ip:
            self._syn_hits[dst_ip].append(now)
            alert = self._check_syn_flood(src_ip, dst_ip, now)
            if alert:
                new_alerts.append(alert)

        if dns and src_ip:
            for q in dns.questions:
                self._dns_queries[src_ip].append((now, q.name))
            alerts = self._check_dns_tunneling(src_ip, dst_ip, dns, now)
            new_alerts.extend(alerts)

        if arp:
            alert = self._check_arp_spoofing(arp, now)
            if alert:
                new_alerts.append(alert)

        if tcp and tcp.is_syn and src_ip:
            if tcp.dst_port in self.config.brute_force_ports:
                key = (src_ip, tcp.dst_port)
                self._auth_hits[key].append(now)
                alert = self._check_brute_force(src_ip, dst_ip, tcp.dst_port, now)
                if alert:
                    new_alerts.append(alert)

        if ip and ip.ttl <= self.config.ttl_max_suspicious:
            alert = self._check_ttl_anomaly(ip, now)
            if alert:
                new_alerts.append(alert)

        if tcp:
            alert = self._check_proto_anomaly(tcp, src_ip, dst_ip, now)
            if alert:
                new_alerts.append(alert)

        if icmp and src_ip:
            self._icmp_hits[src_ip].append((now, len(icmp.payload)))
            alert = self._check_icmp_tunnel(src_ip, dst_ip, icmp, now)
            if alert:
                new_alerts.append(alert)

        for alert in new_alerts:
            self.alerts.append(alert)
            self.stats["threats_detected"] += 1
            self.stats["severity_counts"][alert.severity] += 1

        return new_alerts

    # ── Detection methods (with explainability) ──

    def _check_port_scan(self, src_ip: str, dst_ip: Optional[str], now: float) -> Optional[ThreatEvent]:
        hits = self._port_hits[src_ip]
        cutoff = now - self.config.port_scan_window
        recent = [(t, p) for t, p in hits if t > cutoff]

        unique_ports = len(set(p for _, p in recent))
        if unique_ports < self.config.port_scan_threshold:
            return None

        # Whitelist check: suppress CDN reconnection bursts
        if self._is_whitelisted_ip(src_ip):
            self.stats["whitelist_suppressed"] += 1
            return None

        scan_methods = self._scan_types.get(src_ip, {"SYN"})
        scan_str = "/".join(sorted(scan_methods))
        confidence = min(1.0, unique_ports / (self.config.port_scan_threshold * 2))
        severity = Severity.HIGH if unique_ports > self.config.port_scan_threshold * 2 else Severity.MEDIUM
        rate = len(recent) / self.config.port_scan_window

        port_list = sorted(set(p for _, p in recent))[:20]

        return ThreatEvent(
            timestamp=now, severity=severity, category="PORT_SCAN",
            description=f"{scan_str} scan detected from {src_ip}: {unique_ports} unique ports in {self.config.port_scan_window}s",
            source_ip=src_ip, dest_ip=dst_ip,
            confidence=confidence, mitre_ref=MITRE["port_scan"],
            raw_evidence=f"ports={port_list}",
            explanation=(
                f"Detected because {src_ip} probed {unique_ports} unique destination ports "
                f"within {self.config.port_scan_window}s using {scan_str} scan technique(s). "
                f"This exceeds the threshold of {self.config.port_scan_threshold} unique ports. "
                f"Scan rate: {rate:.1f} probes/sec. Target ports include: {port_list[:10]}."
            ),
            evidence_factors=[
                EvidenceFactor("Unique ports probed", str(unique_ports),
                               f"threshold: {self.config.port_scan_threshold}", 0.5),
                EvidenceFactor("Scan rate", f"{rate:.1f} probes/sec",
                               f"window: {self.config.port_scan_window}s", 0.3),
                EvidenceFactor("Scan method(s)", scan_str, "SYN/FIN/XMAS/NULL", 0.2),
            ],
            detection_logic=f"Count unique dst_port per src_ip within {self.config.port_scan_window}s sliding window",
            response_actions=RESPONSE_ACTIONS["PORT_SCAN"],
        )

    def _check_syn_flood(self, src_ip: Optional[str], dst_ip: str, now: float) -> Optional[ThreatEvent]:
        hits = self._syn_hits[dst_ip]
        cutoff = now - self.config.syn_flood_window
        recent = sum(1 for t in hits if t > cutoff)

        if recent < self.config.syn_flood_threshold:
            return None

        rate = recent / self.config.syn_flood_window
        ack_count = self._syn_ack_count.get(dst_ip, 0)
        confidence = min(1.0, recent / (self.config.syn_flood_threshold * 3))

        return ThreatEvent(
            timestamp=now, severity=Severity.CRITICAL, category="SYN_FLOOD",
            description=f"SYN flood targeting {dst_ip}: {recent} SYNs in {self.config.syn_flood_window}s",
            source_ip=src_ip, dest_ip=dst_ip,
            confidence=confidence, mitre_ref=MITRE["syn_flood"],
            explanation=(
                f"Detected {recent} SYN packets in {self.config.syn_flood_window}s targeting {dst_ip} "
                f"with only {ack_count} corresponding SYN-ACK responses observed. "
                f"Rate: {rate:.1f} SYN/sec exceeds threshold of "
                f"{self.config.syn_flood_threshold / self.config.syn_flood_window:.0f} SYN/sec. "
                f"This asymmetry between SYN and SYN-ACK indicates a denial-of-service attack "
                f"attempting to exhaust the target's TCP connection table."
            ),
            evidence_factors=[
                EvidenceFactor("SYN packet rate", f"{rate:.1f} SYN/sec",
                               f"threshold: {self.config.syn_flood_threshold/self.config.syn_flood_window:.0f}/sec", 0.4),
                EvidenceFactor("SYN count", str(recent),
                               f"threshold: {self.config.syn_flood_threshold}", 0.3),
                EvidenceFactor("SYN-ACK responses", str(ack_count),
                               "expected: proportional to SYNs", 0.2),
                EvidenceFactor("Attack duration", f"{self.config.syn_flood_window}s window",
                               "sustained", 0.1),
            ],
            detection_logic="Count SYN packets per destination IP in sliding window, compare with SYN-ACK responses",
            response_actions=RESPONSE_ACTIONS["SYN_FLOOD"],
        )

    def _check_dns_tunneling(
        self, src_ip: str, dst_ip: Optional[str], dns: DNSMessage, now: float
    ) -> List[ThreatEvent]:
        alerts = []

        for q in dns.questions:
            name = q.name
            # Whitelist: skip known legitimate domains
            if self._is_whitelisted_domain(name):
                self.stats["whitelist_suppressed"] += 1
                continue
            labels = name.split('.')

            for label in labels:
                if len(label) < 8:
                    continue
                entropy = _shannon_entropy(label)

                if entropy >= self.config.dns_entropy_threshold:
                    confidence = min(1.0, (entropy - 3.0) / 2.0)
                    normal_entropy = _shannon_entropy("google")  # ~2.25

                    alerts.append(ThreatEvent(
                        timestamp=now, severity=Severity.HIGH, category="DNS_TUNNEL",
                        description=f"High-entropy DNS query from {src_ip}: '{name}' (entropy={entropy:.2f})",
                        source_ip=src_ip, dest_ip=dst_ip,
                        confidence=confidence, mitre_ref=MITRE["dns_tunnel"],
                        raw_evidence=f"label='{label}' entropy={entropy:.3f}",
                        explanation=(
                            f"DNS query label '{label[:30]}' has Shannon entropy of {entropy:.2f} bits/char, "
                            f"which significantly exceeds the threshold of {self.config.dns_entropy_threshold}. "
                            f"Normal domain labels have entropy ~{normal_entropy:.1f} (e.g., 'google'). "
                            f"High entropy indicates encoded/encrypted data in the DNS query, "
                            f"a hallmark of DNS tunneling tools like iodine, dnscat2, or dns2tcp."
                        ),
                        evidence_factors=[
                            EvidenceFactor("Label entropy", f"{entropy:.2f} bits/char",
                                           f"threshold: {self.config.dns_entropy_threshold}", 0.5),
                            EvidenceFactor("Label length", f"{len(label)} chars",
                                           "normal: < 15 chars", 0.2),
                            EvidenceFactor("Entropy vs normal", f"{entropy:.2f} vs {normal_entropy:.1f}",
                                           f"ratio: {entropy/max(normal_entropy,0.1):.1f}x", 0.3),
                        ],
                        detection_logic=f"Shannon entropy of DNS labels > {self.config.dns_entropy_threshold}",
                        response_actions=RESPONSE_ACTIONS["DNS_TUNNEL"],
                    ))
                    break

            for label in labels:
                if len(label) >= self.config.dns_label_len_threshold:
                    alerts.append(ThreatEvent(
                        timestamp=now, severity=Severity.MEDIUM, category="DNS_EXFIL",
                        description=f"Suspiciously long DNS label from {src_ip}: '{label[:40]}...' ({len(label)} chars)",
                        source_ip=src_ip, dest_ip=dst_ip,
                        confidence=0.6, mitre_ref=MITRE["dns_exfil"],
                        explanation=(
                            f"DNS query contains a label of {len(label)} characters, "
                            f"exceeding the threshold of {self.config.dns_label_len_threshold}. "
                            f"Normal DNS labels rarely exceed 20 characters. Long labels suggest "
                            f"data being encoded into DNS queries for exfiltration."
                        ),
                        evidence_factors=[
                            EvidenceFactor("Label length", f"{len(label)} chars",
                                           f"threshold: {self.config.dns_label_len_threshold}", 0.7),
                            EvidenceFactor("Max normal length", "~20 chars",
                                           "RFC 1035: max 63 chars", 0.3),
                        ],
                        detection_logic=f"DNS label length >= {self.config.dns_label_len_threshold} characters",
                        response_actions=RESPONSE_ACTIONS["DNS_EXFIL"],
                    ))
                    break

        # Query rate
        hits = self._dns_queries[src_ip]
        cutoff = now - self.config.dns_query_window
        recent = sum(1 for t, _ in hits if t > cutoff)
        if recent >= self.config.dns_query_rate_threshold:
            rate = recent / self.config.dns_query_window
            alerts.append(ThreatEvent(
                timestamp=now, severity=Severity.MEDIUM, category="DNS_FLOOD",
                description=f"High DNS query rate from {src_ip}: {recent} queries in {self.config.dns_query_window}s",
                source_ip=src_ip, dest_ip=dst_ip,
                confidence=0.5, mitre_ref=MITRE["dns_tunnel"],
                explanation=(
                    f"Source {src_ip} sent {recent} DNS queries in {self.config.dns_query_window}s "
                    f"({rate:.1f} queries/sec). Normal hosts typically send < 5 queries/sec. "
                    f"Elevated rates may indicate DNS tunneling, reconnaissance, or amplification abuse."
                ),
                evidence_factors=[
                    EvidenceFactor("Query rate", f"{rate:.1f} q/sec",
                                   f"threshold: {self.config.dns_query_rate_threshold/self.config.dns_query_window:.0f}/sec", 0.6),
                    EvidenceFactor("Query count", str(recent),
                                   f"threshold: {self.config.dns_query_rate_threshold}", 0.4),
                ],
                detection_logic=f"DNS queries per source IP > {self.config.dns_query_rate_threshold} in {self.config.dns_query_window}s",
                response_actions=RESPONSE_ACTIONS["DNS_FLOOD"],
            ))

        return alerts

    def _check_arp_spoofing(self, arp: ARPPacket, now: float) -> Optional[ThreatEvent]:
        if arp.opcode != 2:
            return None
        ip = arp.sender_ip
        mac = arp.sender_mac

        if ip in self._arp_table:
            old_mac = self._arp_table[ip]
            if old_mac != mac:
                self._arp_table[ip] = mac
                return ThreatEvent(
                    timestamp=now, severity=Severity.CRITICAL, category="ARP_SPOOF",
                    description=f"ARP spoofing detected: {ip} changed from {old_mac} to {mac}",
                    source_ip=ip, confidence=0.9, mitre_ref=MITRE["arp_spoof"],
                    raw_evidence=f"old_mac={old_mac} new_mac={mac}",
                    explanation=(
                        f"The MAC address for IP {ip} changed from {old_mac} to {mac}. "
                        f"In a legitimate network, IP-to-MAC mappings are stable. "
                        f"This change indicates an ARP spoofing/poisoning attack where an attacker "
                        f"is redirecting traffic intended for {ip} through their own machine "
                        f"(MAC {mac}), enabling man-in-the-middle interception."
                    ),
                    evidence_factors=[
                        EvidenceFactor("MAC change", f"{old_mac} -> {mac}",
                                       "expected: stable mapping", 0.7),
                        EvidenceFactor("ARP opcode", "Reply (2)",
                                       "gratuitous ARP replies are suspicious", 0.3),
                    ],
                    detection_logic="Track IP-to-MAC mappings from ARP replies; alert on changes",
                    response_actions=RESPONSE_ACTIONS["ARP_SPOOF"],
                )

        self._arp_table[ip] = mac
        return None

    def _check_brute_force(
        self, src_ip: str, dst_ip: Optional[str], port: int, now: float
    ) -> Optional[ThreatEvent]:
        key = (src_ip, port)
        hits = self._auth_hits[key]
        cutoff = now - self.config.brute_force_window
        recent = sum(1 for t in hits if t > cutoff)

        if recent < self.config.brute_force_threshold:
            return None

        port_name = {22: "SSH", 3389: "RDP", 21: "FTP", 23: "Telnet",
                     5900: "VNC", 3306: "MySQL", 5432: "PostgreSQL",
                     1433: "MSSQL", 27017: "MongoDB"}.get(port, str(port))
        rate = recent / self.config.brute_force_window
        confidence = min(1.0, recent / (self.config.brute_force_threshold * 2))

        return ThreatEvent(
            timestamp=now, severity=Severity.HIGH, category="BRUTE_FORCE",
            description=f"Brute-force attempt from {src_ip} to {dst_ip}:{port} ({port_name}): {recent} connections in {self.config.brute_force_window}s",
            source_ip=src_ip, dest_ip=dst_ip, dest_port=port,
            confidence=confidence, mitre_ref=MITRE["brute_force"],
            explanation=(
                f"Source {src_ip} initiated {recent} TCP connections to {dst_ip}:{port} ({port_name}) "
                f"within {self.config.brute_force_window}s ({rate:.1f} attempts/sec). "
                f"This exceeds the threshold of {self.config.brute_force_threshold} attempts. "
                f"Rapid authentication attempts to {port_name} strongly suggest automated "
                f"credential brute-forcing using tools like Hydra, Medusa, or similar."
            ),
            evidence_factors=[
                EvidenceFactor("Connection attempts", str(recent),
                               f"threshold: {self.config.brute_force_threshold}", 0.4),
                EvidenceFactor("Attempt rate", f"{rate:.1f}/sec",
                               "normal: < 0.1/sec", 0.3),
                EvidenceFactor("Target service", port_name,
                               "authentication-capable port", 0.2),
                EvidenceFactor("Single source", src_ip,
                               "concentrated from one IP", 0.1),
            ],
            detection_logic=f"SYN count to auth ports per (src_ip,dst_port) > {self.config.brute_force_threshold} in {self.config.brute_force_window}s",
            response_actions=RESPONSE_ACTIONS["BRUTE_FORCE"],
        )

    def _check_ttl_anomaly(self, ip: IPv4Packet, now: float) -> Optional[ThreatEvent]:
        if ip.ttl < self.config.ttl_min_suspicious:
            return None
        # Whitelist: skip multicast destinations (SSDP, mDNS, etc.)
        if ip.dst_ip in self.config.whitelist_multicast_ips:
            self.stats["whitelist_suppressed"] += 1
            return None
        return ThreatEvent(
            timestamp=now, severity=Severity.LOW, category="TTL_ANOMALY",
            description=f"Low TTL={ip.ttl} from {ip.src_ip} -> {ip.dst_ip} (possible traceroute or multi-hop proxy)",
            source_ip=ip.src_ip, dest_ip=ip.dst_ip,
            confidence=0.3, mitre_ref=MITRE["ttl_anomaly"],
            explanation=(
                f"Packet from {ip.src_ip} arrived with TTL={ip.ttl}, which is unusually low. "
                f"Normal TTL values are 64 (Linux), 128 (Windows), or 255 (network devices). "
                f"Low TTL may indicate: (1) traceroute activity, (2) traffic routed through "
                f"many proxy hops, or (3) TTL manipulation for evasion."
            ),
            evidence_factors=[
                EvidenceFactor("TTL value", str(ip.ttl),
                               f"suspicious range: {self.config.ttl_min_suspicious}-{self.config.ttl_max_suspicious}", 0.7),
                EvidenceFactor("Expected TTL", "64/128/255",
                               "standard OS defaults", 0.3),
            ],
            detection_logic=f"IP TTL <= {self.config.ttl_max_suspicious}",
            response_actions=RESPONSE_ACTIONS["TTL_ANOMALY"],
        )

    def _check_proto_anomaly(
        self, tcp: TCPSegment, src_ip: Optional[str],
        dst_ip: Optional[str], now: float
    ) -> Optional[ThreatEvent]:
        if (tcp.flags & TCPFlags.SYN) and (tcp.flags & TCPFlags.FIN):
            return ThreatEvent(
                timestamp=now, severity=Severity.HIGH, category="PROTO_ANOMALY",
                description=f"Invalid TCP flags SYN+FIN from {src_ip}:{tcp.src_port} -> {dst_ip}:{tcp.dst_port}",
                source_ip=src_ip, dest_ip=dst_ip,
                source_port=tcp.src_port, dest_port=tcp.dst_port,
                confidence=0.95, mitre_ref=MITRE["proto_anomaly"],
                explanation=(
                    f"TCP packet has both SYN and FIN flags set simultaneously, which is "
                    f"invalid per RFC 793. No legitimate TCP implementation generates SYN+FIN "
                    f"packets. This is a strong indicator of a crafted probe from scanning "
                    f"tools (e.g., nmap) used for OS fingerprinting or firewall evasion."
                ),
                evidence_factors=[
                    EvidenceFactor("TCP flags", "SYN+FIN", "mutually exclusive per RFC 793", 0.9),
                    EvidenceFactor("Legitimacy", "0% chance legitimate",
                                   "no OS generates this", 0.1),
                ],
                detection_logic="Check for mutually exclusive TCP flag combinations",
                response_actions=RESPONSE_ACTIONS["PROTO_ANOMALY"],
            )

        if (tcp.flags & TCPFlags.RST) and (tcp.flags & TCPFlags.SYN):
            return ThreatEvent(
                timestamp=now, severity=Severity.MEDIUM, category="PROTO_ANOMALY",
                description=f"Invalid TCP flags RST+SYN from {src_ip}:{tcp.src_port} -> {dst_ip}:{tcp.dst_port}",
                source_ip=src_ip, dest_ip=dst_ip,
                confidence=0.85, mitre_ref=MITRE["proto_anomaly"],
                explanation=(
                    f"TCP packet has both RST and SYN flags set. This combination is invalid "
                    f"and indicates a crafted packet, likely from a port-scanning or OS-fingerprinting tool."
                ),
                evidence_factors=[
                    EvidenceFactor("TCP flags", "RST+SYN", "invalid combination", 0.9),
                ],
                detection_logic="Check for RST+SYN flag combination",
                response_actions=RESPONSE_ACTIONS["PROTO_ANOMALY"],
            )
        return None

    def _check_icmp_tunnel(
        self, src_ip: str, dst_ip: Optional[str],
        icmp: ICMPPacket, now: float
    ) -> Optional[ThreatEvent]:
        payload_size = len(icmp.payload)

        if payload_size > self.config.icmp_payload_threshold:
            entropy = _shannon_entropy_bytes(icmp.payload)
            if entropy > 3.5:
                return ThreatEvent(
                    timestamp=now, severity=Severity.HIGH, category="ICMP_TUNNEL",
                    description=f"Possible ICMP tunnel from {src_ip}: payload={payload_size}B, entropy={entropy:.2f}",
                    source_ip=src_ip, dest_ip=dst_ip,
                    confidence=min(1.0, entropy / 5.0), mitre_ref=MITRE["icmp_tunnel"],
                    explanation=(
                        f"ICMP packet from {src_ip} carries a {payload_size}-byte payload with "
                        f"Shannon entropy of {entropy:.2f} bits/byte. Normal ICMP echo payloads "
                        f"are small (~32-56 bytes) with low entropy (padding bytes). "
                        f"Large, high-entropy ICMP payloads indicate covert data being tunneled "
                        f"through ICMP (e.g., ptunnel, icmptunnel, Hans). This bypasses firewalls "
                        f"that allow ICMP echo but block other protocols."
                    ),
                    evidence_factors=[
                        EvidenceFactor("Payload size", f"{payload_size} bytes",
                                       f"threshold: {self.config.icmp_payload_threshold} bytes", 0.4),
                        EvidenceFactor("Payload entropy", f"{entropy:.2f} bits/byte",
                                       "threshold: 3.5 bits/byte", 0.4),
                        EvidenceFactor("Normal payload", "32-56 bytes, low entropy",
                                       "RFC 792 standard", 0.2),
                    ],
                    detection_logic=f"ICMP payload > {self.config.icmp_payload_threshold}B AND entropy > 3.5",
                    response_actions=RESPONSE_ACTIONS["ICMP_TUNNEL"],
                )

        hits = self._icmp_hits[src_ip]
        cutoff = now - self.config.icmp_rate_window
        recent = sum(1 for t, _ in hits if t > cutoff)
        if recent >= self.config.icmp_rate_threshold:
            rate = recent / self.config.icmp_rate_window
            return ThreatEvent(
                timestamp=now, severity=Severity.MEDIUM, category="ICMP_FLOOD",
                description=f"High ICMP rate from {src_ip}: {recent} packets in {self.config.icmp_rate_window}s",
                source_ip=src_ip, dest_ip=dst_ip,
                confidence=0.5, mitre_ref=MITRE["icmp_tunnel"],
                explanation=(
                    f"Source {src_ip} sent {recent} ICMP packets in {self.config.icmp_rate_window}s "
                    f"({rate:.1f} pkt/sec). Normal ICMP rates are < 1 pkt/sec. Elevated rates "
                    f"may indicate ping flood, ICMP tunneling, or network reconnaissance."
                ),
                evidence_factors=[
                    EvidenceFactor("ICMP rate", f"{rate:.1f} pkt/sec",
                                   f"threshold: {self.config.icmp_rate_threshold}/{self.config.icmp_rate_window}s", 0.6),
                    EvidenceFactor("Normal rate", "< 1 pkt/sec", "baseline", 0.4),
                ],
                detection_logic=f"ICMP packets per source > {self.config.icmp_rate_threshold} in {self.config.icmp_rate_window}s",
                response_actions=RESPONSE_ACTIONS["ICMP_FLOOD"],
            )
        return None

    # ── Utility ──

    def get_recent_alerts(self, count: int = 20) -> List[ThreatEvent]:
        return self.alerts[-count:]

    def get_alerts_by_severity(self, min_severity: Severity = Severity.MEDIUM) -> List[ThreatEvent]:
        return [a for a in self.alerts if a.severity >= min_severity]

    @property
    def threat_summary(self) -> dict:
        return {
            "total_threats": self.stats["threats_detected"],
            "packets_analyzed": self.stats["total_packets_analyzed"],
            "by_severity": dict(self.stats["severity_counts"]),
            "arp_table_size": len(self._arp_table),
            "tracked_sources": len(self._port_hits),
        }


# ──────────────────────────────────────────────────────────────────────
# Shannon Entropy helpers
# ──────────────────────────────────────────────────────────────────────

def _shannon_entropy(text: str) -> float:
    if not text:
        return 0.0
    freq = {}
    for ch in text:
        freq[ch] = freq.get(ch, 0) + 1
    length = len(text)
    return -sum((c / length) * math.log2(c / length) for c in freq.values())


def _shannon_entropy_bytes(data: bytes) -> float:
    if not data:
        return 0.0
    freq = [0] * 256
    for b in data:
        freq[b] += 1
    length = len(data)
    return -sum(
        (c / length) * math.log2(c / length)
        for c in freq if c > 0
    )
