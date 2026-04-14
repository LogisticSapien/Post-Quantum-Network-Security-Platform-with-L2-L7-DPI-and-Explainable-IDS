"""
Detection Quality Analysis
============================
Measures IDS accuracy with ground-truth attack data:
  • Confusion matrix
  • False Positive Rate (FPR) / False Negative Rate (FNR)
  • Precision / Recall / F1 score
  • Per-attack detection accuracy
  • TLS cipher weakness analysis (% traffic using weak crypto)

Usage:
  python detection_quality.py
  python -m quantum_sniffer --mode quality
"""

from __future__ import annotations

import os
import random
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from protocols import (
    IPv4Packet, TCPSegment, TCPFlags, UDPDatagram,
    DNSMessage, DNSQuestion, ICMPPacket, ARPPacket,
    TLSClientHello, analyze_tls_handshake,
)
from ids import IDSEngine, ThreatEvent, Severity, SEVERITY_LABELS
import base64


# ──────────────────────────────────────────────────────────────────────
# Result containers
# ──────────────────────────────────────────────────────────────────────

@dataclass
class ConfusionMatrix:
    """Binary confusion matrix."""
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0

    @property
    def total(self) -> int:
        return self.tp + self.fp + self.fn + self.tn

    @property
    def precision(self) -> float:
        d = self.tp + self.fp
        return self.tp / d if d > 0 else 0.0

    @property
    def recall(self) -> float:
        d = self.tp + self.fn
        return self.tp / d if d > 0 else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    @property
    def fpr(self) -> float:
        d = self.fp + self.tn
        return self.fp / d if d > 0 else 0.0

    @property
    def fnr(self) -> float:
        d = self.fn + self.tp
        return self.fn / d if d > 0 else 0.0

    @property
    def accuracy(self) -> float:
        return (self.tp + self.tn) / self.total if self.total > 0 else 0.0


@dataclass
class AttackTestResult:
    """Result from testing a specific attack type."""
    attack_name: str
    samples: int
    detected: int
    missed: int
    accuracy: float
    alerts: List[ThreatEvent] = field(default_factory=list)


# ──────────────────────────────────────────────────────────────────────
# Attack generators (labeled ground truth)
# ──────────────────────────────────────────────────────────────────────

class LabeledDataGenerator:
    """Generates packets with known labels (attack / benign)."""

    def gen_benign_tcp(self, n: int = 500) -> List[Tuple[dict, bool]]:
        """Generate benign TCP packets (normal web browsing)."""
        samples = []
        for i in range(n):
            ip = IPv4Packet(
                version=4, ihl=20, dscp=0, ecn=0, total_length=200 + random.randint(0, 1200),
                identification=i & 0xFFFF, flags=0x02, fragment_offset=0,
                ttl=64, protocol=6, checksum=0,
                src_ip=f"192.168.1.{random.randint(2, 50)}",
                dst_ip=f"93.184.{random.randint(1, 254)}.{random.randint(1, 254)}",
                options=b"", payload=b"",
            )
            # Normal ACK traffic (established connections)
            tcp = TCPSegment(
                src_port=random.randint(49152, 65535),
                dst_port=random.choice([80, 443, 8080]),
                seq_num=random.randint(1000, 999999), ack_num=random.randint(1000, 999999),
                data_offset=20, flags=TCPFlags.ACK | TCPFlags.PSH,
                window=65535, checksum=0, urgent_ptr=0,
                options=b"", payload=b"",
            )
            samples.append(({"ip": ip, "tcp": tcp}, False))  # False = benign
        return samples

    def gen_benign_dns(self, n: int = 200) -> List[Tuple[dict, bool]]:
        """Generate benign DNS queries."""
        domains = ["google.com", "github.com", "stackoverflow.com", "python.org",
                    "wikipedia.org", "amazon.com", "cloudflare.com", "microsoft.com"]
        samples = []
        for i in range(n):
            dns = DNSMessage(
                transaction_id=random.randint(0, 65535),
                is_response=False, opcode=0, rcode=0,
                questions=[DNSQuestion(
                    name=random.choice(domains), qtype=1, qclass=1,
                )],
                answers=[], authorities=[], additionals=[],
            )
            ip = IPv4Packet(
                version=4, ihl=20, dscp=0, ecn=0, total_length=60,
                identification=i & 0xFFFF, flags=0x02, fragment_offset=0,
                ttl=64, protocol=17, checksum=0,
                src_ip="192.168.1.10", dst_ip="8.8.8.8",
                options=b"", payload=b"",
            )
            samples.append(({"ip": ip, "dns": dns}, False))
        return samples

    def gen_port_scan(self, n: int = 60) -> List[Tuple[dict, bool]]:
        """Generate port scan traffic (attack)."""
        scanner = "192.168.1.200"
        target = "192.168.1.100"
        samples = []
        for port in range(20, 20 + n):
            ip = IPv4Packet(
                version=4, ihl=20, dscp=0, ecn=0, total_length=40,
                identification=port, flags=0x02, fragment_offset=0,
                ttl=64, protocol=6, checksum=0,
                src_ip=scanner, dst_ip=target, options=b"", payload=b"",
            )
            flags = random.choice([TCPFlags.SYN, TCPFlags.FIN,
                                    TCPFlags.FIN | TCPFlags.PSH | TCPFlags.URG])
            tcp = TCPSegment(
                src_port=random.randint(40000, 65535), dst_port=port,
                seq_num=0, ack_num=0, data_offset=20, flags=flags,
                window=1024, checksum=0, urgent_ptr=0,
                options=b"", payload=b"",
            )
            samples.append(({"ip": ip, "tcp": tcp}, True))  # True = attack
        return samples

    def gen_syn_flood(self, n: int = 200) -> List[Tuple[dict, bool]]:
        """Generate SYN flood traffic (attack)."""
        target = "192.168.1.100"
        samples = []
        for i in range(n):
            ip = IPv4Packet(
                version=4, ihl=20, dscp=0, ecn=0, total_length=40,
                identification=random.randint(0, 65535), flags=0x02,
                fragment_offset=0, ttl=64, protocol=6, checksum=0,
                src_ip=f"10.{random.randint(1, 254)}.{random.randint(1, 254)}.{random.randint(1, 254)}",
                dst_ip=target, options=b"", payload=b"",
            )
            tcp = TCPSegment(
                src_port=random.randint(1024, 65535), dst_port=80,
                seq_num=random.randint(0, 2**32 - 1), ack_num=0,
                data_offset=20, flags=TCPFlags.SYN, window=65535,
                checksum=0, urgent_ptr=0, options=b"", payload=b"",
            )
            samples.append(({"ip": ip, "tcp": tcp}, True))
        return samples

    def gen_dns_tunnel(self, n: int = 30) -> List[Tuple[dict, bool]]:
        """Generate DNS tunneling traffic (attack)."""
        samples = []
        for i in range(n):
            data = os.urandom(30)
            encoded = base64.b32encode(data).decode().lower().rstrip("=")
            dns = DNSMessage(
                transaction_id=random.randint(0, 65535),
                is_response=False, opcode=0, rcode=0,
                questions=[DNSQuestion(
                    name=f"{encoded[:40]}.tunnel.evil.com", qtype=1, qclass=1,
                )],
                answers=[], authorities=[], additionals=[],
            )
            ip = IPv4Packet(
                version=4, ihl=20, dscp=0, ecn=0, total_length=100,
                identification=i, flags=0x02, fragment_offset=0,
                ttl=64, protocol=17, checksum=0,
                src_ip="192.168.1.10", dst_ip="8.8.8.8",
                options=b"", payload=b"",
            )
            samples.append(({"ip": ip, "dns": dns}, True))
        return samples

    def gen_arp_spoof(self) -> List[Tuple[dict, bool]]:
        """Generate ARP spoofing traffic (attack)."""
        samples = []
        # Legitimate binding
        legit = ARPPacket(
            hw_type=1, proto_type=0x0800, opcode=2,
            sender_mac="aa:bb:cc:dd:ee:ff", sender_ip="192.168.1.1",
            target_mac="11:22:33:44:55:66", target_ip="192.168.1.10",
        )
        samples.append(({"arp": legit}, False))
        # Spoofed
        spoof = ARPPacket(
            hw_type=1, proto_type=0x0800, opcode=2,
            sender_mac="66:77:88:99:aa:bb", sender_ip="192.168.1.1",
            target_mac="11:22:33:44:55:66", target_ip="192.168.1.10",
        )
        samples.append(({"arp": spoof}, True))
        return samples

    def gen_brute_force(self, n: int = 20) -> List[Tuple[dict, bool]]:
        """Generate SSH brute-force traffic (attack)."""
        samples = []
        for i in range(n):
            ip = IPv4Packet(
                version=4, ihl=20, dscp=0, ecn=0, total_length=40,
                identification=i, flags=0x02, fragment_offset=0,
                ttl=64, protocol=6, checksum=0,
                src_ip="10.0.0.50", dst_ip="192.168.1.1",
                options=b"", payload=b"",
            )
            tcp = TCPSegment(
                src_port=random.randint(40000, 65535), dst_port=22,
                seq_num=random.randint(0, 2**32 - 1), ack_num=0,
                data_offset=20, flags=TCPFlags.SYN, window=65535,
                checksum=0, urgent_ptr=0, options=b"", payload=b"",
            )
            samples.append(({"ip": ip, "tcp": tcp}, True))
        return samples

    def gen_icmp_tunnel(self, n: int = 10) -> List[Tuple[dict, bool]]:
        """Generate ICMP tunneling traffic (attack)."""
        samples = []
        for i in range(n):
            ip = IPv4Packet(
                version=4, ihl=20, dscp=0, ecn=0, total_length=276,
                identification=i, flags=0x02, fragment_offset=0,
                ttl=64, protocol=1, checksum=0,
                src_ip="192.168.1.10", dst_ip="10.0.0.1",
                options=b"", payload=b"",
            )
            icmp = ICMPPacket(
                type=8, code=0, checksum=0,
                identifier=1, sequence=i + 1,
                payload=os.urandom(256),
            )
            samples.append(({"ip": ip, "icmp": icmp}, True))
        return samples

    def gen_benign_icmp(self, n: int = 50) -> List[Tuple[dict, bool]]:
        """Generate normal ICMP echo traffic."""
        samples = []
        for i in range(n):
            ip = IPv4Packet(
                version=4, ihl=20, dscp=0, ecn=0, total_length=84,
                identification=i, flags=0x02, fragment_offset=0,
                ttl=64, protocol=1, checksum=0,
                src_ip="192.168.1.10", dst_ip="8.8.8.8",
                options=b"", payload=b"",
            )
            icmp = ICMPPacket(
                type=8, code=0, checksum=0,
                identifier=1, sequence=i + 1,
                payload=b"\x00" * 56,  # Standard 56-byte ping
            )
            samples.append(({"ip": ip, "icmp": icmp}, False))
        return samples


# ──────────────────────────────────────────────────────────────────────
# TLS Weakness Analysis
# ──────────────────────────────────────────────────────────────────────

def analyze_tls_weakness() -> dict:
    """Analyze a simulated TLS cipher suite distribution for quantum safety."""
    # Simulate observed cipher suite distribution in typical traffic
    suites = [
        (0x1301, 35, "TLS_AES_128_GCM_SHA256"),         # TLS 1.3 — safe
        (0x1302, 20, "TLS_AES_256_GCM_SHA384"),         # TLS 1.3 — safe
        (0x1303, 10, "TLS_CHACHA20_POLY1305_SHA256"),   # TLS 1.3 — safe
        (0xC02F, 12, "TLS_ECDHE_RSA_WITH_AES_128_GCM"),# TLS 1.2 ECDHE — moderate
        (0xC030, 8,  "TLS_ECDHE_RSA_WITH_AES_256_GCM"),# TLS 1.2 ECDHE — moderate
        (0x009C, 6,  "TLS_RSA_WITH_AES_128_GCM"),      # RSA — weak (no PFS)
        (0x009D, 4,  "TLS_RSA_WITH_AES_256_GCM"),      # RSA — weak (no PFS)
        (0x002F, 3,  "TLS_RSA_WITH_AES_128_CBC"),      # RSA+CBC — very weak
        (0x0035, 2,  "TLS_RSA_WITH_AES_256_CBC"),      # RSA+CBC — very weak
    ]

    total = sum(pct for _, pct, _ in suites)
    safe = 0
    weak_pfs = 0
    weak_rsa = 0
    details = []

    for cs_id, pct, name in suites:
        a = analyze_tls_handshake(cs_id)
        is_pq_safe = a.pqc_verdict == "SAFE"
        has_pfs = a.forward_secrecy in ("ECDHE", "DHE", "TLS 1.3")

        if is_pq_safe:
            category = "PQ-Safe"
            safe += pct
        elif has_pfs:
            category = "Moderate (has PFS, no PQ)"
            weak_pfs += pct
        else:
            category = "Weak (no PFS, no PQ)"
            weak_rsa += pct

        details.append({
            "suite": name,
            "traffic_pct": pct,
            "grade": a.overall_grade,
            "pqc_verdict": a.pqc_verdict,
            "category": category,
        })

    return {
        "total_suites_analyzed": len(suites),
        "pq_safe_pct": safe,
        "moderate_pct": weak_pfs,
        "weak_pct": weak_rsa,
        "details": details,
    }


# ──────────────────────────────────────────────────────────────────────
# Main Detection Quality Test
# ──────────────────────────────────────────────────────────────────────

def run_detection_quality() -> dict:
    """Run detection quality analysis and print results."""
    print("=" * 80)
    print("  ⚛️  QUANTUM SNIFFER — DETECTION QUALITY ANALYSIS")
    print("=" * 80)

    gen = LabeledDataGenerator()
    ids = IDSEngine(sensitivity="high")

    # Generate all labeled data
    attack_sets = {
        "Port Scan": gen.gen_port_scan(60),
        "SYN Flood": gen.gen_syn_flood(200),
        "DNS Tunneling": gen.gen_dns_tunnel(30),
        "ARP Spoofing": gen.gen_arp_spoof(),
        "Brute Force": gen.gen_brute_force(20),
        "ICMP Tunneling": gen.gen_icmp_tunnel(10),
    }
    benign_sets = {
        "Benign TCP": gen.gen_benign_tcp(500),
        "Benign DNS": gen.gen_benign_dns(200),
        "Benign ICMP": gen.gen_benign_icmp(50),
    }

    # Global confusion matrix
    cm = ConfusionMatrix()
    per_attack: Dict[str, AttackTestResult] = {}

    # ── Test attacks ──
    print("\n  ▶ Per-Attack Detection Accuracy")
    print("  " + "─" * 76)

    for attack_name, samples in attack_sets.items():
        detected = 0
        missed = 0
        alerts = []
        for pkt_kwargs, is_attack in samples:
            result = ids.analyze_packet(**pkt_kwargs)
            got_alert = len(result) > 0
            if is_attack:
                if got_alert:
                    cm.tp += 1
                    detected += 1
                    alerts.extend(result)
                else:
                    cm.fn += 1
                    missed += 1
        acc = detected / max(detected + missed, 1)
        per_attack[attack_name] = AttackTestResult(
            attack_name=attack_name,
            samples=len(samples),
            detected=detected,
            missed=missed,
            accuracy=acc,
            alerts=alerts[:5],
        )
        status = "✅" if acc >= 0.5 else "❌"
        print(f"    {status} {attack_name:<20s}  {detected}/{detected + missed} detected "
              f"({acc * 100:.0f}%)  alerts={len(alerts)}")

    # ── Test benign ──
    print("\n  ▶ Benign Traffic (False Positive Check)")
    print("  " + "─" * 76)

    benign_fps = 0
    benign_total = 0
    for set_name, samples in benign_sets.items():
        fps = 0
        for pkt_kwargs, is_attack in samples:
            result = ids.analyze_packet(**pkt_kwargs)
            got_alert = len(result) > 0
            if not is_attack:
                if got_alert:
                    cm.fp += 1
                    fps += 1
                else:
                    cm.tn += 1
        benign_fps += fps
        benign_total += len(samples)
        fpr = fps / len(samples) * 100
        status = "✅" if fpr < 5 else "⚠️"
        print(f"    {status} {set_name:<20s}  FP: {fps}/{len(samples)} ({fpr:.1f}%)")

    # ── TLS Weakness ──
    print("\n  ▶ TLS Cipher Weakness Analysis")
    print("  " + "─" * 76)

    tls = analyze_tls_weakness()
    for d in tls["details"]:
        grade_icon = {"A+": "🟢", "A": "🟢", "B": "🟡", "C": "🟠", "D": "🔴", "F": "🔴"}.get(d["grade"], "⚪")
        print(f"    {grade_icon} [{d['grade']}] {d['suite']:<45s} {d['traffic_pct']:>3d}% — {d['category']}")

    # ── Summary ──
    print(f"\n{'=' * 80}")
    print(f"  DETECTION QUALITY SUMMARY")
    print(f"{'=' * 80}")
    print(f"\n  ┌─────────────────────────┬──────────────────────────┐")
    print(f"  │                         │ Predicted Attack  Benign │")
    print(f"  ├─────────────────────────┼──────────────────────────┤")
    print(f"  │ Actual Attack           │    TP={cm.tp:<6d}    FN={cm.fn:<5d}│")
    print(f"  │ Actual Benign           │    FP={cm.fp:<6d}    TN={cm.tn:<5d}│")
    print(f"  └─────────────────────────┴──────────────────────────┘")
    print()
    print(f"  Accuracy:     {cm.accuracy:.4f}  ({cm.accuracy * 100:.1f}%)")
    print(f"  Precision:    {cm.precision:.4f}  ({cm.precision * 100:.1f}%)")
    print(f"  Recall:       {cm.recall:.4f}  ({cm.recall * 100:.1f}%)")
    print(f"  F1 Score:     {cm.f1:.4f}")
    print(f"  FPR:          {cm.fpr:.4f}  ({cm.fpr * 100:.1f}%)")
    print(f"  FNR:          {cm.fnr:.4f}  ({cm.fnr * 100:.1f}%)")
    print()
    print(f"  TLS Weakness: {tls['weak_pct']}% weak (no PFS), "
          f"{tls['moderate_pct']}% moderate (PFS, no PQ), "
          f"{tls['pq_safe_pct']}% PQ-safe")
    print(f"{'=' * 80}")

    return {
        "confusion_matrix": {
            "tp": cm.tp, "fp": cm.fp, "fn": cm.fn, "tn": cm.tn,
        },
        "accuracy": cm.accuracy,
        "precision": cm.precision,
        "recall": cm.recall,
        "f1": cm.f1,
        "fpr": cm.fpr,
        "fnr": cm.fnr,
        "per_attack": {
            name: {"samples": r.samples, "detected": r.detected,
                   "missed": r.missed, "accuracy": r.accuracy}
            for name, r in per_attack.items()
        },
        "tls_weakness": tls,
    }


if __name__ == "__main__":
    run_detection_quality()
