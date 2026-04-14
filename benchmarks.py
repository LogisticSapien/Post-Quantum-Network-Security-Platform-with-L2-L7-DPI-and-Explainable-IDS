"""
Detection Quality Benchmarks
==============================
Measures IDS precision, recall, and F1-score using synthetic labeled traffic.
Generates known-attack and known-benign packets, runs them through the IDS
engine, and computes a confusion matrix per detection category.

Usage:
  python __main__.py --quality
  python -m benchmarks            # standalone
"""

from __future__ import annotations

import os
import random
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from ids import IDSEngine, IDSConfig, Severity
from protocols import (
    IPv4Packet, TCPSegment, TCPFlags, ARPPacket,
    DNSMessage, DNSQuestion, ICMPPacket, UDPDatagram,
)

# ──────────────────────────────────────────────────────────────────────
# Synthetic Dataset
# ──────────────────────────────────────────────────────────────────────

@dataclass
class LabeledPacket:
    """A packet with a ground-truth label."""
    label: str               # "PORT_SCAN", "SYN_FLOOD", "BENIGN", etc.
    ip: IPv4Packet = None
    tcp: TCPSegment = None
    udp: UDPDatagram = None
    dns: DNSMessage = None
    arp: ARPPacket = None
    icmp: ICMPPacket = None


class SyntheticDataset:
    """
    Generate deterministic labeled packets for IDS benchmarking.
    All randomness is seeded for reproducibility.
    """

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.rng = random.Random(seed)
        self.packets: List[LabeledPacket] = []

    def _make_ip(self, src: str = "10.0.0.50", dst: str = "192.168.1.100",
                 ttl: int = 64, protocol: int = 6) -> IPv4Packet:
        return IPv4Packet(
            version=4, ihl=20, dscp=0, ecn=0, total_length=40,
            identification=self.rng.randint(0, 65535), flags=2,
            fragment_offset=0, ttl=ttl, protocol=protocol, checksum=0,
            src_ip=src, dst_ip=dst, options=b"", payload=b"",
        )

    def _make_tcp(self, src_port: int = 12345, dst_port: int = 80,
                  flags: int = TCPFlags.SYN) -> TCPSegment:
        return TCPSegment(
            src_port=src_port, dst_port=dst_port,
            seq_num=0, ack_num=0, data_offset=20,
            flags=flags, window=65535, checksum=0, urgent_ptr=0,
            options=b"", payload=b"",
        )

    def _make_dns(self, query_name: str = "www.google.com") -> DNSMessage:
        return DNSMessage(
            transaction_id=self.rng.randint(0, 65535), is_response=False,
            opcode=0, rcode=0,
            questions=[DNSQuestion(name=query_name, qtype=1, qclass=1)],
            answers=[], authorities=[], additionals=[],
        )

    def _make_arp(self, sender_mac: str, sender_ip: str) -> ARPPacket:
        return ARPPacket(
            hw_type=1, proto_type=0x0800,
            opcode=2, sender_mac=sender_mac, sender_ip=sender_ip,
            target_mac="00:00:00:00:00:00", target_ip="0.0.0.0",
        )

    def _make_icmp(self, payload: bytes = b"") -> ICMPPacket:
        return ICMPPacket(
            type=8, code=0, checksum=0, identifier=1, sequence=1, payload=payload,
        )

    def generate(self) -> List[LabeledPacket]:
        """Generate full labeled dataset."""
        self.packets = []
        self._gen_port_scans()
        self._gen_syn_floods()
        self._gen_dns_tunnels()
        self._gen_arp_spoofs()
        self._gen_brute_force()
        self._gen_icmp_tunnels()
        self._gen_proto_anomalies()
        self._gen_benign_tcp()
        self._gen_benign_dns()
        self._gen_benign_icmp()
        return self.packets

    # ── Attack generators ──

    def _gen_port_scans(self, count: int = 3):
        """Generate port scan sequences."""
        for attempt in range(count):
            src = f"10.0.{attempt}.50"
            for port in range(1, 25):  # 25 ports > threshold of 15
                self.packets.append(LabeledPacket(
                    label="PORT_SCAN",
                    ip=self._make_ip(src=src),
                    tcp=self._make_tcp(dst_port=port, flags=TCPFlags.SYN),
                ))

    def _gen_syn_floods(self, count: int = 2):
        """Generate SYN flood bursts."""
        for attempt in range(count):
            dst = f"192.168.1.{100 + attempt}"
            for i in range(250):  # 250 > threshold of 100
                src = f"10.{self.rng.randint(1,254)}.{self.rng.randint(1,254)}.{self.rng.randint(1,254)}"
                self.packets.append(LabeledPacket(
                    label="SYN_FLOOD",
                    ip=self._make_ip(src=src, dst=dst),
                    tcp=self._make_tcp(
                        src_port=self.rng.randint(1024, 65535), dst_port=80,
                    ),
                ))

    def _gen_dns_tunnels(self, count: int = 5):
        """Generate high-entropy DNS queries."""
        import base64
        for i in range(count):
            data = bytes(self.rng.getrandbits(8) for _ in range(30))
            encoded = base64.b32encode(data).decode().lower().rstrip("=")
            query = f"{encoded[:40]}.tunnel.evil{i}.com"
            self.packets.append(LabeledPacket(
                label="DNS_TUNNEL",
                ip=self._make_ip(src="192.168.1.10", protocol=17),
                dns=self._make_dns(query_name=query),
            ))

    def _gen_arp_spoofs(self, count: int = 3):
        """Generate ARP spoof sequences."""
        for i in range(count):
            ip = f"192.168.1.{1 + i}"
            # Legitimate ARP
            self.packets.append(LabeledPacket(
                label="BENIGN",
                arp=self._make_arp(f"aa:bb:cc:dd:ee:{i:02x}", ip),
            ))
            # Spoofed ARP with different MAC
            self.packets.append(LabeledPacket(
                label="ARP_SPOOF",
                arp=self._make_arp(f"66:77:88:99:aa:{i:02x}", ip),
            ))

    def _gen_brute_force(self, count: int = 2):
        """Generate SSH brute force sequences."""
        for attempt in range(count):
            src = f"10.0.{attempt}.99"
            for i in range(20):  # 20 > threshold of 10
                self.packets.append(LabeledPacket(
                    label="BRUTE_FORCE",
                    ip=self._make_ip(src=src, dst="192.168.1.1"),
                    tcp=self._make_tcp(
                        src_port=self.rng.randint(40000, 65535), dst_port=22,
                    ),
                ))

    def _gen_icmp_tunnels(self, count: int = 3):
        """Generate ICMP tunnel packets."""
        for i in range(count):
            payload = bytes(self.rng.getrandbits(8) for _ in range(256))
            self.packets.append(LabeledPacket(
                label="ICMP_TUNNEL",
                ip=self._make_ip(src=f"192.168.1.{20+i}", protocol=1),
                icmp=self._make_icmp(payload=payload),
            ))

    def _gen_proto_anomalies(self, count: int = 3):
        """Generate protocol anomaly packets."""
        for i in range(count):
            self.packets.append(LabeledPacket(
                label="PROTO_ANOMALY",
                ip=self._make_ip(src=f"10.0.{i}.1"),
                tcp=self._make_tcp(flags=TCPFlags.SYN | TCPFlags.FIN),
            ))

    # ── Benign generators ──

    def _gen_benign_tcp(self, count: int = 100):
        """Generate normal TCP traffic."""
        for i in range(count):
            self.packets.append(LabeledPacket(
                label="BENIGN",
                ip=self._make_ip(
                    src=f"192.168.1.{self.rng.randint(1,20)}",
                    dst=f"10.0.0.{self.rng.randint(1,10)}",
                ),
                tcp=self._make_tcp(
                    src_port=self.rng.randint(1024, 65535),
                    dst_port=self.rng.choice([80, 443, 8080, 8443]),
                    flags=TCPFlags.ACK,
                ),
            ))

    def _gen_benign_dns(self, count: int = 20):
        """Generate normal DNS queries."""
        domains = ["www.google.com", "api.github.com", "cdn.jsdelivr.net",
                    "fonts.googleapis.com", "example.com", "stackoverflow.com"]
        for i in range(count):
            self.packets.append(LabeledPacket(
                label="BENIGN",
                ip=self._make_ip(src="192.168.1.5", protocol=17),
                dns=self._make_dns(query_name=self.rng.choice(domains)),
            ))

    def _gen_benign_icmp(self, count: int = 10):
        """Generate normal ICMP pings."""
        for i in range(count):
            self.packets.append(LabeledPacket(
                label="BENIGN",
                ip=self._make_ip(src=f"192.168.1.{i+1}", protocol=1),
                icmp=self._make_icmp(payload=b"\x00" * 32),
            ))


# ──────────────────────────────────────────────────────────────────────
# Detection Benchmark
# ──────────────────────────────────────────────────────────────────────

ATTACK_CATEGORIES = {
    "PORT_SCAN", "SYN_FLOOD", "DNS_TUNNEL", "DNS_EXFIL", "DNS_FLOOD",
    "ARP_SPOOF", "BRUTE_FORCE", "TTL_ANOMALY", "PROTO_ANOMALY",
    "ICMP_TUNNEL", "ICMP_FLOOD",
}


@dataclass
class CategoryMetrics:
    tp: int = 0  # true positive
    fp: int = 0  # false positive
    fn: int = 0  # false negative
    tn: int = 0  # true negative

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) > 0 else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) > 0 else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


class DetectionBenchmark:
    """Run IDS against labeled dataset and compute performance metrics."""

    def __init__(self, sensitivity: str = "medium"):
        # Use a whitelist-free config for benchmarking
        config = IDSConfig(
            whitelist_ip_cidrs=[],
            whitelist_multicast_ips=set(),
            whitelist_dns_domains=set(),
        )
        self.ids = IDSEngine(config=config, sensitivity=sensitivity)
        self.metrics: Dict[str, CategoryMetrics] = defaultdict(CategoryMetrics)

    def run(self, dataset: List[LabeledPacket]) -> Dict[str, CategoryMetrics]:
        """Process all packets and compute metrics."""
        for pkt in dataset:
            alerts = self.ids.analyze_packet(
                ip=pkt.ip, tcp=pkt.tcp, udp=pkt.udp,
                dns=pkt.dns, arp=pkt.arp, icmp=pkt.icmp,
            )
            alert_cats = {a.category for a in alerts}
            is_attack = pkt.label != "BENIGN"

            if is_attack:
                # Check if the correct category was detected
                if pkt.label in alert_cats:
                    self.metrics[pkt.label].tp += 1
                else:
                    self.metrics[pkt.label].fn += 1
                # Any OTHER categories are false positives
                for cat in alert_cats:
                    if cat != pkt.label:
                        self.metrics[cat].fp += 1
            else:
                # Benign packet — any alert is a false positive
                if alert_cats:
                    for cat in alert_cats:
                        self.metrics[cat].fp += 1
                # No alert on benign = true negative (counted globally)

        return dict(self.metrics)


def run_benchmark(sensitivity: str = "medium"):
    """Entry point: generate dataset, run benchmark, print results."""
    print()
    print("=" * 72)
    print("  QUANTUM SNIFFER — Detection Quality Benchmark")
    print("=" * 72)

    ds = SyntheticDataset(seed=42)
    packets = ds.generate()
    print(f"\n  Seed: {ds.seed}")
    print(f"  Total packets: {len(packets)}")

    attack_counts = defaultdict(int)
    for p in packets:
        attack_counts[p.label] += 1
    benign = attack_counts.pop("BENIGN", 0)
    print(f"  Benign: {benign}")
    for cat, count in sorted(attack_counts.items()):
        print(f"  {cat}: {count}")

    print(f"\n  Running IDS (sensitivity={sensitivity})...")
    bench = DetectionBenchmark(sensitivity=sensitivity)
    metrics = bench.run(packets)

    print(f"\n  {'Category':<18} {'TP':>4} {'FP':>4} {'FN':>4}  {'Prec':>6} {'Recall':>6} {'F1':>6}")
    print("  " + "-" * 62)

    total_tp = total_fp = total_fn = 0
    for cat in sorted(metrics.keys()):
        m = metrics[cat]
        total_tp += m.tp
        total_fp += m.fp
        total_fn += m.fn
        print(f"  {cat:<18} {m.tp:4d} {m.fp:4d} {m.fn:4d}  {m.precision:6.1%} {m.recall:6.1%} {m.f1:6.1%}")

    print("  " + "-" * 62)
    overall_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    overall_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    overall_f1 = 2 * overall_p * overall_r / (overall_p + overall_r) if (overall_p + overall_r) > 0 else 0
    print(f"  {'OVERALL':<18} {total_tp:4d} {total_fp:4d} {total_fn:4d}  {overall_p:6.1%} {overall_r:6.1%} {overall_f1:6.1%}")

    print(f"\n  IDS stats: {bench.ids.stats['total_packets_analyzed']} packets analyzed, "
          f"{bench.ids.stats['threats_detected']} threats detected")
    print("=" * 72)
    print()

    return metrics


if __name__ == "__main__":
    run_benchmark()
