"""
Comprehensive Benchmark Suite
================================
Real measurements of system performance:
  • IDS-only throughput (pkt/sec)
  • Full-pipeline throughput (IDS + analytics + dissection)
  • Latency percentiles: avg / p50 / p95 / p99
  • PQC transport overhead (with vs without encryption)
  • CPU + memory profiling under load
  • Drop rate simulation (queue overflow)
  • Key rotation cost analysis

Usage:
  python benchmark_suite.py                # Run all benchmarks
  python -m quantum_sniffer --mode bench   # Via CLI
"""

from __future__ import annotations

import gc
import json
import os
import random
import statistics
import sys
import time
import tracemalloc
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

from protocols import IPv4Packet, TCPSegment, TCPFlags, UDPDatagram, DNSMessage, DNSQuestion
from ids import IDSEngine
from analytics import AnalyticsManager


# ──────────────────────────────────────────────────────────────────────
# Result dataclass
# ──────────────────────────────────────────────────────────────────────

@dataclass
class BenchmarkResult:
    """Result from a single benchmark run."""
    name: str
    packet_count: int
    elapsed_sec: float
    throughput_pps: float
    latency_avg_us: float
    latency_p50_us: float
    latency_p95_us: float
    latency_p99_us: float
    latency_min_us: float
    latency_max_us: float
    memory_peak_kb: float = 0.0
    cpu_time_sec: float = 0.0
    extra: Dict = field(default_factory=dict)

    def summary_line(self) -> str:
        return (
            f"  {self.name:<35s} "
            f"{self.throughput_pps:>10,.0f} pkt/s  "
            f"avg={self.latency_avg_us:>7.1f}µs  "
            f"p50={self.latency_p50_us:>7.1f}µs  "
            f"p95={self.latency_p95_us:>7.1f}µs  "
            f"p99={self.latency_p99_us:>7.1f}µs  "
            f"mem={self.memory_peak_kb:>8.0f}KB"
        )


# ──────────────────────────────────────────────────────────────────────
# Synthetic packet generators
# ──────────────────────────────────────────────────────────────────────

def _gen_tcp_packets(n: int) -> list:
    """Generate n synthetic TCP packets as (IPv4Packet, TCPSegment) tuples."""
    packets = []
    for i in range(n):
        ip = IPv4Packet(
            version=4, ihl=20, dscp=0, ecn=0,
            total_length=60 + random.randint(0, 1400),
            identification=i & 0xFFFF, flags=0x02, fragment_offset=0,
            ttl=64, protocol=6, checksum=0,
            src_ip=f"10.0.{(i // 256) % 256}.{i % 256}",
            dst_ip=f"192.168.{(i // 256) % 256}.{i % 256}",
            options=b"", payload=b"\x00" * 20,
        )
        tcp = TCPSegment(
            src_port=random.randint(1024, 65535),
            dst_port=random.choice([80, 443, 22, 53, 8080]),
            seq_num=i, ack_num=0, data_offset=20,
            flags=TCPFlags.SYN if i % 10 == 0 else TCPFlags.ACK,
            window=65535, checksum=0, urgent_ptr=0,
            options=b"", payload=b"",
        )
        packets.append((ip, tcp))
    return packets


def _gen_alert_payloads(n: int) -> list:
    """Generate n realistic alert dicts for PQC transport benchmarking."""
    alerts = []
    categories = ["PORT_SCAN", "SYN_FLOOD", "DNS_TUNNEL", "ARP_SPOOF", "BRUTE_FORCE"]
    for i in range(n):
        alerts.append({
            "timestamp": time.time(),
            "severity": random.randint(1, 5),
            "category": random.choice(categories),
            "source_ip": f"10.0.{i % 256}.{(i // 256) % 256}",
            "destination_ip": f"192.168.1.{i % 256}",
            "description": f"Simulated alert #{i} for benchmark testing with enough data to be realistic",
            "confidence": round(random.uniform(0.3, 1.0), 2),
            "mitre_ref": "T1046 - Network Service Discovery",
            "evidence": [{"metric": "unique_ports", "value": random.randint(10, 100)}],
        })
    return alerts


# ──────────────────────────────────────────────────────────────────────
# Core benchmarks
# ──────────────────────────────────────────────────────────────────────

def _compute_latency_stats(latencies: list) -> dict:
    """Compute latency statistics from a list of microsecond values."""
    if not latencies:
        return {"avg": 0, "p50": 0, "p95": 0, "p99": 0, "min": 0, "max": 0}
    s = sorted(latencies)
    n = len(s)
    return {
        "avg": statistics.mean(s),
        "p50": s[int(n * 0.50)],
        "p95": s[min(int(n * 0.95), n - 1)],
        "p99": s[min(int(n * 0.99), n - 1)],
        "min": s[0],
        "max": s[-1],
    }


def bench_ids_only(packet_count: int = 50000) -> BenchmarkResult:
    """Benchmark: IDS analysis only (no analytics, no dissection)."""
    packets = _gen_tcp_packets(packet_count)
    ids = IDSEngine(sensitivity="medium")
    latencies = []

    gc.disable()
    tracemalloc.start()
    cpu_start = time.process_time()
    t0 = time.perf_counter()

    for ip, tcp in packets:
        pstart = time.perf_counter()
        ids.analyze_packet(ip=ip, tcp=tcp)
        latencies.append((time.perf_counter() - pstart) * 1_000_000)

    elapsed = time.perf_counter() - t0
    cpu_time = time.process_time() - cpu_start
    mem_peak = tracemalloc.get_traced_memory()[1] / 1024
    tracemalloc.stop()
    gc.enable()

    stats = _compute_latency_stats(latencies)
    return BenchmarkResult(
        name="IDS Only",
        packet_count=packet_count,
        elapsed_sec=elapsed,
        throughput_pps=packet_count / elapsed,
        latency_avg_us=stats["avg"], latency_p50_us=stats["p50"],
        latency_p95_us=stats["p95"], latency_p99_us=stats["p99"],
        latency_min_us=stats["min"], latency_max_us=stats["max"],
        memory_peak_kb=mem_peak, cpu_time_sec=cpu_time,
        extra={"alerts_generated": ids.stats["threats_detected"]},
    )


def bench_analytics_only(packet_count: int = 50000) -> BenchmarkResult:
    """Benchmark: Analytics recording only."""
    packets = _gen_tcp_packets(packet_count)
    analytics = AnalyticsManager()
    latencies = []

    gc.disable()
    tracemalloc.start()
    cpu_start = time.process_time()
    t0 = time.perf_counter()

    for ip, tcp in packets:
        pstart = time.perf_counter()
        analytics.record_packet("TCP", ip.src_ip, ip.dst_ip, ip.total_length,
                                tcp.src_port, tcp.dst_port)
        latencies.append((time.perf_counter() - pstart) * 1_000_000)

    elapsed = time.perf_counter() - t0
    cpu_time = time.process_time() - cpu_start
    mem_peak = tracemalloc.get_traced_memory()[1] / 1024
    tracemalloc.stop()
    gc.enable()

    stats = _compute_latency_stats(latencies)
    return BenchmarkResult(
        name="Analytics Only",
        packet_count=packet_count,
        elapsed_sec=elapsed,
        throughput_pps=packet_count / elapsed,
        latency_avg_us=stats["avg"], latency_p50_us=stats["p50"],
        latency_p95_us=stats["p95"], latency_p99_us=stats["p99"],
        latency_min_us=stats["min"], latency_max_us=stats["max"],
        memory_peak_kb=mem_peak, cpu_time_sec=cpu_time,
    )


def bench_full_pipeline(packet_count: int = 50000) -> BenchmarkResult:
    """Benchmark: Full pipeline = IDS + Analytics."""
    packets = _gen_tcp_packets(packet_count)
    ids = IDSEngine(sensitivity="medium")
    analytics = AnalyticsManager()
    latencies = []

    gc.disable()
    tracemalloc.start()
    cpu_start = time.process_time()
    t0 = time.perf_counter()

    for ip, tcp in packets:
        pstart = time.perf_counter()
        ids.analyze_packet(ip=ip, tcp=tcp)
        analytics.record_packet("TCP", ip.src_ip, ip.dst_ip, ip.total_length,
                                tcp.src_port, tcp.dst_port)
        latencies.append((time.perf_counter() - pstart) * 1_000_000)

    elapsed = time.perf_counter() - t0
    cpu_time = time.process_time() - cpu_start
    mem_peak = tracemalloc.get_traced_memory()[1] / 1024
    tracemalloc.stop()
    gc.enable()

    stats = _compute_latency_stats(latencies)
    return BenchmarkResult(
        name="Full Pipeline (IDS + Analytics)",
        packet_count=packet_count,
        elapsed_sec=elapsed,
        throughput_pps=packet_count / elapsed,
        latency_avg_us=stats["avg"], latency_p50_us=stats["p50"],
        latency_p95_us=stats["p95"], latency_p99_us=stats["p99"],
        latency_min_us=stats["min"], latency_max_us=stats["max"],
        memory_peak_kb=mem_peak, cpu_time_sec=cpu_time,
        extra={"alerts_generated": ids.stats["threats_detected"]},
    )


def bench_pqc_transport(message_count: int = 500) -> dict:
    """Benchmark: PQC transport encryption/decryption overhead."""
    from pqc_transport import PQCTransport

    alerts = _gen_alert_payloads(message_count)

    # ── Plaintext baseline ──
    latencies_plain = []
    t0 = time.perf_counter()
    for alert in alerts:
        pstart = time.perf_counter()
        _ = json.dumps(alert).encode("utf-8")
        latencies_plain.append((time.perf_counter() - pstart) * 1_000_000)
    elapsed_plain = time.perf_counter() - t0

    # ── PQC encrypted ──
    agg = PQCTransport()
    pk = agg.keygen()
    sensor = PQCTransport()
    sensor.set_peer_public_key(pk)

    latencies_enc = []
    latencies_dec = []
    tracemalloc.start()
    t0 = time.perf_counter()
    for i, alert in enumerate(alerts):
        # Encrypt
        pstart = time.perf_counter()
        encrypted = sensor.encrypt_payload(alert)
        latencies_enc.append((time.perf_counter() - pstart) * 1_000_000)
        # Decrypt
        pstart = time.perf_counter()
        _ = agg.decrypt_payload(encrypted, sender_id=f"bench-{i % 10}")
        latencies_dec.append((time.perf_counter() - pstart) * 1_000_000)
    elapsed_pqc = time.perf_counter() - t0
    mem_peak = tracemalloc.get_traced_memory()[1] / 1024
    tracemalloc.stop()

    enc_stats = _compute_latency_stats(latencies_enc)
    dec_stats = _compute_latency_stats(latencies_dec)
    plain_stats = _compute_latency_stats(latencies_plain)

    overhead_factor = elapsed_pqc / max(elapsed_plain, 0.001)

    return {
        "message_count": message_count,
        "plaintext": BenchmarkResult(
            name="Plaintext Serialization",
            packet_count=message_count,
            elapsed_sec=elapsed_plain,
            throughput_pps=message_count / elapsed_plain,
            latency_avg_us=plain_stats["avg"], latency_p50_us=plain_stats["p50"],
            latency_p95_us=plain_stats["p95"], latency_p99_us=plain_stats["p99"],
            latency_min_us=plain_stats["min"], latency_max_us=plain_stats["max"],
        ),
        "encryption": BenchmarkResult(
            name="PQC Encrypt (AES-256-GCM)",
            packet_count=message_count,
            elapsed_sec=sum(latencies_enc) / 1e6,
            throughput_pps=message_count / (sum(latencies_enc) / 1e6),
            latency_avg_us=enc_stats["avg"], latency_p50_us=enc_stats["p50"],
            latency_p95_us=enc_stats["p95"], latency_p99_us=enc_stats["p99"],
            latency_min_us=enc_stats["min"], latency_max_us=enc_stats["max"],
        ),
        "decryption": BenchmarkResult(
            name="PQC Decrypt (KEM + AES-GCM)",
            packet_count=message_count,
            elapsed_sec=sum(latencies_dec) / 1e6,
            throughput_pps=message_count / (sum(latencies_dec) / 1e6),
            latency_avg_us=dec_stats["avg"], latency_p50_us=dec_stats["p50"],
            latency_p95_us=dec_stats["p95"], latency_p99_us=dec_stats["p99"],
            latency_min_us=dec_stats["min"], latency_max_us=dec_stats["max"],
        ),
        "overhead_factor": overhead_factor,
        "memory_peak_kb": mem_peak,
        "session_stats": sensor.stats,
    }


def bench_key_rotation(rotations: int = 20) -> dict:
    """Benchmark: Cost of KEM key rotation."""
    from pqc_transport import PQCTransport
    from pqc import KyberKEM

    # Measure raw KEM keygen + encapsulate cost
    kem = KyberKEM()
    kg_latencies = []
    enc_latencies = []

    for _ in range(rotations):
        t0 = time.perf_counter()
        pk, sk = kem.keygen()
        kg_latencies.append((time.perf_counter() - t0) * 1_000_000)

        t0 = time.perf_counter()
        ct, ss = kem.encapsulate(pk)
        enc_latencies.append((time.perf_counter() - t0) * 1_000_000)

    # Measure rotation in PQCTransport context
    agg = PQCTransport()
    pk = agg.keygen()
    sensor = PQCTransport()
    sensor.set_peer_public_key(pk)

    rot_latencies = []
    for _ in range(rotations):
        t0 = time.perf_counter()
        sensor.rotate_key()
        rot_latencies.append((time.perf_counter() - t0) * 1_000_000)

    return {
        "rotations": rotations,
        "keygen_avg_us": statistics.mean(kg_latencies),
        "keygen_p99_us": sorted(kg_latencies)[min(int(len(kg_latencies) * 0.99), len(kg_latencies) - 1)],
        "encapsulate_avg_us": statistics.mean(enc_latencies),
        "encapsulate_p99_us": sorted(enc_latencies)[min(int(len(enc_latencies) * 0.99), len(enc_latencies) - 1)],
        "rotation_avg_us": statistics.mean(rot_latencies),
        "rotation_p99_us": sorted(rot_latencies)[min(int(len(rot_latencies) * 0.99), len(rot_latencies) - 1)],
        "rotation_cost_vs_message_pct": (
            statistics.mean(rot_latencies) /
            max(statistics.mean(enc_latencies), 0.01) * 100
        ),
    }


def bench_drop_rate(packet_count: int = 100000, queue_size: int = 1000) -> dict:
    """Simulate queue overflow and measure drop rate."""
    from queue import Queue, Full

    q = Queue(maxsize=queue_size)
    dropped = 0
    accepted = 0

    # Simulate burst-producing faster than consumer can drain
    for i in range(packet_count):
        try:
            q.put_nowait(i)
            accepted += 1
        except Full:
            dropped += 1
        # Simulate slow consumer — drain every 10th packet
        if i % 10 == 0:
            try:
                q.get_nowait()
            except Exception:
                pass

    return {
        "total_produced": packet_count,
        "accepted": accepted,
        "dropped": dropped,
        "drop_rate_pct": (dropped / packet_count) * 100,
        "queue_size": queue_size,
    }


# ──────────────────────────────────────────────────────────────────────
# Main runner
# ──────────────────────────────────────────────────────────────────────

def run_all(packet_count: int = 50000) -> dict:
    """Run the complete benchmark suite and print results."""
    print("=" * 80)
    print("  QUANTUM SNIFFER - COMPREHENSIVE BENCHMARK SUITE")
    print("=" * 80)
    print(f"  Packet count: {packet_count:,}")
    print(f"  Python: {sys.version.split()[0]}")
    print()

    results = {}

    # ── Phase 1: Pipeline throughput ──
    print(f"  > Phase 1: Pipeline Throughput")
    print("  " + "─" * 76)

    for name, fn in [
        ("IDS Only", bench_ids_only),
        ("Analytics Only", bench_analytics_only),
        ("Full Pipeline", bench_full_pipeline),
    ]:
        print(f"    Running {name}...", end="", flush=True)
        r = fn(packet_count)
        results[name] = r
        print(f" {r.throughput_pps:,.0f} pkt/s")
        print(f"    {r.summary_line()}")

    # ── Phase 2: PQC Transport Overhead ──
    msg_count = min(packet_count // 100, 500)
    print(f"\n  ▶ Phase 2: PQC Transport Overhead ({msg_count} messages)")
    print("  " + "─" * 76)

    pqc_results = bench_pqc_transport(msg_count)
    results["PQC Transport"] = pqc_results
    print(f"    {pqc_results['plaintext'].summary_line()}")
    print(f"    {pqc_results['encryption'].summary_line()}")
    print(f"    {pqc_results['decryption'].summary_line()}")
    print(f"    Overhead factor: {pqc_results['overhead_factor']:.1f}x vs plaintext")
    print(f"    Peak memory: {pqc_results['memory_peak_kb']:.0f} KB")

    # ── Phase 3: Key Rotation Cost ──
    print(f"\n  ▶ Phase 3: Key Rotation Cost (20 rotations)")
    print("  " + "─" * 76)

    kr = bench_key_rotation(20)
    results["Key Rotation"] = kr
    print(f"    KEM keygen:      avg={kr['keygen_avg_us']:>10.0f}µs  p99={kr['keygen_p99_us']:>10.0f}µs")
    print(f"    KEM encapsulate: avg={kr['encapsulate_avg_us']:>10.0f}µs  p99={kr['encapsulate_p99_us']:>10.0f}µs")
    print(f"    Full rotation:   avg={kr['rotation_avg_us']:>10.0f}µs  p99={kr['rotation_p99_us']:>10.0f}µs")
    print(f"    Rotation cost:   {kr['rotation_cost_vs_message_pct']:.0f}% of one message encrypt")

    # ── Phase 4: Drop Rate ──
    print(f"\n  ▶ Phase 4: Drop Rate Simulation")
    print("  " + "─" * 76)

    dr = bench_drop_rate(100000, 1000)
    results["Drop Rate"] = dr
    print(f"    Queue size: {dr['queue_size']}  |  Produced: {dr['total_produced']:,}")
    print(f"    Accepted: {dr['accepted']:,}  |  Dropped: {dr['dropped']:,}  |  Drop rate: {dr['drop_rate_pct']:.1f}%")

    # ── Summary ──
    print(f"\n{'=' * 80}")
    print(f"  BENCHMARK SUMMARY")
    print(f"{'=' * 80}")
    print(f"  {'Component':<35s} {'pkt/sec':>12s} {'avg lat':>10s} {'p99 lat':>10s} {'memory':>10s}")
    print(f"  {'─' * 35} {'─' * 12} {'─' * 10} {'─' * 10} {'─' * 10}")
    for name in ["IDS Only", "Analytics Only", "Full Pipeline"]:
        r = results[name]
        print(f"  {r.name:<35s} {r.throughput_pps:>12,.0f} {r.latency_avg_us:>8.1f}µs {r.latency_p99_us:>8.1f}µs {r.memory_peak_kb:>8.0f}KB")

    print(f"\n  PQC Transport overhead: {pqc_results['overhead_factor']:.1f}x")
    print(f"  Key rotation cost: {kr['rotation_avg_us']:.0f}µs/rotation")
    print(f"  Drop rate (burst): {dr['drop_rate_pct']:.1f}% @ queue={dr['queue_size']}")
    print(f"{'=' * 80}")

    return results


if __name__ == "__main__":
    run_all()
