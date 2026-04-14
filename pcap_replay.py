"""
PCAP Replay Engine
===================
Replay captured network traffic from .pcap/.pcapng files:
  - Feed packets through the full IDS + analytics pipeline
  - Real-time or max-speed replay
  - Support for Wireshark/tcpdump captures
  - Dataset testing (CICIDS, UNSW-NB15 CSV format)

Usage:
  python pcap_replay.py capture.pcap [--speed 1.0] [--max-speed]
  python __main__.py --pcap capture.pcap
"""

from __future__ import annotations

import csv
import os
import sys
import time
from pathlib import Path
from typing import List, Optional

try:
    from scapy.all import rdpcap, PcapReader, Ether, Raw
    HAS_SCAPY = True
except ImportError:
    HAS_SCAPY = False

from protocols import (
    parse_ethernet, parse_ipv4, parse_tcp, parse_udp,
    parse_dns, parse_icmp, parse_arp, parse_tls,
    EtherType, TLSClientHello, TLSServerHello,
)
from ids import IDSEngine, Severity, SEVERITY_LABELS
from analytics import AnalyticsManager


class PcapReplayer:
    """Replay PCAP files through the IDS and analytics pipeline."""

    def __init__(
        self,
        sensitivity: str = "high",
        max_speed: bool = False,
        speed_factor: float = 1.0,
        verbose: bool = True,
    ):
        self.ids = IDSEngine(sensitivity=sensitivity)
        self.analytics = AnalyticsManager()
        self.max_speed = max_speed
        self.speed_factor = speed_factor
        self.verbose = verbose

        # Stats
        self.packets_processed = 0
        self.packets_failed = 0
        self.alerts_generated = 0
        self.start_time = 0.0

    def replay_file(self, pcap_path: str) -> dict:
        """Replay a PCAP file and return analysis results."""
        if not HAS_SCAPY:
            print("  ERROR: Scapy not installed. pip install scapy")
            return {"error": "scapy not installed"}

        path = Path(pcap_path)
        if not path.exists():
            print(f"  ERROR: File not found: {pcap_path}")
            return {"error": f"file not found: {pcap_path}"}

        file_size = path.stat().st_size
        print("=" * 70)
        print("  PCAP REPLAY ENGINE")
        print("=" * 70)
        print(f"  File: {path.name}")
        print(f"  Size: {file_size / 1024:.1f} KB")
        print(f"  Mode: {'max-speed' if self.max_speed else f'{self.speed_factor}x real-time'}")
        print(f"  IDS Sensitivity: {self.ids.config.port_scan_threshold} port threshold")
        print()

        self.start_time = time.time()
        all_alerts = []
        prev_ts = None

        try:
            # Use PcapReader for memory-efficient streaming
            with PcapReader(str(path)) as reader:
                for pkt in reader:
                    # Timing control
                    if not self.max_speed and prev_ts is not None:
                        pkt_ts = float(pkt.time)
                        delay = (pkt_ts - prev_ts) / self.speed_factor
                        if delay > 0 and delay < 2.0:  # Cap at 2s
                            time.sleep(delay)
                    if hasattr(pkt, 'time'):
                        prev_ts = float(pkt.time)

                    # Process packet
                    try:
                        raw = bytes(pkt)
                        alerts = self._process_packet(raw)
                        all_alerts.extend(alerts)
                        self.packets_processed += 1
                    except Exception:
                        self.packets_failed += 1

                    # Progress
                    if self.verbose and self.packets_processed % 1000 == 0:
                        elapsed = time.time() - self.start_time
                        pps = self.packets_processed / max(elapsed, 0.001)
                        print(f"  ... {self.packets_processed:,} packets, "
                              f"{len(all_alerts)} alerts, "
                              f"{pps:.0f} pkt/s")

        except Exception as e:
            print(f"  ERROR reading PCAP: {e}")
            return {"error": str(e)}

        elapsed = time.time() - self.start_time
        return self._print_summary(all_alerts, elapsed)

    def _process_packet(self, raw: bytes) -> list:
        """Process a single raw packet through IDS + analytics."""
        alerts = []

        eth = parse_ethernet(raw)
        if eth is None:
            return alerts

        payload = eth.payload

        # ARP
        if eth.ether_type == EtherType.ARP:
            arp = parse_arp(payload)
            if arp:
                self.analytics.record_packet("ARP", arp.sender_ip, arp.target_ip, len(raw))
                new = self.ids.analyze_packet(arp=arp)
                alerts.extend(new)
            return alerts

        # IPv4
        if eth.ether_type == EtherType.IPv4:
            ip = parse_ipv4(payload)
            if ip is None:
                return alerts

            src, dst = ip.src_ip, ip.dst_ip
            transport = ip.payload

            # ICMP
            if ip.protocol == 1:
                icmp = parse_icmp(transport)
                if icmp:
                    self.analytics.record_packet("ICMP", src, dst, len(raw))
                    new = self.ids.analyze_packet(ip=ip, icmp=icmp)
                    alerts.extend(new)
                return alerts

            # TCP
            if ip.protocol == 6:
                tcp = parse_tcp(transport)
                if tcp:
                    self.analytics.record_packet("TCP", src, dst, len(raw),
                                                 tcp.src_port, tcp.dst_port)

                    # TLS detection
                    if tcp.payload and tcp.dst_port in (443, 8443):
                        tls = parse_tls(tcp.payload)
                        if isinstance(tls, TLSClientHello):
                            self.analytics.record_packet("TLS", src, dst, len(tcp.payload))

                    new = self.ids.analyze_packet(ip=ip, tcp=tcp)
                    alerts.extend(new)
                return alerts

            # UDP
            if ip.protocol == 17:
                udp = parse_udp(transport)
                if udp:
                    self.analytics.record_packet("UDP", src, dst, len(raw),
                                                 udp.src_port, udp.dst_port)

                    # DNS
                    if udp.src_port == 53 or udp.dst_port == 53:
                        dns = parse_dns(udp.payload)
                        if dns:
                            self.analytics.record_packet("DNS", src, dst, len(udp.payload))
                            new = self.ids.analyze_packet(ip=ip, udp=udp, dns=dns)
                            alerts.extend(new)
                            return alerts

                    new = self.ids.analyze_packet(ip=ip, udp=udp)
                    alerts.extend(new)

        return alerts

    def _print_summary(self, alerts: list, elapsed: float) -> dict:
        """Print replay summary and return results dict."""
        pps = self.packets_processed / max(elapsed, 0.001)

        print()
        print("=" * 70)
        print("  REPLAY SUMMARY")
        print("=" * 70)
        print(f"  Packets processed: {self.packets_processed:,}")
        print(f"  Packets failed:    {self.packets_failed:,}")
        print(f"  Elapsed time:      {elapsed:.2f}s")
        print(f"  Throughput:        {pps:,.0f} packets/sec")
        print()

        # Alert summary
        print(f"  Total alerts:      {len(alerts)}")
        if alerts:
            by_severity = {}
            by_category = {}
            for a in alerts:
                sev = SEVERITY_LABELS.get(a.severity, str(a.severity))
                by_severity[sev] = by_severity.get(sev, 0) + 1
                by_category[a.category] = by_category.get(a.category, 0) + 1

            print(f"  By severity:")
            for sev in ["CRITICAL", "HIGH", "MEDIUM", "LOW", "INFO"]:
                if sev in by_severity:
                    print(f"    {sev}: {by_severity[sev]}")

            print(f"  By category:")
            for cat, count in sorted(by_category.items(), key=lambda x: -x[1]):
                print(f"    {cat}: {count}")

            print(f"\n  Top 5 alerts:")
            seen = set()
            for a in alerts:
                key = (a.category, a.source_ip)
                if key not in seen:
                    seen.add(key)
                    sev = SEVERITY_LABELS.get(a.severity, str(a.severity))
                    print(f"    [{sev}] {a.description[:80]}")
                    if len(seen) >= 5:
                        break

        # Protocol stats
        protos = self.analytics.protocols.top_protocols
        if protos:
            print(f"\n  Protocol distribution:")
            for proto, count in protos[:8]:
                pct = (count / max(self.packets_processed, 1)) * 100
                print(f"    {proto}: {count:,} ({pct:.1f}%)")

        # Top talkers
        senders = self.analytics.talkers.top_senders[:5]
        if senders:
            print(f"\n  Top senders:")
            for ip, byt in senders:
                print(f"    {ip}: {byt / 1024:.1f} KB")

        print("=" * 70)

        return {
            "packets_processed": self.packets_processed,
            "packets_failed": self.packets_failed,
            "elapsed": elapsed,
            "throughput_pps": pps,
            "total_alerts": len(alerts),
            "alerts_by_severity": by_severity if alerts else {},
            "alerts_by_category": by_category if alerts else {},
            "protocols": dict(self.analytics.protocols.counts),
            "ids_stats": dict(self.ids.stats),
        }


class DatasetTester:
    """Test IDS against labeled datasets (CICIDS, UNSW-NB15 CSV format)."""

    def __init__(self, sensitivity: str = "high"):
        self.ids = IDSEngine(sensitivity=sensitivity)
        self.true_positives = 0
        self.false_positives = 0
        self.false_negatives = 0
        self.true_negatives = 0
        self.total = 0

    def test_csv(self, csv_path: str, label_column: str = "Label",
                 src_ip_col: str = "Src IP", dst_ip_col: str = "Dst IP",
                 src_port_col: str = "Src Port", dst_port_col: str = "Dst Port",
                 protocol_col: str = "Protocol") -> dict:
        """Test against a labeled CSV dataset."""
        print("=" * 70)
        print("  DATASET TESTING")
        print("=" * 70)
        print(f"  File: {csv_path}")

        try:
            with open(csv_path, 'r', encoding='utf-8-sig') as f:
                reader = csv.DictReader(f)

                for row in reader:
                    self.total += 1
                    label = row.get(label_column, "BENIGN").strip()
                    is_attack = label.upper() != "BENIGN"

                    # Create minimal packet for IDS
                    from protocols import IPv4Packet, TCPSegment, TCPFlags
                    src_ip = row.get(src_ip_col, "0.0.0.0").strip()
                    dst_ip = row.get(dst_ip_col, "0.0.0.0").strip()
                    try:
                        src_port = int(float(row.get(src_port_col, 0)))
                        dst_port = int(float(row.get(dst_port_col, 0)))
                    except (ValueError, TypeError):
                        src_port, dst_port = 0, 0

                    ip = IPv4Packet(
                        version=4, ihl=20, dscp=0, ecn=0, total_length=60,
                        identification=self.total, flags=0x02, fragment_offset=0,
                        ttl=64, protocol=6, checksum=0,
                        src_ip=src_ip, dst_ip=dst_ip, options=b"", payload=b"",
                    )
                    tcp = TCPSegment(
                        src_port=src_port, dst_port=dst_port,
                        seq_num=0, ack_num=0, data_offset=20,
                        flags=TCPFlags.SYN, window=65535,
                        checksum=0, urgent_ptr=0, options=b"", payload=b"",
                    )

                    alerts = self.ids.analyze_packet(ip=ip, tcp=tcp)
                    detected = len(alerts) > 0

                    if is_attack and detected:
                        self.true_positives += 1
                    elif is_attack and not detected:
                        self.false_negatives += 1
                    elif not is_attack and detected:
                        self.false_positives += 1
                    else:
                        self.true_negatives += 1

                    if self.total % 10000 == 0:
                        print(f"  ... {self.total:,} rows processed")

        except Exception as e:
            print(f"  ERROR: {e}")
            return {"error": str(e)}

        return self._print_results()

    def _print_results(self) -> dict:
        """Print confusion matrix and metrics."""
        precision = self.true_positives / max(self.true_positives + self.false_positives, 1)
        recall = self.true_positives / max(self.true_positives + self.false_negatives, 1)
        f1 = 2 * precision * recall / max(precision + recall, 0.001)
        accuracy = (self.true_positives + self.true_negatives) / max(self.total, 1)

        print(f"\n  Results ({self.total:,} samples):")
        print(f"  {'':>20} Predicted Attack  Predicted Benign")
        print(f"  {'Actual Attack':>20}  {self.true_positives:>10,}     {self.false_negatives:>10,}")
        print(f"  {'Actual Benign':>20}  {self.false_positives:>10,}     {self.true_negatives:>10,}")
        print()
        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1 Score:  {f1:.4f}")
        print("=" * 70)

        return {
            "total": self.total,
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "true_negatives": self.true_negatives,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }


def run_pcap_replay(pcap_path: str, speed: float = 1.0, max_speed: bool = False):
    """Entry point for PCAP replay."""
    replayer = PcapReplayer(max_speed=max_speed, speed_factor=speed)
    return replayer.replay_file(pcap_path)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="PCAP Replay Engine")
    parser.add_argument("pcap", help="Path to .pcap/.pcapng file")
    parser.add_argument("--speed", type=float, default=1.0, help="Replay speed multiplier")
    parser.add_argument("--max-speed", action="store_true", help="Replay at maximum speed")
    args = parser.parse_args()

    run_pcap_replay(args.pcap, args.speed, args.max_speed)
