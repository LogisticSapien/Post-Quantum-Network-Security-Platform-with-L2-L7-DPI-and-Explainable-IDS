"""
Attack Simulator
=================
Generates realistic attack traffic and feeds it through the IDS engine
to demonstrate detection working live. Supports:
  • SYN flood
  • DNS tunneling (high-entropy encoded subdomains)
  • ARP spoofing (IP→MAC changes)
  • Port scan (SYN/FIN/XMAS/NULL)
  • Brute force (SSH/RDP rapid connections)
  • ICMP tunnel (large high-entropy payloads)
"""

from __future__ import annotations

import base64
import os
import random
import string
import time
from typing import Dict, List, Tuple

from protocols import (
    ARPPacket, DNSMessage, DNSQuestion, ICMPPacket, IPv4Packet,
    TCPSegment, TCPFlags, UDPDatagram, TLSClientHello,
    analyze_tls_handshake,
)
from ids import IDSEngine, IDSConfig, Severity, ThreatEvent, SEVERITY_LABELS


class AttackSimulator:
    """Simulates network attacks and tests IDS detection."""

    def __init__(self):
        self.ids = IDSEngine(sensitivity="high")
        self.results: Dict[str, dict] = {}

    def run_all(self) -> Dict[str, dict]:
        """Run all attack simulations and return results."""
        print("=" * 70)
        print("  ATTACK SIMULATION + LIVE DETECTION")
        print("=" * 70)

        sims = [
            ("SYN Flood",       self._sim_syn_flood),
            ("Port Scan",       self._sim_port_scan),
            ("DNS Tunneling",   self._sim_dns_tunnel),
            ("ARP Spoofing",    self._sim_arp_spoof),
            ("Brute Force",     self._sim_brute_force),
            ("ICMP Tunnel",     self._sim_icmp_tunnel),
        ]

        all_passed = True
        for name, sim_fn in sims:
            print(f"\n{'-' * 70}")
            print(f"  ATTACK: {name}")
            print(f"{'-' * 70}")
            result = sim_fn()
            self.results[name] = result
            status = "DETECTED" if result["detected"] else "MISSED"
            icon = "OK" if result["detected"] else "FAIL"
            print(f"\n  [{icon}] {name}: {status}")
            if result["detected"]:
                print(f"      Alerts generated: {result['alert_count']}")
                for alert in result.get("alerts", [])[:3]:
                    sev = SEVERITY_LABELS.get(alert.severity, str(alert.severity))
                    print(f"      [{sev}] {alert.description}")
                    if alert.explanation:
                        # Print first 120 chars of explanation
                        expl = alert.explanation[:120] + "..." if len(alert.explanation) > 120 else alert.explanation
                        print(f"        WHY: {expl}")
                    if alert.evidence_factors:
                        for ef in alert.evidence_factors[:2]:
                            print(f"        - {ef.metric}: {ef.observed} ({ef.threshold})")
                    if alert.response_actions:
                        print(f"        ACTION: {alert.response_actions[0]}")
            else:
                all_passed = False
                print(f"      WARNING: IDS did not detect this attack")

        # Summary
        print(f"\n{'=' * 70}")
        print(f"  SIMULATION SUMMARY")
        print(f"{'=' * 70}")
        detected = sum(1 for r in self.results.values() if r["detected"])
        total = len(self.results)
        print(f"  Attacks simulated: {total}")
        print(f"  Detected: {detected}/{total}")
        print(f"  Total alerts: {self.ids.stats['threats_detected']}")
        print(f"  Packets analyzed: {self.ids.stats['total_packets_analyzed']}")

        if all_passed:
            print(f"\n  ALL ATTACKS DETECTED")
        else:
            missed = [n for n, r in self.results.items() if not r["detected"]]
            print(f"\n  MISSED: {', '.join(missed)}")

        # Deep TLS demo
        print(f"\n{'-' * 70}")
        print(f"  DEEP TLS ANALYSIS DEMO")
        print(f"{'-' * 70}")
        self._demo_tls_analysis()

        print(f"\n{'=' * 70}")
        return self.results

    def _sim_syn_flood(self) -> dict:
        """Simulate a SYN flood attack."""
        target = "192.168.1.100"
        alerts = []

        print(f"  Target: {target}")
        print(f"  Sending 200 SYN packets from spoofed IPs...")

        now = time.time()
        for i in range(200):
            src_ip = f"10.{random.randint(1,254)}.{random.randint(1,254)}.{random.randint(1,254)}"
            ip = IPv4Packet(
                version=4, ihl=20, dscp=0, ecn=0, total_length=40,
                identification=random.randint(0, 65535), flags=0x02,
                fragment_offset=0, ttl=64, protocol=6, checksum=0,
                src_ip=src_ip, dst_ip=target, options=b"", payload=b"",
            )
            tcp = TCPSegment(
                src_port=random.randint(1024, 65535), dst_port=80,
                seq_num=random.randint(0, 2**32-1), ack_num=0,
                data_offset=20, flags=TCPFlags.SYN, window=65535,
                checksum=0, urgent_ptr=0, options=b"", payload=b"",
            )
            new = self.ids.analyze_packet(ip=ip, tcp=tcp)
            alerts.extend(new)

        return {"detected": len(alerts) > 0, "alert_count": len(alerts), "alerts": alerts}

    def _sim_port_scan(self) -> dict:
        """Simulate a port scan."""
        scanner = "192.168.1.50"
        target = "192.168.1.100"
        alerts = []

        print(f"  Scanner: {scanner} -> Target: {target}")
        print(f"  Scanning 50 ports with SYN/FIN/XMAS patterns...")

        scan_types = [
            (TCPFlags.SYN, "SYN"),
            (TCPFlags.FIN, "FIN"),
            (TCPFlags.FIN | TCPFlags.PSH | TCPFlags.URG, "XMAS"),
        ]

        for port in range(20, 70):
            flags, _ = random.choice(scan_types)
            ip = IPv4Packet(
                version=4, ihl=20, dscp=0, ecn=0, total_length=40,
                identification=random.randint(0, 65535), flags=0x02,
                fragment_offset=0, ttl=64, protocol=6, checksum=0,
                src_ip=scanner, dst_ip=target, options=b"", payload=b"",
            )
            tcp = TCPSegment(
                src_port=random.randint(40000, 65535), dst_port=port,
                seq_num=0, ack_num=0, data_offset=20, flags=flags,
                window=1024, checksum=0, urgent_ptr=0, options=b"", payload=b"",
            )
            new = self.ids.analyze_packet(ip=ip, tcp=tcp)
            alerts.extend(new)

        return {"detected": len(alerts) > 0, "alert_count": len(alerts), "alerts": alerts}

    def _sim_dns_tunnel(self) -> dict:
        """Simulate DNS tunneling with high-entropy queries."""
        src = "192.168.1.10"
        alerts = []

        print(f"  Source: {src}")
        print(f"  Sending 20 high-entropy DNS queries (simulating dnscat2)...")

        for i in range(20):
            # Generate base64-like random subdomain
            data = os.urandom(30)
            encoded = base64.b32encode(data).decode().lower().rstrip("=")
            query_name = f"{encoded[:40]}.tunnel.evil.com"

            dns = DNSMessage(
                transaction_id=random.randint(0, 65535),
                is_response=False, opcode=0, rcode=0,
                questions=[DNSQuestion(name=query_name, qtype=1, qclass=1)],
                answers=[], authorities=[], additionals=[],
            )
            ip = IPv4Packet(
                version=4, ihl=20, dscp=0, ecn=0, total_length=100,
                identification=random.randint(0, 65535), flags=0x02,
                fragment_offset=0, ttl=64, protocol=17, checksum=0,
                src_ip=src, dst_ip="8.8.8.8", options=b"", payload=b"",
            )
            new = self.ids.analyze_packet(ip=ip, dns=dns)
            alerts.extend(new)

        return {"detected": len(alerts) > 0, "alert_count": len(alerts), "alerts": alerts}

    def _sim_arp_spoof(self) -> dict:
        """Simulate ARP spoofing (gateway MAC change)."""
        alerts = []

        print(f"  Simulating gateway 192.168.1.1 MAC change...")
        print(f"  Legitimate: aa:bb:cc:dd:ee:ff -> Attacker: 66:77:88:99:aa:bb")

        # First: legitimate ARP reply
        legit = ARPPacket(
            hw_type=1, proto_type=0x0800, opcode=2,
            sender_mac="aa:bb:cc:dd:ee:ff", sender_ip="192.168.1.1",
            target_mac="11:22:33:44:55:66", target_ip="192.168.1.10",
        )
        self.ids.analyze_packet(arp=legit)

        # Then: spoofed ARP reply (different MAC for same IP)
        spoof = ARPPacket(
            hw_type=1, proto_type=0x0800, opcode=2,
            sender_mac="66:77:88:99:aa:bb", sender_ip="192.168.1.1",
            target_mac="11:22:33:44:55:66", target_ip="192.168.1.10",
        )
        new = self.ids.analyze_packet(arp=spoof)
        alerts.extend(new)

        return {"detected": len(alerts) > 0, "alert_count": len(alerts), "alerts": alerts}

    def _sim_brute_force(self) -> dict:
        """Simulate SSH brute-force attack."""
        attacker = "10.0.0.50"
        target = "192.168.1.1"
        alerts = []

        print(f"  Attacker: {attacker} -> Target: {target}:22 (SSH)")
        print(f"  Sending 15 rapid connection attempts...")

        for i in range(15):
            ip = IPv4Packet(
                version=4, ihl=20, dscp=0, ecn=0, total_length=40,
                identification=random.randint(0, 65535), flags=0x02,
                fragment_offset=0, ttl=64, protocol=6, checksum=0,
                src_ip=attacker, dst_ip=target, options=b"", payload=b"",
            )
            tcp = TCPSegment(
                src_port=random.randint(40000, 65535), dst_port=22,
                seq_num=random.randint(0, 2**32-1), ack_num=0,
                data_offset=20, flags=TCPFlags.SYN, window=65535,
                checksum=0, urgent_ptr=0, options=b"", payload=b"",
            )
            new = self.ids.analyze_packet(ip=ip, tcp=tcp)
            alerts.extend(new)

        return {"detected": len(alerts) > 0, "alert_count": len(alerts), "alerts": alerts}

    def _sim_icmp_tunnel(self) -> dict:
        """Simulate ICMP tunneling with large high-entropy payloads."""
        src = "192.168.1.10"
        alerts = []

        print(f"  Source: {src}")
        print(f"  Sending 5 ICMP packets with 256B high-entropy payloads...")

        for i in range(5):
            payload = os.urandom(256)  # High entropy random data
            ip = IPv4Packet(
                version=4, ihl=20, dscp=0, ecn=0, total_length=276,
                identification=random.randint(0, 65535), flags=0x02,
                fragment_offset=0, ttl=64, protocol=1, checksum=0,
                src_ip=src, dst_ip="10.0.0.1", options=b"", payload=b"",
            )
            icmp = ICMPPacket(
                type=8, code=0, checksum=0,
                identifier=1, sequence=i+1, payload=payload,
            )
            new = self.ids.analyze_packet(ip=ip, icmp=icmp)
            alerts.extend(new)

        return {"detected": len(alerts) > 0, "alert_count": len(alerts), "alerts": alerts}

    def _demo_tls_analysis(self):
        """Demo the deep TLS analysis on different cipher suites."""
        demos = [
            (0x002F, 0x0302, None, "Legacy RSA (TLS 1.1)"),
            (0x1301, 0x0304, [29, 0x6399], "Upgraded ECDHE + PQ Hybrid (TLS 1.3)"),
            (0x1301, 0x0304, [29, 0x6399], "Quantum-safe (TLS 1.3 + PQ)"),
            (0x1301, 0x0304, [29], "Upgraded ECDHE-GCM (TLS 1.3)"),
        ]

        for cs_id, ver, groups, label in demos:
            a = analyze_tls_handshake(cs_id, tls_version=ver, supported_groups=groups)
            grade_icon = {"A+": "A+", "A": "A", "B": "B", "C": "C", "D": "D", "F": "F"}.get(a.overall_grade, "?")
            print(f"\n  [{grade_icon}] {label}")
            print(f"    Suite: {a.cipher_suite_name}")
            print(f"    KEX: {a.kex_type} ({a.kex_description}) | FS: {a.forward_secrecy}")
            print(f"    Cipher: {a.cipher_name} [{a.cipher_strength}] {a.cipher_bits}-bit {a.cipher_mode}")
            print(f"    TLS: {a.tls_version} [{a.tls_version_risk}]")
            print(f"    PQC: [{a.pqc_verdict}] {a.pqc_explanation[:100]}...")
            if a.recommendations:
                print(f"    Recommendations: {'; '.join(a.recommendations[:2])}")


def run_simulation():
    """Entry point for attack simulation."""
    sim = AttackSimulator()
    return sim.run_all()


if __name__ == "__main__":
    run_simulation()
