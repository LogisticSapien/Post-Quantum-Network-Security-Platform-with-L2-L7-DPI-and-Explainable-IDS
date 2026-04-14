"""
Multiprocessing Packet Pipeline
=================================
Bypasses the GIL by offloading packet dissection to worker processes:
  • Main process: Scapy capture → raw bytes into shared Queue
  • Worker pool: parallel dissection via multiprocessing.Pool
  • Main process: collects results, feeds IDS/analytics/PQC logger
  • Graceful shutdown with queue drain and PQC finalization

Usage:
  python __main__.py --workers 4
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import signal
import sys
import time
from dataclasses import dataclass
from queue import Empty
from typing import Dict, List, Optional, Tuple

# Worker-process dissection function (must be top-level for pickling)

def _dissect_worker(raw: bytes) -> Optional[Dict]:
    """Dissect a single raw packet in a worker process.

    Returns a dict with parsed fields, or None if unparseable.
    This function is intentionally lightweight — it only does
    protocol parsing, NOT IDS/analytics (which need shared state).
    """
    try:
        from protocols import (
            parse_ethernet, parse_ipv4, parse_ipv6, parse_tcp, parse_udp,
            parse_dns, parse_http, parse_tls, parse_icmp, parse_arp,
            parse_quic, parse_ssh_banner, parse_dhcp,
            EtherType, TLSClientHello, TLSServerHello,
        )

        eth = parse_ethernet(raw)
        if eth is None:
            return None

        result = {
            "raw_size": len(raw),
            "src_mac": eth.src_mac,
            "dst_mac": eth.dst_mac,
            "ether_type": eth.ether_type,
            "protocol": None,
            "src_ip": None,
            "dst_ip": None,
            "src_port": None,
            "dst_port": None,
            "summary": "",
            "extra": "",
            "ip_proto": None,
            "tcp_flags": 0,
            "tcp_flag_str": "",
            "payload_size": 0,
            "tls_info": None,
            "dns_info": None,
        }

        payload = eth.payload

        # ARP
        if eth.ether_type == EtherType.ARP:
            arp = parse_arp(payload)
            if arp:
                result["protocol"] = "ARP"
                result["src_ip"] = arp.sender_ip
                result["dst_ip"] = arp.target_ip
                result["summary"] = f"{arp.sender_ip} ({arp.sender_mac}) → {arp.target_ip} [{arp.opcode_name}]"
            return result

        # IPv4
        if eth.ether_type == EtherType.IPv4:
            ip = parse_ipv4(payload)
            if ip is None:
                return None
            result["src_ip"] = ip.src_ip
            result["dst_ip"] = ip.dst_ip
            result["ip_proto"] = ip.protocol

            transport = ip.payload

            # ICMP
            if ip.protocol == 1:
                icmp = parse_icmp(transport)
                if icmp:
                    result["protocol"] = "ICMP"
                    result["summary"] = f"{ip.src_ip} → {ip.dst_ip} {icmp.type_name} id={icmp.identifier} seq={icmp.sequence}"
                return result

            # TCP
            if ip.protocol == 6:
                tcp = parse_tcp(transport)
                if tcp is None:
                    return None
                result["protocol"] = "TCP"
                result["src_port"] = tcp.src_port
                result["dst_port"] = tcp.dst_port
                result["tcp_flags"] = tcp.flags
                result["tcp_flag_str"] = tcp.flag_str
                result["payload_size"] = len(tcp.payload) if tcp.payload else 0
                conn = f"{ip.src_ip}:{tcp.src_port} → {ip.dst_ip}:{tcp.dst_port}"
                result["summary"] = f"{conn} [{tcp.flag_str}] win={tcp.window}"
                result["extra"] = f"seq={tcp.seq_num}"

                # TLS detection
                if tcp.payload and (tcp.dst_port in (443, 8443) or tcp.src_port in (443, 8443)):
                    tls = parse_tls(tcp.payload)
                    if isinstance(tls, TLSClientHello):
                        result["tls_info"] = {
                            "type": "client_hello",
                            "sni": tls.sni,
                            "ja3": tls.ja3_hash,
                            "version": tls.tls_version_name,
                            "has_pq": tls.has_post_quantum,
                            "cipher_suites": tls.cipher_suites,
                        }
                    elif isinstance(tls, TLSServerHello):
                        result["tls_info"] = {
                            "type": "server_hello",
                            "cipher": tls.cipher_suite,
                            "cipher_name": tls.cipher_suite_name,
                        }

                # HTTP detection
                if tcp.payload and (tcp.src_port in (80, 8080) or tcp.dst_port in (80, 8080)):
                    http = parse_http(tcp.payload)
                    if http and http.is_request:
                        result["protocol"] = "HTTP"
                        result["summary"] = f"{http.method} {http.uri}"
                        result["extra"] = f"Host: {http.host}"

                return result

            # UDP
            if ip.protocol == 17:
                udp = parse_udp(transport)
                if udp is None:
                    return None
                result["protocol"] = "UDP"
                result["src_port"] = udp.src_port
                result["dst_port"] = udp.dst_port
                conn = f"{ip.src_ip}:{udp.src_port} → {ip.dst_ip}:{udp.dst_port}"
                result["summary"] = conn

                # DNS
                if udp.src_port == 53 or udp.dst_port == 53:
                    dns = parse_dns(udp.payload)
                    if dns:
                        result["protocol"] = "DNS"
                        result["dns_info"] = {
                            "is_response": dns.is_response,
                            "query_names": dns.query_names,
                            "rcode": dns.rcode_name,
                        }
                        if dns.is_response:
                            answers = ', '.join(f"{a.rdata}({a.type_name})" for a in dns.answers[:3])
                            result["summary"] = f"Response: {' '.join(dns.query_names)}"
                            result["extra"] = answers or dns.rcode_name
                        else:
                            result["summary"] = f"Query: {' '.join(dns.query_names)}"

                # QUIC
                elif udp.dst_port == 443 or udp.src_port == 443:
                    quic = parse_quic(udp.payload)
                    if quic:
                        result["protocol"] = "QUIC"
                        result["summary"] = f"{conn} {quic.version_name}"
                        result["extra"] = f"DCID={quic.dcid[:8].hex()}"

                return result

        # IPv6
        if eth.ether_type == EtherType.IPv6:
            ipv6 = parse_ipv6(payload)
            if ipv6:
                result["protocol"] = "IPv6"
                result["src_ip"] = ipv6.src_ip
                result["dst_ip"] = ipv6.dst_ip
                result["summary"] = f"{ipv6.src_ip} → {ipv6.dst_ip} nh={ipv6.next_header_name}"
            return result

        return result

    except Exception:
        return None


class MultiprocessCaptureEngine:
    """
    Multiprocessing capture engine — bypasses the GIL.

    Architecture:
      Main process:  Scapy AsyncSniffer → raw bytes → mp.Queue
      Worker pool:   N processes running _dissect_worker()
      Main process:  Collects parsed results → IDS/Analytics/PQC Logger

    IDS, Analytics, and PQC Logger are NOT parallelised (they hold shared
    mutable state). Only the CPU-heavy packet dissection is distributed.
    """

    def __init__(
        self,
        interface: Optional[str] = None,
        bpf_filter: Optional[str] = None,
        workers: int = 0,
        pqc_enabled: bool = True,
        log_dir: str = "./pqc_logs",
        sensitivity: str = "medium",
        geo_enabled: bool = False,
        export_path: Optional[str] = None,
    ):
        self.interface = interface
        self.bpf_filter = bpf_filter
        self.workers = workers if workers > 0 else max(2, (os.cpu_count() or 4) - 1)
        self.export_path = export_path

        # Components (main process only)
        from ids import IDSEngine
        from analytics import AnalyticsManager
        from pqc import PQCSecureLogger, QuantumThreatAnalyzer

        self.ids = IDSEngine(sensitivity=sensitivity)
        self.analytics = AnalyticsManager(geo_enabled=geo_enabled)
        self.qt_analyzer = QuantumThreatAnalyzer()

        if pqc_enabled:
            self.pqc_logger = PQCSecureLogger(log_dir=log_dir)
        else:
            self.pqc_logger = None

        # Internal state
        self._running = False
        self._pool: Optional[mp.Pool] = None
        self._packets_processed = 0
        self._packets_dropped = 0
        self._start_time = 0.0

    def start(self):
        """Start multiprocessing capture engine."""
        from scapy.all import AsyncSniffer

        self._running = True
        self._start_time = time.time()

        # Signal handlers
        signal.signal(signal.SIGINT, self._shutdown_handler)
        try:
            signal.signal(signal.SIGTERM, self._shutdown_handler)
        except (OSError, ValueError):
            pass

        # Create worker pool
        self._pool = mp.Pool(processes=self.workers)

        print("\n" + "=" * 60)
        print("  ⚛️  QUANTUM SNIFFER — Multiprocess Engine")
        print("=" * 60)
        pqc_status = "ENABLED 🔐 (CCA2)" if self.pqc_logger else "DISABLED"
        print(f"  PQC Encryption: {pqc_status}")
        print(f"  Interface: {self.interface or 'default'}")
        print(f"  Workers: {self.workers} processes")
        if self.bpf_filter:
            print(f"  BPF Filter: {self.bpf_filter}")
        print(f"  Press Ctrl+C to stop.\n")

        # Collect pending async results
        pending_results = []
        last_flush = time.time()
        last_status = time.time()

        try:
            sniffer_kwargs = {
                "prn": lambda pkt: self._on_packet(pkt, pending_results),
                "store": False,
            }
            if self.interface:
                sniffer_kwargs["iface"] = self.interface
            if self.bpf_filter:
                sniffer_kwargs["filter"] = self.bpf_filter

            sniffer = AsyncSniffer(**sniffer_kwargs)
            sniffer.start()

            while self._running:
                time.sleep(0.1)

                # Harvest completed results
                self._harvest_results(pending_results)

                now = time.time()

                # Periodic flush
                if now - last_flush >= 60:
                    if self.pqc_logger:
                        self.pqc_logger.flush_to_disk()
                    if hasattr(self.analytics, 'flows'):
                        self.analytics.flows.cleanup_stale()
                    last_flush = now

                # Periodic status
                if now - last_status >= 15:
                    elapsed = now - self._start_time
                    pps = self._packets_processed / max(elapsed, 0.001)
                    print(f"  [{elapsed:.0f}s] {self._packets_processed:,} packets "
                          f"({pps:,.0f} pkt/s) | {self._packets_dropped} dropped "
                          f"| {self.ids.stats['threats_detected']} threats")
                    last_status = now

        except PermissionError:
            print("\n❌ Permission denied! Run as Administrator (Windows) or root (Linux).")
            sys.exit(1)
        except Exception as e:
            print(f"\n❌ Capture error: {e}")
            sys.exit(1)
        finally:
            # Stop sniffer
            try:
                sniffer.stop()
            except Exception:
                pass

            # Final harvest
            self._harvest_results(pending_results)

            # Shutdown pool
            if self._pool:
                self._pool.terminate()
                self._pool.join()

            # Finalize PQC
            if self.pqc_logger:
                filename = self.pqc_logger.finalize()
                if filename:
                    print(f"\n📝 PQC logs finalized to: {filename}")

            # Export analytics
            if self.export_path:
                try:
                    data = self.analytics.export_json()
                    with open(self.export_path, 'w') as f:
                        json.dump(data, f, indent=2, default=str)
                    print(f"📊 Analytics exported to: {self.export_path}")
                except Exception as e:
                    print(f"❌ Export failed: {e}")

            # Print summary
            elapsed = time.time() - self._start_time
            print(f"\n{'='*60}")
            print(f"  ⚛️  QUANTUM SNIFFER — Session Summary (Multiprocess)")
            print(f"{'='*60}")
            print(f"  Workers used:   {self.workers}")
            print(f"  Total packets:  {self._packets_processed:,}")
            print(f"  Dropped:        {self._packets_dropped:,}")
            print(f"  Duration:       {elapsed:.1f}s")
            print(f"  Throughput:     {self._packets_processed / max(elapsed, 0.001):,.0f} pkt/s")
            print(f"  Threats found:  {self.ids.stats['threats_detected']}")
            qt = self.qt_analyzer.vulnerability_summary
            print(f"  Cipher suites:  {qt['total_analyzed']} analyzed, "
                  f"{qt['quantum_vulnerable']} quantum-vulnerable")
            print(f"{'='*60}\n")

    def _on_packet(self, packet, pending: list):
        """Scapy callback — submit raw bytes to worker pool."""
        try:
            raw = bytes(packet)
            if self._pool:
                result = self._pool.apply_async(_dissect_worker, (raw,))
                pending.append(result)
        except Exception:
            self._packets_dropped += 1

    def _harvest_results(self, pending: list):
        """Collect completed dissection results and feed to IDS/analytics."""
        completed = []
        still_pending = []

        for r in pending:
            if r.ready():
                completed.append(r)
            else:
                still_pending.append(r)

        pending.clear()
        pending.extend(still_pending)

        for r in completed:
            try:
                result = r.get(timeout=0.01)
                if result is not None:
                    self._process_result(result)
                    self._packets_processed += 1
            except Exception:
                self._packets_dropped += 1

    def _process_result(self, result: dict):
        """Process a dissected packet result in the main process."""
        proto = result.get("protocol")
        src_ip = result.get("src_ip", "")
        dst_ip = result.get("dst_ip", "")
        size = result.get("raw_size", 0)
        src_port = result.get("src_port")
        dst_port = result.get("dst_port")

        if proto:
            self.analytics.record_packet(
                proto, src_ip, dst_ip, size,
                src_port, dst_port,
            )

            # TLS vulnerability analysis
            tls_info = result.get("tls_info")
            if tls_info and tls_info.get("type") == "client_hello":
                suites = tls_info.get("cipher_suites", [])
                self.qt_analyzer.analyze_cipher_list(suites)

            # IDS analysis (simplified — uses IP/TCP when available)
            if proto in ("TCP", "HTTP", "TLS") and src_port and dst_port:
                self._ids_tcp(src_ip, dst_ip, src_port, dst_port,
                              result.get("tcp_flags", 0), size)
            elif proto in ("UDP", "DNS") and src_port and dst_port:
                self._ids_udp(src_ip, dst_ip, src_port, dst_port, size)

            # PQC log
            if self.pqc_logger:
                summary = result.get("summary", "")
                self.pqc_logger.log(f"{proto} {summary}")

    def _ids_tcp(self, src, dst, sport, dport, flags, size):
        """Feed TCP packet info to IDS."""
        from protocols import IPv4Packet, TCPSegment
        ip = IPv4Packet(
            version=4, ihl=20, dscp=0, ecn=0, total_length=size,
            identification=0, flags=0x02, fragment_offset=0,
            ttl=64, protocol=6, checksum=0,
            src_ip=src, dst_ip=dst, options=b"", payload=b"",
        )
        tcp = TCPSegment(
            src_port=sport, dst_port=dport,
            seq_num=0, ack_num=0, data_offset=20,
            flags=flags, window=65535,
            checksum=0, urgent_ptr=0, options=b"", payload=b"",
        )
        self.ids.analyze_packet(ip=ip, tcp=tcp)

    def _ids_udp(self, src, dst, sport, dport, size):
        """Feed UDP packet info to IDS."""
        from protocols import IPv4Packet, UDPDatagram
        ip = IPv4Packet(
            version=4, ihl=20, dscp=0, ecn=0, total_length=size,
            identification=0, flags=0x02, fragment_offset=0,
            ttl=64, protocol=17, checksum=0,
            src_ip=src, dst_ip=dst, options=b"", payload=b"",
        )
        udp = UDPDatagram(
            src_port=sport, dst_port=dport,
            length=size, checksum=0, payload=b"",
        )
        self.ids.analyze_packet(ip=ip, udp=udp)

    def _shutdown_handler(self, signum, frame):
        """Graceful shutdown on SIGINT/SIGTERM."""
        self._running = False
