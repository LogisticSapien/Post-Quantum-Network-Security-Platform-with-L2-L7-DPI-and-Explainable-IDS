"""
Packet Capture Engine
======================
Cross-platform packet capture with Scapy:
  • AsyncSniffer for non-blocking capture
  • BPF filter support
  • Threaded processing pipeline
  • Automatic protocol dissection & routing
  • Integration with IDS, Analytics, Dashboard, and PQC logger
"""

from __future__ import annotations

import json
import signal
import sys
import threading
import time
from pathlib import Path
from queue import Queue, Empty
from typing import List, Optional

from scapy.all import AsyncSniffer, conf, get_if_list, Raw, Ether

from protocols import (
    parse_ethernet, parse_ipv4, parse_ipv6, parse_tcp, parse_udp,
    parse_dns, parse_http, parse_tls, parse_icmp, parse_arp,
    parse_quic, parse_ssh_banner, parse_dhcp,
    EtherType, TLSClientHello, TLSServerHello,
)
from ids import IDSEngine, Severity
from analytics import AnalyticsManager
from pqc import PQCSecureLogger, QuantumThreatAnalyzer
from dashboard import Dashboard, SimplePrinter, PacketFeed
from iforest_detector import IForestNetworkDetector
from flow_tracker import FlowFeatureTracker


class CaptureEngine:
    """
    Core capture engine.

    Captures packets via Scapy, dissects them through the protocol engine,
    feeds results to the IDS, analytics, PQC logger, and dashboard.
    """

    def __init__(
        self,
        interface: Optional[str] = None,
        bpf_filter: Optional[str] = None,
        use_dashboard: bool = True,
        pqc_enabled: bool = True,
        log_dir: str = "./pqc_logs",
        sensitivity: str = "medium",
        geo_enabled: bool = False,
        export_path: Optional[str] = None,
        iforest_enabled: bool = True,
    ):
        self.interface = interface
        self.bpf_filter = bpf_filter
        self.use_dashboard = use_dashboard
        self.export_path = export_path

        # ── Components ──
        self.ids = IDSEngine(sensitivity=sensitivity)
        self.analytics = AnalyticsManager(geo_enabled=geo_enabled)
        self.qt_analyzer = QuantumThreatAnalyzer()

        # Isolation Forest zero-day detector
        if iforest_enabled:
            self.iforest = IForestNetworkDetector()
            self.flow_tracker = FlowFeatureTracker(max_flows=10000)
        else:
            self.iforest = None
            self.flow_tracker = None

        if pqc_enabled:
            self.pqc_logger = PQCSecureLogger(log_dir=log_dir)
        else:
            self.pqc_logger = None

        if use_dashboard:
            self.dashboard = Dashboard(
                self.analytics, self.ids,
                self.pqc_logger, self.qt_analyzer,
            )
            self.printer = None
        else:
            self.dashboard = None
            self.printer = SimplePrinter()

        # ── Internal state ──
        self._sniffer: Optional[AsyncSniffer] = None
        self._running = False
        self._packet_queue: Queue = Queue(maxsize=10000)
        self._worker_thread: Optional[threading.Thread] = None
        self._stats_thread: Optional[threading.Thread] = None
        self._flush_interval = 60  # seconds
        self._packets_dropped = 0

    def start(self):
        """Start packet capture and processing."""
        self._running = True

        # Signal handlers
        signal.signal(signal.SIGINT, self._shutdown_handler)
        try:
            signal.signal(signal.SIGTERM, self._shutdown_handler)
        except (OSError, ValueError):
            pass  # SIGTERM not available on Windows in some contexts

        # Worker thread
        self._worker_thread = threading.Thread(
            target=self._processing_loop, daemon=True
        )
        self._worker_thread.start()

        # Periodic tasks thread
        self._stats_thread = threading.Thread(
            target=self._periodic_loop, daemon=True
        )
        self._stats_thread.start()

        # Dashboard
        if self.dashboard:
            self.dashboard.start()
        else:
            print("\n" + "=" * 60)
            print("  ⚛️  QUANTUM SNIFFER — Post-Quantum Protected Analyzer")
            print("=" * 60)
            pqc_status = "ENABLED 🔐" if self.pqc_logger else "DISABLED"
            print(f"  PQC Encryption: {pqc_status}")
            print(f"  Interface: {self.interface or 'default'}")
            if self.bpf_filter:
                print(f"  BPF Filter: {self.bpf_filter}")
            print(f"  Press Ctrl+C to stop.\n")

        # Start Scapy sniffer
        try:
            sniffer_kwargs = {
                "prn": self._packet_callback,
                "store": False,
            }
            if self.interface:
                sniffer_kwargs["iface"] = self.interface
            if self.bpf_filter:
                sniffer_kwargs["filter"] = self.bpf_filter

            self._sniffer = AsyncSniffer(**sniffer_kwargs)
            self._sniffer.start()

            # Block until stopped
            while self._running:
                time.sleep(0.5)
                if self.dashboard:
                    self.dashboard.update()

        except PermissionError:
            self._stop_display()
            print("\n❌ Permission denied! Run as Administrator (Windows) or root (Linux).")
            print("   Windows: Install Npcap from https://npcap.com/")
            sys.exit(1)
        except Exception as e:
            self._stop_display()
            print(f"\n❌ Capture error: {e}")
            print("   Ensure Npcap (Windows) or libpcap (Linux) is installed.")
            sys.exit(1)
        finally:
            self._cleanup()

    def _packet_callback(self, packet):
        """Scapy callback — enqueue raw bytes for processing.

        Uses drop-based backpressure: if queue is full, packet is dropped
        and counted. This prevents memory exhaustion under burst traffic.
        """
        try:
            raw = bytes(packet)
            self._packet_queue.put_nowait(raw)
        except Exception:
            self._packets_dropped += 1
            if self._packets_dropped % 1000 == 1:
                import logging
                logging.getLogger(__name__).warning(
                    f"Queue full — {self._packets_dropped} packets dropped "
                    f"(queue_size={self._packet_queue.maxsize})"
                )

    def _processing_loop(self):
        """Worker thread — dequeue and process packets."""
        while self._running:
            try:
                raw = self._packet_queue.get(timeout=0.5)
                self._process_packet(raw)
            except Empty:
                continue
            except Exception:
                continue

    def _process_packet(self, raw: bytes):
        """Dissect one packet through all layers."""
        # ── Layer 2 ──
        eth = parse_ethernet(raw)
        if eth is None:
            return

        payload = eth.payload

        # ── ARP ──
        if eth.ether_type == EtherType.ARP:
            arp = parse_arp(payload)
            if arp:
                self._output("ARP", f"{arp.sender_ip} ({arp.sender_mac}) → {arp.target_ip} [{arp.opcode_name}]")
                self.analytics.record_packet("ARP", arp.sender_ip, arp.target_ip, len(raw))
                alerts = self.ids.analyze_packet(arp=arp)
                self._handle_alerts(alerts)
                self._pqc_log(f"ARP {arp.opcode_name} {arp.sender_ip}->{arp.target_ip}")
            return

        # ── IPv4 ──
        if eth.ether_type == EtherType.IPv4:
            ip = parse_ipv4(payload)
            if ip is None:
                return
            self._process_ip_payload(ip, len(raw))
            return

        # ── IPv6 ──
        if eth.ether_type == EtherType.IPv6:
            ipv6 = parse_ipv6(payload)
            if ipv6 is None:
                return
            self._output("IPv6", f"{ipv6.src_ip} → {ipv6.dst_ip} nh={ipv6.next_header_name}")
            self.analytics.record_packet("IPv6", ipv6.src_ip, ipv6.dst_ip, len(raw))
            self._pqc_log(f"IPv6 {ipv6.src_ip}->{ipv6.dst_ip}")
            return

    def _process_ip_payload(self, ip, pkt_size: int):
        """Process IPv4 transport layer."""
        src, dst = ip.src_ip, ip.dst_ip
        transport = ip.payload

        # ── ICMP ──
        if ip.protocol == 1:
            icmp = parse_icmp(transport)
            if icmp:
                self._output("ICMP", f"{src} → {dst} {icmp.type_name} id={icmp.identifier} seq={icmp.sequence}")
                self.analytics.record_packet("ICMP", src, dst, pkt_size)
                alerts = self.ids.analyze_packet(ip=ip, icmp=icmp)
                self._handle_alerts(alerts)
                self._pqc_log(f"ICMP {icmp.type_name} {src}->{dst}")
            return

        # ── TCP ──
        if ip.protocol == 6:
            tcp = parse_tcp(transport)
            if tcp is None:
                return

            conn = f"{src}:{tcp.src_port} → {dst}:{tcp.dst_port}"
            self._output("TCP", f"{conn} [{tcp.flag_str}] win={tcp.window}", f"seq={tcp.seq_num}")
            self.analytics.record_packet("TCP", src, dst, pkt_size, tcp.src_port, tcp.dst_port)

            sni_for_flow = None
            ja3_for_flow = None

            # ── HTTP detection ──
            if tcp.payload and (tcp.src_port in (80, 8080) or tcp.dst_port in (80, 8080)):
                http = parse_http(tcp.payload)
                if http:
                    if http.is_request:
                        self._output("HTTP", f"{http.method} {http.uri}", f"Host: {http.host}")
                        self.analytics.record_packet("HTTP", src, dst, len(tcp.payload))
                        if http.host and http.uri:
                            self.analytics.record_http(http.method, http.uri, http.host or "", src)
                        self._pqc_log(f"HTTP {http.method} {http.host}{http.uri}")
                    else:
                        self._output("HTTP", f"Response {http.status_code} {http.status_text}",
                                     f"Content-Type: {http.content_type}")
                        self.analytics.record_packet("HTTP", src, dst, len(tcp.payload))

            # ── TLS detection ──
            if tcp.payload and (tcp.src_port == 443 or tcp.dst_port == 443
                                or tcp.src_port == 8443 or tcp.dst_port == 8443):
                tls = parse_tls(tcp.payload)
                if isinstance(tls, TLSClientHello):
                    extra_parts = []
                    if tls.sni:
                        extra_parts.append(f"SNI={tls.sni}")
                        sni_for_flow = tls.sni
                    extra_parts.append(f"JA3={tls.ja3_hash}")
                    ja3_for_flow = tls.ja3_hash
                    extra_parts.append(tls.tls_version_name)
                    if tls.has_post_quantum:
                        extra_parts.append("🛡️PQ-KEM")
                    self._output("TLS", f"ClientHello {conn}", " | ".join(extra_parts))
                    self.analytics.record_packet("TLS", src, dst, len(tcp.payload))
                    if tls.sni:
                        self.analytics.record_tls_sni(tls.sni, tls.ja3_hash, src)

                    # Quantum vulnerability analysis
                    vuln_reports = self.qt_analyzer.analyze_cipher_list(tls.cipher_suites)
                    for r in vuln_reports:
                        if r.quantum_vulnerable:
                            self._output("TLS", f"⚠️ Quantum-vulnerable: {r.cipher_name}", r.risk_level)

                    self._pqc_log(f"TLS ClientHello SNI={tls.sni} JA3={tls.ja3_hash}")

                elif isinstance(tls, TLSServerHello):
                    self._output("TLS", f"ServerHello {conn}", f"cipher={tls.cipher_suite_name}")
                    self.analytics.record_packet("TLS", src, dst, len(tcp.payload))
                    vuln = self.qt_analyzer.analyze_cipher_suite(tls.cipher_suite)
                    if vuln and vuln.quantum_vulnerable:
                        self._output("TLS", f"⚠️ Server selected quantum-vulnerable cipher: {vuln.cipher_name}",
                                     vuln.risk_level)

            # ── SSH detection ──
            if tcp.payload and (tcp.src_port == 22 or tcp.dst_port == 22):
                ssh = parse_ssh_banner(tcp.payload)
                if ssh:
                    self._output("SSH", f"{conn} {ssh.raw}")
                    self.analytics.record_packet("SSH", src, dst, len(tcp.payload))
                    self._pqc_log(f"SSH {ssh.raw} {src}->{dst}")

            # Flow tracking
            self.analytics.record_tcp_flow(
                src, tcp.src_port, dst, tcp.dst_port,
                tcp.flags, len(tcp.payload),
                sni_for_flow, ja3_for_flow,
            )

            # IDS analysis
            alerts = self.ids.analyze_packet(ip=ip, tcp=tcp)
            self._handle_alerts(alerts)

            # Isolation Forest detection + flow tracking
            if self.iforest:
                is_syn = bool(getattr(tcp, 'is_syn', False))
                iforest_alert = self.iforest.record_packet(
                    "TCP", src, dst, pkt_size, tcp.src_port, tcp.dst_port, is_syn
                )
                if iforest_alert:
                    self._handle_alerts([iforest_alert])

                # Per-flow tracking for slow exfil detection
                if self.flow_tracker:
                    self.flow_tracker.record_packet(
                        src, dst, tcp.dst_port, pkt_size,
                        protocol="TCP", is_syn=is_syn,
                    )

            self._pqc_log(f"TCP {conn} [{tcp.flag_str}]")
            return

        # ── UDP ──
        if ip.protocol == 17:
            udp = parse_udp(transport)
            if udp is None:
                return

            conn = f"{src}:{udp.src_port} → {dst}:{udp.dst_port}"
            self.analytics.record_packet("UDP", src, dst, pkt_size, udp.src_port, udp.dst_port)

            # ── DNS ──
            if udp.src_port == 53 or udp.dst_port == 53:
                dns = parse_dns(udp.payload)
                if dns:
                    if dns.is_response:
                        answers = ', '.join(f"{a.rdata}({a.type_name})" for a in dns.answers[:3])
                        self._output("DNS", f"Response: {' '.join(dns.query_names)}", answers or dns.rcode_name)
                    else:
                        for q in dns.questions:
                            self._output("DNS", f"Query: {q.name}", q.type_name)
                            self.analytics.record_dns(q.name, q.type_name, src)
                    self.analytics.record_packet("DNS", src, dst, len(udp.payload))

                    # IDS analysis for DNS
                    alerts = self.ids.analyze_packet(ip=ip, udp=udp, dns=dns)
                    self._handle_alerts(alerts)
                    self._pqc_log(f"DNS {' '.join(dns.query_names)}")
                    return

            # ── DHCP ──
            if udp.src_port in (67, 68) or udp.dst_port in (67, 68):
                dhcp = parse_dhcp(udp.payload)
                if dhcp:
                    extra = f"hostname={dhcp.hostname}" if dhcp.hostname else ""
                    if dhcp.requested_ip:
                        extra += f" req={dhcp.requested_ip}"
                    self._output("DHCP", f"{dhcp.msg_type_name} xid=0x{dhcp.xid:08x} {dhcp.client_mac}", extra)
                    self.analytics.record_packet("DHCP", src, dst, len(udp.payload))
                    self._pqc_log(f"DHCP {dhcp.msg_type_name}")
                    return

            # ── QUIC ──
            if udp.dst_port == 443 or udp.src_port == 443:
                quic = parse_quic(udp.payload)
                if quic:
                    self._output("QUIC", f"{conn} {quic.version_name}",
                                 f"DCID={quic.dcid[:8].hex()}")
                    self.analytics.record_packet("QUIC", src, dst, len(udp.payload))
                    self._pqc_log(f"QUIC {quic.version_name} {src}->{dst}")
                    return

            self._output("UDP", conn)
            alerts = self.ids.analyze_packet(ip=ip, udp=udp)
            self._handle_alerts(alerts)
            self._pqc_log(f"UDP {conn}")

    # ── Output helpers ──

    def _output(self, protocol: str, summary: str, extra: Optional[str] = None):
        """Send packet info to dashboard or printer."""
        if self.dashboard:
            self.dashboard.feed.add(protocol, summary, extra)
        elif self.printer:
            self.printer.print_packet(protocol, summary, extra)

    def _handle_alerts(self, alerts: List):
        """Handle IDS alerts."""
        for alert in alerts:
            if self.dashboard:
                pass  # Dashboard renders alerts from IDS directly
            elif self.printer:
                self.printer.print_alert(alert)
            self._pqc_log(
                f"ALERT [{alert.severity.name}] {alert.category}: {alert.description}",
                level="ALERT"
            )

    def _pqc_log(self, data: str, level: str = "INFO"):
        """Write to PQC-encrypted log."""
        if self.pqc_logger:
            self.pqc_logger.log(data, level)

    # ── Periodic tasks ──

    def _periodic_loop(self):
        """Periodic maintenance tasks."""
        last_flush = time.time()
        last_cleanup = time.time()

        while self._running:
            time.sleep(1)
            now = time.time()

            # Flush PQC logs
            if now - last_flush >= self._flush_interval:
                if self.pqc_logger:
                    self.pqc_logger.flush_to_disk()
                last_flush = now

            # Cleanup stale flows
            if now - last_cleanup >= 30:
                self.analytics.flows.cleanup_stale()
                last_cleanup = now

    # ── Shutdown ──

    def _shutdown_handler(self, signum, frame):
        """Graceful shutdown on SIGINT/SIGTERM."""
        self._running = False

    def _stop_display(self):
        if self.dashboard:
            self.dashboard.stop()

    def _drain_queue(self):
        """Process any remaining packets in the queue."""
        drained = 0
        while not self._packet_queue.empty():
            try:
                raw = self._packet_queue.get_nowait()
                self._process_packet(raw)
                drained += 1
            except Empty:
                break
            except Exception:
                continue
        if drained > 0:
            print(f"   Drained {drained} buffered packets")

    def _cleanup(self):
        """Clean up resources with proper queue drain and PQC finalization."""
        self._running = False

        # 1. Stop the sniffer
        if self._sniffer:
            try:
                self._sniffer.stop()
            except Exception:
                pass

        self._stop_display()

        # 2. Drain remaining packet queue
        self._drain_queue()

        # 3. Stop flow evictor thread
        try:
            self.analytics.flows.stop()
        except Exception:
            pass

        # 4. Finalize PQC logger (sentinel entry + flush)
        if self.pqc_logger:
            filename = self.pqc_logger.finalize()
            if filename:
                print(f"\n📝 PQC logs finalized to: {filename}")
            stats = self.pqc_logger.stats
            print(f"   Entries: {stats['entries_logged']}, "
                  f"Chain intact: {stats['chain_intact']}, "
                  f"Key rotations: {stats['key_rotations']}, "
                  f"Clean close: {stats['finalized']}")

        # 5. Export analytics
        if self.export_path:
            try:
                data = self.analytics.export_json()
                with open(self.export_path, 'w') as f:
                    json.dump(data, f, indent=2, default=str)
                print(f"📊 Analytics exported to: {self.export_path}")
            except Exception as e:
                print(f"❌ Export failed: {e}")

        # 6. Print summary
        summary = self.analytics.summary
        print(f"\n{'='*60}")
        print(f"  ⚛️  QUANTUM SNIFFER — Session Summary")
        print(f"{'='*60}")
        bw = self.analytics.bandwidth
        print(f"  Total packets:  {summary['total_packets']:,}")
        print(f"  Total bytes:    {bw.format_bytes(summary['total_bytes'])}")
        print(f"  Duration:       {summary['uptime_seconds']:.1f}s")
        print(f"  Threats found:  {self.ids.stats['threats_detected']}")
        print(f"  Flows evicted:  {self.analytics.flows.flows_evicted_total}")

        qt = self.qt_analyzer.vulnerability_summary
        print(f"  Cipher suites:  {qt['total_analyzed']} analyzed, "
              f"{qt['quantum_vulnerable']} quantum-vulnerable")

        protos = self.analytics.protocols.top_protocols
        if protos:
            print(f"  Top protocols:  {', '.join(f'{p}({c})' for p, c in protos[:5])}")
        print(f"{'='*60}\n")

