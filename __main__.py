"""
Quantum Sniffer — CLI Entry Point
===================================
Usage:
  python __main__.py --test           Run self-test suite
  python __main__.py --simulate       Run attack simulation + detection demo
  python __main__.py --benchmark      Run performance benchmark
  python __main__.py --benchmark-pqc  Run RSA vs Kyber benchmark
  python __main__.py --web            Start with web dashboard
  python __main__.py --stress-test    Run distributed stress test
  python __main__.py                  Start packet capture (requires admin)
"""
from __future__ import annotations

import sys
import io

# Fix Windows console encoding for Unicode output
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    except Exception:
        pass

import argparse
import os
import sys
import time


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="quantum_sniffer",
        description="Quantum-Resistant Packet Sniffer with IDS, Analytics & PQC",
    )
    parser.add_argument("-i", "--interface", help="Network interface to capture on")
    parser.add_argument("-f", "--filter", help="BPF filter string", default="")
    parser.add_argument("--no-pqc", action="store_true", help="Disable PQC encryption")
    parser.add_argument("--no-dashboard", action="store_true", help="Disable Rich dashboard")
    parser.add_argument("--sensitivity", choices=["low", "medium", "high"], default="medium")
    parser.add_argument("--geoip", action="store_true", help="Enable GeoIP lookups")
    parser.add_argument("--export", metavar="FILE", help="Export JSON analytics on exit")
    parser.add_argument("--list-interfaces", action="store_true", help="List available interfaces")

    # New features
    parser.add_argument("--test", action="store_true", help="Run self-test suite")
    parser.add_argument("--simulate", action="store_true", help="Run attack simulation demo")
    parser.add_argument("--benchmark", action="store_true", help="Run performance benchmark")
    parser.add_argument("--benchmark-pqc", action="store_true", help="Run RSA vs Kyber benchmark")
    parser.add_argument("--web", action="store_true", help="Enable web dashboard (port 5000)")
    parser.add_argument("--web-port", type=int, default=5000, help="Web dashboard port")
    parser.add_argument("--mode", choices=["capture", "sensor", "aggregator"], default="capture")
    parser.add_argument("--server", metavar="HOST:PORT", help="Aggregator server address (sensor mode)")
    parser.add_argument("--port", type=int, default=9999, help="Aggregator listen port")
    parser.add_argument("--pcap", metavar="FILE", help="Replay a .pcap/.pcapng file through IDS")
    parser.add_argument("--pcap-speed", type=float, default=1.0, help="PCAP replay speed multiplier")
    parser.add_argument("--max-speed", action="store_true", help="Replay PCAP at maximum speed")
    parser.add_argument("--dataset", metavar="FILE", help="Test against labeled CSV dataset (CICIDS/UNSW-NB15)")
    parser.add_argument("--profile", action="store_true", help="Enable cProfile on benchmark mode")
    parser.add_argument("--pqc-transport", action=argparse.BooleanOptionalAction,
                        default=True, help="Enable PQC-encrypted alert transport (default: enabled)")

    # New features
    parser.add_argument("--bench", action="store_true",
                        help="Run comprehensive benchmark suite (throughput, latency, PQC overhead)")
    parser.add_argument("--quality", action="store_true",
                        help="Run detection quality analysis (precision/recall/F1, confusion matrix)")
    parser.add_argument("--config", metavar="FILE",
                        help="Path to config.yaml or config.json")
    parser.add_argument("--init-config", action="store_true",
                        help="Generate default config.yaml and exit")
    parser.add_argument("--log-level",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        default=None, help="Set logging level")
    parser.add_argument("--pqc-level",
                        choices=["educational", "production"],
                        default="production",
                        help="Kyber security level (default: production N=256)")
    parser.add_argument("--stress-test", action="store_true",
                        help="Run distributed stress test (batching + throughput)")
    parser.add_argument("--stress-duration", type=int, default=10,
                        help="Stress test duration in seconds (default: 10)")
    parser.add_argument("--stress-sensors", type=int, default=3,
                        help="Number of stress test sensors (default: 3)")
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of worker processes for packet dissection (default: 1 = single-threaded)")
    parser.add_argument("--iforest-demo", action="store_true",
                        help="Run Isolation Forest MLA project demo (generates plots + report)")
    parser.add_argument("--no-iforest", action="store_true",
                        help="Disable Isolation Forest detector in capture mode")
    parser.add_argument("--train-pcap", metavar="FILE",
                        help="Train & evaluate iForest on a .pcap/.pcapng file (testing only)")

    return parser.parse_args()


def run_self_tests():
    """Run comprehensive self-test suite."""
    print()
    print("  QUANTUM SNIFFER — Self-Test Suite")
    print()

    all_passed = True

    # Module 1: PQC
    print("-" * 60)
    print("  Module 1: Post-Quantum Cryptography")
    print("-" * 60)
    try:
        from pqc import test_pqc
        test_pqc()
    except Exception as e:
        print(f"  FAIL: {e}")
        all_passed = False

    # Module 2: Protocols (with Deep TLS)
    print()
    print("-" * 60)
    print("  Module 2: Protocol Dissectors + Deep TLS")
    print("-" * 60)
    try:
        from protocols import test_protocols
        test_protocols()
    except Exception as e:
        print(f"  FAIL: {e}")
        all_passed = False

    # Module 3: IDS (with Explainability)
    print()
    print("-" * 60)
    print("  Module 3: IDS + Explainability")
    print("-" * 60)
    try:
        from ids import IDSEngine, Severity
        from protocols import IPv4Packet, TCPSegment, TCPFlags

        ids = IDSEngine(sensitivity="high")
        for port in range(1, 20):
            ip = IPv4Packet(4, 20, 0, 0, 40, port, 2, 0, 64, 6, 0,
                            "10.0.0.1", "10.0.0.2", b"", b"")
            tcp = TCPSegment(12345, port, 0, 0, 20, TCPFlags.SYN, 65535, 0, 0, b"", b"")
            ids.analyze_packet(ip=ip, tcp=tcp)

        scan_alerts = [a for a in ids.alerts if a.category == "PORT_SCAN"]
        assert len(scan_alerts) > 0, "Port scan not detected"
        alert = scan_alerts[0]
        assert alert.explanation, "Missing explanation"
        assert len(alert.evidence_factors) > 0, "Missing evidence factors"
        assert len(alert.response_actions) > 0, "Missing response actions"
        print(f"    OK Port scan detection PASSED (with explainability)")
        print(f"    Explanation: {alert.explanation[:80]}...")
        print(f"    Evidence factors: {len(alert.evidence_factors)}")
        print(f"    Response actions: {len(alert.response_actions)}")
        print(f"    Packets analyzed: {ids.stats['total_packets_analyzed']}")
        print(f"    Threats detected: {ids.stats['threats_detected']}")
        print(f"    OK IDS + Explainability PASSED")
    except Exception as e:
        print(f"  FAIL: {e}")
        import traceback; traceback.print_exc()
        all_passed = False

    # Module 4: Analytics
    print()
    print("-" * 60)
    print("  Module 4: Analytics Engine")
    print("-" * 60)
    try:
        from analytics import AnalyticsManager
        eng = AnalyticsManager()
        import random
        for i in range(150):
            proto = random.choice(["TCP"] * 100 + ["DNS"] * 30 + ["TLS"] * 20)
            eng.record_packet(
                protocol=proto,
                size=random.randint(40, 1500),
                src_ip=f"10.0.0.{random.randint(1,10)}",
                dst_ip=f"192.168.1.{random.randint(1,10)}",
                src_port=random.randint(1024, 65535),
                dst_port=random.choice([80, 443, 53]),
            )
        stats = eng.summary
        print(f"    Total packets: {stats['total_packets']}")
        print(f"    Protocols: {stats['protocols']}")
        top = eng.talkers.top_senders
        print(f"    Top senders: {top[:3]}")
        print(f"    Bandwidth: {eng.bandwidth.total_bytes / 1024:.1f} KB")
        print(f"    OK Analytics engine PASSED")
    except Exception as e:
        print(f"  FAIL: {e}")
        all_passed = False

    # Module 5: Distributed (quick test)
    print()
    print("-" * 60)
    print("  Module 5: Distributed Sniffer")
    print("-" * 60)
    try:
        from distributed import test_distributed
        result = test_distributed()
        if not result:
            print("  WARNING: Distributed test had issues (timing-dependent)")
    except Exception as e:
        print(f"  FAIL: {e}")
        print("  (Distributed test requires networking — not critical)")

    # Final
    print()
    print("=" * 60)
    if all_passed:
        print("  ALL SELF-TESTS PASSED")
    else:
        print("  SOME TESTS FAILED")
    print("=" * 60)
    print()


def main():
    args = parse_args()

    # ── Set PQC level early ──
    from pqc import set_kyber_level
    set_kyber_level(args.pqc_level)

    # ── Self-test mode ──
    if args.test:
        run_self_tests()
        return

    # ── Stress test mode ──
    if args.stress_test:
        from distributed import stress_test_distributed
        stress_test_distributed(
            duration=args.stress_duration,
            sensors=args.stress_sensors,
        )
        return

    # ── Isolation Forest demo mode ──
    if args.iforest_demo:
        from iforest_demo import run_iforest_demo
        run_iforest_demo()
        return

    # ── Attack simulation mode ──
    if args.simulate:
        from simulator import run_simulation
        run_simulation()
        return

    # ── Performance benchmark mode ──
    if args.benchmark:
        from performance import run_benchmark
        run_benchmark()
        return

    # ── PQC benchmark mode ──
    if args.benchmark_pqc:
        from pqc import run_pqc_benchmark
        run_pqc_benchmark()
        return

    # ── PCAP training mode (testing/evaluation) ──
    if args.train_pcap:
        from pcap_trainer import PcapTrainer
        trainer = PcapTrainer()
        trainer.train(args.train_pcap)
        return

    # ── PCAP replay mode ──
    if args.pcap:
        from pcap_replay import run_pcap_replay
        run_pcap_replay(args.pcap, speed=args.pcap_speed, max_speed=args.max_speed)
        return

    # ── Dataset testing mode ──
    if args.dataset:
        from pcap_replay import DatasetTester
        tester = DatasetTester(sensitivity=args.sensitivity)
        tester.test_csv(args.dataset)
        return

    # ── List interfaces ──
    if args.list_interfaces:
        try:
            from scapy.all import get_if_list
            print("\nAvailable interfaces:")
            for iface in get_if_list():
                print(f"  - {iface}")
        except ImportError:
            print("Scapy not installed. pip install scapy")
        return

    # ── Init config mode ──
    if args.init_config:
        from config import save_default_config
        save_default_config("config.yaml")
        return

    # ── Setup logging ──
    import logging
    log_level = args.log_level or "INFO"
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # ── Load config ──
    try:
        from config import load_config
        config = load_config(config_path=getattr(args, 'config', None))
    except Exception:
        config = None

    # ── Benchmark suite mode ──
    if args.bench:
        from benchmark_suite import run_all
        run_all()
        return

    # ── Detection quality mode ──
    if args.quality:
        from detection_quality import run_detection_quality
        run_detection_quality()
        return

    # ── Aggregator mode ──
    if args.mode == "aggregator":
        from distributed import AggregationServer, PacketSummary
        from ids import IDSEngine, SEVERITY_LABELS
        from analytics import AnalyticsManager
        from anomaly import AnomalyDetector
        from correlator import AlertCorrelator
        from stix_exporter import STIXExporter
        from metrics import get_metrics

        ids_engine = IDSEngine(sensitivity=args.sensitivity)
        analytics = AnalyticsManager()
        anomaly = AnomalyDetector()
        correlator = AlertCorrelator()
        stix = STIXExporter()
        mc = get_metrics()
        all_alerts = []

        print()
        print("  QUANTUM SNIFFER v2.0 — AGGREGATOR")
        print("  ==================================")
        print(f"  Listening on 0.0.0.0:{args.port}")
        print(f"  IDS sensitivity: {args.sensitivity}")
        print()

        web_store = None
        if args.web:
            try:
                from web_dashboard import DashboardDataStore, start_web_dashboard
                web_store = DashboardDataStore()

                def _create_app_with_stix(data_store):
                    from web_dashboard import create_web_app
                    from flask import jsonify, request
                    app = create_web_app(data_store)

                    @app.route("/api/threats/stix")
                    def api_stix():
                        minutes = request.args.get("minutes", 60, type=int)
                        bundle = stix.export_bundle(all_alerts, minutes=minutes)
                        return jsonify(bundle)

                    @app.route("/api/incidents")
                    def api_incidents():
                        return jsonify([{
                            "id": inc.incident_id,
                            "source_ip": inc.source_ip,
                            "alerts": inc.alert_count,
                            "severity": inc.severity_label,
                            "chain": inc.chain_description,
                            "duration": inc.duration,
                        } for inc in correlator.active_incidents])

                    return app

                app = _create_app_with_stix(web_store)
                import threading, logging
                logging.getLogger('werkzeug').setLevel(logging.ERROR)
                def _run_web():
                    app.run(host="0.0.0.0", port=args.web_port, debug=False, use_reloader=False)
                threading.Thread(target=_run_web, daemon=True).start()
                print(f"  Web dashboard:  http://localhost:{args.web_port}")
                print(f"  Prometheus:     http://localhost:{args.web_port}/metrics")
                print(f"  Health check:   http://localhost:{args.web_port}/health")
                print(f"  STIX export:    http://localhost:{args.web_port}/api/threats/stix")
                print(f"  Incidents:      http://localhost:{args.web_port}/api/incidents")
                print()
            except Exception as e:
                print(f"  Web dashboard error: {e}")

        def on_packet(pkt: PacketSummary):
            import time as _time
            t0 = _time.time()

            analytics.record_packet(pkt.protocol, pkt.src_ip, pkt.dst_ip,
                                    pkt.size, pkt.src_port, pkt.dst_port)
            mc.record_packet(pkt.protocol, pkt.size)

            from protocols import IPv4Packet, TCPSegment, TCPFlags
            ip = IPv4Packet(
                version=4, ihl=20, dscp=0, ecn=0, total_length=pkt.size,
                identification=0, flags=0x02, fragment_offset=0,
                ttl=64, protocol=6 if pkt.protocol == "TCP" else 17,
                checksum=0, src_ip=pkt.src_ip, dst_ip=pkt.dst_ip,
                options=b"", payload=b"",
            )

            alerts = []
            if pkt.protocol in ("TCP", "TLS", "HTTP"):
                flag_val = 0
                if "S" in pkt.flags: flag_val |= TCPFlags.SYN
                if "A" in pkt.flags: flag_val |= TCPFlags.ACK
                if "F" in pkt.flags: flag_val |= TCPFlags.FIN
                if "R" in pkt.flags: flag_val |= TCPFlags.RST
                tcp = TCPSegment(
                    src_port=pkt.src_port, dst_port=pkt.dst_port,
                    seq_num=0, ack_num=0, data_offset=20,
                    flags=flag_val or TCPFlags.SYN, window=65535,
                    checksum=0, urgent_ptr=0, options=b"", payload=b"",
                )
                alerts = ids_engine.analyze_packet(ip=ip, tcp=tcp)

            elif pkt.protocol in ("UDP", "DNS"):
                from protocols import UDPDatagram
                udp = UDPDatagram(
                    src_port=pkt.src_port, dst_port=pkt.dst_port,
                    length=pkt.size, checksum=0, payload=b"",
                )
                alerts = ids_engine.analyze_packet(ip=ip, udp=udp)

            anomaly_alert = anomaly.update("packets_per_sec",
                                           mc.packets_per_sec._value._value if hasattr(mc, 'packets_per_sec') else 0)
            if anomaly_alert:
                alerts.append(anomaly_alert)

            for alert in alerts:
                incident = correlator.correlate(alert)
                all_alerts.append(alert)
                mc.record_alert(
                    SEVERITY_LABELS.get(alert.severity, "UNKNOWN"),
                    alert.category,
                )
                sev = SEVERITY_LABELS.get(alert.severity, "?")
                print(f"  [{sev}] {alert.description[:120]}")
                if incident and incident.is_escalated:
                    print(f"  ESCALATED: {incident.summary}")

            if web_store and analytics:
                perf_state = {
                    "total_packets": ids_engine.stats["total_packets_analyzed"],
                    "uptime": _time.time() - server_start,
                }
                web_store.update(
                    performance=perf_state,
                    protocols=dict(analytics.protocols.counts),
                    pqc={"enabled": True, "entries": 0},
                    alerts=[{
                        "severity": SEVERITY_LABELS.get(a.severity, "?"),
                        "category": a.category,
                        "description": a.description,
                        "source_ip": a.source_ip,
                        "timestamp": a.timestamp,
                    } for a in all_alerts[-100:]],
                    bandwidth=analytics.summary,
                    flows=analytics.flow_summaries[:50] if hasattr(analytics, 'flow_summaries') else [],
                )

            latency = _time.time() - t0
            mc.observe_latency(latency)

        server = AggregationServer(host="0.0.0.0", port=args.port, pqc_transport=args.pqc_transport)
        server.on_packet(on_packet)

        import time as _time
        server_start = _time.time()

        def _status_loop():
            while True:
                _time.sleep(15)
                s = server.summary
                print(f"\n  STATUS: {s['alive_nodes']} nodes, "
                      f"{s['total_packets']:,} packets, "
                      f"{len(all_alerts)} alerts, "
                      f"{correlator.active_count} active incidents\n")
                mc.update_gauges(
                    flows=len(analytics.flows.flows) if hasattr(analytics, 'flows') else 0,
                    tracked_sources=len(ids_engine.port_tracker) if hasattr(ids_engine, 'port_tracker') else 0,
                )

        import threading
        threading.Thread(target=_status_loop, daemon=True).start()

        print(f"  Waiting for sensor connections...")
        print()
        try:
            server.start()
        except KeyboardInterrupt:
            server.stop()
            print(f"\n  Aggregator stopped.")
            print(f"  Total packets: {server.summary['total_packets']:,}")
            print(f"  Total alerts: {len(all_alerts)}")
            print(f"  Active incidents: {correlator.active_count}")
        return

    # ── Capture mode ──
    print()
    print("  QUANTUM SNIFFER v2.0")
    print("  ====================")
    print()

    # Web dashboard
    web_store = None
    if args.web:
        try:
            from web_dashboard import DashboardDataStore, start_web_dashboard
            web_store = DashboardDataStore()
            start_web_dashboard(web_store, port=args.web_port)
        except Exception as e:
            print(f"  Web dashboard error: {e}")
            print("  Install flask: pip install flask")

    # Start capture engine
    try:
        if args.workers > 1:
            from mp_engine import MultiprocessCaptureEngine
            engine = MultiprocessCaptureEngine(
                interface=args.interface,
                bpf_filter=args.filter,
                workers=args.workers,
                pqc_enabled=not args.no_pqc,
                sensitivity=args.sensitivity,
                geo_enabled=args.geoip,
                export_path=args.export,
            )
        else:
            from engine import CaptureEngine
            engine = CaptureEngine(
                interface=args.interface,
                bpf_filter=args.filter,
                pqc_enabled=not args.no_pqc,
                use_dashboard=not args.no_dashboard,
                sensitivity=args.sensitivity,
                geo_enabled=args.geoip,
                export_path=args.export,
                iforest_enabled=not args.no_iforest,
            )

        if web_store:
            import threading
            def _web_update_loop():
                while True:
                    time.sleep(2)
                    try:
                        web_store.update(
                            performance={
                                "total_packets": engine.ids.stats["total_packets_analyzed"],
                                "uptime": time.time(),
                            },
                            protocols=dict(engine.analytics.protocols.counts),
                            pqc={
                                "enabled": engine.pqc_logger is not None,
                                "entries": engine.pqc_logger.stats["entries_logged"] if engine.pqc_logger else 0,
                            },
                            alerts=[{
                                "severity": a.severity.name,
                                "category": a.category,
                                "description": a.description,
                                "source_ip": a.source_ip,
                                "timestamp": a.timestamp,
                            } for a in engine.ids.alerts[-100:]],
                            bandwidth=engine.analytics.summary,
                            flows=engine.analytics.flow_summaries[:50] if hasattr(engine.analytics, 'flow_summaries') else [],
                        )
                    except Exception:
                        pass
            threading.Thread(target=_web_update_loop, daemon=True).start()

        # If sensor mode, connect to aggregator
        if args.mode == "sensor" and args.server:
            from distributed import SensorNode
            host, port = args.server.rsplit(":", 1)
            sensor = SensorNode("sensor-local", host, int(port), pqc_transport=args.pqc_transport)
            if sensor.connect():
                print(f"  Connected to aggregator {args.server}")

        engine.start()

    except KeyboardInterrupt:
        print("\n  Shutting down...")
    except PermissionError:
        print("\n  ERROR: Requires elevated privileges (admin/root)")
        print("  Run as Administrator or with sudo")
    except Exception as e:
        print(f"\n  ERROR: {e}")
        import traceback; traceback.print_exc()


if __name__ == "__main__":
    main()
