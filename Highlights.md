## Highlights

- **6/6 attack scenarios detected** - 267 alerts from 292 packets across SYN flood, DNS tunneling, ARP spoofing, ICMP tunneling, brute force, and port scan
- **50K-packet benchmark** - full pipeline processing with p50/p95/p99 latency percentiles and pipeline overhead reporting
- **10 MITRE ATT&CK-mapped detectors** - with per-alert confidence scoring, structured evidence factors, and natural-language explainability
- **Kyber-512 KEM implemented from scratch** - NTT-optimized O(n log n) polynomial multiplication, no PQC library used
- **AES-256-GCM encrypted logging** - key derived via Kyber KEM with automatic rotation every 10,000 entries
- **SHA3-256 hash-chain integrity** - blockchain-style tamper-evident log chain, every entry verifiable
- **Deep TLS handshake analysis** - 35+ cipher suites, SNI extraction, JA3 fingerprinting, grades A+ through F
- **Distributed sensor-aggregator architecture** - multi-node TCP capture with heartbeat health monitoring
- **Dual dashboards** - Rich terminal UI and Flask + Chart.js web dashboard with REST API
- **Self-tests all passed** - PQC, Protocols, IDS, Analytics, Distributed
