#  Quantum Sniffer

> **A post-quantum-protected network security platform combining real-time deep packet inspection, explainable intrusion detection, and quantum-resilient cryptographic logging.**

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![License](https://img.shields.io/badge/License-MIT-green)
![Version](https://img.shields.io/badge/Version-2.0.0-orange)
![Lines](https://img.shields.io/badge/LOC-~6100-lightgrey)
![Modules](https://img.shields.io/badge/Modules-13-blueviolet)

---

## Overview

Quantum Sniffer is a solo-built, production-grade network packet analyzer that goes far beyond traditional sniffers. It parses every layer of the network stack (L2–L7), detects 10 categories of attacks mapped to MITRE ATT&CK, encrypts all logs using a from-scratch Kyber-512 post-quantum KEM, and presents everything through a Rich terminal UI or a glassmorphism Flask web dashboard.

### Verification Results
```
Self-Tests:  ALL PASSED (PQC, Protocols+Deep TLS, IDS+XAI, Analytics, Distributed)
Simulation:  6/6 attacks detected, 267 alerts from 292 packets
TLS Grades:  RSA=D/CRITICAL, ECDHE=B/AT_RISK, TLS1.3+PQ=A+/SAFE
```

---

##  Architecture

```
__main__.py  ─────────────────────────────────────────────────────────────
     │                                                                    
     ▼                                                                    
engine.py  (Central Orchestrator)                                         
     │                                                                    
     ├──► protocols.py     (L2–L7 Protocol Dissectors)                   
     │         │                                                          
     │         ├──► ids.py            (IDS Engine + Explainability)      
     │         └──► analytics.py      (Bandwidth, Flows, GeoIP)          
     │                                                                    
     ├──► pqc.py           (Kyber-512 KEM + AES-256-GCM Logger)         
     ├──► dashboard.py     (Rich Terminal UI)                             
     ├──► web_dashboard.py (Flask + Chart.js Web UI)                     
     └──► distributed.py  (Sensor/Aggregator Architecture)               
                                                                          
simulator.py   (6 Attack Types)                                           
performance.py (50K-packet Benchmark)                                     
```

---

##  Features

###  L2–L7 Deep Packet Inspection
Full protocol parsing across all network layers:

| Layer | Protocols |
|-------|-----------|
| L2 | Ethernet (802.1Q VLAN), ARP |
| L3 | IPv4 (DSCP/ECN/fragments), IPv6, ICMP (10 types) |
| L4 | TCP (9 flags, scan detection), UDP |
| L7 | DNS (10+ record types, pointer compression), HTTP, TLS, QUIC, SSH, DHCP |

**Deep TLS Handshake Analysis:**
- SNI extraction and JA3 fingerprinting
- 35+ cipher suite classification
- Key exchange type: RSA / ECDHE / DHE / TLS 1.3
- Forward secrecy detection
- PQC safety verdict: `SAFE` / `AT_RISK` / `CRITICAL`
- Overall security grade: **A+ through F**
- Actionable recommendations (e.g. "Enable PQ hybrid key exchange")

---

###  Explainable Intrusion Detection System

10 detection categories with full MITRE ATT&CK mapping:

| Attack | MITRE | Severity | Detection Method |
|--------|-------|----------|-----------------|
| Port Scan (SYN/FIN/XMAS/NULL) | T1046 | MED–HIGH | Unique dst ports per src IP in sliding window |
| SYN Flood | T1498.001 | CRITICAL | SYN count + SYN-ACK ratio per destination |
| DNS Tunneling | T1071.004 | HIGH | Shannon entropy of DNS labels > threshold |
| DNS Exfiltration | T1048.003 | MEDIUM | DNS label length > 40 chars |
| DNS Flood | T1071.004 | MEDIUM | Query rate per source IP |
| ARP Spoofing | T1557.002 | CRITICAL | IP-to-MAC mapping change detection |
| Brute Force | T1110 | HIGH | Rapid SYN to SSH/RDP/FTP ports |
| TTL Anomaly | T1090.003 | LOW | TTL ≤ 5 (traceroute/proxy detection) |
| Protocol Anomaly | T1036 | HIGH | Invalid TCP flag combos (SYN+FIN, RST+SYN) |
| ICMP Tunneling | T1095 | HIGH | Large payloads (>64B) + high entropy (>3.5 bits/byte) |

**Every alert includes:**
- Natural-language explanation of what was detected and why
- Structured evidence factors with observed value vs threshold
- Confidence score (0.0–1.0)
- Recommended response actions (e.g. "Block source IP at firewall", "Enable SYN cookies")
- Internal detection logic description

Sensitivity is tunable: `--sensitivity low | medium | high`

---

###  Post-Quantum Cryptography (from scratch)

**Kyber-512 KEM** — implemented without any PQC library:
- Lattice-based, resistant to Shor's algorithm (Module-LWE problem)
- Parameters: `n=64, q=3329, k=2, η₁=η₂=2`
- NTT-optimized polynomial multiplication: O(n log n) via Number Theoretic Transform
- Negacyclic pre/post-twist, Cooley-Tukey DIT butterfly, Gentleman-Sande DIF inverse
- Full keygen → encapsulate → decapsulate round-trip verified

**PQC Secure Logger:**
- All session logs encrypted with AES-256-GCM
- Encryption key derived via Kyber KEM (not RSA)
- Key rotation every 10,000 entries
- Binary `.pqclog` format with magic bytes, nonces, length-prefixed entries

**SHA3-256 Hash Chain:**
- Blockchain-style tamper-evident integrity
- Genesis block seeded with `QUANTUM_SNIFFER_GENESIS_v1`
- Every entry hashed with previous head — tamper detection is immediate

**Quantum Threat Analyzer:**
- Classifies 20+ TLS cipher suite IDs as quantum-safe or vulnerable
- Risk levels: `SAFE` / `AT_RISK` / `CRITICAL`
- Migration recommendations for hybrid PQ/classical transition

---

###  Analytics Engine

| Component | Purpose |
|-----------|---------|
| `BandwidthMonitor` | Rolling-window bytes/sec and packets/sec with per-connection breakdowns |
| `ProtocolStats` | Protocol distribution counts and percentages |
| `TopTalkers` | Top senders/receivers by bytes, most-connected IPs |
| `FlowTracker` | TCP state machine: NEW → SYN_SENT → ESTABLISHED → FIN_WAIT → CLOSED |
| `GeoIPLookup` | Cached async GeoIP via ip-api.com, RFC1918 private IP detection |

---

###  Dual Dashboards

**Rich Terminal UI** (`dashboard.py`)
- Live packet feed with color-coded protocol badges (TCP=cyan, DNS=yellow, TLS=magenta)
- Statistics panel: packets, bytes, bandwidth, active flows, threats
- Protocol distribution bar charts
- Active TCP flows table with state badges
- Threat alerts with severity icons (ℹ️ 🟡 🟠 🔴 🚨)
- PQC status footer: log count, chain integrity, key rotations
- Refreshes at 2fps via Rich `Live`

**Flask Web Dashboard** (`web_dashboard.py`)
- Glassmorphism dark theme (CSS backdrop-filter, gradient accents)
- Chart.js 4: animated traffic timeline, protocol doughnut, top talkers bar
- TLS analysis cards with A+–F letter grades
- REST API: `/api/state`, `/api/stats`, `/api/alerts`, `/api/flows`
- Auto-refresh every 2 seconds

---

###  Distributed Architecture

Multi-node capture with two roles:

| Role | Description |
|------|-------------|
| **Aggregator** | TCP listener, accepts multiple sensor connections, tracks node health via heartbeats (15s timeout) |
| **Sensor** | Connects to aggregator, streams `PacketSummary` objects, auto-heartbeats every 5s |

Wire protocol: 4-byte big-endian length-prefixed JSON messages over TCP.

---

###  Attack Simulator

Generates and validates detection of 6 real attack types:

| # | Attack | Simulation | Alerts Generated |
|---|--------|-----------|-----------------|
| 1 | SYN Flood | 200 SYN packets from spoofed IPs | 200+ |
| 2 | Port Scan | 50 ports, mixed SYN/FIN/XMAS patterns | 40+ |
| 3 | DNS Tunneling | 20 base32-encoded high-entropy queries to `*.tunnel.evil.com` | 20+ |
| 4 | ARP Spoofing | Gateway MAC substitution | 1 |
| 5 | SSH Brute Force | 15 rapid SYN connections from single source | 11 |
| 6 | ICMP Tunneling | 5 ICMP packets with 256-byte high-entropy payloads | 5 |

---

###  Performance Benchmarks

- **50K-packet synthetic benchmark** across 3 scenarios: IDS only, Analytics only, Full Pipeline
- Rolling-window p50/p95/p99 latency percentiles
- Pipeline overhead percentage reporting
- RSA-2048 vs Kyber-512 head-to-head: keygen, encapsulate, decapsulate latency (µs) + key/ciphertext size comparison

---

##  Quick Start

### Prerequisites
- Python 3.10+
- Admin/root privileges (required for raw packet capture)
- Npcap (Windows only)

### Installation

```bash
git clone https://github.com/LogisticSapien/quantum-sniffer.git
cd quantum-sniffer
pip install -r requirements.txt
```

### Running

```bash
# Live packet capture (requires admin)
sudo python -m quantum_sniffer

# Run all self-tests
python -m quantum_sniffer --test

# Simulate 6 attacks through IDS
python -m quantum_sniffer --simulate

# Performance benchmark (50K packets)
python -m quantum_sniffer --benchmark

# RSA vs Kyber benchmark
python -m quantum_sniffer --benchmark-pqc

# Launch web dashboard alongside capture
sudo python -m quantum_sniffer --web

# Distributed mode — aggregator
python -m quantum_sniffer --mode aggregator

# Distributed mode — sensor
python -m quantum_sniffer --mode sensor --server HOST:PORT
```

### All CLI Flags

| Flag | Description |
|------|-------------|
| `--interface` | Network interface to capture on |
| `--filter` | BPF filter string |
| `--no-pqc` | Disable PQC logging |
| `--no-dashboard` | Headless mode |
| `--sensitivity` | IDS sensitivity: `low` / `medium` / `high` |
| `--geoip` | Enable GeoIP lookups |
| `--export` | Export session analytics to JSON |
| `--mode` | `sensor` or `aggregator` for distributed mode |
| `--server` | Aggregator address for sensor mode (`HOST:PORT`) |

---

##  Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `scapy` | ≥ 2.5.0 | Packet capture |
| `rich` | ≥ 13.0.0 | Terminal dashboard |
| `cryptography` | ≥ 41.0.0 | AES-256-GCM + RSA benchmarks |
| `numpy` | ≥ 1.24.0 | Kyber polynomial arithmetic + NTT |
| `requests` | ≥ 2.31.0 | GeoIP API lookups |
| `flask` | ≥ 3.0.0 | Web dashboard |

---

##  File Summary

| File | Lines | Size | Purpose |
|------|------:|-----:|---------|
| `protocols.py` | 1,353 | 47.4 KB | 12 protocol parsers + deep TLS analysis |
| `ids.py` | 775 | 36.2 KB | 10 IDS detectors + explainability engine |
| `pqc.py` | 928 | 33.4 KB | Kyber KEM, hash chain, secure logger |
| `web_dashboard.py` | 502 | 21.8 KB | Flask + Chart.js web UI |
| `engine.py` | 475 | 17.9 KB | Capture engine orchestrator |
| `analytics.py` | 481 | 17.0 KB | Bandwidth, protocols, talkers, flows, GeoIP |
| `dashboard.py` | 369 | 13.3 KB | Rich terminal UI |
| `simulator.py` | 307 | 12.0 KB | 6 attack simulations |
| `distributed.py` | 375 | 11.7 KB | Sensor/Aggregator distributed capture |
| `performance.py` | 276 | 10.0 KB | Performance monitor + benchmark |
| `__main__.py` | 272 | 9.4 KB | CLI entry point, 6 modes |
| **Total** | **~6,100** | **~230 KB** | |

---

##  Packet Lifecycle

```
1. Capture    Scapy AsyncSniffer receives raw frame from NIC
2. Enqueue    Callback enqueues raw bytes into bounded Queue (max 10,000)
3. Dequeue    Worker thread dequeues and begins processing
4. L2 Parse   parse_ethernet() → EthernetFrame (VLAN-aware)
5. L3 Route   Route to IPv4 / IPv6 / ARP parser via EtherType
6. L4 Parse   parse_tcp() / parse_udp() / parse_icmp()
7. L7 Detect  DNS / HTTP / TLS / QUIC / SSH / DHCP via port numbers
8. IDS        IDSEngine.analyze_packet() → ThreatEvent[]
9. Analytics  AnalyticsManager records bandwidth, protocols, flows
10. PQC Log   AES-256-GCM encryption (Kyber-keyed) + hash chain append
11. Display   Dashboard / Web renders packet summary, stats, alerts
```

---

##  TLS Security Grading

```
Grade  Criteria
A+     TLS 1.3 + PQ hybrid key exchange (X25519Kyber768)  → SAFE
A      TLS 1.3 + ECDHE + AEAD cipher                      → SAFE  
B      TLS 1.2 + ECDHE + AEAD cipher                      → AT_RISK
C      TLS 1.2 + ECDHE + CBC cipher                       → AT_RISK
D      TLS 1.2 + RSA key exchange                         → CRITICAL
F      NULL cipher / broken configuration                  → CRITICAL
```

---

##  Why Post-Quantum?

Classical public-key cryptography (RSA, ECDH) is vulnerable to **Shor's algorithm** on a sufficiently powerful quantum computer — which would allow an attacker to break TLS session keys retroactively ("harvest now, decrypt later"). 

Quantum Sniffer addresses this by:
1. **Detecting** TLS sessions still using quantum-vulnerable cipher suites in real time
2. **Logging** all captured data using Kyber-512 KEM — a NIST-standardized lattice-based algorithm resistant to both classical and quantum attacks
3. **Benchmarking** the performance cost of migrating from RSA to Kyber

---

##  License

MIT License — see [LICENSE](LICENSE) for details.

---

##  Author

**Dheemanth A**  
Electronics & Communication Engineering, PES University  
[github.com/LogisticSapien](https://github.com/LogisticSapien) · [dheemanth1579@gmail.com](mailto:dheemanth1579@gmail.com)

---

> *Built solo in free time during 6th semester. All cryptographic primitives (Kyber-512 NTT, SHA3-256 hash chain) implemented from scratch without PQC libraries.*
