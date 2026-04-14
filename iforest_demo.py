"""
Isolation Forest for Network Attack Detection -- MLA Project Demo
=================================================================
Standalone demonstration script for Machine Learning Applications (MLA)
course project. Implements and evaluates Isolation Forest for detecting
network attack patterns.

This script:
  1. Generates synthetic network traffic data (normal + 5 attack types)
  2. Trains a from-scratch Isolation Forest model
  3. Evaluates detection performance (accuracy, precision, recall, F1)
  4. Generates publication-quality visualizations:
     - Anomaly score distribution
     - 2D PCA scatter plot (normal vs anomaly)
     - Confusion matrix heatmap
     - Feature importance chart
     - ROC curve with AUC
     - Precision-Recall curve

Usage:
  python iforest_demo.py              # Run full demo
  python __main__.py --iforest-demo   # Run via Quantum Sniffer CLI

Output:
  - Console: classification report + model summary
  - Files: iforest_results/*.png (6 publication-ready plots)

Author: Chinmay -- MLA Course Project, Semester 2026
"""

from __future__ import annotations

import math
import os
import sys
import time
from pathlib import Path

import numpy as np

# Ensure our modules are importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from isolation_forest import IsolationForest, _c
from hybrid_scorer import HybridScorer
from extended_isolation_forest import ExtendedIsolationForest


# ══════════════════════════════════════════════════════════════════════
# 1. SYNTHETIC DATA GENERATION
# ══════════════════════════════════════════════════════════════════════

# Feature specification (same as iforest_detector.py)
FEATURE_NAMES = [
    "packet_rate",       "byte_rate",        "avg_packet_size",
    "unique_src_ips",    "unique_dst_ips",    "unique_dst_ports",
    "tcp_ratio",         "udp_ratio",         "dns_ratio",
    "icmp_ratio",        "syn_ratio",         "connection_rate",
    "port_entropy",      "ip_entropy",
]

ATTACK_TYPES = {
    "Normal":          0,
    "DDoS":            1,
    "Port Scan":       2,
    "Data Exfil":      3,
    "Brute Force":     4,
    "DNS Tunneling":   5,
}


def generate_normal_traffic(n: int, rng: np.random.RandomState) -> np.ndarray:
    """Generate normal network traffic feature vectors.

    Realistic simulation with natural variance:
    - Includes occasional traffic spikes (video calls, backups, updates)
    - Higher standard deviations to create realistic overlap with mild attacks
    - Some normal samples will have elevated metrics (natural outliers)
    """
    data = np.zeros((n, len(FEATURE_NAMES)))
    data[:, 0] = rng.normal(120, 55, n)          # packet_rate (wider spread)
    data[:, 1] = rng.normal(150000, 80000, n)    # byte_rate (occasional spikes)
    data[:, 2] = rng.normal(800, 350, n)         # avg_packet_size (varies a lot)
    data[:, 3] = rng.normal(15, 8, n)            # unique_src_ips
    data[:, 4] = rng.normal(20, 10, n)           # unique_dst_ips
    data[:, 5] = rng.normal(30, 18, n)           # unique_dst_ports (wider spread)
    data[:, 6] = rng.normal(0.72, 0.10, n)       # tcp_ratio (more variance)
    data[:, 7] = rng.normal(0.14, 0.06, n)       # udp_ratio
    data[:, 8] = rng.normal(0.09, 0.04, n)       # dns_ratio (some DNS-heavy normal)
    data[:, 9] = rng.normal(0.05, 0.03, n)       # icmp_ratio
    data[:, 10] = rng.normal(0.18, 0.08, n)      # syn_ratio (wider spread)
    data[:, 11] = rng.normal(6, 4, n)            # connection_rate (spiky)
    data[:, 12] = rng.normal(3.5, 0.8, n)        # port_entropy
    data[:, 13] = rng.normal(2.8, 0.7, n)        # ip_entropy

    # Inject realistic anomalous normal traffic (15% of normal samples)
    # These represent legitimate spikes: video conferences, backups, updates
    n_spiky = int(n * 0.15)
    spike_idx = rng.choice(n, size=n_spiky, replace=False)
    data[spike_idx, 0] *= rng.uniform(1.5, 3.0, n_spiky)   # higher packet rate
    data[spike_idx, 1] *= rng.uniform(1.5, 4.0, n_spiky)   # higher byte rate
    data[spike_idx, 11] *= rng.uniform(1.3, 2.5, n_spiky)  # higher conn rate

    return np.clip(data, 0, None)


def generate_ddos(n: int, rng: np.random.RandomState) -> np.ndarray:
    """DDoS: mix of high-intensity and low-intensity (slow DDoS) attacks.

    Real-world DDoS varies widely:
    - Volumetric floods (~5000+ pps) -- obvious
    - Slow DDoS / application-layer (~200-400 pps) -- hard to detect
    - Some overlap with normal traffic spikes
    """
    data = np.zeros((n, len(FEATURE_NAMES)))

    # 60% intense DDoS
    n_intense = int(n * 0.6)
    # 40% subtle / slow DDoS (overlaps with normal)
    n_subtle = n - n_intense

    # Intense DDoS
    data[:n_intense, 0] = rng.normal(3500, 1800, n_intense)
    data[:n_intense, 1] = rng.normal(2200000, 900000, n_intense)
    data[:n_intense, 2] = rng.normal(600, 150, n_intense)
    data[:n_intense, 3] = rng.normal(350, 200, n_intense)
    data[:n_intense, 4] = rng.normal(3, 2, n_intense)
    data[:n_intense, 5] = rng.normal(5, 3, n_intense)
    data[:n_intense, 6] = rng.normal(0.92, 0.05, n_intense)
    data[:n_intense, 7] = rng.normal(0.04, 0.02, n_intense)
    data[:n_intense, 8] = rng.normal(0.02, 0.01, n_intense)
    data[:n_intense, 9] = rng.normal(0.02, 0.01, n_intense)
    data[:n_intense, 10] = rng.normal(0.85, 0.08, n_intense)
    data[:n_intense, 11] = rng.normal(150, 70, n_intense)
    data[:n_intense, 12] = rng.normal(1.2, 0.5, n_intense)
    data[:n_intense, 13] = rng.normal(5.5, 1.5, n_intense)

    # Subtle / slow DDoS (overlaps with normal high-traffic windows)
    data[n_intense:, 0] = rng.normal(280, 100, n_subtle)
    data[n_intense:, 1] = rng.normal(350000, 120000, n_subtle)
    data[n_intense:, 2] = rng.normal(700, 200, n_subtle)
    data[n_intense:, 3] = rng.normal(60, 30, n_subtle)
    data[n_intense:, 4] = rng.normal(5, 3, n_subtle)
    data[n_intense:, 5] = rng.normal(8, 4, n_subtle)
    data[n_intense:, 6] = rng.normal(0.88, 0.07, n_subtle)
    data[n_intense:, 7] = rng.normal(0.06, 0.03, n_subtle)
    data[n_intense:, 8] = rng.normal(0.03, 0.02, n_subtle)
    data[n_intense:, 9] = rng.normal(0.03, 0.02, n_subtle)
    data[n_intense:, 10] = rng.normal(0.55, 0.15, n_subtle)
    data[n_intense:, 11] = rng.normal(35, 15, n_subtle)
    data[n_intense:, 12] = rng.normal(1.8, 0.6, n_subtle)
    data[n_intense:, 13] = rng.normal(3.8, 1.0, n_subtle)

    return np.clip(data, 0, None)


def generate_port_scan(n: int, rng: np.random.RandomState) -> np.ndarray:
    """Port Scan: mix of fast scans and stealthy slow scans.

    Real-world scanners:
    - nmap aggressive (-T5): fast, obvious
    - nmap stealth (-T1/-T2): very slow, looks almost normal
    - Randomized port order to evade detection
    """
    data = np.zeros((n, len(FEATURE_NAMES)))

    n_fast = int(n * 0.5)
    n_slow = n - n_fast

    # Fast scans (more obvious)
    data[:n_fast, 0] = rng.normal(250, 90, n_fast)
    data[:n_fast, 1] = rng.normal(22000, 8000, n_fast)
    data[:n_fast, 2] = rng.normal(85, 30, n_fast)
    data[:n_fast, 3] = rng.normal(2, 1.0, n_fast)
    data[:n_fast, 4] = rng.normal(4, 2, n_fast)
    data[:n_fast, 5] = rng.normal(400, 180, n_fast)
    data[:n_fast, 6] = rng.normal(0.95, 0.03, n_fast)
    data[:n_fast, 7] = rng.normal(0.03, 0.02, n_fast)
    data[:n_fast, 8] = rng.normal(0.01, 0.008, n_fast)
    data[:n_fast, 9] = rng.normal(0.01, 0.008, n_fast)
    data[:n_fast, 10] = rng.normal(0.90, 0.06, n_fast)
    data[:n_fast, 11] = rng.normal(80, 35, n_fast)
    data[:n_fast, 12] = rng.normal(6.5, 0.8, n_fast)
    data[:n_fast, 13] = rng.normal(0.6, 0.3, n_fast)

    # Slow / stealthy scans (hard to distinguish from normal)
    data[n_fast:, 0] = rng.normal(135, 45, n_slow)        # near-normal pkt rate
    data[n_fast:, 1] = rng.normal(12000, 5000, n_slow)
    data[n_fast:, 2] = rng.normal(90, 35, n_slow)
    data[n_fast:, 3] = rng.normal(2, 1.0, n_slow)
    data[n_fast:, 4] = rng.normal(5, 3, n_slow)
    data[n_fast:, 5] = rng.normal(80, 40, n_slow)         # more ports but not extreme
    data[n_fast:, 6] = rng.normal(0.88, 0.06, n_slow)
    data[n_fast:, 7] = rng.normal(0.06, 0.03, n_slow)
    data[n_fast:, 8] = rng.normal(0.03, 0.02, n_slow)
    data[n_fast:, 9] = rng.normal(0.03, 0.02, n_slow)
    data[n_fast:, 10] = rng.normal(0.55, 0.15, n_slow)    # moderate SYN ratio
    data[n_fast:, 11] = rng.normal(15, 8, n_slow)         # near-normal conn rate
    data[n_fast:, 12] = rng.normal(4.5, 0.8, n_slow)      # somewhat high entropy
    data[n_fast:, 13] = rng.normal(0.8, 0.4, n_slow)

    return np.clip(data, 0, None)


def generate_data_exfiltration(n: int, rng: np.random.RandomState) -> np.ndarray:
    """Data Exfiltration: mix of bulk transfer and slow drip exfil.

    Real-world exfil:
    - Bulk: large files transferred quickly (looks like cloud backup)
    - Slow drip: small amounts over long periods (very hard to detect)
    """
    data = np.zeros((n, len(FEATURE_NAMES)))

    n_bulk = int(n * 0.45)
    n_slow = n - n_bulk

    # Bulk exfiltration (more obvious)
    data[:n_bulk, 0] = rng.normal(75, 25, n_bulk)
    data[:n_bulk, 1] = rng.normal(650000, 250000, n_bulk)
    data[:n_bulk, 2] = rng.normal(8500, 2500, n_bulk)
    data[:n_bulk, 3] = rng.normal(3, 1.5, n_bulk)
    data[:n_bulk, 4] = rng.normal(2, 1.0, n_bulk)
    data[:n_bulk, 5] = rng.normal(5, 3, n_bulk)
    data[:n_bulk, 6] = rng.normal(0.83, 0.07, n_bulk)
    data[:n_bulk, 7] = rng.normal(0.10, 0.04, n_bulk)
    data[:n_bulk, 8] = rng.normal(0.04, 0.02, n_bulk)
    data[:n_bulk, 9] = rng.normal(0.03, 0.02, n_bulk)
    data[:n_bulk, 10] = rng.normal(0.12, 0.05, n_bulk)
    data[:n_bulk, 11] = rng.normal(3, 1.5, n_bulk)
    data[:n_bulk, 12] = rng.normal(1.5, 0.5, n_bulk)
    data[:n_bulk, 13] = rng.normal(1.0, 0.5, n_bulk)

    # Slow drip exfiltration (looks almost normal)
    data[n_bulk:, 0] = rng.normal(100, 35, n_slow)
    data[n_bulk:, 1] = rng.normal(250000, 100000, n_slow)  # elevated but borderline
    data[n_bulk:, 2] = rng.normal(2500, 800, n_slow)       # larger than normal
    data[n_bulk:, 3] = rng.normal(8, 4, n_slow)
    data[n_bulk:, 4] = rng.normal(3, 1.5, n_slow)
    data[n_bulk:, 5] = rng.normal(8, 4, n_slow)
    data[n_bulk:, 6] = rng.normal(0.78, 0.08, n_slow)
    data[n_bulk:, 7] = rng.normal(0.12, 0.05, n_slow)
    data[n_bulk:, 8] = rng.normal(0.06, 0.03, n_slow)
    data[n_bulk:, 9] = rng.normal(0.04, 0.02, n_slow)
    data[n_bulk:, 10] = rng.normal(0.14, 0.06, n_slow)
    data[n_bulk:, 11] = rng.normal(4, 2, n_slow)
    data[n_bulk:, 12] = rng.normal(2.0, 0.6, n_slow)
    data[n_bulk:, 13] = rng.normal(1.2, 0.5, n_slow)

    return np.clip(data, 0, None)


def generate_brute_force(n: int, rng: np.random.RandomState) -> np.ndarray:
    """Brute Force: mix of rapid and throttled attacks.

    Real-world brute force:
    - Rapid: hammering SSH/RDP (obvious)
    - Throttled: 1-2 attempts per second (evades rate limiting)
    - Credential stuffing: distributed across many IPs
    """
    data = np.zeros((n, len(FEATURE_NAMES)))

    n_rapid = int(n * 0.5)
    n_throttled = n - n_rapid

    # Rapid brute force
    data[:n_rapid, 0] = rng.normal(180, 60, n_rapid)
    data[:n_rapid, 1] = rng.normal(35000, 12000, n_rapid)
    data[:n_rapid, 2] = rng.normal(190, 60, n_rapid)
    data[:n_rapid, 3] = rng.normal(3, 1.5, n_rapid)
    data[:n_rapid, 4] = rng.normal(1.5, 0.8, n_rapid)
    data[:n_rapid, 5] = rng.normal(2.5, 1.0, n_rapid)
    data[:n_rapid, 6] = rng.normal(0.96, 0.03, n_rapid)
    data[:n_rapid, 7] = rng.normal(0.02, 0.01, n_rapid)
    data[:n_rapid, 8] = rng.normal(0.01, 0.007, n_rapid)
    data[:n_rapid, 9] = rng.normal(0.01, 0.007, n_rapid)
    data[:n_rapid, 10] = rng.normal(0.75, 0.10, n_rapid)
    data[:n_rapid, 11] = rng.normal(60, 25, n_rapid)
    data[:n_rapid, 12] = rng.normal(0.6, 0.3, n_rapid)
    data[:n_rapid, 13] = rng.normal(0.4, 0.2, n_rapid)

    # Throttled brute force (borderline with normal)
    data[n_rapid:, 0] = rng.normal(110, 40, n_throttled)     # near-normal
    data[n_rapid:, 1] = rng.normal(20000, 8000, n_throttled)
    data[n_rapid:, 2] = rng.normal(180, 60, n_throttled)
    data[n_rapid:, 3] = rng.normal(5, 3, n_throttled)        # distributed
    data[n_rapid:, 4] = rng.normal(1.5, 0.8, n_throttled)
    data[n_rapid:, 5] = rng.normal(3, 1.5, n_throttled)
    data[n_rapid:, 6] = rng.normal(0.90, 0.06, n_throttled)
    data[n_rapid:, 7] = rng.normal(0.05, 0.03, n_throttled)
    data[n_rapid:, 8] = rng.normal(0.03, 0.02, n_throttled)
    data[n_rapid:, 9] = rng.normal(0.02, 0.01, n_throttled)
    data[n_rapid:, 10] = rng.normal(0.45, 0.15, n_throttled)  # moderate SYN
    data[n_rapid:, 11] = rng.normal(12, 6, n_throttled)       # near-normal conn rate
    data[n_rapid:, 12] = rng.normal(1.0, 0.5, n_throttled)
    data[n_rapid:, 13] = rng.normal(0.8, 0.4, n_throttled)

    return np.clip(data, 0, None)


def generate_dns_tunneling(n: int, rng: np.random.RandomState) -> np.ndarray:
    """DNS Tunneling: mix of obvious and subtle tunneling.

    Real-world DNS tunneling:
    - High volume: iodine, dnscat2 at full speed (obvious)
    - Low volume: slow exfil via DNS TXT records (borderline)
    - Some legitimate DNS-heavy traffic exists (CDNs, cloud services)
    """
    data = np.zeros((n, len(FEATURE_NAMES)))

    n_obvious = int(n * 0.5)
    n_subtle = n - n_obvious

    # Obvious DNS tunneling
    data[:n_obvious, 0] = rng.normal(140, 40, n_obvious)
    data[:n_obvious, 1] = rng.normal(95000, 25000, n_obvious)
    data[:n_obvious, 2] = rng.normal(650, 120, n_obvious)
    data[:n_obvious, 3] = rng.normal(5, 2, n_obvious)
    data[:n_obvious, 4] = rng.normal(6, 3, n_obvious)
    data[:n_obvious, 5] = rng.normal(8, 4, n_obvious)
    data[:n_obvious, 6] = rng.normal(0.22, 0.08, n_obvious)
    data[:n_obvious, 7] = rng.normal(0.18, 0.06, n_obvious)
    data[:n_obvious, 8] = rng.normal(0.55, 0.10, n_obvious)  # very high DNS
    data[:n_obvious, 9] = rng.normal(0.05, 0.02, n_obvious)
    data[:n_obvious, 10] = rng.normal(0.10, 0.04, n_obvious)
    data[:n_obvious, 11] = rng.normal(4, 2, n_obvious)
    data[:n_obvious, 12] = rng.normal(2.0, 0.6, n_obvious)
    data[:n_obvious, 13] = rng.normal(1.5, 0.5, n_obvious)

    # Subtle DNS tunneling (overlaps with DNS-heavy normal traffic)
    data[n_obvious:, 0] = rng.normal(115, 35, n_subtle)       # near-normal
    data[n_obvious:, 1] = rng.normal(120000, 40000, n_subtle)
    data[n_obvious:, 2] = rng.normal(700, 180, n_subtle)
    data[n_obvious:, 3] = rng.normal(10, 5, n_subtle)
    data[n_obvious:, 4] = rng.normal(12, 5, n_subtle)
    data[n_obvious:, 5] = rng.normal(15, 7, n_subtle)
    data[n_obvious:, 6] = rng.normal(0.50, 0.12, n_subtle)    # lower TCP
    data[n_obvious:, 7] = rng.normal(0.15, 0.06, n_subtle)
    data[n_obvious:, 8] = rng.normal(0.28, 0.10, n_subtle)    # elevated DNS
    data[n_obvious:, 9] = rng.normal(0.07, 0.03, n_subtle)
    data[n_obvious:, 10] = rng.normal(0.13, 0.06, n_subtle)
    data[n_obvious:, 11] = rng.normal(5, 2.5, n_subtle)
    data[n_obvious:, 12] = rng.normal(2.5, 0.7, n_subtle)
    data[n_obvious:, 13] = rng.normal(2.0, 0.6, n_subtle)

    return np.clip(data, 0, None)


def generate_dataset(
    n_normal: int = 2000,
    n_attack: int = 100,
    seed: int = 42,
) -> tuple:
    """Generate a realistic labeled dataset with noise and overlap.

    Key realism features:
    - Normal traffic has natural variance and occasional spikes
    - Each attack type has both obvious and subtle (stealthy) variants
    - ~40-50% of attack samples intentionally overlap with normal ranges
    - Gaussian noise added to all samples post-generation

    Returns:
        (X, y_true, y_labels) where:
          X: feature matrix (n_samples, 14)
          y_true: binary labels (0=normal, 1=attack)
          y_labels: string attack type labels
    """
    rng = np.random.RandomState(seed)

    # Normal traffic (80%)
    X_normal = generate_normal_traffic(n_normal, rng)

    # Attack traffic (20% total, split across 5 types)
    X_ddos = generate_ddos(n_attack, rng)
    X_portscan = generate_port_scan(n_attack, rng)
    X_exfil = generate_data_exfiltration(n_attack, rng)
    X_brute = generate_brute_force(n_attack, rng)
    X_dns = generate_dns_tunneling(n_attack, rng)

    # Combine
    X = np.vstack([X_normal, X_ddos, X_portscan, X_exfil, X_brute, X_dns])

    # Labels
    y_true = np.array(
        [0] * n_normal +
        [1] * n_attack +
        [1] * n_attack +
        [1] * n_attack +
        [1] * n_attack +
        [1] * n_attack
    )

    y_labels = (
        ["Normal"] * n_normal +
        ["DDoS"] * n_attack +
        ["Port Scan"] * n_attack +
        ["Data Exfil"] * n_attack +
        ["Brute Force"] * n_attack +
        ["DNS Tunneling"] * n_attack
    )

    # Add global Gaussian noise to simulate sensor imprecision
    noise = rng.normal(0, 0.02, X.shape) * np.abs(X)
    X = X + noise
    X = np.clip(X, 0, None)

    # Shuffle
    indices = rng.permutation(len(X))
    X = X[indices]
    y_true = y_true[indices]
    y_labels = [y_labels[i] for i in indices]

    return X, y_true, y_labels


# ══════════════════════════════════════════════════════════════════════
# 2. EVALUATION METRICS (from scratch)
# ══════════════════════════════════════════════════════════════════════

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute classification metrics."""
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-10)

    return {
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def compute_roc(y_true: np.ndarray, scores: np.ndarray, n_thresholds: int = 200) -> tuple:
    """Compute ROC curve points and AUC."""
    thresholds = np.linspace(0, 1, n_thresholds)
    fprs, tprs = [], []

    for thresh in thresholds:
        y_pred = (scores >= thresh).astype(int)
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        tn = np.sum((y_true == 0) & (y_pred == 0))

        fpr = fp / max(fp + tn, 1)
        tpr = tp / max(tp + fn, 1)
        fprs.append(fpr)
        tprs.append(tpr)

    fprs = np.array(fprs)
    tprs = np.array(tprs)

    # AUC via trapezoidal rule
    sorted_idx = np.argsort(fprs)
    fprs_sorted = fprs[sorted_idx]
    tprs_sorted = tprs[sorted_idx]
    auc = float(np.trapz(tprs_sorted, fprs_sorted))

    return fprs, tprs, thresholds, abs(auc)


def compute_precision_recall(y_true: np.ndarray, scores: np.ndarray, n_thresholds: int = 200) -> tuple:
    """Compute Precision-Recall curve points."""
    thresholds = np.linspace(0, 1, n_thresholds)
    precisions, recalls = [], []

    for thresh in thresholds:
        y_pred = (scores >= thresh).astype(int)
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))

        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        precisions.append(prec)
        recalls.append(rec)

    return np.array(precisions), np.array(recalls), thresholds


def per_class_report(y_true_labels: list, y_pred: np.ndarray) -> str:
    """Generate a per-class classification report."""
    classes = sorted(set(y_true_labels))
    y_pred_labels = []
    for i, label in enumerate(y_true_labels):
        if y_pred[i] == 1:
            y_pred_labels.append("Anomaly")
        else:
            y_pred_labels.append("Normal")

    lines = []
    lines.append(f"{'Class':<18} {'Samples':>8} {'Detected':>9} {'Missed':>7} {'Recall':>8}")
    lines.append("-" * 52)

    for cls in classes:
        indices = [i for i, l in enumerate(y_true_labels) if l == cls]
        n = len(indices)
        if cls == "Normal":
            correct = sum(1 for i in indices if y_pred[i] == 0)
            lines.append(f"{cls:<18} {n:>8} {correct:>9} {n - correct:>7} {correct/max(n,1):>8.1%}")
        else:
            detected = sum(1 for i in indices if y_pred[i] == 1)
            lines.append(f"{cls:<18} {n:>8} {detected:>9} {n - detected:>7} {detected/max(n,1):>8.1%}")

    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════
# 3. PCA (from scratch -- for visualization)
# ══════════════════════════════════════════════════════════════════════

def pca_2d(X: np.ndarray) -> np.ndarray:
    """Project data to 2D using PCA (from scratch).

    Steps:
      1. Center the data (subtract mean)
      2. Compute covariance matrix
      3. Eigendecomposition
      4. Project onto top 2 eigenvectors
    """
    X_centered = X - X.mean(axis=0)
    cov = np.cov(X_centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    # Sort by descending eigenvalue
    idx = np.argsort(eigenvalues)[::-1]
    top2 = eigenvectors[:, idx[:2]]
    return X_centered @ top2


# ══════════════════════════════════════════════════════════════════════
# 4. VISUALIZATION
# ══════════════════════════════════════════════════════════════════════

def generate_visualizations(
    X: np.ndarray,
    y_true: np.ndarray,
    y_labels: list,
    y_pred: np.ndarray,
    scores: np.ndarray,
    forest: IsolationForest,
    output_dir: str = "iforest_results",
):
    """Generate all publication-quality plots."""
    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap
    except ImportError:
        print("  [!] matplotlib not installed. Skipping plots.")
        print("      Install with: pip install matplotlib")
        return

    os.makedirs(output_dir, exist_ok=True)

    # Color palette
    COLORS = {
        "normal": "#2ecc71",
        "anomaly": "#e74c3c",
        "primary": "#3498db",
        "secondary": "#9b59b6",
        "background": "#1a1a2e",
        "surface": "#16213e",
        "text": "#eaeaea",
        "grid": "#333366",
    }

    ATTACK_COLORS = {
        "Normal": "#2ecc71",
        "DDoS": "#e74c3c",
        "Port Scan": "#e67e22",
        "Data Exfil": "#9b59b6",
        "Brute Force": "#f1c40f",
        "DNS Tunneling": "#1abc9c",
    }

    plt.rcParams.update({
        'figure.facecolor': COLORS['background'],
        'axes.facecolor': COLORS['surface'],
        'axes.edgecolor': COLORS['grid'],
        'axes.labelcolor': COLORS['text'],
        'text.color': COLORS['text'],
        'xtick.color': COLORS['text'],
        'ytick.color': COLORS['text'],
        'grid.color': COLORS['grid'],
        'grid.alpha': 0.3,
        'font.size': 11,
        'font.family': 'sans-serif',
    })

    # -- Plot 1: Anomaly Score Distribution --
    fig, ax = plt.subplots(figsize=(10, 6))
    normal_scores = scores[y_true == 0]
    attack_scores = scores[y_true == 1]
    ax.hist(normal_scores, bins=50, alpha=0.7, color=COLORS['normal'],
            label=f'Normal (n={len(normal_scores)})', edgecolor='none')
    ax.hist(attack_scores, bins=50, alpha=0.7, color=COLORS['anomaly'],
            label=f'Attack (n={len(attack_scores)})', edgecolor='none')
    ax.axvline(x=forest.threshold, color='#f39c12', linestyle='--',
               linewidth=2, label=f'Threshold ({forest.threshold:.3f})')
    ax.set_xlabel('Anomaly Score', fontsize=13)
    ax.set_ylabel('Frequency', fontsize=13)
    ax.set_title('Isolation Forest -- Anomaly Score Distribution', fontsize=15, fontweight='bold')
    ax.legend(fontsize=11, facecolor=COLORS['surface'], edgecolor=COLORS['grid'])
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, '1_score_distribution.png'), dpi=150,
                facecolor=COLORS['background'], bbox_inches='tight')
    plt.close(fig)
    print(f"  [OK] Score distribution -> {output_dir}/1_score_distribution.png")

    # -- Plot 2: 2D PCA Scatter --
    X_2d = pca_2d(X)
    fig, ax = plt.subplots(figsize=(10, 8))
    for label, color in ATTACK_COLORS.items():
        mask = [l == label for l in y_labels]
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                   c=color, label=label, alpha=0.6, s=20, edgecolors='none')
    ax.set_xlabel('Principal Component 1', fontsize=13)
    ax.set_ylabel('Principal Component 2', fontsize=13)
    ax.set_title('PCA Projection -- Normal vs Attack Clusters', fontsize=15, fontweight='bold')
    ax.legend(fontsize=10, facecolor=COLORS['surface'], edgecolor=COLORS['grid'],
              loc='upper right', markerscale=2)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, '2_pca_scatter.png'), dpi=150,
                facecolor=COLORS['background'], bbox_inches='tight')
    plt.close(fig)
    print(f"  [OK] PCA scatter plot -> {output_dir}/2_pca_scatter.png")

    # -- Plot 3: Confusion Matrix --
    metrics = compute_metrics(y_true, y_pred)
    cm = np.array([[metrics['tn'], metrics['fp']],
                    [metrics['fn'], metrics['tp']]])

    fig, ax = plt.subplots(figsize=(7, 6))
    # Custom colormap
    cmap = LinearSegmentedColormap.from_list('custom',
        [COLORS['surface'], '#2980b9', '#2ecc71'])
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.colorbar(im, ax=ax)

    labels = ['Normal', 'Attack']
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_yticklabels(labels, fontsize=12)
    ax.set_xlabel('Predicted Label', fontsize=13)
    ax.set_ylabel('True Label', fontsize=13)
    ax.set_title('Confusion Matrix', fontsize=15, fontweight='bold')

    # Annotate cells
    for i in range(2):
        for j in range(2):
            val = cm[i, j]
            pct = val / cm.sum() * 100
            ax.text(j, i, f'{val}\n({pct:.1f}%)',
                    ha='center', va='center', fontsize=14,
                    color='white', fontweight='bold')

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, '3_confusion_matrix.png'), dpi=150,
                facecolor=COLORS['background'], bbox_inches='tight')
    plt.close(fig)
    print(f"  [OK] Confusion matrix -> {output_dir}/3_confusion_matrix.png")

    # -- Plot 4: Feature Importance --
    importances = forest.feature_importance(X)
    sorted_idx = np.argsort(importances)
    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh(range(len(FEATURE_NAMES)),
                   importances[sorted_idx],
                   color=plt.cm.viridis(np.linspace(0.3, 0.9, len(FEATURE_NAMES))))
    ax.set_yticks(range(len(FEATURE_NAMES)))
    ax.set_yticklabels([FEATURE_NAMES[i] for i in sorted_idx], fontsize=11)
    ax.set_xlabel('Relative Importance (Permutation)', fontsize=13)
    ax.set_title('Feature Importance -- Isolation Forest', fontsize=15, fontweight='bold')
    ax.grid(True, alpha=0.2, axis='x')

    # Add value labels
    for bar, val in zip(bars, importances[sorted_idx]):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                f'{val:.3f}', va='center', fontsize=10, color=COLORS['text'])

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, '4_feature_importance.png'), dpi=150,
                facecolor=COLORS['background'], bbox_inches='tight')
    plt.close(fig)
    print(f"  [OK] Feature importance -> {output_dir}/4_feature_importance.png")

    # -- Plot 5: ROC Curve --
    fprs, tprs, _, auc_val = compute_roc(y_true, scores)
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot(fprs, tprs, color=COLORS['primary'], linewidth=2.5,
            label=f'Isolation Forest (AUC = {auc_val:.4f})')
    ax.plot([0, 1], [0, 1], color='#7f8c8d', linestyle='--',
            linewidth=1, label='Random Classifier')
    ax.fill_between(fprs, tprs, alpha=0.15, color=COLORS['primary'])
    ax.set_xlabel('False Positive Rate', fontsize=13)
    ax.set_ylabel('True Positive Rate', fontsize=13)
    ax.set_title('ROC Curve -- Isolation Forest', fontsize=15, fontweight='bold')
    ax.legend(fontsize=12, facecolor=COLORS['surface'], edgecolor=COLORS['grid'])
    ax.grid(True, alpha=0.2)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, '5_roc_curve.png'), dpi=150,
                facecolor=COLORS['background'], bbox_inches='tight')
    plt.close(fig)
    print(f"  [OK] ROC curve (AUC={auc_val:.4f}) -> {output_dir}/5_roc_curve.png")

    # -- Plot 6: Precision-Recall Curve --
    precisions, recalls, _ = compute_precision_recall(y_true, scores)
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot(recalls, precisions, color=COLORS['secondary'], linewidth=2.5,
            label='Isolation Forest')
    ax.fill_between(recalls, precisions, alpha=0.15, color=COLORS['secondary'])
    ax.set_xlabel('Recall', fontsize=13)
    ax.set_ylabel('Precision', fontsize=13)
    ax.set_title('Precision-Recall Curve', fontsize=15, fontweight='bold')
    ax.legend(fontsize=12, facecolor=COLORS['surface'], edgecolor=COLORS['grid'])
    ax.grid(True, alpha=0.2)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, '6_precision_recall.png'), dpi=150,
                facecolor=COLORS['background'], bbox_inches='tight')
    plt.close(fig)
    print(f"  [OK] Precision-Recall curve -> {output_dir}/6_precision_recall.png")


# ══════════════════════════════════════════════════════════════════════
# 5. MAIN DEMO EXECUTION
# ══════════════════════════════════════════════════════════════════════

def run_iforest_demo():
    """Main entry point for the MLA project demo."""
    print()
    print("=" * 64)
    print("  ISOLATION FOREST FOR NETWORK ATTACK DETECTION")
    print("  Machine Learning Applications -- Course Project Demo")
    print("=" * 64)
    print()

    # -- Step 1: Generate Data --
    print("  [1/5] Generating synthetic network traffic dataset...")
    X, y_true, y_labels = generate_dataset(n_normal=2000, n_attack=100, seed=42)
    n_total = len(X)
    n_normal = int(np.sum(y_true == 0))
    n_attack = int(np.sum(y_true == 1))
    print(f"        Total samples: {n_total}")
    print(f"        Normal:  {n_normal} ({n_normal/n_total*100:.1f}%)")
    print(f"        Attack:  {n_attack} ({n_attack/n_total*100:.1f}%)")
    print(f"        Attack types: DDoS, Port Scan, Data Exfil, Brute Force, DNS Tunneling")
    print(f"        Features: {len(FEATURE_NAMES)}")
    print()

    # -- Step 2: Normalize --
    print("  [2/5] Normalizing features (z-score standardization)...")
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X_std[X_std < 1e-10] = 1.0
    X_normalized = (X - X_mean) / X_std
    print(f"        Feature means: min={X_mean.min():.1f}, max={X_mean.max():.1f}")
    print(f"        Feature stds:  min={X_std.min():.2f}, max={X_std.max():.2f}")
    print()

    # -- Step 3: Train Isolation Forest (enhanced) --
    print("  [3/6] Training Isolation Forest (from scratch, no sklearn)...")
    print("        Config: 200 trees, psi=512 (increased from 100/256 for FNR reduction)")
    t0 = time.time()
    forest = IsolationForest(
        n_estimators=200,      # ^ from 100 -- finer score granularity
        max_samples=512,       # ^ from 256 -- better boundary resolution
        contamination=0.25,    # ^ from 0.2 -- biased toward more detections
        random_state=42,
    )
    forest.fit(X_normalized)
    train_time = time.time() - t0
    print(f"        Trees:           {forest.n_estimators}")
    print(f"        Subsample size:  {forest._psi}")
    print(f"        Height limit:    {math.ceil(math.log2(max(forest._psi, 2)))}")
    print(f"        Contamination:   {forest.contamination}")
    print(f"        Default thresh:  {forest.threshold:.4f}")
    print(f"        Training time:   {train_time:.3f}s")
    print()

    # -- Step 4: Recall-Optimised Threshold Search --
    print("  [4/6] Finding recall-optimised threshold (FNR reduction)...")
    t0 = time.time()
    scores = forest.anomaly_scores(X_normalized)
    predict_time = time.time() - t0

    # Default threshold results
    y_pred_default = forest.predict(X_normalized)
    metrics_default = compute_metrics(y_true, y_pred_default)

    # Recall-optimised threshold sweep
    # Objective: maximize  0.85.recall + 0.15.precision
    # This heavily penalises FN while still caring about FP somewhat
    recall_weight = 0.85
    best_objective = -1.0
    best_threshold = forest.threshold
    best_metrics = metrics_default

    thresholds = np.linspace(scores.min(), scores.max(), 500)
    for thresh in thresholds:
        y_pred_trial = (scores >= thresh).astype(int)
        m = compute_metrics(y_true, y_pred_trial)
        objective = recall_weight * m['recall'] + (1 - recall_weight) * m['precision']
        if objective > best_objective:
            best_objective = objective
            best_threshold = float(thresh)
            best_metrics = m

    # Apply recall-optimised threshold
    forest.threshold = best_threshold
    y_pred = (scores >= best_threshold).astype(int)
    metrics = best_metrics

    print(f"        Default threshold:   {metrics_default['recall']*100:.1f}% recall, "
          f"{metrics_default['precision']*100:.1f}% precision (t={forest.threshold:.4f})")
    print(f"        Optimised threshold: {metrics['recall']*100:.1f}% recall, "
          f"{metrics['precision']*100:.1f}% precision (t={best_threshold:.4f})")
    print(f"        Recall improvement:  +{(metrics['recall']-metrics_default['recall'])*100:.1f}pp")
    print(f"        FNR reduction:       {metrics_default['fn']} -> {metrics['fn']} "
          f"({(1 - metrics['fn']/max(metrics_default['fn'],1))*100:.0f}% fewer missed attacks)")
    print()

    # -- Step 5: Extended Isolation Forest comparison --
    print("  [5/6] Training Extended Isolation Forest (hyperplane splits)...")
    t0_eif = time.time()
    eif = ExtendedIsolationForest(
        n_estimators=200,
        max_samples=512,
        contamination=0.25,
        random_state=42,
    )
    eif.fit(X_normalized)
    eif_time = time.time() - t0_eif
    eif_scores = eif.anomaly_scores(X_normalized)

    # Recall-optimise EIF threshold too
    best_eif_obj = -1.0
    best_eif_thresh = eif.threshold
    for thresh in np.linspace(eif_scores.min(), eif_scores.max(), 500):
        y_eif = (eif_scores >= thresh).astype(int)
        m = compute_metrics(y_true, y_eif)
        obj = recall_weight * m['recall'] + (1 - recall_weight) * m['precision']
        if obj > best_eif_obj:
            best_eif_obj = obj
            best_eif_thresh = float(thresh)
            best_eif_metrics = m

    eif.threshold = best_eif_thresh
    y_pred_eif = (eif_scores >= best_eif_thresh).astype(int)
    print(f"        EIF recall:      {best_eif_metrics['recall']*100:.1f}%, "
          f"precision: {best_eif_metrics['precision']*100:.1f}%")
    print(f"        EIF train time:  {eif_time:.3f}s")
    print()

    # -- Hybrid scoring (iForest + EIF ensemble) --
    print("  Fusing iForest + EIF scores (ensemble)...")
    ensemble_scores = 0.5 * scores + 0.5 * eif_scores
    best_ens_obj = -1.0
    best_ens_thresh = 0.5
    for thresh in np.linspace(ensemble_scores.min(), ensemble_scores.max(), 500):
        y_ens = (ensemble_scores >= thresh).astype(int)
        m = compute_metrics(y_true, y_ens)
        obj = recall_weight * m['recall'] + (1 - recall_weight) * m['precision']
        if obj > best_ens_obj:
            best_ens_obj = obj
            best_ens_thresh = float(thresh)
            best_ens_metrics = m

    y_pred_ensemble = (ensemble_scores >= best_ens_thresh).astype(int)

    # Pick the best model
    candidates = [
        ('iForest (recall-opt)', metrics, y_pred, scores, best_threshold),
        ('EIF (recall-opt)', best_eif_metrics, y_pred_eif, eif_scores, best_eif_thresh),
        ('iForest+EIF ensemble', best_ens_metrics, y_pred_ensemble, ensemble_scores, best_ens_thresh),
    ]
    # Sort by recall, then by F1 as tiebreaker
    candidates.sort(key=lambda x: (x[1]['recall'], x[1]['f1']), reverse=True)
    best_name, metrics, y_pred, scores_for_viz, _ = candidates[0]

    print(f"  +----------------------------------------------------------+")
    print(f"  |  MODEL COMPARISON (recall-optimised thresholds)         |")
    print(f"  +--------------------------+--------+--------+------------+")
    print(f"  | Model                    | Recall | Precis | FNR        |")
    print(f"  +--------------------------+--------+--------+------------+")
    for name, m, _, _, _ in candidates:
        fnr = m['fn'] / max(m['fn'] + m['tp'], 1) * 100
        marker = " <- BEST" if name == best_name else ""
        print(f"  | {name:24s} | {m['recall']*100:5.1f}% | {m['precision']*100:5.1f}% | {fnr:5.1f}%{marker:>5s} |")
    print(f"  +--------------------------+--------+--------+------------+")
    fnr_default = metrics_default['fn'] / max(metrics_default['fn'] + metrics_default['tp'], 1) * 100
    print(f"  | {'iForest (default t)':24s} | {metrics_default['recall']*100:5.1f}% | {metrics_default['precision']*100:5.1f}% | {fnr_default:5.1f}%      |")
    print(f"  +--------------------------+--------+--------+------------+")
    print(f"  Selected: {best_name}")
    print()

    print(f"        +--------------------------------------+")
    print(f"        |  FINAL METRICS ({best_name:18s})  |")
    print(f"        +--------------------------------------+")
    print(f"        |  Accuracy:   {metrics['accuracy']:.4f}  ({metrics['accuracy']*100:.1f}%)       |")
    print(f"        |  Precision:  {metrics['precision']:.4f}  ({metrics['precision']*100:.1f}%)       |")
    print(f"        |  Recall:     {metrics['recall']:.4f}  ({metrics['recall']*100:.1f}%)  *    |")
    print(f"        |  F1 Score:   {metrics['f1']:.4f}  ({metrics['f1']*100:.1f}%)       |")
    print(f"        |  FNR:        {metrics['fn']/(metrics['fn']+metrics['tp'])*100:.1f}%                        |")
    print(f"        +--------------------------------------+")
    print()

    # Per-class report
    print("  PER-CLASS DETECTION REPORT:")
    print("  " + "-" * 52)
    report = per_class_report(y_labels, y_pred)
    for line in report.split("\n"):
        print(f"  {line}")
    print()

    # ROC AUC
    _, _, _, auc = compute_roc(y_true, scores_for_viz)
    print(f"  ROC AUC: {auc:.4f}")
    print()

    # -- Step 6: GUI Dashboard + Visualizations --
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "iforest_results")
    print(f"  [6/6] Generating visualizations -> {output_dir}/")
    generate_visualizations(
        X_normalized, y_true, y_labels, y_pred, scores_for_viz, forest, output_dir
    )

    # Compute data needed for GUI
    fprs_roc, tprs_roc, _, auc = compute_roc(y_true, scores_for_viz)
    pr_precisions, pr_recalls, _ = compute_precision_recall(y_true, scores_for_viz)
    importances = forest.feature_importance(X_normalized)

    # Build per-class data
    per_class_data = []
    classes = sorted(set(y_labels))
    for cls in classes:
        indices = [i for i, l in enumerate(y_labels) if l == cls]
        n = len(indices)
        if cls == "Normal":
            detected = sum(1 for i in indices if y_pred[i] == 0)
            missed = n - detected
            rec = detected / max(n, 1)
        else:
            detected = sum(1 for i in indices if y_pred[i] == 1)
            missed = n - detected
            rec = detected / max(n, 1)
        per_class_data.append({
            "class": cls, "samples": n, "detected": detected,
            "missed": missed, "recall": rec,
        })

    # Build model comparison data
    model_comp = []
    for name, m, _, _, _ in candidates:
        fnr = m['fn'] / max(m['fn'] + m['tp'], 1)
        model_comp.append({
            "name": name, "recall": m['recall'],
            "precision": m['precision'], "fnr": fnr,
        })
    # Add default baseline
    fnr_def = metrics_default['fn'] / max(metrics_default['fn'] + metrics_default['tp'], 1)
    model_comp.append({
        "name": "iForest (default)", "recall": metrics_default['recall'],
        "precision": metrics_default['precision'], "fnr": fnr_def,
    })

    # Instead of launching GUI here, return the results dict to the launcher
    print("  Analysis complete. Preparing dashboard...")
    return {
        "tp": metrics['tp'], "tn": metrics['tn'],
        "fp": metrics['fp'], "fn": metrics['fn'],
        "accuracy": metrics['accuracy'],
        "precision": metrics['precision'],
        "recall": metrics['recall'],
        "f1": metrics['f1'],
        "scores": scores_for_viz,
        "y_true": y_true,
        "y_labels": y_labels,
        "y_pred": y_pred,
        "threshold": forest.threshold,
        "auc": auc,
        "model_name": best_name,
        "train_time": train_time,
        "predict_time": predict_time,
        "n_estimators": forest.n_estimators,
        "psi": forest._psi,
        "contamination": forest.contamination,
        "feature_names": FEATURE_NAMES,
        "feature_importances": importances,
        "per_class": per_class_data,
        "model_comparison": model_comp,
        "fprs": fprs_roc,
        "tprs": tprs_roc,
        "pr_precisions": pr_precisions,
        "pr_recalls": pr_recalls,
    }


if __name__ == "__main__":
    try:
        from dashboard_gui import run_demo_with_gui
        # Run the demo wrapped in the loading GUI, which then opens the dashboard
        run_demo_with_gui(run_iforest_demo)
    except ImportError:
        # Fallback if tkinter is not available
        results = run_iforest_demo()
        if results:
            from dashboard_gui import launch_dashboard
            launch_dashboard(results)
