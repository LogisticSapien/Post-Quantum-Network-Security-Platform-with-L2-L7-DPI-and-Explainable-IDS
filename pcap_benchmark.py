"""
PCAP-Based ML Benchmark
=========================
Evaluates the ML stack directly on CICIDS2017 raw PCAPs using
FlowFeatureExtractor + iForest/EIF scoring.

Since PCAPs don't have per-flow labels, we use the published
CICIDS2017 attack time windows as ground truth:

  Friday-WorkingHours.pcap:
    BENIGN:    09:00 - 10:02
    Botnet:    10:02 - 11:16
    PortScan:  13:55 - 14:35 (approx)
    DDoS:      15:56 - 16:16

  Tuesday-WorkingHours.pcap:
    BENIGN:    09:00 - 09:20
    BruteForce (FTP): 09:20 - 10:20
    BruteForce (SSH): 14:00 - 15:00
    BENIGN:    rest

Usage:
  python pcap_benchmark.py --pcap Friday-WorkingHours.pcap --day friday
  python pcap_benchmark.py --pcap Tuesday-WorkingHours.pcap --day tuesday
  python pcap_benchmark.py --pcap Friday-WorkingHours.pcap --day friday --max-packets 500000
"""

from __future__ import annotations

import os
import sys
import time
import math
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flow_feature_extractor import FlowFeatureExtractor, NUM_FEATURES, FLOW_FEATURE_NAMES
from isolation_forest import IsolationForest
from extended_isolation_forest import ExtendedIsolationForest


# ──────────────────────────────────────────────────────────────────────
# CICIDS2017 Ground Truth Time Windows
# ──────────────────────────────────────────────────────────────────────

# CICIDS2017 was captured July 3-7 2017 (EDT = UTC-4)
# All times in "hour:minute" of that day

FRIDAY_ATTACKS = [
    # (start_h, start_m, end_h, end_m, label)
    (10, 2, 11, 16, "Botnet"),
    (13, 55, 14, 35, "PortScan"),
    (15, 56, 16, 16, "DDoS"),
]

TUESDAY_ATTACKS = [
    (9, 20, 10, 20, "BruteForce-FTP"),
    (14, 0, 15, 0, "BruteForce-SSH"),
]


def _time_to_seconds(h: int, m: int) -> int:
    """Convert hour:minute to seconds since midnight."""
    return h * 3600 + m * 60


def classify_timestamp(ts: float, day: str) -> str:
    """Classify a Unix timestamp into attack type or BENIGN.

    Uses the published CICIDS2017 attack windows.

    Args:
        ts: Unix timestamp from packet.
        day: "friday" or "tuesday".

    Returns:
        Attack label string or "BENIGN".
    """
    # Convert to time of day (seconds since midnight)
    # CICIDS2017 timestamps are in EDT (UTC-4)
    dt = datetime.fromtimestamp(ts, tz=timezone(timedelta(hours=-4)))
    sod = dt.hour * 3600 + dt.minute * 60 + dt.second

    attacks = FRIDAY_ATTACKS if day == "friday" else TUESDAY_ATTACKS

    for start_h, start_m, end_h, end_m, label in attacks:
        start = _time_to_seconds(start_h, start_m)
        end = _time_to_seconds(end_h, end_m)
        if start <= sod <= end:
            return label

    return "BENIGN"


# ──────────────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))

    recall = tp / max(tp + fn, 1)
    precision = tp / max(tp + fp, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-10)
    fpr = fp / max(fp + tn, 1)
    fnr = fn / max(fn + tp, 1)
    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)

    denom = math.sqrt(max((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn), 1))
    mcc = (tp*tn - fp*fn) / denom

    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "recall": recall, "precision": precision,
        "f1": f1, "fpr": fpr, "fnr": fnr,
        "accuracy": accuracy, "mcc": mcc,
    }


def find_best_threshold(scores, y_true, n_thresh=500):
    """Find threshold maximizing Youden's J statistic (TPR - FPR).

    Youden's J avoids the degenerate case where recall-heavy objectives
    push the threshold to the minimum score, classifying everything as
    anomalous (TPR=1.0, FPR=1.0, FN=0, TN=0).

    When J values are tied (e.g. no score separation between classes),
    prefers higher threshold (lower FPR) for conservative predictions.
    """
    thresholds = np.linspace(scores.min() - 0.001, scores.max() + 0.001, n_thresh)

    n_pos = max(int(np.sum(y_true == 1)), 1)
    n_neg = max(int(np.sum(y_true == 0)), 1)

    best_j = -2.0
    best_t = float(np.median(scores))
    best_fpr = 1.0

    for t in thresholds:
        pred = (scores >= t).astype(int)
        tp = int(np.sum((y_true == 1) & (pred == 1)))
        fp = int(np.sum((y_true == 0) & (pred == 1)))

        tpr = tp / n_pos
        fpr = fp / n_neg
        j = tpr - fpr

        # Pick best J; break ties by preferring lower FPR (higher threshold)
        if j > best_j + 1e-6 or (abs(j - best_j) <= 1e-6 and fpr < best_fpr):
            best_j = j
            best_t = float(t)
            best_fpr = fpr

    return best_t


def compute_auc_roc(y_true, scores, n_thresh=500):
    """Compute ROC AUC via trapezoidal rule."""
    thresholds = np.linspace(scores.min() - 0.01, scores.max() + 0.01, n_thresh)
    n_pos = max(int(np.sum(y_true == 1)), 1)
    n_neg = max(int(np.sum(y_true == 0)), 1)
    fprs, tprs = [], []
    for t in thresholds:
        pred = (scores >= t).astype(int)
        tp = np.sum((y_true == 1) & (pred == 1))
        fp = np.sum((y_true == 0) & (pred == 1))
        tprs.append(tp / n_pos)
        fprs.append(fp / n_neg)
    fprs = np.array(fprs)
    tprs = np.array(tprs)
    order = np.argsort(fprs)
    return abs(float(np.trapz(tprs[order], fprs[order])))


# ──────────────────────────────────────────────────────────────────────
# Main Benchmark
# ──────────────────────────────────────────────────────────────────────

def run_pcap_benchmark(
    pcap_path: str,
    day: str = "friday",
    window_seconds: float = 30.0,
    max_packets: Optional[int] = None,
    n_estimators: int = 200,
    max_samples: int = 256,
    contamination: float = 0.015,
):
    """Run the full PCAP-based benchmark.

    Steps:
      1. Stream packets from PCAP via Scapy PcapReader
      2. Group into 30s windows via FlowFeatureExtractor
      3. First N benign windows → training set
      4. All windows → test scoring with iForest + EIF
      5. Label each flow-window using CICIDS2017 attack time windows
      6. Report metrics
    """
    try:
        from scapy.all import PcapReader
        from scapy.layers.inet import IP
    except ImportError:
        print("ERROR: Scapy required. Install: pip install scapy")
        return

    print()
    print("=" * 72)
    print("  PCAP-BASED ML BENCHMARK")
    print("  Quantum Sniffer v4.0 — Flow Feature Extractor + iForest/EIF")
    print("=" * 72)
    print(f"  PCAP:        {os.path.basename(pcap_path)}")
    print(f"  Day:         {day}")
    print(f"  Window:      {window_seconds}s")
    print(f"  Max packets: {max_packets or 'all'}")
    print(f"  Models:      iForest (t={n_estimators}), EIF (t={n_estimators})")
    print(f"  Contam.:     {contamination}")
    print()

    # ── Phase 1: Stream packets and extract flow features per window ──
    print("  [1/4] Streaming packets and extracting flow features...")
    extractor = FlowFeatureExtractor(
        window_seconds=window_seconds,
        normalize=False,
        min_packets_per_flow=2,
    )

    all_window_features = []     # List of (features_matrix, window_midtime)
    window_start_ts = None
    packets_read = 0
    packets_skipped = 0

    t0 = time.time()

    with PcapReader(str(pcap_path)) as reader:
        for pkt in reader:
            if max_packets and packets_read >= max_packets:
                break

            try:
                if not pkt.haslayer(IP):
                    packets_skipped += 1
                    continue

                ts = float(pkt.time)

                if window_start_ts is None:
                    window_start_ts = ts

                # Feed to extractor
                key = extractor.ingest_packet(pkt)
                packets_read += 1

                # Check window boundary
                if ts - window_start_ts >= window_seconds:
                    features, keys = extractor.flush_window()
                    if features.shape[0] > 0:
                        window_mid = window_start_ts + window_seconds / 2
                        all_window_features.append((features, window_mid))
                    window_start_ts = ts

                # Progress
                if packets_read % 100_000 == 0:
                    elapsed = time.time() - t0
                    pps = packets_read / max(elapsed, 0.1)
                    n_windows = len(all_window_features)
                    n_flows = sum(f.shape[0] for f, _ in all_window_features)
                    print(f"    ... {packets_read:>10,} packets | "
                          f"{n_windows:>5} windows | "
                          f"{n_flows:>7,} flows | "
                          f"{pps:>8,.0f} pkt/s")

            except Exception:
                packets_skipped += 1

    # Flush last window
    features, keys = extractor.flush_window()
    if features.shape[0] > 0:
        window_mid = (window_start_ts or 0) + window_seconds / 2
        all_window_features.append((features, window_mid))

    read_time = time.time() - t0
    total_flows = sum(f.shape[0] for f, _ in all_window_features)

    print(f"\n  Packet reading complete:")
    print(f"    Packets read:    {packets_read:,}")
    print(f"    Packets skipped: {packets_skipped:,}")
    print(f"    Windows:         {len(all_window_features)}")
    print(f"    Total flows:     {total_flows:,}")
    print(f"    Read time:       {read_time:.1f}s ({packets_read/max(read_time,0.1):,.0f} pkt/s)")
    print()

    if not all_window_features:
        print("  ERROR: No flow features extracted. Aborting.")
        return

    # ── Phase 2: Build training set from benign-ONLY windows ──
    # FIX: Train exclusively on benign traffic. Isolation Forest is a
    # one-class learner — it must learn what "normal" looks like.
    # Training on mixed traffic (73% attacks on Friday) causes the
    # model to learn attacks as normal and flag benign as anomalous.
    print("  [2/4] Building training set from benign-only windows...")

    # Label each window by its midpoint timestamp
    window_labels = []
    for features, mid_ts in all_window_features:
        label = classify_timestamp(mid_ts, day)
        window_labels.append(label)

    # Collect ALL benign windows for training (one-class baseline)
    train_features = []
    n_train_windows = 0
    for i, (features, mid_ts) in enumerate(all_window_features):
        if window_labels[i] == "BENIGN":
            train_features.append(features)
            n_train_windows += 1

    if not train_features:
        # Fallback: use first 10 windows regardless of label
        print("  WARNING: No pure benign windows found in expected time range.")
        print("           Using first 10 windows as training baseline.")
        train_features = [f for f, _ in all_window_features[:10]]
        n_train_windows = len(train_features)

    X_train = np.vstack(train_features)
    print(f"  Training set: {n_train_windows} windows, {X_train.shape[0]:,} flows")

    # ── Phase 3: Normalize + Train models ──
    print("  [3/4] Training models...")

    # Z-score normalization
    feat_means = X_train.mean(axis=0)
    feat_stds = X_train.std(axis=0)
    feat_stds[feat_stds < 1e-10] = 1.0
    X_train_norm = (X_train - feat_means) / feat_stds

    # Isolation Forest
    t0 = time.time()
    iforest = IsolationForest(
        n_estimators=n_estimators,
        max_samples=min(max_samples, X_train_norm.shape[0]),
        contamination=contamination,
        random_state=42,
    )
    iforest.fit(X_train_norm)
    if_train_time = time.time() - t0
    print(f"    iForest trained in {if_train_time:.2f}s")

    # Extended Isolation Forest
    t0 = time.time()
    eif = ExtendedIsolationForest(
        n_estimators=n_estimators,
        max_samples=min(max_samples, X_train_norm.shape[0]),
        contamination=contamination,
        random_state=42,
    )
    eif.fit(X_train_norm)
    eif_train_time = time.time() - t0
    print(f"    EIF trained in {eif_train_time:.2f}s")
    print()

    # ── Phase 4: Score all windows and evaluate ──
    print("  [4/4] Scoring all flows and evaluating...")

    all_if_scores = []
    all_eif_scores = []
    all_labels = []      # Per-flow labels
    all_label_strs = []  # Per-flow attack type strings

    t0 = time.time()
    for features, mid_ts in all_window_features:
        label_str = classify_timestamp(mid_ts, day)
        y_label = 0 if label_str == "BENIGN" else 1

        X_norm = (features - feat_means) / feat_stds

        if_scores = iforest.anomaly_scores(X_norm)
        eif_scores = eif.anomaly_scores(X_norm)

        all_if_scores.append(if_scores)
        all_eif_scores.append(eif_scores)
        all_labels.extend([y_label] * features.shape[0])
        all_label_strs.extend([label_str] * features.shape[0])

    score_time = time.time() - t0

    if_scores = np.concatenate(all_if_scores)
    eif_scores = np.concatenate(all_eif_scores)
    y_true = np.array(all_labels)

    # Combined score (Fix 3 weights: EIF=0.50, iF=0.15, no AE here)
    # Redistribute AE weight (0.35) → EIF gets 70%, iF gets 30%
    combined_scores = 0.7385 * eif_scores + 0.2615 * if_scores
    combined_scores = np.clip(combined_scores, 0, 1)

    n_attack = int(y_true.sum())
    n_benign = len(y_true) - n_attack
    print(f"  Total flows scored: {len(y_true):,}")
    print(f"  Benign: {n_benign:,} | Attack: {n_attack:,}")
    print(f"  Score time: {score_time:.2f}s ({len(y_true)/max(score_time,0.01):,.0f} flows/s)")
    print()

    # ── Results ──
    print("=" * 72)
    print("  RESULTS")
    print("=" * 72)
    print()

    models = {
        "IsolationForest": if_scores,
        "ExtendedIForest": eif_scores,
        "Combined(EIF+iF)": combined_scores,
    }

    model_objects = {
        "IsolationForest": iforest,
        "ExtendedIForest": eif,
    }

    # ── Score Distribution Diagnostics ──
    benign_mask = y_true == 0
    attack_mask = y_true == 1
    print("  SCORE DISTRIBUTION DIAGNOSTICS:")
    print("  " + "-" * 72)
    any_inverted = False
    for name, scores in models.items():
        b_mean = np.mean(scores[benign_mask]) if benign_mask.any() else 0
        a_mean = np.mean(scores[attack_mask]) if attack_mask.any() else 0
        b_std = np.std(scores[benign_mask]) if benign_mask.any() else 0
        a_std = np.std(scores[attack_mask]) if attack_mask.any() else 0
        sep = a_mean - b_mean
        # Cohen's d effect size
        pooled_std = math.sqrt((b_std**2 + a_std**2) / 2) if (b_std + a_std) > 0 else 1.0
        cohens_d = sep / pooled_std
        auc = compute_auc_roc(y_true, scores)
        status = "✓ OK" if sep > 0.01 else ("⚠ WEAK" if sep > 0 else "✗ INVERTED")
        if sep <= 0:
            any_inverted = True
        print(f"    {name:20s}: benign={b_mean:.4f}±{b_std:.4f}, "
              f"attack={a_mean:.4f}±{a_std:.4f}, "
              f"sep={sep:+.4f}, d={cohens_d:+.3f}, AUC={auc:.4f}  [{status}]")
    print("  " + "-" * 72)

    if any_inverted:
        print()
        print("  ⚠ WARNING: Score distributions are INVERTED or overlapping!")
        print("    Benign traffic scores HIGHER (more anomalous) than attack traffic.")
        print("    This means the model cannot distinguish attacks from normal traffic")
        print("    in the current feature space. Possible causes:")
        print("      1. Flow features from PCAP don't capture attack signatures")
        print("      2. Time-window labeling may be misaligned with actual traffic")
        print("      3. Attack traffic (Botnet/PortScan) may have regular patterns")
        print("         that look 'normal' to the Isolation Forest")
        print("    → Consider using CSV-based benchmark with per-flow labels instead.")
        print()

    # ── Per-Model Results ──
    # Report with BOTH model-calibrated threshold AND Youden's J optimal
    for model_name, scores in models.items():
        # Youden's J optimal threshold (test-set oracle)
        thresh_j = find_best_threshold(scores, y_true)
        y_pred_j = (scores >= thresh_j).astype(int)
        m_j = compute_metrics(y_true, y_pred_j)

        # Model's own contamination-calibrated threshold (operational)
        if model_name in model_objects:
            model_thresh = model_objects[model_name].threshold
        else:
            # Combined: use weighted average of component thresholds
            model_thresh = 0.7385 * eif.threshold + 0.2615 * iforest.threshold

        y_pred_m = (scores >= model_thresh).astype(int)
        m_m = compute_metrics(y_true, y_pred_m)

        auc = compute_auc_roc(y_true, scores)

        print(f"  --- {model_name} ---")
        print(f"    AUC-ROC:    {auc:.4f}")
        print()
        print(f"    [Model Threshold = {model_thresh:.4f}]  (contamination-calibrated)")
        print(f"      Recall:     {m_m['recall']:.4f}")
        print(f"      Precision:  {m_m['precision']:.4f}")
        print(f"      F1:         {m_m['f1']:.4f}")
        print(f"      MCC:        {m_m['mcc']:.4f}")
        print(f"      FPR:        {m_m['fpr']:.4f}")
        print(f"      TP={m_m['tp']:,} FP={m_m['fp']:,} FN={m_m['fn']:,} TN={m_m['tn']:,}")
        print()
        print(f"    [Youden's J Threshold = {thresh_j:.4f}]  (best separation)")
        print(f"      Recall:     {m_j['recall']:.4f}")
        print(f"      Precision:  {m_j['precision']:.4f}")
        print(f"      F1:         {m_j['f1']:.4f}")
        print(f"      MCC:        {m_j['mcc']:.4f}")
        print(f"      FPR:        {m_j['fpr']:.4f}")
        print(f"      TP={m_j['tp']:,} FP={m_j['fp']:,} FN={m_j['fn']:,} TN={m_j['tn']:,}")
        print()

    # Per-attack breakdown (using Combined model threshold)
    best_scores = combined_scores
    combined_model_thresh = 0.7385 * eif.threshold + 0.2615 * iforest.threshold
    best_pred = (best_scores >= combined_model_thresh).astype(int)

    categories = sorted(set(all_label_strs))
    print("  PER-ATTACK BREAKDOWN (Combined EIF+iF, model threshold):")
    print("  " + "-" * 72)
    print(f"  {'Category':<20s} {'Total':>8s} {'Detect':>8s} {'Missed':>8s} {'Recall':>8s} {'AvgScore':>9s}")
    print("  " + "-" * 72)

    for cat in categories:
        mask = np.array([l == cat for l in all_label_strs])
        cat_pred = best_pred[mask]
        cat_scores = best_scores[mask]
        n = int(mask.sum())

        if cat == "BENIGN":
            correct = int(np.sum(cat_pred == 0))  # True negatives
            false_pos = n - correct
            print(f"  {cat:<20s} {n:>8,} {correct:>8,} {false_pos:>8,} "
                  f"{correct/max(n,1):>8.4f} {np.mean(cat_scores):>9.4f}")
        else:
            detected = int(np.sum(cat_pred == 1))
            missed = n - detected
            print(f"  {cat:<20s} {n:>8,} {detected:>8,} {missed:>8,} "
                  f"{detected/max(n,1):>8.4f} {np.mean(cat_scores):>9.4f}")

    print("  " + "-" * 72)
    print()
    print("=" * 72)
    print("  Benchmark complete.")
    print("=" * 72)


# ──────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="PCAP-based ML Benchmark")
    parser.add_argument("--pcap", required=True, help="Path to CICIDS2017 PCAP")
    parser.add_argument("--day", choices=["friday", "tuesday"], default="friday",
                        help="Which day (for ground truth labeling)")
    parser.add_argument("--max-packets", type=int, default=None,
                        help="Max packets to read (default: all)")
    parser.add_argument("--window", type=float, default=30.0,
                        help="Window duration in seconds (default: 30)")
    parser.add_argument("--n-estimators", type=int, default=200,
                        help="Trees per forest (default: 200)")
    parser.add_argument("--contamination", type=float, default=0.015,
                        help="Contamination (default: 0.015)")
    args = parser.parse_args()

    run_pcap_benchmark(
        pcap_path=args.pcap,
        day=args.day,
        window_seconds=args.window,
        max_packets=args.max_packets,
        n_estimators=args.n_estimators,
        contamination=args.contamination,
    )
