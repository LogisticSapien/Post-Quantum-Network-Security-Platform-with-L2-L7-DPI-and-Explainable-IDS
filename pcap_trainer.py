"""
PCAP Model Trainer & Evaluator
================================
Trains and evaluates the Isolation Forest model on real Wireshark
capture files (PCAP/PCAPNG). This is a TESTING/EVALUATION tool —
after training + evaluation, the model continues to take live input
from the sniffer engine as before.

Usage:
  python pcap_trainer.py Friday-WorkingHours.pcap
  python pcap_trainer.py capture.pcap --window 15 --trees 200
  python __main__.py --train-pcap Friday-WorkingHours.pcap

What this does:
  1. Streams packets from the PCAP (memory-efficient, works with multi-GB files)
  2. Extracts 14-feature time-windowed vectors using WindowAccumulator
  3. Trains the from-scratch Isolation Forest on extracted features
  4. Scores every window and reports detection results
  5. Saves model baseline (means, stds, threshold) to JSON for reference
  6. Generates score distribution plot

The sniffer engine's live pipeline is NOT modified — this is purely
for offline testing/benchmarking against real captured traffic.
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
import pickle
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

# Ensure our modules are importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from scapy.all import PcapReader, Ether, IP, TCP, UDP, ICMP, DNS
    HAS_SCAPY = True
except ImportError:
    HAS_SCAPY = False

from isolation_forest import IsolationForest
from iforest_detector import (
    IForestNetworkDetector,
    WindowAccumulator,
    FEATURE_NAMES,
    NUM_FEATURES,
    _shannon_entropy,
)
from datetime import datetime, timezone, timedelta


# ──────────────────────────────────────────────────────────────────────
# CICIDS2017 Ground Truth Time Windows (for benign-only training)
# ──────────────────────────────────────────────────────────────────────

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
    Assumes CICIDS2017 EDT (UTC-4) timezone.
    """
    if not day:
        return "BENIGN"  # Assume everything is normal if no day context

    dt = datetime.fromtimestamp(ts, tz=timezone(timedelta(hours=-4)))
    sod = dt.hour * 3600 + dt.minute * 60 + dt.second

    attacks = FRIDAY_ATTACKS if day.lower() == "friday" else TUESDAY_ATTACKS

    for start_h, start_m, end_h, end_m, label in attacks:
        start = _time_to_seconds(start_h, start_m)
        end = _time_to_seconds(end_h, end_m)
        if start <= sod <= end:
            return label

    return "BENIGN"


# ──────────────────────────────────────────────────────────────────────
# PCAP Feature Extractor
# ──────────────────────────────────────────────────────────────────────

class PcapFeatureExtractor:
    """Extract time-windowed feature vectors from a PCAP file.

    Streams packets through a WindowAccumulator and emits a
    14-feature vector every `window_seconds`.
    """

    def __init__(self, window_seconds: float = 30.0, verbose: bool = True):
        self.window_seconds = window_seconds
        self.verbose = verbose

    def extract(self, pcap_path: str) -> Tuple[np.ndarray, List[dict]]:
        """Extract feature vectors from a PCAP file.

        Args:
            pcap_path: Path to .pcap/.pcapng file.

        Returns:
            (feature_matrix, window_metadata) where:
              feature_matrix: shape (n_windows, 14)
              window_metadata: list of dicts with timing info per window
        """
        if not HAS_SCAPY:
            raise RuntimeError("Scapy not installed. pip install scapy")

        path = Path(pcap_path)
        if not path.exists():
            raise FileNotFoundError(f"PCAP file not found: {pcap_path}")

        file_size = path.stat().st_size
        print(f"  File: {path.name}")
        print(f"  Size: {file_size / (1024**2):.1f} MB")
        print(f"  Window: {self.window_seconds}s")
        print()

        accumulator = WindowAccumulator()
        features_list: List[np.ndarray] = []
        metadata_list: List[dict] = []

        packets_processed = 0
        packets_failed = 0
        window_start_ts = None
        t0 = time.time()

        try:
            with PcapReader(str(path)) as reader:
                for pkt in reader:
                    try:
                        pkt_ts = float(pkt.time) if hasattr(pkt, 'time') else 0.0

                        # Initialize window start from first packet timestamp
                        if window_start_ts is None:
                            window_start_ts = pkt_ts
                            accumulator.start_time = time.time()

                        # Check if window elapsed (using PCAP timestamps)
                        elapsed_pcap = pkt_ts - window_start_ts
                        if elapsed_pcap >= self.window_seconds and accumulator.packet_count > 0:
                            # Extract features for this window
                            fv = accumulator.to_feature_vector(elapsed_pcap)
                            features_list.append(fv)
                            metadata_list.append({
                                "window_idx": len(features_list) - 1,
                                "start_ts": window_start_ts,
                                "end_ts": pkt_ts,
                                "duration": elapsed_pcap,
                                "packet_count": accumulator.packet_count,
                                "total_bytes": accumulator.total_bytes,
                            })

                            # Reset for next window
                            accumulator.reset()
                            window_start_ts = pkt_ts

                        # Extract packet metadata and record
                        self._record_packet(pkt, accumulator)
                        packets_processed += 1

                    except Exception:
                        packets_failed += 1

                    # Progress reporting
                    if self.verbose and packets_processed % 50000 == 0 and packets_processed > 0:
                        elapsed_wall = time.time() - t0
                        pps = packets_processed / max(elapsed_wall, 0.001)
                        print(f"    ... {packets_processed:>10,} packets | "
                              f"{len(features_list):>5} windows | "
                              f"{pps:,.0f} pkt/s | "
                              f"{packets_failed} failed")

            # Flush final window
            if accumulator.packet_count > 0 and window_start_ts is not None:
                elapsed_pcap = max(self.window_seconds, 1.0)
                fv = accumulator.to_feature_vector(elapsed_pcap)
                features_list.append(fv)
                metadata_list.append({
                    "window_idx": len(features_list) - 1,
                    "start_ts": window_start_ts,
                    "end_ts": window_start_ts + elapsed_pcap,
                    "duration": elapsed_pcap,
                    "packet_count": accumulator.packet_count,
                    "total_bytes": accumulator.total_bytes,
                })

        except Exception as e:
            print(f"  ERROR reading PCAP: {e}")
            if not features_list:
                raise

        elapsed_total = time.time() - t0
        print()
        print(f"  Extraction complete:")
        print(f"    Packets processed: {packets_processed:,}")
        print(f"    Packets failed:    {packets_failed:,}")
        print(f"    Windows extracted: {len(features_list)}")
        print(f"    Wall time:         {elapsed_total:.1f}s")
        print(f"    Throughput:        {packets_processed / max(elapsed_total, 0.001):,.0f} pkt/s")

        if not features_list:
            raise ValueError("No feature windows extracted from PCAP")

        X = np.vstack(features_list)
        return X, metadata_list

    @staticmethod
    def _record_packet(pkt, accumulator: WindowAccumulator):
        """Extract metadata from Scapy packet and record it."""
        size = len(pkt)

        # Try to get IP layer
        if not pkt.haslayer(IP):
            return

        ip_layer = pkt[IP]
        src_ip = ip_layer.src
        dst_ip = ip_layer.dst
        protocol = "OTHER"
        src_port = 0
        dst_port = 0
        is_syn = False

        # TCP
        if pkt.haslayer(TCP):
            tcp_layer = pkt[TCP]
            protocol = "TCP"
            src_port = tcp_layer.sport
            dst_port = tcp_layer.dport
            is_syn = bool(tcp_layer.flags & 0x02)  # SYN flag

            # Check for common application protocols
            if dst_port == 53 or src_port == 53:
                protocol = "DNS"
            elif dst_port in (80, 8080, 8000):
                protocol = "HTTP"
            elif dst_port in (443, 8443):
                protocol = "TLS"
            elif dst_port == 22:
                protocol = "SSH"

        # UDP
        elif pkt.haslayer(UDP):
            udp_layer = pkt[UDP]
            protocol = "UDP"
            src_port = udp_layer.sport
            dst_port = udp_layer.dport

            if dst_port == 53 or src_port == 53:
                protocol = "DNS"

        # ICMP
        elif pkt.haslayer(ICMP):
            protocol = "ICMP"

        accumulator.record(
            protocol=protocol,
            src_ip=src_ip,
            dst_ip=dst_ip,
            size=size,
            src_port=src_port,
            dst_port=dst_port,
            is_syn=is_syn,
        )


# ──────────────────────────────────────────────────────────────────────
# PCAP Trainer
# ──────────────────────────────────────────────────────────────────────

class PcapTrainer:
    """Train and evaluate the Isolation Forest on PCAP capture data.

    This is a TESTING tool. The live sniffer engine pipeline is
    not modified — it continues to feed the model in real-time
    via IForestNetworkDetector.record_packet() as before.

    Args:
        n_estimators: Number of isolation trees.
        max_samples: Subsample size per tree.
        contamination: Expected anomaly ratio.
        window_seconds: Feature extraction window duration.
    """

    def __init__(
        self,
        n_estimators: int = 200,
        max_samples: int = 512,
        contamination: float = 0.015,
        window_seconds: float = 30.0,
    ):
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.contamination = contamination
        self.window_seconds = window_seconds

    def train(self, pcap_path: str, output_dir: str = "pcap_training_results", day: Optional[str] = None) -> dict:
        """Train iForest on PCAP and evaluate detection quality.

        Steps:
          1. Extract features from PCAP
          2. Filter for benign windows (if day provided)
          3. Normalize features (z-score)
          4. Train Isolation Forest
          5. Score all windows
          6. Report results and save baseline

        Args:
            pcap_path: Path to .pcap/.pcapng file.
            output_dir: Directory for output files.
            day: "friday", "tuesday", or None (if none, all windows used).

        Returns:
            Results dictionary.
        """
        print()
        print("=" * 64)
        print("  ISOLATION FOREST — PCAP TRAINING & EVALUATION")
        print("  Testing model on real Wireshark capture data")
        print("=" * 64)
        print()

        # ── Step 1: Extract features ──
        cache_path = os.path.join(output_dir, "features_cache.pkl")
        if os.path.exists(cache_path):
            print(f"  [1/5] Loading features from cache: {cache_path}")
            with open(cache_path, 'rb') as f:
                X, metadata = pickle.load(f)
        else:
            print("  [1/5] Extracting features from PCAP (this may take time)...")
            extractor = PcapFeatureExtractor(
                window_seconds=self.window_seconds,
                verbose=True,
            )
            X, metadata = extractor.extract(pcap_path)
            
            # Save cache
            os.makedirs(output_dir, exist_ok=True)
            with open(cache_path, 'wb') as f:
                pickle.dump((X, metadata), f)
            print(f"    Saved features cache -> {cache_path}")

        n_windows = X.shape[0]
        print(f"  Feature matrix: {X.shape} ({n_windows} windows x {NUM_FEATURES} features)")
        print()

        # ── Step 2: Build training set from benign-ONLY windows ──
        # FIX: Train exclusively on benign traffic. Isolation Forest is a
        # one-class learner — it must learn what "normal" looks like.
        # Training on mixed traffic (73% attacks on Friday) causes the
        # model to learn attacks as normal and flag benign as anomalous.
        if day:
            print(f"  [2/5] Filtering for benign-only training windows ({day})...")
            benign_features = []
            for i, features in enumerate(X):
                mid_ts = metadata[i]['start_time'] + (metadata[i]['duration'] / 2)
                if classify_timestamp(mid_ts, day) == "BENIGN":
                    benign_features.append(features)
            
            if not benign_features:
                print("    WARNING: No pure benign windows found. Falling back to all windows.")
                X_train_src = X
            else:
                X_train_src = np.vstack(benign_features)
                print(f"    Selected {len(X_train_src)} benign windows for training baseline.")
        else:
            print("  [2/5] Using all windows for training (no day filter)...")
            X_train_src = X

        # ── Step 3: Normalize ──
        print("  [3/5] Z-score normalization...")
        X_mean = X_train_src.mean(axis=0)
        X_std = X_train_src.std(axis=0)
        X_std[X_std < 1e-10] = 1.0
        X_norm = (X - X_mean) / X_std
        X_train_norm = (X_train_src - X_mean) / X_std

        print(f"    Feature means: min={X_mean.min():.1f}, max={X_mean.max():.1f}")
        print(f"    Feature stds:  min={X_std.min():.2f}, max={X_std.max():.2f}")
        print()

        # ── Step 3: Train ──
        print(f"  [3/5] Training Isolation Forest (t={self.n_estimators}, "
              f"psi={self.max_samples}, c={self.contamination})...")
        t0 = time.time()
        forest = IsolationForest(
            n_estimators=self.n_estimators,
            max_samples=self.max_samples,
            contamination=self.contamination,
            random_state=42,
        )
        forest.fit(X_train_norm)
        train_time = time.time() - t0

        print(f"    Trees:           {forest.n_estimators}")
        print(f"    Subsample size:  {forest._psi}")
        print(f"    Height limit:    {math.ceil(math.log2(max(forest._psi, 2)))}")
        print(f"    Default thresh:  {forest.threshold:.4f}")
        print(f"    Training time:   {train_time:.3f}s")
        print()

        # ── Step 4: Score ──
        print("  [4/5] Scoring all windows...")
        t0 = time.time()
        scores = forest.anomaly_scores(X_norm)
        score_time = time.time() - t0

        labels = forest.predict(X_norm)
        n_anomalous = int(labels.sum())
        n_normal = n_windows - n_anomalous

        print(f"    Scoring time:    {score_time:.3f}s")
        print(f"    Score range:     [{scores.min():.4f}, {scores.max():.4f}]")
        print(f"    Score mean:      {scores.mean():.4f}")
        print(f"    Score std:       {scores.std():.4f}")
        print()
        print(f"    Normal windows:  {n_normal} ({n_normal/n_windows*100:.1f}%)")
        print(f"    Anomalous windows: {n_anomalous} ({n_anomalous/n_windows*100:.1f}%)")
        print()

        # ── Step 5: Detailed results ──
        print("  [5/5] Detection results...")
        print()

        # Top anomalous windows
        top_indices = np.argsort(scores)[::-1][:20]
        print("  TOP 20 MOST ANOMALOUS WINDOWS:")
        print("  " + "-" * 58)
        print(f"  {'Window':<8} {'Score':>7} {'Packets':>9} "
              f"{'Bytes':>12} {'Duration':>10}")
        print("  " + "-" * 58)

        for idx in top_indices:
            meta = metadata[idx]
            print(f"  {meta['window_idx']:<8} {scores[idx]:>7.4f} "
                  f"{meta['packet_count']:>9,} "
                  f"{meta['total_bytes']:>12,} "
                  f"{meta['duration']:>9.1f}s")

        print()

        # Feature statistics for anomalous windows
        if n_anomalous > 0:
            anomalous_mask = labels == 1
            normal_mask = labels == 0

            print("  FEATURE COMPARISON (anomalous vs normal):")
            print("  " + "-" * 62)
            print(f"  {'Feature':<22} {'Normal Mean':>12} {'Anomaly Mean':>13} {'Ratio':>7}")
            print("  " + "-" * 62)

            for i, name in enumerate(FEATURE_NAMES):
                normal_mean = X[normal_mask, i].mean() if n_normal > 0 else 0
                anomaly_mean = X[anomalous_mask, i].mean() if n_anomalous > 0 else 0
                ratio = anomaly_mean / max(normal_mean, 1e-10)
                flag = " [WARN]" if abs(ratio - 1.0) > 0.5 else ""
                print(f"  {name:<22} {normal_mean:>12.2f} "
                      f"{anomaly_mean:>13.2f} {ratio:>6.2f}x{flag}")

            print()

        # Score distribution stats
        percentiles = [10, 25, 50, 75, 90, 95, 99]
        print("  SCORE DISTRIBUTION:")
        for p in percentiles:
            val = np.percentile(scores, p)
            print(f"    P{p:<3}: {val:.4f}")
        print()

        # ── Save results ──
        os.makedirs(output_dir, exist_ok=True)

        # Save model baseline
        baseline = {
            "pcap_file": os.path.basename(pcap_path),
            "pcap_size_bytes": Path(pcap_path).stat().st_size,
            "window_seconds": self.window_seconds,
            "n_windows": n_windows,
            "n_anomalous": n_anomalous,
            "model_params": forest.get_params(),
            "feature_names": FEATURE_NAMES,
            "feature_means": X_mean.tolist(),
            "feature_stds": X_std.tolist(),
            "score_stats": {
                "min": float(scores.min()),
                "max": float(scores.max()),
                "mean": float(scores.mean()),
                "std": float(scores.std()),
                "threshold": float(forest.threshold),
            },
            "train_time": train_time,
            "score_time": score_time,
        }

        baseline_path = os.path.join(output_dir, "model_baseline.json")
        with open(baseline_path, 'w') as f:
            json.dump(baseline, f, indent=2)
        print(f"  Saved model baseline -> {baseline_path}")

        # Save per-window scores
        scores_path = os.path.join(output_dir, "window_scores.csv")
        with open(scores_path, 'w') as f:
            f.write("window_idx,score,label,packets,bytes,duration\n")
            for i in range(n_windows):
                meta = metadata[i]
                f.write(f"{i},{scores[i]:.6f},{labels[i]},"
                        f"{meta['packet_count']},{meta['total_bytes']},"
                        f"{meta['duration']:.2f}\n")
        print(f"  Saved window scores  -> {scores_path}")

        # Generate plot
        self._generate_plot(scores, labels, forest.threshold, output_dir)

        # Summary
        print()
        print("=" * 64)
        print("  TRAINING COMPLETE")
        print("=" * 64)
        print(f"  Windows:    {n_windows} (from {os.path.basename(pcap_path)})")
        print(f"  Normal:     {n_normal} ({n_normal/n_windows*100:.1f}%)")
        print(f"  Anomalous:  {n_anomalous} ({n_anomalous/n_windows*100:.1f}%)")
        print(f"  Threshold:  {forest.threshold:.4f}")
        print(f"  AUC (est):  -  (no ground-truth labels in raw PCAP)")
        print()
        print("  NOTE: This was a TEST/EVALUATION run. The sniffer engine's")
        print("  live pipeline (engine.py → iforest_detector) is unchanged.")
        print("  The model still takes input from the sniffer in real-time.")
        print("=" * 64)
        print()

        return {
            "baseline": baseline,
            "scores": scores,
            "labels": labels,
            "features": X,
            "metadata": metadata,
        }

    def _generate_plot(
        self,
        scores: np.ndarray,
        labels: np.ndarray,
        threshold: float,
        output_dir: str,
    ):
        """Generate score distribution plot."""
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        except ImportError:
            print("  [!] matplotlib not installed, skipping plot")
            return

        COLORS = {
            "normal": "#2ecc71",
            "anomaly": "#e74c3c",
            "background": "#1a1a2e",
            "surface": "#16213e",
            "text": "#eaeaea",
            "grid": "#333366",
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
        })

        fig, ax = plt.subplots(figsize=(10, 6))

        normal_scores = scores[labels == 0]
        anomaly_scores = scores[labels == 1]

        ax.hist(normal_scores, bins=50, alpha=0.7, color=COLORS['normal'],
                label=f'Normal (n={len(normal_scores)})', edgecolor='none')
        if len(anomaly_scores) > 0:
            ax.hist(anomaly_scores, bins=50, alpha=0.7, color=COLORS['anomaly'],
                    label=f'Anomalous (n={len(anomaly_scores)})', edgecolor='none')

        ax.axvline(x=threshold, color='#f39c12', linestyle='--',
                   linewidth=2, label=f'Threshold ({threshold:.3f})')

        ax.set_xlabel('Anomaly Score', fontsize=13)
        ax.set_ylabel('Frequency', fontsize=13)
        ax.set_title('iForest Score Distribution — PCAP Training',
                      fontsize=15, fontweight='bold')
        ax.legend(fontsize=11, facecolor=COLORS['surface'],
                  edgecolor=COLORS['grid'])
        ax.grid(True, alpha=0.2)

        plt.tight_layout()
        plot_path = os.path.join(output_dir, 'score_distribution.png')
        fig.savefig(plot_path, dpi=150, facecolor=COLORS['background'],
                    bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved score plot     -> {plot_path}")


# ──────────────────────────────────────────────────────────────────────
# CLI Entry Point
# ──────────────────────────────────────────────────────────────────────

def main():
    """CLI entry point for PCAP training."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Train & evaluate Isolation Forest on Wireshark PCAP capture"
    )
    parser.add_argument("pcap", help="Path to .pcap/.pcapng file")
    parser.add_argument("--window", type=float, default=30.0,
                        help="Feature window duration in seconds (default: 30)")
    parser.add_argument("--trees", type=int, default=200,
                        help="Number of isolation trees (default: 200)")
    parser.add_argument("--psi", type=int, default=512,
                        help="Subsample size per tree (default: 512)")
    parser.add_argument("--contamination", type=float, default=0.015,
                        help="Expected anomaly ratio (default: 0.10)")
    parser.add_argument("--output-dir", default="pcap_training_results",
                        help="Output directory (default: pcap_training_results)")
    parser.add_argument("--day", help="Benchmark day (friday, tuesday) for filtering labels")
    args = parser.parse_args()

    trainer = PcapTrainer(
        n_estimators=args.trees,
        max_samples=args.psi,
        contamination=args.contamination,
        window_seconds=args.window,
    )
    trainer.train(args.pcap, output_dir=args.output_dir, day=args.day)


if __name__ == "__main__":
    main()
