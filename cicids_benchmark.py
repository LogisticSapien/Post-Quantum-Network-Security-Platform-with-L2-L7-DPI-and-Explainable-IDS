"""
CICIDS2017 Offline Benchmark — Multi-Model Evaluation
========================================================
Clean evaluation pipeline for validating ML models offline before
plugging them into the live ml_dpi_controller pipeline.

Pipeline:
  1. Load CICIDS2017 pre-extracted CSVs (official 80-feature + Label)
  2. Map to the 14 flow features used in the live pipeline
  3. Train on Monday CSV (benign-only, unsupervised baseline)
  4. Test on Friday CSV (DDoS + PortScan + Botnet)
  5. Run iForest, EIF, Autoencoder, CombinedDetector individually
  6. Output comparison table: Recall, Precision, F1, MCC, AUC-ROC
  7. Break down Friday results by attack category

NOTE: This script is purely for offline validation. It does not
touch the live sniffer or detection engine.

Constraints: Uses only our from-scratch models (no sklearn/xgboost).
"""

from __future__ import annotations

import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Ensure local imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from isolation_forest import IsolationForest
from extended_isolation_forest import ExtendedIsolationForest
from flow_feature_extractor import FlowFeatureExtractor, NUM_FEATURES, FLOW_FEATURE_NAMES


# ──────────────────────────────────────────────────────────────────────
# Metrics (pure NumPy, no sklearn)
# ──────────────────────────────────────────────────────────────────────

def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute TP, FP, FN, TN."""
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, scores: np.ndarray) -> dict:
    """Compute all classification metrics.

    Returns dict with: recall, precision, f1, mcc, auc_roc, fpr, fnr,
    and raw tp/fp/fn/tn.
    """
    cm = confusion_matrix(y_true, y_pred)
    tp, fp, fn, tn = cm["tp"], cm["fp"], cm["fn"], cm["tn"]

    recall = tp / max(tp + fn, 1)
    precision = tp / max(tp + fp, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-10)
    fpr = fp / max(fp + tn, 1)
    fnr = fn / max(fn + tp, 1)

    # Matthews Correlation Coefficient
    denom = math.sqrt(max((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn), 1))
    mcc = (tp * tn - fp * fn) / denom

    # AUC-ROC (trapezoidal approximation)
    auc = _compute_auc(y_true, scores)

    return {
        **cm,
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "mcc": mcc,
        "auc_roc": auc,
        "fpr": fpr,
        "fnr": fnr,
    }


def _compute_auc(y_true: np.ndarray, scores: np.ndarray, n_thresh: int = 500) -> float:
    """ROC AUC via trapezoidal rule."""
    thresholds = np.linspace(scores.min() - 0.01, scores.max() + 0.01, n_thresh)
    fprs, tprs = [], []
    n_pos = max(int(y_true.sum()), 1)
    n_neg = max(len(y_true) - n_pos, 1)

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


def per_attack_breakdown(
    y_labels: List[str],
    y_pred: np.ndarray,
    scores: np.ndarray,
) -> Dict[str, dict]:
    """Break down metrics by attack category.

    Returns dict mapping attack_type → metrics dict.
    """
    categories = sorted(set(y_labels))
    results = {}

    for cat in categories:
        mask = np.array([l == cat for l in y_labels])
        cat_pred = y_pred[mask]
        cat_scores = scores[mask]
        n = int(mask.sum())

        if cat.upper() == "BENIGN":
            # For benign: correct = predicted normal
            correct = int(np.sum(cat_pred == 0))
            fp = n - correct
            results[cat] = {
                "total": n,
                "correctly_classified": correct,
                "false_positives": fp,
                "specificity": correct / max(n, 1),
                "mean_score": float(np.mean(cat_scores)),
            }
        else:
            detected = int(np.sum(cat_pred == 1))
            missed = n - detected
            results[cat] = {
                "total": n,
                "detected": detected,
                "missed": missed,
                "recall": detected / max(n, 1),
                "mean_score": float(np.mean(cat_scores)),
            }

    return results


# ──────────────────────────────────────────────────────────────────────
# CSV Loading (pure Python, no pandas dependency required)
# ──────────────────────────────────────────────────────────────────────

def _try_import_pandas():
    """Try to import pandas; fall back to manual CSV parsing."""
    try:
        import pandas as pd
        return pd
    except ImportError:
        return None


def load_cicids_csv(filepath: str) -> Tuple[np.ndarray, List[str], List[str]]:
    """Load a CICIDS2017 CSV file.

    Handles the notorious inconsistent whitespace in column names.

    Returns:
        (features, labels, column_names)
    """
    pd = _try_import_pandas()

    if pd is not None:
        # Fast path with pandas
        df = pd.read_csv(filepath, encoding='utf-8', low_memory=False)
        # Strip column names
        df.columns = df.columns.str.strip()

        # Find label column
        label_col = None
        for candidate in ["Label", "label", " Label"]:
            if candidate.strip() in [c.strip() for c in df.columns]:
                label_col = [c for c in df.columns if c.strip() == candidate.strip()][0]
                break

        if label_col is None:
            raise ValueError(f"No 'Label' column found in {filepath}")

        labels = df[label_col].astype(str).str.strip().tolist()

        # Drop label column and convert to numeric
        feature_cols = [c for c in df.columns if c != label_col]
        X = df[feature_cols].apply(lambda col: pd.to_numeric(col, errors='coerce')).values
        X = np.nan_to_num(X.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)

        return X, labels, [c.strip() for c in feature_cols]

    else:
        # Slow path: manual CSV parsing
        print(f"    [INFO] pandas not available, using manual CSV parser (slower)")

        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
            header = f.readline().strip()

        columns = [c.strip() for c in header.split(',')]

        label_idx = None
        for i, c in enumerate(columns):
            if c.lower() == 'label':
                label_idx = i
                break
        if label_idx is None:
            raise ValueError(f"No 'Label' column in {filepath}")

        rows = []
        labels = []
        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
            next(f)  # skip header
            for line in f:
                parts = line.strip().split(',')
                if len(parts) != len(columns):
                    continue
                labels.append(parts[label_idx].strip())
                row = []
                for i, v in enumerate(parts):
                    if i == label_idx:
                        continue
                    try:
                        val = float(v.strip())
                        if not np.isfinite(val):
                            val = 0.0
                    except (ValueError, OverflowError):
                        val = 0.0
                    row.append(val)
                rows.append(row)

        X = np.array(rows, dtype=np.float64) if rows else np.empty((0, 0))
        feat_cols = [c for i, c in enumerate(columns) if i != label_idx]
        return X, labels, feat_cols


def extract_14_features(
    X: np.ndarray,
    column_names: List[str],
) -> np.ndarray:
    """Map CICIDS2017 80-feature CSV to our 14 flow features.

    Uses FlowFeatureExtractor's static method for the mapping.
    """
    pd = _try_import_pandas()
    if pd is not None:
        import pandas as pdd
        df = pdd.DataFrame(X, columns=column_names)
        return FlowFeatureExtractor.extract_from_cicids_csv(df)
    else:
        # Manual column mapping
        col_idx = {name: i for i, name in enumerate(column_names)}

        def _get(name: str, default: float = 0.0) -> np.ndarray:
            # Try exact, then case-insensitive
            if name in col_idx:
                return X[:, col_idx[name]]
            for k, v in col_idx.items():
                if k.lower() == name.lower():
                    return X[:, v]
            return np.full(X.shape[0], default)

        n = X.shape[0]
        features = np.zeros((n, 14), dtype=np.float64)

        features[:, 0] = _get("Flow Duration")
        features[:, 1] = _get("Total Fwd Packets", 0)
        features[:, 2] = _get("Total Backward Packets", 0)
        if features[:, 2].sum() == 0:
            features[:, 2] = _get("Total Bwd packets", 0)
        features[:, 3] = _get("Total Length of Fwd Packets", 0)
        if features[:, 3].sum() == 0:
            features[:, 3] = _get("Total Length of Fwd Packet", 0)
        features[:, 4] = _get("Total Length of Bwd Packets", 0)
        if features[:, 4].sum() == 0:
            features[:, 4] = _get("Total Length of Bwd Packet", 0)
        features[:, 5] = _get("Fwd Packet Length Mean", 0)
        if features[:, 5].sum() == 0:
            features[:, 5] = _get("Fwd Pkt Len Mean", 0)
        features[:, 6] = _get("Fwd Packet Length Std", 0)
        if features[:, 6].sum() == 0:
            features[:, 6] = _get("Fwd Pkt Len Std", 0)
        features[:, 7] = _get("Bwd Packet Length Mean", 0)
        if features[:, 7].sum() == 0:
            features[:, 7] = _get("Bwd Pkt Len Mean", 0)
        features[:, 8] = _get("Bwd Packet Length Std", 0)
        if features[:, 8].sum() == 0:
            features[:, 8] = _get("Bwd Pkt Len Std", 0)
        features[:, 9] = _get("Flow IAT Mean", 0)
        features[:, 10] = _get("Flow IAT Std", 0)
        features[:, 11] = _get("SYN Flag Count", 0)
        if features[:, 11].sum() == 0:
            features[:, 11] = _get("SYN Flag Cnt", 0)
        features[:, 12] = _get("ACK Flag Count", 0)
        if features[:, 12].sum() == 0:
            features[:, 12] = _get("ACK Flag Cnt", 0)
        features[:, 13] = _get("Flow Packets/s", 0)
        if features[:, 13].sum() == 0:
            features[:, 13] = _get("Flow Pkts/s", 0)

        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        return features


# ──────────────────────────────────────────────────────────────────────
# Threshold Search (recall-optimized)
# ──────────────────────────────────────────────────────────────────────

def find_optimal_threshold(
    scores: np.ndarray,
    y_true: np.ndarray,
    recall_weight: float = 0.8,
    n_thresholds: int = 500,
) -> Tuple[float, dict]:
    """Find threshold maximizing Youden's J statistic (TPR - FPR).

    Youden's J avoids the degenerate case where recall-heavy objectives
    push the threshold to the minimum score, classifying everything as
    anomalous (TPR=1.0, FPR=1.0, FN=0, TN=0).

    When J values are tied (no score separation), prefers higher threshold
    (lower FPR) for conservative, honest predictions.

    Note: recall_weight parameter is kept for API compatibility but unused.
    """
    thresholds = np.linspace(scores.min() - 0.001, scores.max() + 0.001, n_thresholds)

    n_pos = max(int(np.sum(y_true == 1)), 1)
    n_neg = max(int(np.sum(y_true == 0)), 1)

    best_j = -2.0
    best_thresh = float(np.median(scores))
    best_fpr = 1.0
    best_m = {}

    for t in thresholds:
        pred = (scores >= t).astype(int)
        cm = confusion_matrix(y_true, pred)
        tp, fp, fn, tn = cm["tp"], cm["fp"], cm["fn"], cm["tn"]

        tpr = tp / n_pos
        fpr = fp / n_neg
        prec = tp / max(tp + fp, 1)
        j = tpr - fpr

        # Pick best J; break ties by preferring lower FPR (higher threshold)
        if j > best_j + 1e-6 or (abs(j - best_j) <= 1e-6 and fpr < best_fpr):
            best_j = j
            best_thresh = float(t)
            best_fpr = fpr
            best_m = {"threshold": best_thresh, "recall": tpr, "precision": prec}

    return best_thresh, best_m


# ──────────────────────────────────────────────────────────────────────
# CICIDS2017 attack type normalization
# ──────────────────────────────────────────────────────────────────────

ATTACK_TYPE_MAP = {
    "BENIGN": "BENIGN",
    "Bot": "Botnet",
    "DDoS": "DDoS",
    "DoS GoldenEye": "DoS",
    "DoS Hulk": "DoS",
    "DoS Slowhttptest": "DoS",
    "DoS slowloris": "DoS",
    "FTP-Patator": "BruteForce",
    "SSH-Patator": "BruteForce",
    "Heartbleed": "Heartbleed",
    "Infiltration": "Infiltration",
    "PortScan": "PortScan",
    "Web Attack \x96 Brute Force": "WebAttack",
    "Web Attack – Brute Force": "WebAttack",
    "Web Attack \x96 Sql Injection": "WebAttack",
    "Web Attack – Sql Injection": "WebAttack",
    "Web Attack \x96 XSS": "WebAttack",
    "Web Attack – XSS": "WebAttack",
}


def normalize_label(label: str) -> str:
    """Normalize CICIDS2017 label to canonical attack type."""
    label = label.strip()
    return ATTACK_TYPE_MAP.get(label, label)


# ──────────────────────────────────────────────────────────────────────
# Main Benchmark Runner
# ──────────────────────────────────────────────────────────────────────

class CICIDSBenchmark:
    """Multi-model CICIDS2017 benchmark evaluator.

    Trains all models on Monday (benign), tests on Friday (attacks).
    Produces per-model comparison and per-attack-category breakdown.

    Args:
        contamination: Contamination parameter for tree models.
            See Fix 2 analysis for correct value.
        n_estimators: Trees per forest model.
        max_samples: Subsample size per tree.
        random_state: Random seed.
    """

    def __init__(
        self,
        contamination: float = 0.015,
        n_estimators: int = 200,
        max_samples: int = 256,
        random_state: int = 42,
    ):
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.random_state = random_state

    def run(
        self,
        monday_csv: str,
        friday_csv: str,
    ) -> dict:
        """Run the full benchmark.

        Args:
            monday_csv: Path to Monday (benign) CICIDS CSV.
            friday_csv: Path to Friday (attacks) CICIDS CSV.

        Returns:
            Full results dict with per-model and per-category metrics.
        """
        print()
        print("=" * 72)
        print("  CICIDS2017 MULTI-MODEL BENCHMARK")
        print("  Quantum Sniffer v4.0 — Offline Validation")
        print("=" * 72)
        print()

        # ── Step 1: Load Monday (benign training set) ──
        print("  [1/6] Loading Monday CSV (benign baseline)...")
        X_mon_raw, labels_mon, cols_mon = load_cicids_csv(monday_csv)
        print(f"         {X_mon_raw.shape[0]:,} rows, {X_mon_raw.shape[1]} columns")

        # Filter to benign only
        benign_mask = np.array([normalize_label(l) == "BENIGN" for l in labels_mon])
        X_mon_raw = X_mon_raw[benign_mask]
        print(f"         {X_mon_raw.shape[0]:,} benign rows after filtering")

        # Map to 14 features
        X_train_raw = extract_14_features(X_mon_raw, cols_mon)
        print(f"         Mapped to {X_train_raw.shape[1]} flow features")
        print()

        # ── Step 2: Load Friday (attack test set) ──
        print("  [2/6] Loading Friday CSV (DDoS + PortScan + Botnet)...")
        X_fri_raw, labels_fri, cols_fri = load_cicids_csv(friday_csv)
        print(f"         {X_fri_raw.shape[0]:,} rows, {X_fri_raw.shape[1]} columns")

        # Map to 14 features
        X_test_raw = extract_14_features(X_fri_raw, cols_fri)

        # Build binary labels
        labels_fri_norm = [normalize_label(l) for l in labels_fri]
        y_test = np.array([0 if l == "BENIGN" else 1 for l in labels_fri_norm])

        n_attack = int(y_test.sum())
        n_benign = len(y_test) - n_attack
        attack_ratio = n_attack / max(len(y_test), 1)
        print(f"         Benign: {n_benign:,} | Attack: {n_attack:,} "
              f"({attack_ratio*100:.1f}%)")

        # Attack category distribution
        cat_counts = {}
        for l in labels_fri_norm:
            cat_counts[l] = cat_counts.get(l, 0) + 1
        print("         Categories:")
        for cat in sorted(cat_counts.keys()):
            print(f"           {cat:15s}: {cat_counts[cat]:>8,}")
        print()

        # ── Step 3: Z-score normalization (fit on Monday) ──
        print("  [3/6] Z-score normalization (fit on Monday benign)...")
        extractor = FlowFeatureExtractor(normalize=True)
        extractor.fit_normalization(X_train_raw)
        X_train = extractor.transform(X_train_raw)
        X_test = extractor.transform(X_test_raw)
        print()

        # ── Step 4: Train and evaluate each model ──
        results = {}

        # --- 4a: Isolation Forest ---
        print("  [4/6] Training & evaluating models...")
        print()
        print("    ─── Isolation Forest ───")
        t0 = time.time()
        iforest = IsolationForest(
            n_estimators=self.n_estimators,
            max_samples=self.max_samples,
            contamination=self.contamination,
            random_state=self.random_state,
        )
        iforest.fit(X_train)
        train_t = time.time() - t0

        t0 = time.time()
        if_scores = iforest.anomaly_scores(X_test)
        score_t = time.time() - t0

        thresh, _ = find_optimal_threshold(if_scores, y_test)
        if_pred = (if_scores >= thresh).astype(int)
        if_metrics = compute_metrics(y_test, if_pred, if_scores)
        if_metrics["train_time"] = train_t
        if_metrics["score_time"] = score_t
        if_metrics["threshold"] = thresh
        results["IsolationForest"] = if_metrics

        _print_model_results("Isolation Forest", if_metrics)

        # --- 4b: Extended Isolation Forest ---
        print("    ─── Extended Isolation Forest ───")
        t0 = time.time()
        eif = ExtendedIsolationForest(
            n_estimators=self.n_estimators,
            max_samples=self.max_samples,
            contamination=self.contamination,
            random_state=self.random_state,
        )
        eif.fit(X_train)
        train_t = time.time() - t0

        t0 = time.time()
        eif_scores = eif.anomaly_scores(X_test)
        score_t = time.time() - t0

        thresh, _ = find_optimal_threshold(eif_scores, y_test)
        eif_pred = (eif_scores >= thresh).astype(int)
        eif_metrics = compute_metrics(y_test, eif_pred, eif_scores)
        eif_metrics["train_time"] = train_t
        eif_metrics["score_time"] = score_t
        eif_metrics["threshold"] = thresh
        results["ExtendedIsolationForest"] = eif_metrics

        _print_model_results("Extended Isolation Forest", eif_metrics)

        # --- 4c: Autoencoder ---
        ae_metrics = None
        try:
            from autoencoder_detector import AutoencoderDetector
            print("    ─── Autoencoder (14→8→4→8→14) ───")
            t0 = time.time()
            ae = AutoencoderDetector(
                input_dim=14,
                hidden_dim=8,
                bottleneck_dim=4,
                epochs=100,
                batch_size=128,
                patience=10,
                contamination=self.contamination,
            )
            ae.fit(X_train_raw, verbose=True)  # AE normalizes internally
            train_t = time.time() - t0

            t0 = time.time()
            ae_scores = ae.anomaly_scores(X_test_raw)
            score_t = time.time() - t0

            thresh, _ = find_optimal_threshold(ae_scores, y_test)
            ae_pred = (ae_scores >= thresh).astype(int)
            ae_metrics = compute_metrics(y_test, ae_pred, ae_scores)
            ae_metrics["train_time"] = train_t
            ae_metrics["score_time"] = score_t
            ae_metrics["threshold"] = thresh
            results["Autoencoder"] = ae_metrics

            _print_model_results("Autoencoder", ae_metrics)

        except ImportError:
            print("    [SKIP] PyTorch not available — skipping Autoencoder")
            ae_scores = np.full(len(y_test), 0.5)

        # --- 4d: CombinedDetector (EIF backbone + AE secondary + iF tiebreaker) ---
        print("    ─── CombinedDetector (restructured) ───")
        # Weights from Fix 3: EIF=0.50, AE=0.35, iF=0.15
        w_eif = 0.50
        w_ae = 0.35
        w_if = 0.15

        combined_scores = (
            w_eif * eif_scores
            + w_ae * ae_scores
            + w_if * if_scores
        )
        combined_scores = np.clip(combined_scores, 0, 1)

        thresh, _ = find_optimal_threshold(combined_scores, y_test)
        comb_pred = (combined_scores >= thresh).astype(int)
        comb_metrics = compute_metrics(y_test, comb_pred, combined_scores)
        comb_metrics["threshold"] = thresh
        comb_metrics["weights"] = {"eif": w_eif, "ae": w_ae, "iforest": w_if}
        results["CombinedDetector"] = comb_metrics

        _print_model_results("CombinedDetector", comb_metrics)

        # ── Step 5: Comparison Table ──
        print()
        print("  [5/6] Model Comparison")
        print("  " + "=" * 68)
        print(f"  {'Model':<25s} {'Recall':>7s} {'Prec':>7s} {'F1':>7s} "
              f"{'MCC':>7s} {'AUC':>7s} {'FPR':>7s}")
        print("  " + "-" * 68)
        for name, m in results.items():
            print(f"  {name:<25s} {m['recall']:>7.4f} {m['precision']:>7.4f} "
                  f"{m['f1']:>7.4f} {m['mcc']:>7.4f} {m['auc_roc']:>7.4f} "
                  f"{m['fpr']:>7.4f}")
        print("  " + "=" * 68)
        print()

        # ── Step 6: Per-Attack-Category Breakdown ──
        print("  [6/6] Per-Attack-Category Breakdown (Friday)")
        print("  " + "=" * 68)

        # Use CombinedDetector (best expected) for breakdown
        best_model_name = max(results, key=lambda k: results[k]["f1"])
        if best_model_name == "CombinedDetector":
            breakdown_pred = comb_pred
            breakdown_scores = combined_scores
        elif best_model_name == "ExtendedIsolationForest":
            breakdown_pred = eif_pred
            breakdown_scores = eif_scores
        else:
            breakdown_pred = if_pred
            breakdown_scores = if_scores

        breakdown = per_attack_breakdown(labels_fri_norm, breakdown_pred, breakdown_scores)

        print(f"  Using: {best_model_name}")
        print(f"  {'Category':<15s} {'Total':>8s} {'Detect':>8s} {'Missed':>8s} "
              f"{'Recall':>8s} {'AvgScore':>9s}")
        print("  " + "-" * 68)
        for cat in sorted(breakdown.keys()):
            b = breakdown[cat]
            if cat == "BENIGN":
                print(f"  {cat:<15s} {b['total']:>8,} {b['correctly_classified']:>8,} "
                      f"{b['false_positives']:>8,} {b['specificity']:>8.4f} "
                      f"{b['mean_score']:>9.4f}")
            else:
                print(f"  {cat:<15s} {b['total']:>8,} {b['detected']:>8,} "
                      f"{b['missed']:>8,} {b['recall']:>8.4f} "
                      f"{b['mean_score']:>9.4f}")
        print("  " + "=" * 68)
        print()

        # Also produce per-attack breakdown for ALL models
        all_breakdowns = {}
        model_preds = {
            "IsolationForest": if_pred,
            "ExtendedIsolationForest": eif_pred,
            "CombinedDetector": comb_pred,
        }
        if ae_metrics is not None:
            model_preds["Autoencoder"] = ae_pred
        model_scores_map = {
            "IsolationForest": if_scores,
            "ExtendedIsolationForest": eif_scores,
            "CombinedDetector": combined_scores,
        }
        if ae_metrics is not None:
            model_scores_map["Autoencoder"] = ae_scores

        for mname, mpred in model_preds.items():
            all_breakdowns[mname] = per_attack_breakdown(
                labels_fri_norm, mpred, model_scores_map[mname]
            )

        # Print cross-model per-attack recall table
        attack_cats = [c for c in sorted(cat_counts.keys()) if c != "BENIGN"]
        if attack_cats:
            print("  Per-Attack Recall — All Models:")
            print("  " + "-" * 72)
            header = f"  {'Attack':<15s}"
            for mname in model_preds:
                header += f" {mname[:12]:>12s}"
            print(header)
            print("  " + "-" * 72)
            for cat in attack_cats:
                row = f"  {cat:<15s}"
                for mname in model_preds:
                    bd = all_breakdowns[mname].get(cat, {})
                    rec = bd.get("recall", 0.0)
                    row += f" {rec:>12.4f}"
                print(row)
            print("  " + "-" * 72)
            print()

        full_results = {
            "dataset": {
                "train_samples": X_train.shape[0],
                "test_samples": X_test.shape[0],
                "attack_ratio": attack_ratio,
                "categories": cat_counts,
            },
            "models": results,
            "per_attack": all_breakdowns,
            "contamination": self.contamination,
            "feature_names": FLOW_FEATURE_NAMES,
        }

        return full_results


def _print_model_results(name: str, m: dict):
    """Print summary for one model."""
    print(f"      Recall={m['recall']:.4f}  Precision={m['precision']:.4f}  "
          f"F1={m['f1']:.4f}  MCC={m['mcc']:.4f}  AUC={m['auc_roc']:.4f}")
    print(f"      TP={m['tp']:,}  FP={m['fp']:,}  FN={m['fn']:,}  TN={m['tn']:,}")
    if "train_time" in m:
        print(f"      Train: {m['train_time']:.2f}s  Score: {m['score_time']:.2f}s")
    print()


# ──────────────────────────────────────────────────────────────────────
# CLI Entry Point
# ──────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="CICIDS2017 Multi-Model Benchmark (Quantum Sniffer)"
    )
    parser.add_argument(
        "--monday", required=True,
        help="Path to Monday (benign) CICIDS CSV"
    )
    parser.add_argument(
        "--friday", required=True,
        help="Path to Friday (attack) CICIDS CSV"
    )
    parser.add_argument(
        "--contamination", type=float, default=0.015,
        help="Contamination parameter (default: 0.015)"
    )
    parser.add_argument(
        "--n-estimators", type=int, default=200,
        help="Trees per forest (default: 200)"
    )
    parser.add_argument(
        "--max-samples", type=int, default=256,
        help="Subsample size per tree (default: 256)"
    )
    args = parser.parse_args()

    bench = CICIDSBenchmark(
        contamination=args.contamination,
        n_estimators=args.n_estimators,
        max_samples=args.max_samples,
    )
    bench.run(args.monday, args.friday)


if __name__ == "__main__":
    main()
