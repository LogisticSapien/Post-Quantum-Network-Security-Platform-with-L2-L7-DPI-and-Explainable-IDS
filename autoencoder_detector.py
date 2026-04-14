"""
Autoencoder Anomaly Detector
===============================
PyTorch autoencoder for detecting behavioral anomalies that the
Isolation Forest may miss. Learns a compressed representation of
normal traffic and flags samples with high reconstruction error.

Architecture: 14 → 8 → 4 → 8 → 14 (symmetric bottleneck)
  - Encoder: 14→8→4 with ReLU + BatchNorm
  - Decoder: 4→8→14 with ReLU + BatchNorm (final layer: Sigmoid)

Anomaly score = MSE reconstruction error per sample.
High reconstruction error → the autoencoder cannot reproduce the
input well → input is unlike the training distribution → anomaly.

Fusion with iForest:
  combined = w₁·s_iforest + w₂·σ(recon_error / baseline_error)
  - iForest catches structural/volumetric anomalies
  - Autoencoder catches subtle behavioral pattern changes

This module requires PyTorch. If PyTorch is not installed,
a stub class is provided that raises ImportError on use.

Constraint: PyTorch required (Tier 3 exception to NumPy-only rule).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

# Try to import PyTorch
_TORCH_AVAILABLE = False
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    _TORCH_AVAILABLE = True
except ImportError:
    pass


# ──────────────────────────────────────────────────────────────────────
# Autoencoder Architecture (PyTorch)
# ──────────────────────────────────────────────────────────────────────

if _TORCH_AVAILABLE:

    class AnomalyAutoencoder(nn.Module):
        """Symmetric autoencoder for anomaly detection.

        Architecture: input_dim → hidden1 → bottleneck → hidden1 → input_dim

        Args:
            input_dim: Number of input features (default: 14).
            hidden_dim: First hidden layer size (default: 8).
            bottleneck_dim: Bottleneck size (default: 4).
            dropout: Dropout rate (default: 0.1).
        """

        def __init__(
            self,
            input_dim: int = 14,
            hidden_dim: int = 8,
            bottleneck_dim: int = 4,
            dropout: float = 0.1,
        ):
            super().__init__()

            # Encoder
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, bottleneck_dim),
                nn.BatchNorm1d(bottleneck_dim),
                nn.ReLU(),
            )

            # Decoder
            self.decoder = nn.Sequential(
                nn.Linear(bottleneck_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, input_dim),
                # No activation on output — unbounded reconstruction
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Forward pass: encode then decode.

            Args:
                x: Input tensor, shape (batch_size, input_dim).

            Returns:
                Reconstructed tensor, same shape as input.
            """
            encoded = self.encoder(x)
            decoded = self.decoder(encoded)
            return decoded

        def encode(self, x: torch.Tensor) -> torch.Tensor:
            """Get bottleneck representation.

            Args:
                x: Input tensor.

            Returns:
                Encoded representation, shape (batch_size, bottleneck_dim).
            """
            return self.encoder(x)


# ──────────────────────────────────────────────────────────────────────
# Autoencoder Detector
# ──────────────────────────────────────────────────────────────────────

@dataclass
class AutoencoderResult:
    """Result from autoencoder anomaly detection."""
    reconstruction_error: float
    anomaly_score: float
    is_anomaly: bool
    threshold: float


class AutoencoderDetector:
    """Autoencoder-based anomaly detector with training pipeline.

    Trains a symmetric autoencoder on normal traffic data, then
    uses reconstruction error as the anomaly score.

    Args:
        input_dim: Number of features (default: 14).
        hidden_dim: Hidden layer size (default: 8).
        bottleneck_dim: Bottleneck size (default: 4).
        learning_rate: AdamW learning rate (default: 0.001).
        epochs: Maximum training epochs (default: 100).
        batch_size: Training batch size (default: 64).
        patience: Early stopping patience (default: 10).
        contamination: Expected anomaly ratio for threshold (default: 0.05).
        dropout: Dropout rate (default: 0.1).
    """

    def __init__(
        self,
        input_dim: int = 14,
        hidden_dim: int = 8,
        bottleneck_dim: int = 4,
        learning_rate: float = 0.001,
        epochs: int = 100,
        batch_size: int = 64,
        patience: int = 10,
        contamination: float = 0.05,
        dropout: float = 0.1,
    ):
        if not _TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is required for AutoencoderDetector. "
                "Install with: pip install torch"
            )

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.bottleneck_dim = bottleneck_dim
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.contamination = contamination

        self.model = AnomalyAutoencoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            bottleneck_dim=bottleneck_dim,
            dropout=dropout,
        )

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self._device)

        self._fitted = False
        self._threshold: float = 0.0
        self._baseline_error: float = 1.0
        self._training_history: List[dict] = []

        # Normalization stats
        self._feature_means: Optional[np.ndarray] = None
        self._feature_stds: Optional[np.ndarray] = None

    def fit(
        self,
        X: np.ndarray,
        validation_split: float = 0.2,
        verbose: bool = True,
    ) -> dict:
        """Train the autoencoder on normal data.

        Args:
            X: Training data, shape (n_samples, input_dim).
                Should be primarily normal (non-anomalous) samples.
            validation_split: Fraction of data for validation.
            verbose: Print training progress.

        Returns:
            Training history dict.
        """
        X = np.asarray(X, dtype=np.float32)

        # Store normalization stats
        self._feature_means = X.mean(axis=0).astype(np.float32)
        self._feature_stds = X.std(axis=0).astype(np.float32)
        self._feature_stds[self._feature_stds < 1e-7] = 1.0

        # Normalize
        X_norm = (X - self._feature_means) / self._feature_stds

        # Train/val split
        n = len(X_norm)
        n_val = max(1, int(n * validation_split))
        indices = np.random.permutation(n)
        X_train = X_norm[indices[n_val:]]
        X_val = X_norm[indices[:n_val]]

        train_tensor = torch.FloatTensor(X_train).to(self._device)
        val_tensor = torch.FloatTensor(X_val).to(self._device)

        train_dataset = TensorDataset(train_tensor, train_tensor)
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )

        optimizer = optim.AdamW(
            self.model.parameters(), lr=self.learning_rate, weight_decay=1e-5
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=5, factor=0.5
        )
        criterion = nn.MSELoss()

        # Training loop with early stopping
        best_val_loss = float('inf')
        patience_counter = 0
        best_state = None

        if verbose:
            print(f"    Training autoencoder ({self._device})...")
            print(f"    Train: {len(X_train)}, Val: {len(X_val)}")

        t0 = time.time()
        for epoch in range(self.epochs):
            # Train
            self.model.train()
            train_loss = 0.0
            n_batches = 0
            for batch_x, _ in train_loader:
                optimizer.zero_grad()
                output = self.model(batch_x)
                loss = criterion(output, batch_x)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                n_batches += 1
            train_loss /= max(n_batches, 1)

            # Validate
            self.model.eval()
            with torch.no_grad():
                val_output = self.model(val_tensor)
                val_loss = criterion(val_output, val_tensor).item()

            scheduler.step(val_loss)

            self._training_history.append({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
            })

            if verbose and (epoch + 1) % 10 == 0:
                print(f"    Epoch {epoch+1:3d}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    if verbose:
                        print(f"    Early stopping at epoch {epoch+1}")
                    break

        train_time = time.time() - t0

        # Restore best model
        if best_state:
            self.model.load_state_dict(best_state)

        # Calibrate threshold
        self.model.eval()
        with torch.no_grad():
            train_recon = self.model(train_tensor)
            errors = torch.mean((train_tensor - train_recon) ** 2, dim=1).cpu().numpy()

        self._baseline_error = float(np.mean(errors))
        self._threshold = float(np.percentile(errors, 100 * (1 - self.contamination)))
        self._fitted = True

        if verbose:
            print(f"    Training complete in {train_time:.2f}s")
            print(f"    Baseline error: {self._baseline_error:.6f}")
            print(f"    Threshold: {self._threshold:.6f}")

        return {
            "epochs_trained": len(self._training_history),
            "best_val_loss": best_val_loss,
            "train_time": train_time,
            "baseline_error": self._baseline_error,
            "threshold": self._threshold,
        }

    def reconstruction_error(self, X: np.ndarray) -> np.ndarray:
        """Compute per-sample reconstruction error.

        Args:
            X: Data, shape (n_samples, input_dim).

        Returns:
            MSE per sample, shape (n_samples,).
        """
        if not self._fitted:
            raise RuntimeError("AutoencoderDetector not fitted. Call fit() first.")

        X = np.asarray(X, dtype=np.float32)
        X_norm = (X - self._feature_means) / self._feature_stds

        self.model.eval()
        with torch.no_grad():
            tensor = torch.FloatTensor(X_norm).to(self._device)
            recon = self.model(tensor)
            errors = torch.mean((tensor - recon) ** 2, dim=1).cpu().numpy()

        return errors

    def anomaly_scores(self, X: np.ndarray) -> np.ndarray:
        """Compute anomaly scores (normalized reconstruction error).

        Scores are sigmoid-normalized so they fall in [0, 1].

        Args:
            X: Data, shape (n_samples, input_dim).

        Returns:
            Anomaly scores in [0, 1], shape (n_samples,).
        """
        errors = self.reconstruction_error(X)
        # Sigmoid normalize relative to baseline
        ratio = errors / max(self._baseline_error, 1e-10)
        # Map via sigmoid: ratio=1 → ~0.5, ratio>1 → >0.5
        scores = 1.0 / (1.0 + np.exp(-(ratio - 1.0) * 3.0))
        return scores

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict anomaly labels.

        Args:
            X: Data, shape (n_samples, input_dim).

        Returns:
            Labels: 1 = anomaly, 0 = normal.
        """
        errors = self.reconstruction_error(X)
        return (errors >= self._threshold).astype(int)

    def predict_single(self, x: np.ndarray) -> AutoencoderResult:
        """Predict for a single sample with full details.

        Args:
            x: Single sample, shape (input_dim,) or (1, input_dim).

        Returns:
            AutoencoderResult.
        """
        x = x.reshape(1, -1)
        error = float(self.reconstruction_error(x)[0])
        score = float(self.anomaly_scores(x)[0])
        return AutoencoderResult(
            reconstruction_error=error,
            anomaly_score=score,
            is_anomaly=error >= self._threshold,
            threshold=self._threshold,
        )

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    @property
    def threshold(self) -> float:
        return self._threshold

    @threshold.setter
    def threshold(self, value: float):
        self._threshold = value

    def get_params(self) -> dict:
        return {
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "bottleneck_dim": self.bottleneck_dim,
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "patience": self.patience,
            "contamination": self.contamination,
            "fitted": self._fitted,
            "threshold": self._threshold,
            "baseline_error": self._baseline_error,
            "device": str(self._device),
        }


# ──────────────────────────────────────────────────────────────────────
# Combined Detector (EIF backbone + Autoencoder secondary + iForest voter)
# ──────────────────────────────────────────────────────────────────────

class CombinedDetector:
    """Weighted tri-model fusion detector.

    Restructured weight hierarchy (Fix 3):
      - EIF (backbone):      0.50 — handles clustered high-volume attacks
        (hyperplane splits detect diagonal anomalies iForest misses)
      - Autoencoder (sec.):  0.35 — catches behavioral drift via recon error
      - iForest (tiebreak):  0.15 — weak voter, breaks ties on borderline cases

    Previous weights (iF=0.6, AE=0.4) over-relied on iForest which
    structurally assumes anomalies are globally isolated and rare —
    wrong for DDoS/PortScan where attacks form dense clusters.

    Backward-compatible: score() still accepts (iforest, autoencoder)
    for legacy callers. New callers should use score_triple().

    Args:
        eif_weight: Weight for EIF backbone (default: 0.50).
        autoencoder_weight: Weight for autoencoder secondary (default: 0.35).
        iforest_weight: Weight for iForest tiebreaker (default: 0.15).
        threshold: Decision threshold on combined score (default: 0.5).
    """

    def __init__(
        self,
        eif_weight: float = 0.50,
        autoencoder_weight: float = 0.35,
        iforest_weight: float = 0.15,
        threshold: float = 0.5,
    ):
        total = eif_weight + autoencoder_weight + iforest_weight
        self.eif_weight = eif_weight / total
        self.autoencoder_weight = autoencoder_weight / total
        self.iforest_weight = iforest_weight / total
        self._threshold = threshold

    def score_triple(
        self,
        eif_scores: np.ndarray,
        autoencoder_scores: np.ndarray,
        iforest_scores: np.ndarray,
    ) -> np.ndarray:
        """Compute combined score from all three models.

        Args:
            eif_scores: ExtendedIsolationForest scores, shape (n,).
            autoencoder_scores: Autoencoder scores, shape (n,).
            iforest_scores: IsolationForest scores, shape (n,).

        Returns:
            Combined scores ∈ [0, 1], shape (n,).
        """
        return np.clip(
            self.eif_weight * np.asarray(eif_scores)
            + self.autoencoder_weight * np.asarray(autoencoder_scores)
            + self.iforest_weight * np.asarray(iforest_scores),
            0.0, 1.0,
        )

    def score(
        self,
        iforest_scores: np.ndarray,
        autoencoder_scores: np.ndarray,
    ) -> np.ndarray:
        """Legacy 2-model API (backward compatible).

        When EIF scores are not available, redistributes EIF weight
        proportionally: AE gets ~70%, iF gets ~30% of the EIF share.
        Effective weights: iF ≈ 0.30, AE ≈ 0.70.

        Args:
            iforest_scores: iForest scores, shape (n,).
            autoencoder_scores: Autoencoder scores, shape (n,).

        Returns:
            Combined scores, shape (n,).
        """
        # Redistribute EIF weight: 70% to AE, 30% to iF
        w_if = self.iforest_weight + 0.30 * self.eif_weight
        w_ae = self.autoencoder_weight + 0.70 * self.eif_weight
        return np.clip(
            w_if * np.asarray(iforest_scores)
            + w_ae * np.asarray(autoencoder_scores),
            0.0, 1.0,
        )

    def predict_triple(
        self,
        eif_scores: np.ndarray,
        autoencoder_scores: np.ndarray,
        iforest_scores: np.ndarray,
    ) -> np.ndarray:
        """Predict labels from tri-model fusion.

        Returns:
            Labels: 1 = anomaly, 0 = normal.
        """
        combined = self.score_triple(eif_scores, autoencoder_scores, iforest_scores)
        return (combined >= self._threshold).astype(int)

    def predict(
        self,
        iforest_scores: np.ndarray,
        autoencoder_scores: np.ndarray,
    ) -> np.ndarray:
        """Legacy 2-model predict (backward compatible).

        Returns:
            Labels: 1 = anomaly, 0 = normal.
        """
        combined = self.score(iforest_scores, autoencoder_scores)
        return (combined >= self._threshold).astype(int)

    def calibrate_threshold(
        self,
        combined_scores: np.ndarray,
        y_true: np.ndarray,
        recall_weight: float = 0.8,
        n_thresholds: int = 300,
    ) -> float:
        """Find recall-biased optimal threshold.

        Objective: recall_weight*recall + (1-recall_weight)*precision

        Args:
            combined_scores: Fused scores to threshold.
            y_true: Binary ground truth (1=anomaly).
            recall_weight: FN penalty weight (default 0.8).
            n_thresholds: Search resolution.

        Returns:
            Optimal threshold.
        """
        thresholds = np.linspace(
            combined_scores.min(), combined_scores.max(), n_thresholds
        )
        best_obj = -1.0
        best_t = 0.5

        for t in thresholds:
            pred = (combined_scores >= t).astype(int)
            tp = np.sum((y_true == 1) & (pred == 1))
            fp = np.sum((y_true == 0) & (pred == 1))
            fn = np.sum((y_true == 1) & (pred == 0))
            rec = tp / max(tp + fn, 1)
            prec = tp / max(tp + fp, 1)
            obj = recall_weight * rec + (1 - recall_weight) * prec
            if obj > best_obj:
                best_obj = obj
                best_t = float(t)

        self._threshold = best_t
        return best_t

    @property
    def threshold(self) -> float:
        return self._threshold

    @threshold.setter
    def threshold(self, value: float):
        self._threshold = value

    def get_params(self) -> dict:
        """Get detector configuration."""
        return {
            "eif_weight": self.eif_weight,
            "autoencoder_weight": self.autoencoder_weight,
            "iforest_weight": self.iforest_weight,
            "threshold": self._threshold,
        }
