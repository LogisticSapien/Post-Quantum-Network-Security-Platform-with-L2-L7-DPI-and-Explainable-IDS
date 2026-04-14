"""
Isolation Forest — Pure-Python Implementation
================================================
From-scratch Isolation Forest for unsupervised anomaly detection.
No sklearn dependency — uses only numpy.

Algorithm (Liu, Ting & Zhou, 2008):
  1. Build an ensemble of Isolation Trees (iTrees) on random subsamples
  2. Each iTree recursively partitions data by random feature + random split
  3. Anomalies are isolated in fewer splits → shorter path lengths
  4. Anomaly score: s(x, n) = 2^(-E(h(x)) / c(n))
     - E(h(x)) = average path length across all trees
     - c(n) = average unsuccessful BST search path length (normalization)

Key properties:
  - Linear time complexity O(n·t·ψ) where t=trees, ψ=subsample size
  - No distance/density computation — purely isolation-based
  - Effective for high-dimensional data with sublinear sample sizes
  - Naturally handles irrelevant features (random splits dilute them)

Design rationale:
  Isolation Forest over other anomaly detectors because:
  1. Works well with unlabeled network data (unsupervised)
  2. Handles mixed feature types naturally
  3. Linear time — suitable for real-time network monitoring
  4. No assumption about data distribution (unlike z-score/EWMA)
  5. Detects novel (zero-day) attack patterns without signatures
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


# ──────────────────────────────────────────────────────────────────────
# Constants & Utility
# ──────────────────────────────────────────────────────────────────────

def _c(n: int) -> float:
    """Average path length of unsuccessful search in a Binary Search Tree.

    Used as normalization factor for anomaly scores.

    c(n) = 2·H(n-1) - 2(n-1)/n
    where H(i) ≈ ln(i) + 0.5772156649 (Euler-Mascheroni constant)

    Args:
        n: Number of data points (external nodes).

    Returns:
        Average path length c(n). Returns 0 for n <= 1.
    """
    if n <= 1:
        return 0.0
    if n == 2:
        return 1.0
    H = math.log(n - 1) + 0.5772156649  # Euler-Mascheroni constant
    return 2.0 * H - 2.0 * (n - 1) / n


# ──────────────────────────────────────────────────────────────────────
# Isolation Tree Node
# ──────────────────────────────────────────────────────────────────────

@dataclass
class ITreeNode:
    """A node in an Isolation Tree.

    Internal nodes store the split attribute and value.
    External (leaf) nodes store the sample count at that leaf.
    """
    # Split info (internal nodes only)
    split_attr: Optional[int] = None
    split_value: Optional[float] = None
    left: Optional['ITreeNode'] = None
    right: Optional['ITreeNode'] = None

    # Leaf info
    size: int = 0  # number of samples at this external node
    is_external: bool = True


# ──────────────────────────────────────────────────────────────────────
# Isolation Tree
# ──────────────────────────────────────────────────────────────────────

class IsolationTree:
    """A single Isolation Tree.

    Recursively partitions data using random attribute and random split
    value until:
      - Node has 0 or 1 samples (perfectly isolated), or
      - Maximum depth (height limit) is reached

    The tree structure encodes how many splits are needed to isolate
    each sample — anomalies require fewer splits.
    """

    def __init__(self, height_limit: int):
        """
        Args:
            height_limit: Maximum tree depth. Typically ceil(log2(ψ))
                         where ψ is the subsample size.
        """
        self.height_limit = height_limit
        self.root: Optional[ITreeNode] = None
        self.n_features: int = 0

    def fit(self, X: np.ndarray) -> 'IsolationTree':
        """Build the isolation tree from data.

        Args:
            X: Training data of shape (n_samples, n_features).

        Returns:
            self
        """
        self.n_features = X.shape[1]
        self.root = self._build(X, depth=0)
        return self

    def _build(self, X: np.ndarray, depth: int) -> ITreeNode:
        """Recursively build the tree.

        Args:
            X: Data subset at this node.
            depth: Current depth in the tree.

        Returns:
            ITreeNode (internal or external).
        """
        n_samples = X.shape[0]

        # Base cases → external node
        if n_samples <= 1 or depth >= self.height_limit:
            return ITreeNode(size=n_samples, is_external=True)

        # Check if all values are identical (can't split)
        if np.all(X == X[0]):
            return ITreeNode(size=n_samples, is_external=True)

        # Random attribute selection
        n_features = X.shape[1]

        # Try to find a feature with variance (avoid degenerate splits)
        attempts = 0
        while attempts < n_features:
            q = random.randint(0, n_features - 1)
            col = X[:, q]
            col_min, col_max = col.min(), col.max()
            if col_max > col_min:
                break
            attempts += 1
        else:
            # All features are constant → external node
            return ITreeNode(size=n_samples, is_external=True)

        # Random split value between min and max
        p = random.uniform(col_min, col_max)

        # Partition
        left_mask = X[:, q] < p
        right_mask = ~left_mask

        node = ITreeNode(
            split_attr=q,
            split_value=p,
            is_external=False,
        )
        node.left = self._build(X[left_mask], depth + 1)
        node.right = self._build(X[right_mask], depth + 1)

        return node

    def path_length(self, x: np.ndarray) -> float:
        """Compute the path length for a single sample.

        The path length h(x) is the number of edges traversed from root
        to the external node where x ends up. If the external node has
        size > 1, we add c(size) as an estimate of the remaining
        unbuilt subtree.

        Args:
            x: Single sample of shape (n_features,).

        Returns:
            Path length h(x).
        """
        return self._path_length(x, self.root, 0)

    def _path_length(self, x: np.ndarray, node: ITreeNode, depth: int) -> float:
        """Recursive path length computation.

        Args:
            x: Sample.
            node: Current tree node.
            depth: Current depth (edges traversed so far).

        Returns:
            Path length from root to external node.
        """
        if node.is_external:
            # At external node, add c(node.size) for unbuilt subtree
            return depth + _c(node.size)

        # Traverse left or right based on split
        if x[node.split_attr] < node.split_value:
            return self._path_length(x, node.left, depth + 1)
        else:
            return self._path_length(x, node.right, depth + 1)

    def path_lengths_batch(self, X: np.ndarray) -> np.ndarray:
        """Compute path lengths for multiple samples.

        Args:
            X: Data of shape (n_samples, n_features).

        Returns:
            Array of path lengths of shape (n_samples,).
        """
        return np.array([self.path_length(x) for x in X])


# ──────────────────────────────────────────────────────────────────────
# Isolation Forest (Ensemble)
# ──────────────────────────────────────────────────────────────────────

class IsolationForest:
    """Isolation Forest anomaly detector.

    Ensemble of Isolation Trees trained on random subsamples.
    Anomaly score is computed from average path length across all trees.

    Parameters:
        n_estimators: Number of Isolation Trees (default: 100).
        max_samples: Subsample size ψ for each tree (default: 256).
                     If 'auto', uses min(256, n_samples).
        contamination: Expected proportion of anomalies (default: 0.1).
                       Used to set the decision threshold.
        random_state: Random seed for reproducibility.

    Usage:
        >>> forest = IsolationForest(n_estimators=100, max_samples=256)
        >>> forest.fit(X_train)
        >>> scores = forest.anomaly_scores(X_test)  # 0-1, higher = more anomalous
        >>> labels = forest.predict(X_test)          # 1 = anomaly, 0 = normal
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_samples: int = 256,
        contamination: float = 0.015,
        random_state: Optional[int] = None,
    ):
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.contamination = contamination
        self.random_state = random_state

        self.trees: List[IsolationTree] = []
        self._threshold: float = 0.5
        self._fitted = False
        self._n_samples: int = 0
        self._n_features: int = 0
        self._psi: int = 0  # actual subsample size used

    def fit(self, X: np.ndarray) -> 'IsolationForest':
        """Fit the Isolation Forest on training data.

        Args:
            X: Training data of shape (n_samples, n_features).
               Should ideally be "normal" data (low contamination).

        Returns:
            self
        """
        if self.random_state is not None:
            random.seed(self.random_state)
            np.random.seed(self.random_state)

        X = np.asarray(X, dtype=np.float64)
        n_samples, n_features = X.shape
        self._n_samples = n_samples
        self._n_features = n_features

        # Determine actual subsample size
        if isinstance(self.max_samples, str) and self.max_samples == 'auto':
            self._psi = min(256, n_samples)
        else:
            self._psi = min(self.max_samples, n_samples)

        # Height limit: ceil(log2(ψ))
        height_limit = max(1, math.ceil(math.log2(max(self._psi, 2))))

        # Build trees
        self.trees = []
        for _ in range(self.n_estimators):
            # Random subsample (with replacement if n < psi, else without)
            if n_samples <= self._psi:
                X_sub = X.copy()
            else:
                indices = np.random.choice(n_samples, size=self._psi, replace=False)
                X_sub = X[indices]

            tree = IsolationTree(height_limit=height_limit)
            tree.fit(X_sub)
            self.trees.append(tree)

        self._fitted = True

        # Calibrate threshold using training data scores
        if self.contamination > 0:
            scores = self._compute_scores(X)
            self._threshold = float(np.percentile(
                scores, 100 * (1 - self.contamination)
            ))
        else:
            self._threshold = 0.5

        return self

    def _compute_scores(self, X: np.ndarray) -> np.ndarray:
        """Internal scoring — no fitted check (used during fit)."""
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        # Average path length across all trees
        avg_path_lengths = np.zeros(X.shape[0])
        for tree in self.trees:
            avg_path_lengths += tree.path_lengths_batch(X)
        avg_path_lengths /= len(self.trees)

        # Normalization factor
        cn = _c(self._psi)
        if cn == 0:
            return np.full(X.shape[0], 0.5)

        # Anomaly score: s(x) = 2^(-E(h(x)) / c(ψ))
        return np.power(2, -avg_path_lengths / cn)

    def anomaly_scores(self, X: np.ndarray) -> np.ndarray:
        """Compute anomaly scores for samples.

        Score formula: s(x, n) = 2^(-E(h(x)) / c(ψ))

        Returns scores in [0, 1]:
          - Score ≈ 1.0 → definite anomaly
          - Score ≈ 0.5 → borderline / uncertain
          - Score ≈ 0.0 → definitely normal

        Args:
            X: Data of shape (n_samples, n_features).

        Returns:
            Anomaly scores of shape (n_samples,).
        """
        if not self._fitted:
            raise RuntimeError("IsolationForest not fitted. Call fit() first.")
        return self._compute_scores(X)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Raw anomaly scores (same as anomaly_scores).

        Provided for API compatibility.
        """
        return self.anomaly_scores(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict anomaly labels.

        Args:
            X: Data of shape (n_samples, n_features).

        Returns:
            Labels: 1 = anomaly, 0 = normal.
        """
        scores = self.anomaly_scores(X)
        return (scores >= self._threshold).astype(int)

    def feature_importance(self, X: np.ndarray) -> np.ndarray:
        """Estimate per-feature anomaly contribution.

        For each feature, compute how much the anomaly score
        changes when that feature is randomly permuted.
        Higher values = feature contributes more to anomaly detection.

        Args:
            X: Data of shape (n_samples, n_features).

        Returns:
            Importance scores of shape (n_features,).
        """
        X = np.asarray(X, dtype=np.float64)
        baseline_scores = self.anomaly_scores(X)
        importances = np.zeros(X.shape[1])

        for feat_idx in range(X.shape[1]):
            X_permuted = X.copy()
            np.random.shuffle(X_permuted[:, feat_idx])
            permuted_scores = self.anomaly_scores(X_permuted)
            # Importance = drop in detection ability when feature is shuffled
            importances[feat_idx] = np.mean(np.abs(baseline_scores - permuted_scores))

        # Normalize
        total = importances.sum()
        if total > 0:
            importances /= total

        return importances

    @property
    def threshold(self) -> float:
        """Current anomaly threshold."""
        return self._threshold

    @threshold.setter
    def threshold(self, value: float):
        self._threshold = value

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    def get_params(self) -> dict:
        """Get model parameters (for display/serialization)."""
        return {
            "n_estimators": self.n_estimators,
            "max_samples": self.max_samples,
            "contamination": self.contamination,
            "psi_actual": self._psi,
            "n_trees_built": len(self.trees),
            "threshold": self._threshold,
            "fitted": self._fitted,
        }
