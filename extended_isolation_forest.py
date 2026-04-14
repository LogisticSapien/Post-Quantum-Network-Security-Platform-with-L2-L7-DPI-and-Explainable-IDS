"""
Extended Isolation Forest (EIF)
================================
Hyperplane-split Isolation Forest (Hariri, Kind & Brunner, 2019).
From scratch, no sklearn dependency — uses only NumPy.

Key difference from standard Isolation Forest:
  Standard iForest: axis-parallel splits (x[q] < p)
  Extended iForest:  hyperplane splits  (n·x < p)

This enables detection of anomalies that lie along diagonal or
correlated feature directions — a blind spot of axis-parallel splits.

Math:
  Split at internal node:
    1. Sample random normal vector: n ~ N(0, I_d)
    2. Project data onto n: projections = X · n
    3. Sample random intercept: p ~ Uniform(min(proj), max(proj))
    4. Left child: {x : n·x < p}, Right child: {x : n·x >= p}

  Anomaly score is identical to standard iForest:
    s(x, ψ) = 2^(-E[h(x)] / c(ψ))

Reference:
  Hariri, Kind & Brunner (2019). "Extended Isolation Forest."
  IEEE Transactions on Knowledge and Data Engineering.

Constraints: NumPy only, no sklearn.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from isolation_forest import _c


# ──────────────────────────────────────────────────────────────────────
# Extended Isolation Tree Node
# ──────────────────────────────────────────────────────────────────────

@dataclass
class EITreeNode:
    """A node in an Extended Isolation Tree.

    Internal nodes store a random normal vector and intercept
    for hyperplane splitting. External (leaf) nodes store the
    sample count.
    """
    # Hyperplane split info (internal nodes only)
    normal: Optional[np.ndarray] = None   # Random normal vector, shape (n_features,)
    intercept: Optional[float] = None     # Split intercept p
    left: Optional['EITreeNode'] = None
    right: Optional['EITreeNode'] = None

    # Leaf info
    size: int = 0
    is_external: bool = True


# ──────────────────────────────────────────────────────────────────────
# Extended Isolation Tree
# ──────────────────────────────────────────────────────────────────────

class ExtendedIsolationTree:
    """A single Extended Isolation Tree with hyperplane splits.

    Instead of axis-parallel splits (standard iForest), uses random
    hyperplane splits that can capture diagonal anomaly patterns.

    Args:
        height_limit: Maximum tree depth. Typically ceil(log2(ψ)).
        extension_level: Number of features to use in each split.
            0 = standard iForest (1 feature per split)
            n_features-1 = full EIF (all features in each split)
            Default: n_features-1 (full extension)
    """

    def __init__(self, height_limit: int, extension_level: Optional[int] = None):
        self.height_limit = height_limit
        self.extension_level = extension_level
        self.root: Optional[EITreeNode] = None
        self.n_features: int = 0

    def fit(self, X: np.ndarray) -> 'ExtendedIsolationTree':
        """Build the extended isolation tree from data.

        Args:
            X: Training data of shape (n_samples, n_features).

        Returns:
            self
        """
        self.n_features = X.shape[1]
        if self.extension_level is None:
            self.extension_level = self.n_features - 1
        self.extension_level = min(self.extension_level, self.n_features - 1)
        self.root = self._build(X, depth=0)
        return self

    def _build(self, X: np.ndarray, depth: int) -> EITreeNode:
        """Recursively build the tree with hyperplane splits.

        Args:
            X: Data subset at this node.
            depth: Current depth.

        Returns:
            EITreeNode.
        """
        n_samples, n_features = X.shape

        # Base cases → external node
        if n_samples <= 1 or depth >= self.height_limit:
            return EITreeNode(size=n_samples, is_external=True)

        # Check if all points are identical
        if np.all(X == X[0]):
            return EITreeNode(size=n_samples, is_external=True)

        # Generate random normal vector
        # For extension_level < n_features-1, zero out some dimensions
        n = np.random.randn(n_features)

        if self.extension_level < n_features - 1:
            # Zero out (n_features - 1 - extension_level) random dimensions
            n_zero = n_features - 1 - self.extension_level
            zero_dims = np.random.choice(n_features, size=n_zero, replace=False)
            n[zero_dims] = 0.0

        # Ensure normal vector is not zero
        norm = np.linalg.norm(n)
        if norm < 1e-10:
            n = np.random.randn(n_features)
            norm = np.linalg.norm(n)
            if norm < 1e-10:
                return EITreeNode(size=n_samples, is_external=True)

        n = n / norm  # Unit normal

        # Project data onto normal vector
        projections = X @ n

        # Random intercept between min and max projection
        p_min, p_max = projections.min(), projections.max()
        if abs(p_max - p_min) < 1e-12:
            return EITreeNode(size=n_samples, is_external=True)

        p = np.random.uniform(p_min, p_max)

        # Partition
        left_mask = projections < p
        right_mask = ~left_mask

        # Avoid empty partitions
        if not np.any(left_mask) or not np.any(right_mask):
            return EITreeNode(size=n_samples, is_external=True)

        node = EITreeNode(
            normal=n,
            intercept=p,
            is_external=False,
        )
        node.left = self._build(X[left_mask], depth + 1)
        node.right = self._build(X[right_mask], depth + 1)

        return node

    def path_length(self, x: np.ndarray) -> float:
        """Compute path length for a single sample.

        Args:
            x: Single sample of shape (n_features,).

        Returns:
            Path length h(x).
        """
        return self._path_length(x, self.root, 0)

    def _path_length(self, x: np.ndarray, node: EITreeNode, depth: int) -> float:
        """Recursive path length with hyperplane splits.

        Args:
            x: Sample.
            node: Current node.
            depth: Current depth.

        Returns:
            Path length from root to external node.
        """
        if node.is_external:
            return depth + _c(node.size)

        # Hyperplane split: n·x < p → left, else right
        projection = np.dot(node.normal, x)
        if projection < node.intercept:
            return self._path_length(x, node.left, depth + 1)
        else:
            return self._path_length(x, node.right, depth + 1)

    def path_lengths_batch(self, X: np.ndarray) -> np.ndarray:
        """Compute path lengths for multiple samples.

        Args:
            X: Data of shape (n_samples, n_features).

        Returns:
            Array of path lengths, shape (n_samples,).
        """
        return np.array([self.path_length(x) for x in X])


# ──────────────────────────────────────────────────────────────────────
# Extended Isolation Forest (Ensemble)
# ──────────────────────────────────────────────────────────────────────

class ExtendedIsolationForest:
    """Extended Isolation Forest with hyperplane splits.

    Drop-in replacement for IsolationForest with identical API.
    Uses hyperplane splits to detect diagonal/correlated anomalies
    that axis-parallel splits miss.

    Args:
        n_estimators: Number of Extended Isolation Trees (default: 100).
        max_samples: Subsample size per tree (default: 256).
        contamination: Expected anomaly ratio (default: 0.1).
        extension_level: Dimensions per split. None = full extension.
        random_state: Random seed.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_samples: int = 256,
        contamination: float = 0.1,
        extension_level: Optional[int] = None,
        random_state: Optional[int] = None,
    ):
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.contamination = contamination
        self.extension_level = extension_level
        self.random_state = random_state

        self.trees: List[ExtendedIsolationTree] = []
        self._threshold: float = 0.5
        self._fitted = False
        self._n_samples: int = 0
        self._n_features: int = 0
        self._psi: int = 0

    def fit(self, X: np.ndarray) -> 'ExtendedIsolationForest':
        """Fit the Extended Isolation Forest on training data.

        Args:
            X: Training data of shape (n_samples, n_features).

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

        self._psi = min(self.max_samples, n_samples)
        height_limit = max(1, math.ceil(math.log2(max(self._psi, 2))))

        self.trees = []
        for _ in range(self.n_estimators):
            if n_samples <= self._psi:
                X_sub = X.copy()
            else:
                indices = np.random.choice(n_samples, size=self._psi, replace=False)
                X_sub = X[indices]

            tree = ExtendedIsolationTree(
                height_limit=height_limit,
                extension_level=self.extension_level,
            )
            tree.fit(X_sub)
            self.trees.append(tree)

        self._fitted = True

        if self.contamination > 0:
            scores = self._compute_scores(X)
            self._threshold = float(np.percentile(
                scores, 100 * (1 - self.contamination)
            ))
        else:
            self._threshold = 0.5

        return self

    def _compute_scores(self, X: np.ndarray) -> np.ndarray:
        """Internal scoring."""
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        avg_path_lengths = np.zeros(X.shape[0])
        for tree in self.trees:
            avg_path_lengths += tree.path_lengths_batch(X)
        avg_path_lengths /= len(self.trees)

        cn = _c(self._psi)
        if cn == 0:
            return np.full(X.shape[0], 0.5)

        return np.power(2, -avg_path_lengths / cn)

    def anomaly_scores(self, X: np.ndarray) -> np.ndarray:
        """Compute anomaly scores for samples.

        Returns scores in [0, 1]: ~1.0 = anomaly, ~0.5 = normal.

        Args:
            X: Data of shape (n_samples, n_features).

        Returns:
            Anomaly scores of shape (n_samples,).
        """
        if not self._fitted:
            raise RuntimeError("ExtendedIsolationForest not fitted. Call fit() first.")
        return self._compute_scores(X)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Raw anomaly scores (alias for anomaly_scores)."""
        return self.anomaly_scores(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict anomaly labels: 1 = anomaly, 0 = normal.

        Args:
            X: Data of shape (n_samples, n_features).

        Returns:
            Labels of shape (n_samples,).
        """
        scores = self.anomaly_scores(X)
        return (scores >= self._threshold).astype(int)

    def feature_importance(self, X: np.ndarray) -> np.ndarray:
        """Permutation-based feature importance.

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
            importances[feat_idx] = np.mean(np.abs(baseline_scores - permuted_scores))

        total = importances.sum()
        if total > 0:
            importances /= total

        return importances

    @property
    def threshold(self) -> float:
        return self._threshold

    @threshold.setter
    def threshold(self, value: float):
        self._threshold = value

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    def get_params(self) -> dict:
        """Get model parameters."""
        return {
            "n_estimators": self.n_estimators,
            "max_samples": self.max_samples,
            "contamination": self.contamination,
            "extension_level": self.extension_level,
            "psi_actual": self._psi,
            "n_trees_built": len(self.trees),
            "threshold": self._threshold,
            "fitted": self._fitted,
        }
