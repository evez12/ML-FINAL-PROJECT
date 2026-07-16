"""CART-style decision tree classifier implemented from scratch.

This module is the P1 deliverable.  It intentionally uses only NumPy and
standard-library code for the model implementation.  Scikit-learn may be used
by experiments or tests as a reference baseline, but never by this class.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


# Small number for gain checks and log2, so math stays safe.
_EPSILON = 1e-12


@dataclass
class _Split:
    # I keep split info in one small object.
    feature_index: int
    threshold: float
    gain: float
    left_mask: np.ndarray


@dataclass
class _Node:
    # One object means one place in the tree.
    value: np.ndarray
    samples: int
    weighted_samples: float
    impurity: float
    depth: int
    feature_index: Optional[int] = None
    threshold: Optional[float] = None
    left: Optional["_Node"] = None
    right: Optional["_Node"] = None
    gain: float = 0.0

    @property
    def is_leaf(self) -> bool:
        return self.left is None and self.right is None


class DecisionTree:
    """Binary decision tree classifier using CART split search.

    Parameters mirror the project statement.  The implementation also accepts
    ``sample_weight`` in ``fit`` so AdaBoost can train weighted stumps.
    """

    def __init__(
        self,
        max_depth: int | None = None,
        min_samples_split: int = 2,
        criterion: str = "gini",
        max_features: int | str | None = None,
        random_state: int | None = None,
    ) -> None:
        # I check settings early, because bad values give strange trees.
        if max_depth is not None and max_depth < 0:
            raise ValueError("max_depth must be None or a non-negative integer")
        if min_samples_split < 1:
            raise ValueError("min_samples_split must be at least 1")
        if criterion not in {"gini", "entropy"}:
            raise ValueError("criterion must be 'gini' or 'entropy'")

        # These settings are used later while the tree grows.
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.max_features = max_features
        self.random_state = random_state

        # They are empty now and get values after fit().
        self.root_: _Node | None = None
        self.classes_: np.ndarray | None = None
        self.n_features_in_: int | None = None
        self._rng: np.random.Generator | None = None
        self._raw_importances: np.ndarray | None = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> "DecisionTree":
        """Fit the tree on continuous features and class labels."""

        # First I make the input arrays clean.
        X_checked = self._check_X(X)
        y_checked = self._check_y(y, X_checked.shape[0])
        weights = self._check_sample_weight(sample_weight, X_checked.shape[0])

        # Labels are changed to numbers, so counting is easier.
        self.classes_, y_encoded = np.unique(y_checked, return_inverse=True)
        self.n_features_in_ = X_checked.shape[1]

        # One random generator keeps feature sampling repeatable.
        self._rng = np.random.default_rng(self.random_state)
        self._raw_importances = np.zeros(self.n_features_in_, dtype=float)

        # Training starts from the root.
        self.root_ = self._build_tree(X_checked, y_encoded, weights, depth=0)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return the majority class predicted by each reached leaf."""

        self._require_fitted()

        # The biggest probability becomes the predicted class.
        probabilities = self.predict_proba(X)
        encoded = np.argmax(probabilities, axis=1)
        return self.classes_[encoded]  # type: ignore[index]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return empirical class probabilities from reached leaves."""

        self._require_fitted()
        X_checked = self._check_X(X, fitting=False)
        proba = np.zeros((X_checked.shape[0], len(self.classes_)), dtype=float)  # type: ignore[arg-type]

        # Each row walks down to one leaf.
        for i, row in enumerate(X_checked):
            node = self._traverse(row)
            total = float(np.sum(node.value))

            # This is only a backup for an empty weighted leaf.
            if total <= 0.0:
                proba[i] = np.full(len(node.value), 1.0 / len(node.value))
            else:
                proba[i] = node.value / total
        return proba

    @property
    def depth(self) -> int:
        """Maximum split depth.  A root-only tree has depth 0."""

        self._require_fitted()
        return self._node_depth(self.root_)  # type: ignore[arg-type]

    @property
    def n_leaves(self) -> int:
        """Number of terminal leaves in the fitted tree."""

        self._require_fitted()
        return self._count_leaves(self.root_)  # type: ignore[arg-type]

    def feature_importances(self) -> np.ndarray:
        """Normalized total impurity reduction assigned to each feature."""

        self._require_fitted()

        # Turn raw gains into a normal importance score.
        importances = self._raw_importances.copy()  # type: ignore[union-attr]
        total = float(np.sum(importances))
        if total <= 0.0:
            return importances
        return importances / total

    def __repr__(self) -> str:
        # Before fit, only show the setup.
        if self.root_ is None:
            return (
                "DecisionTree("
                f"max_depth={self.max_depth}, criterion='{self.criterion}', "
                "fitted=False)"
            )

        # Big trees are hard to read, so I print only summary.
        if self.depth > 4:
            return (
                "DecisionTree("
                f"depth={self.depth}, n_leaves={self.n_leaves}, "
                f"criterion='{self.criterion}')"
            )
        return "DecisionTree\n" + self._format_node(self.root_)

    def _build_tree(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray,
        depth: int,
    ) -> _Node:
        # Count class weight in the current node.
        value = self._class_counts(y, sample_weight)
        impurity = self._impurity_from_counts(value)

        # Make the node first, then decide if it must split.
        node = _Node(
            value=value,
            samples=len(y),
            weighted_samples=float(np.sum(sample_weight)),
            impurity=impurity,
            depth=depth,
        )

        # If a stop rule is true, this node stays as leaf.
        if self._should_stop(X, y, depth):
            return node

        # Try to find the best split for this node.
        split = self._best_split(X, y, sample_weight, impurity)
        if split is None:
            return node

        # Save the chosen split inside the node.
        node.feature_index = split.feature_index
        node.threshold = split.threshold
        node.gain = split.gain

        # This gain is used later for feature importance.
        self._raw_importances[split.feature_index] += split.gain  # type: ignore[index]

        # Now build left and right child nodes.
        left_mask = split.left_mask
        right_mask = ~left_mask
        node.left = self._build_tree(
            X[left_mask],
            y[left_mask],
            sample_weight[left_mask],
            depth + 1,
        )
        node.right = self._build_tree(
            X[right_mask],
            y[right_mask],
            sample_weight[right_mask],
            depth + 1,
        )
        return node

    def _should_stop(self, X: np.ndarray, y: np.ndarray, depth: int) -> bool:
        # A node stops when one of these simple rules is true.
        if self.max_depth is not None and depth >= self.max_depth:
            return True
        if len(y) < self.min_samples_split:
            return True
        if np.unique(y).size == 1:
            return True
        if X.shape[0] <= 1:
            return True
        if np.unique(X, axis=0).shape[0] == 1:
            return True
        return False

    def _best_split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray,
        parent_impurity: float,
    ) -> _Split | None:
        # I keep a tiny starting gain, so zero-change split is ignored.
        best: _Split | None = None
        best_gain = _EPSILON
        total_weight = float(np.sum(sample_weight))

        # Random Forest may give only some features for this node.
        feature_indices = self._node_feature_indices(X.shape[1])

        for feature_index in feature_indices:
            # Sort this feature to test thresholds one by one.
            feature = X[:, feature_index]
            order = np.argsort(feature, kind="mergesort")
            sorted_feature = feature[order]
            sorted_y = y[order]
            sorted_weight = sample_weight[order]

            # Same value everywhere means no useful split.
            if sorted_feature[0] == sorted_feature[-1]:
                continue

            # At first, all rows are on the right side.
            right_counts = self._class_counts(sorted_y, sorted_weight)
            left_counts = np.zeros_like(right_counts)

            for position in range(len(sorted_y) - 1):
                # Move one row to the left side.
                class_index = sorted_y[position]
                weight = sorted_weight[position]
                left_counts[class_index] += weight
                right_counts[class_index] -= weight

                # I split only between different neighbour values.
                if sorted_feature[position] == sorted_feature[position + 1]:
                    continue

                left_weight = float(np.sum(left_counts))
                right_weight = float(np.sum(right_counts))

                # Empty left or right side is not a real split.
                if left_weight <= 0.0 or right_weight <= 0.0:
                    continue

                # Check impurity after this possible split.
                left_impurity = self._impurity_from_counts(left_counts)
                right_impurity = self._impurity_from_counts(right_counts)
                weighted_child_impurity = (
                    (left_weight / total_weight) * left_impurity
                    + (right_weight / total_weight) * right_impurity
                )
                gain = parent_impurity - weighted_child_impurity

                # If this split is better, remember it.
                if gain > best_gain:
                    threshold = (
                        float(sorted_feature[position])
                        + float(sorted_feature[position + 1])
                    ) / 2.0
                    left_mask = X[:, feature_index] <= threshold
                    best = _Split(
                        feature_index=int(feature_index),
                        threshold=threshold,
                        gain=float(gain),
                        left_mask=left_mask,
                    )
                    best_gain = float(gain)

        return best

    def _node_feature_indices(self, n_features: int) -> np.ndarray:
        # Choose which features this node may use.
        count = self._resolve_max_features(n_features)
        if count == n_features:
            return np.arange(n_features)
        return self._rng.choice(n_features, size=count, replace=False)  # type: ignore[union-attr]

    def _resolve_max_features(self, n_features: int) -> int:
        # Convert max_features setting to an actual count.
        if self.max_features is None:
            return n_features
        if isinstance(self.max_features, int):
            if self.max_features < 1:
                raise ValueError("max_features as int must be at least 1")
            return min(self.max_features, n_features)
        if self.max_features == "sqrt":
            return max(1, int(np.sqrt(n_features)))
        if self.max_features == "log2":
            return max(1, int(np.log2(n_features)))
        raise ValueError("max_features must be None, int, 'sqrt', or 'log2'")

    def _class_counts(self, y: np.ndarray, sample_weight: np.ndarray) -> np.ndarray:
        # Weighted counts are needed for AdaBoost also.
        n_classes = len(self.classes_)  # type: ignore[arg-type]
        return np.bincount(y, weights=sample_weight, minlength=n_classes).astype(float)

    def _impurity_from_counts(self, counts: np.ndarray) -> float:
        # Impurity is calculated from class probabilities.
        total = float(np.sum(counts))
        if total <= 0.0:
            return 0.0
        probabilities = counts / total

        # Use gini or entropy, based on the setting.
        if self.criterion == "gini":
            return float(1.0 - np.sum(probabilities**2))
        non_zero = probabilities[probabilities > 0.0]
        return float(-np.sum(non_zero * np.log2(non_zero + _EPSILON)))

    def _traverse(self, row: np.ndarray) -> _Node:
        # Go down the tree until there is no split left.
        node = self.root_
        while node is not None and not node.is_leaf:
            if row[node.feature_index] <= node.threshold:  # type: ignore[index,operator]
                node = node.left
            else:
                node = node.right
        return node  # type: ignore[return-value]

    def _node_depth(self, node: _Node) -> int:
        # Depth is the deepest leaf under this node.
        if node.is_leaf:
            return node.depth
        return max(
            self._node_depth(node.left),  # type: ignore[arg-type]
            self._node_depth(node.right),  # type: ignore[arg-type]
        )

    def _count_leaves(self, node: _Node) -> int:
        # Count only final leaf nodes.
        if node.is_leaf:
            return 1
        return self._count_leaves(node.left) + self._count_leaves(node.right)  # type: ignore[arg-type]

    def _format_node(self, node: _Node, indent: str = "") -> str:
        # This helps to print small trees for debugging.
        distribution = np.array2string(node.value, precision=3, separator=", ")
        if node.is_leaf:
            return (
                f"{indent}leaf(samples={node.samples}, "
                f"{self.criterion}={node.impurity:.4f}, value={distribution})\n"
            )
        text = (
            f"{indent}if x[{node.feature_index}] <= {node.threshold:.6g} "
            f"(samples={node.samples}, {self.criterion}={node.impurity:.4f}, "
            f"gain={node.gain:.4f}, value={distribution})\n"
        )
        text += self._format_node(node.left, indent + "  ")  # type: ignore[arg-type]
        text += f"{indent}else\n"
        text += self._format_node(node.right, indent + "  ")  # type: ignore[arg-type]
        return text

    def _check_X(self, X: np.ndarray, fitting: bool = True) -> np.ndarray:
        # X must be numeric and two-dimensional.
        array = np.asarray(X, dtype=float)
        if array.ndim == 1:
            array = array.reshape(-1, 1)
        if array.ndim != 2:
            raise ValueError("X must be a 2D array")
        if array.shape[0] == 0:
            raise ValueError("X must contain at least one sample")
        if not np.isfinite(array).all():
            raise ValueError("X must contain only finite values")
        if not fitting and self.n_features_in_ is not None:
            if array.shape[1] != self.n_features_in_:
                raise ValueError(
                    f"X has {array.shape[1]} features, expected {self.n_features_in_}"
                )
        return array

    def _check_y(self, y: np.ndarray, n_samples: int) -> np.ndarray:
        # y must have exactly one label for each row.
        array = np.asarray(y)
        if array.ndim != 1:
            raise ValueError("y must be a 1D array")
        if len(array) != n_samples:
            raise ValueError("X and y must contain the same number of samples")
        if len(array) == 0:
            raise ValueError("y must contain at least one label")
        return array

    def _check_sample_weight(
        self,
        sample_weight: np.ndarray | None,
        n_samples: int,
    ) -> np.ndarray:
        # Without sample_weight, all rows count the same.
        if sample_weight is None:
            return np.ones(n_samples, dtype=float)

        # Custom weights must match rows and stay non-negative.
        weights = np.asarray(sample_weight, dtype=float)
        if weights.ndim != 1 or len(weights) != n_samples:
            raise ValueError("sample_weight must be a 1D array matching y")
        if np.any(weights < 0.0) or not np.isfinite(weights).all():
            raise ValueError("sample_weight must contain finite non-negative values")
        if float(np.sum(weights)) <= 0.0:
            raise ValueError("sample_weight must have positive total weight")
        return weights

    def _require_fitted(self) -> None:
        # Predict methods should not run before fit().
        if self.root_ is None or self.classes_ is None:
            raise ValueError("DecisionTree is not fitted yet")
