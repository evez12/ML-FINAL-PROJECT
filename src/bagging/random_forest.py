"""Random Forest classifier implemented with P1 DecisionTree base learners."""

from __future__ import annotations

from multiprocessing import Pool
from typing import Any

import numpy as np

from src.trees import DecisionTree


# Pool works better when this helper stays outside the class.
def _fit_single_tree(args: tuple[Any, ...]) -> tuple[DecisionTree, np.ndarray]:
    """Fit one tree and return it with its out-of-bag indices."""

    # Each task has all info for one tree.
    (
        X,
        y,
        sample_indices,
        oob_indices,
        max_depth,
        max_features,
        min_samples_split,
        random_state,
    ) = args

    # The forest uses our P1 DecisionTree as base model.
    tree = DecisionTree(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        criterion="gini",
        max_features=max_features,
        random_state=random_state,
    )

    # This tree learns only from its sampled rows.
    tree.fit(X[sample_indices], y[sample_indices])
    return tree, oob_indices


class RandomForestClassifier:
    """Bootstrap aggregation over the project DecisionTree implementation."""

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int | None = None,
        max_features: int | str | None = "sqrt",
        min_samples_split: int = 2,
        bootstrap: bool = True,
        oob_score: bool = False,
        n_jobs: int = 1,
        random_state: int | None = None,
    ) -> None:
        # I check settings here, so fit does not fail later.
        if n_estimators < 1:
            raise ValueError("n_estimators must be at least 1")
        if max_depth is not None and max_depth < 0:
            raise ValueError("max_depth must be None or a non-negative integer")
        if min_samples_split < 1:
            raise ValueError("min_samples_split must be at least 1")
        if n_jobs < 1:
            raise ValueError("n_jobs must be at least 1")
        if oob_score and not bootstrap:
            raise ValueError("oob_score=True requires bootstrap=True")

        # Save all options for training.
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.min_samples_split = min_samples_split
        self.bootstrap = bootstrap
        self.oob_score = oob_score
        self.n_jobs = n_jobs
        self.random_state = random_state

        # These are empty before fit().
        self.estimators_: list[DecisionTree] = []
        self.oob_indices_: list[np.ndarray] = []
        self.classes_: np.ndarray | None = None
        self.n_features_in_: int | None = None
        self._oob_score: float | None = None
        self._feature_importances: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForestClassifier":
        """Fit each tree on a bootstrap sample of the training data."""

        # First make sure X and y have proper shape.
        X_checked = self._check_X(X)
        y_checked = self._check_y(y, X_checked.shape[0])

        # Save basic dataset information for predict time.
        self.classes_ = np.unique(y_checked)
        self.n_features_in_ = X_checked.shape[1]

        # Decide feature subset size for every tree split.
        max_features = self._resolve_max_features(self.n_features_in_)
        rng = np.random.default_rng(self.random_state)

        # Give every tree its own seed, but keep forest repeatable.
        tree_seeds = rng.integers(
            low=0,
            high=np.iinfo(np.int32).max,
            size=self.n_estimators,
            dtype=np.int64,
        )

        tasks = []
        n_samples = X_checked.shape[0]
        all_indices = np.arange(n_samples)
        for seed in tree_seeds:
            # A separate generator makes each tree a little different.
            tree_rng = np.random.default_rng(int(seed))
            if self.bootstrap:
                # Bagging means sampling rows with replacement.
                sample_indices = tree_rng.integers(0, n_samples, size=n_samples)

                # Rows not sampled can be used as OOB rows.
                in_bag = np.zeros(n_samples, dtype=bool)
                in_bag[sample_indices] = True
                oob_indices = all_indices[~in_bag]
            else:
                # If bootstrap is off, the tree gets all rows.
                sample_indices = all_indices.copy()
                oob_indices = np.array([], dtype=int)

            # Save the job; later it can run in one process or more.
            tasks.append(
                (
                    X_checked,
                    y_checked,
                    sample_indices,
                    oob_indices,
                    self.max_depth,
                    max_features,
                    self.min_samples_split,
                    int(seed),
                )
            )

        # Train the forest, either normal way or with Pool.
        if self.n_jobs == 1:
            fitted = [_fit_single_tree(task) for task in tasks]
        else:
            with Pool(processes=self.n_jobs) as pool:
                fitted = pool.map(_fit_single_tree, tasks)

        # Keep fitted trees and their OOB index lists.
        self.estimators_ = [tree for tree, _ in fitted]
        self.oob_indices_ = [indices for _, indices in fitted]

        # Prepare values that user can read after fit.
        self._feature_importances = self._compute_feature_importances()
        self._oob_score = self._compute_oob_score(X_checked, y_checked)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict classes by averaging tree probabilities."""

        # Final class comes from the largest average probability.
        probabilities = self.predict_proba(X)
        encoded = np.argmax(probabilities, axis=1)
        return self.classes_[encoded]  # type: ignore[index]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Average class probabilities from all fitted trees."""

        self._require_fitted()
        X_checked = self._check_X(X, fitting=False)
        total = np.zeros((X_checked.shape[0], len(self.classes_)), dtype=float)  # type: ignore[arg-type]

        # Sum probabilities from every tree.
        for tree in self.estimators_:
            total += self._aligned_tree_proba(tree, X_checked)

        # Divide by tree count to get the average.
        return total / len(self.estimators_)

    @property
    def oob_score_(self) -> float:
        """Out-of-bag accuracy computed from trees where each sample was OOB."""

        self._require_fitted()
        if self._oob_score is None:
            raise AttributeError("oob_score_ is available only after fit")
        return self._oob_score

    @property
    def feature_importances_(self) -> np.ndarray:
        """Mean normalized feature importances across all trees."""

        self._require_fitted()
        return self._feature_importances.copy()  # type: ignore[union-attr]

    def _compute_oob_score(self, X: np.ndarray, y: np.ndarray) -> float | None:
        # OOB score is optional, so return None when disabled.
        if not self.oob_score:
            return None

        # A row can vote only from trees that did not train on it.
        votes = np.zeros((X.shape[0], len(self.classes_)), dtype=float)  # type: ignore[arg-type]
        counts = np.zeros(X.shape[0], dtype=int)
        for tree, oob_indices in zip(self.estimators_, self.oob_indices_):
            if len(oob_indices) == 0:
                continue
            votes[oob_indices] += self._aligned_tree_proba(tree, X[oob_indices])
            counts[oob_indices] += 1

        # With few trees, some rows may get no OOB votes.
        has_vote = counts > 0
        if not np.any(has_vote):
            return float("nan")

        # Average OOB probabilities and compare with true labels.
        averaged = votes[has_vote] / counts[has_vote, None]
        predictions = self.classes_[np.argmax(averaged, axis=1)]  # type: ignore[index]
        return float(np.mean(predictions == y[has_vote]))

    def _compute_feature_importances(self) -> np.ndarray:
        # Start with zero score for every feature.
        importances = np.zeros(self.n_features_in_, dtype=float)  # type: ignore[arg-type]

        # Add importance values from all trees.
        for tree in self.estimators_:
            importances += tree.feature_importances()
        importances /= len(self.estimators_)

        # Normalize again so the final sum is one.
        total = float(np.sum(importances))
        if total <= 0.0:
            return importances
        return importances / total

    def _aligned_tree_proba(self, tree: DecisionTree, X: np.ndarray) -> np.ndarray:
        # One tree may not see every class in bootstrap data.
        raw = tree.predict_proba(X)
        aligned = np.zeros((X.shape[0], len(self.classes_)), dtype=float)  # type: ignore[arg-type]

        # Put tree columns into the forest class order.
        class_to_position = {label: i for i, label in enumerate(self.classes_)}  # type: ignore[union-attr]
        for tree_position, label in enumerate(tree.classes_):
            aligned[:, class_to_position[label]] = raw[:, tree_position]
        return aligned

    def _resolve_max_features(self, n_features: int) -> int | None:
        # Convert max_features to the value DecisionTree expects.
        if self.max_features is None:
            return None
        if isinstance(self.max_features, int):
            if self.max_features < 1:
                raise ValueError("max_features as int must be at least 1")
            return min(self.max_features, n_features)
        if self.max_features == "sqrt":
            return max(1, int(np.sqrt(n_features)))
        if self.max_features == "log2":
            return max(1, int(np.log2(n_features)))
        raise ValueError("max_features must be None, int, 'sqrt', or 'log2'")

    def _check_X(self, X: np.ndarray, fitting: bool = True) -> np.ndarray:
        # X must be a numeric 2D array.
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
        # y must contain one label for each row.
        array = np.asarray(y)
        if array.ndim != 1:
            raise ValueError("y must be a 1D array")
        if len(array) != n_samples:
            raise ValueError("X and y must contain the same number of samples")
        if len(array) == 0:
            raise ValueError("y must contain at least one label")
        return array

    def _require_fitted(self) -> None:
        # Predict and properties need fitted trees.
        if not self.estimators_ or self.classes_ is None:
            raise ValueError("RandomForestClassifier is not fitted yet")
