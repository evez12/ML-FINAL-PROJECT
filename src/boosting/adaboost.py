"""AdaBoost classifier with DecisionTree stumps.

This file is the P2 part of the project.  The model is written from scratch
with NumPy and uses the P1 DecisionTree only as a weighted decision stump.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from src.trees import DecisionTree


# Small value for safe division and log.
_EPSILON = 1e-10


class DecisionStump(DecisionTree):
    """Depth-1 decision tree used as the weak learner."""

    def __init__(
        self,
        criterion: str = "gini",
        random_state: int | None = None,
    ) -> None:
        # A stump is just a tree with one split level.
        super().__init__(
            max_depth=1,
            min_samples_split=2,
            criterion=criterion,
            random_state=random_state,
        )


class AdaBoostClassifier:
    """Discrete SAMME AdaBoost using weighted decision stumps."""

    def __init__(
        self,
        n_estimators: int = 50,
        learning_rate: float = 1.0,
        criterion: str = "gini",
        random_state: int | None = None,
    ) -> None:
        # I check these values now, because boosting is sensitive to bad setup.
        if n_estimators < 1:
            raise ValueError("n_estimators must be at least 1")
        if learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if criterion not in {"gini", "entropy"}:
            raise ValueError("criterion must be 'gini' or 'entropy'")

        # Save the main options for fit().
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.criterion = criterion
        self.random_state = random_state

        # These are filled after training.
        self.estimators_: list[DecisionStump] = []
        self.classes_: np.ndarray | None = None
        self.n_features_in_: int | None = None
        self._estimator_weights: np.ndarray | None = None
        self._estimator_errors: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "AdaBoostClassifier":
        """Fit boosted decision stumps on the training data."""

        # First I clean the inputs and store class information.
        X_checked = self._check_X(X)
        y_checked = self._check_y(y, X_checked.shape[0])
        self.classes_ = np.unique(y_checked)
        self.n_features_in_ = X_checked.shape[1]

        # AdaBoost needs at least two classes.
        if len(self.classes_) < 2:
            raise ValueError("AdaBoostClassifier needs at least two classes")

        # All rows start with the same weight.
        n_samples = X_checked.shape[0]
        sample_weight = np.full(n_samples, 1.0 / n_samples, dtype=float)

        # Reset old fitted state if fit() is called again.
        self.estimators_ = []
        weights: list[float] = []
        errors: list[float] = []
        n_classes = len(self.classes_)
        stop_error = 1.0 - (1.0 / n_classes)

        for round_index in range(self.n_estimators):
            # This makes every stump repeatable but still different.
            stump_seed = (
                None
                if self.random_state is None
                else int(self.random_state + round_index)
            )
            stump = DecisionStump(
                criterion=self.criterion,
                random_state=stump_seed,
            )

            # The P1 tree reads sample_weight inside its split search.
            stump.fit(X_checked, y_checked, sample_weight=sample_weight)
            prediction = stump.predict(X_checked)
            missed = prediction != y_checked

            # Weighted error is the total weight of wrong rows.
            error = float(np.sum(sample_weight[missed]) / np.sum(sample_weight))

            # If the first stump is not better than random, boosting cannot start.
            if error >= stop_error:
                if not self.estimators_:
                    raise ValueError("first weak learner is not better than random")
                break

            # Perfect stump is clipped, so alpha stays finite.
            clipped_error = float(np.clip(error, _EPSILON, 1.0 - _EPSILON))
            alpha = np.log((1.0 - clipped_error) / clipped_error)
            if n_classes > 2:
                alpha += np.log(n_classes - 1.0)
            alpha *= self.learning_rate

            # Store the stump and its round statistics.
            self.estimators_.append(stump)
            weights.append(float(alpha))
            errors.append(float(error))

            # If it is already perfect, more stumps are not needed.
            if error <= _EPSILON:
                break

            # Wrong rows get bigger weight for the next stump.
            sample_weight *= np.exp(alpha * missed.astype(float))
            sample_weight_sum = float(np.sum(sample_weight))
            if sample_weight_sum <= 0.0 or not np.isfinite(sample_weight_sum):
                raise ValueError("sample weights became invalid during boosting")
            sample_weight /= sample_weight_sum

        # Save arrays for analysis and public properties.
        self._estimator_weights = np.asarray(weights, dtype=float)
        self._estimator_errors = np.asarray(errors, dtype=float)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict classes by weighted stump votes."""

        # The class with the highest vote score is returned.
        scores = self._decision_scores(X)
        encoded = np.argmax(scores, axis=1)
        return self.classes_[encoded]  # type: ignore[index]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return softmax-normalized SAMME vote scores."""

        # Discrete SAMME gives scores, so I turn them into probabilities.
        scores = self._decision_scores(X)
        scores = scores - np.max(scores, axis=1, keepdims=True)
        exp_scores = np.exp(scores)
        totals = np.sum(exp_scores, axis=1, keepdims=True)
        return exp_scores / totals

    @property
    def estimator_weights(self) -> np.ndarray:
        """Return alpha values for fitted stumps."""

        self._require_fitted()
        return self._estimator_weights.copy()  # type: ignore[union-attr]

    @property
    def estimator_errors(self) -> np.ndarray:
        """Return weighted errors for fitted stumps."""

        self._require_fitted()
        return self._estimator_errors.copy()  # type: ignore[union-attr]

    def staged_predict(self, X: np.ndarray) -> Iterator[np.ndarray]:
        """Yield predictions after each boosting round."""

        self._require_fitted()
        X_checked = self._check_X(X, fitting=False)
        scores = np.zeros((X_checked.shape[0], len(self.classes_)), dtype=float)  # type: ignore[arg-type]

        # Add one stump at a time and yield the current ensemble prediction.
        for stump, alpha in zip(self.estimators_, self.estimator_weights):
            stump_prediction = stump.predict(X_checked)
            self._add_votes(scores, stump_prediction, alpha)
            encoded = np.argmax(scores, axis=1)
            yield self.classes_[encoded]  # type: ignore[index]

    def _decision_scores(self, X: np.ndarray) -> np.ndarray:
        # Build class vote scores from all fitted stumps.
        self._require_fitted()
        X_checked = self._check_X(X, fitting=False)
        scores = np.zeros((X_checked.shape[0], len(self.classes_)), dtype=float)  # type: ignore[arg-type]

        for stump, alpha in zip(self.estimators_, self.estimator_weights):
            stump_prediction = stump.predict(X_checked)
            self._add_votes(scores, stump_prediction, alpha)
        return scores

    def _add_votes(
        self,
        scores: np.ndarray,
        prediction: np.ndarray,
        alpha: float,
    ) -> None:
        # Put each stump vote into the right class column.
        class_to_position = {
            label: index for index, label in enumerate(self.classes_)  # type: ignore[union-attr]
        }
        for row_index, label in enumerate(prediction):
            scores[row_index, class_to_position[label]] += alpha

    def _check_X(self, X: np.ndarray, fitting: bool = True) -> np.ndarray:
        # X must be a clean numeric matrix.
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
        # y must match the rows of X.
        array = np.asarray(y)
        if array.ndim != 1:
            raise ValueError("y must be a 1D array")
        if len(array) != n_samples:
            raise ValueError("X and y must contain the same number of samples")
        if len(array) == 0:
            raise ValueError("y must contain at least one label")
        return array

    def _require_fitted(self) -> None:
        # Prediction methods need at least one trained stump.
        if not self.estimators_ or self.classes_ is None:
            raise ValueError("AdaBoostClassifier is not fitted yet")
