import numpy as np
import pytest

from src.bagging import RandomForestClassifier


# I reuse this small data in the tests below.
def _toy_data() -> tuple[np.ndarray, np.ndarray]:
    # Points are simple, so wrong results are easy to notice.
    X = np.array(
        [
            [0.0, 0.0],
            [0.1, 0.2],
            [0.2, 0.1],
            [1.0, 1.0],
            [1.1, 0.9],
            [0.9, 1.2],
            [2.0, 0.0],
            [2.1, 0.1],
        ]
    )

    # This is a small binary target.
    y = np.array([0, 0, 0, 1, 1, 1, 0, 0])
    return X, y


# Basic fit and predict should work first.
def test_fit_predict_on_small_dataset() -> None:
    # Use the small dataset from the helper.
    X, y = _toy_data()

    # A few trees are enough for this test.
    forest = RandomForestClassifier(
        n_estimators=15,
        max_depth=3,
        random_state=11,
    ).fit(X, y)

    # Prediction count must match label count.
    predictions = forest.predict(X)
    assert predictions.shape == y.shape
    assert np.mean(predictions == y) >= 0.75


# Probability output must look normal.
def test_predict_proba_rows_sum_to_one() -> None:
    # Use same small dataset again.
    X, y = _toy_data()

    # Fit the forest and ask for probabilities.
    forest = RandomForestClassifier(n_estimators=9, random_state=2).fit(X, y)
    probabilities = forest.predict_proba(X)

    # Every row should be a real probability distribution.
    assert probabilities.shape == (len(X), 2)
    assert np.allclose(probabilities.sum(axis=1), 1.0)


# Same random_state should give same result.
def test_random_state_makes_predictions_deterministic() -> None:
    # Use same data for both forests.
    X, y = _toy_data()

    # Same seed means same bootstrap samples and tree seeds.
    first = RandomForestClassifier(n_estimators=11, random_state=42).fit(X, y)
    second = RandomForestClassifier(n_estimators=11, random_state=42).fit(X, y)

    # Both outputs should match exactly or very close.
    assert np.array_equal(first.predict(X), second.predict(X))
    assert np.allclose(first.predict_proba(X), second.predict_proba(X))


# Try every max_features option we support.
@pytest.mark.parametrize("max_features", [None, "sqrt", "log2", 1])
def test_supported_max_features_options(max_features: int | str | None) -> None:
    # Same toy data is enough here.
    X, y = _toy_data()

    # Fit with the current option from parametrize.
    forest = RandomForestClassifier(
        n_estimators=7,
        max_features=max_features,
        random_state=5,
    ).fit(X, y)

    # Prediction should not break for any option.
    assert forest.predict(X).shape == y.shape


# OOB score should be created when user asks for it.
def test_oob_score_is_computed_when_requested() -> None:
    # Use small data, but more trees for OOB.
    X, y = _toy_data()

    # Bootstrap is on by default, so OOB can work.
    forest = RandomForestClassifier(
        n_estimators=25,
        oob_score=True,
        random_state=9,
    ).fit(X, y)

    # Score should be between 0 and 1.
    assert 0.0 <= forest.oob_score_ <= 1.0


# Feature importance should be a clean vector.
def test_feature_importances_are_normalized_when_splits_exist() -> None:
    # Use data where trees can split.
    X, y = _toy_data()

    # Limit depth but still allow useful splits.
    forest = RandomForestClassifier(
        n_estimators=13,
        max_depth=3,
        random_state=4,
    ).fit(X, y)

    # Importance values should be non-negative and sum to one.
    importances = forest.feature_importances_
    assert importances.shape == (X.shape[1],)
    assert np.all(importances >= 0.0)
    assert np.isclose(importances.sum(), 1.0)


# Parallel mode should not change the model result.
def test_parallel_and_sequential_predictions_match() -> None:
    # Use the same input for both runs.
    X, y = _toy_data()

    # One run is sequential, the other uses two jobs.
    sequential = RandomForestClassifier(
        n_estimators=8,
        n_jobs=1,
        random_state=123,
    ).fit(X, y)
    parallel = RandomForestClassifier(
        n_estimators=8,
        n_jobs=2,
        random_state=123,
    ).fit(X, y)

    # Same seed means outputs should match.
    assert np.array_equal(sequential.predict(X), parallel.predict(X))
    assert np.allclose(sequential.predict_proba(X), parallel.predict_proba(X))


# OOB without bootstrap is not allowed.
def test_oob_requires_bootstrap() -> None:
    # There would be no out-of-bag rows in this case.
    with pytest.raises(ValueError, match="bootstrap"):
        RandomForestClassifier(bootstrap=False, oob_score=True)
