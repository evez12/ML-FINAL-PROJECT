import numpy as np
import pytest

from src.boosting import AdaBoostClassifier, DecisionStump


# I reuse this simple binary data in several tests.
def _binary_data() -> tuple[np.ndarray, np.ndarray]:
    # A stump can separate most of these rows.
    X = np.array(
        [
            [0.0, 0.2],
            [0.1, 0.1],
            [0.2, 0.3],
            [1.0, 0.9],
            [1.1, 1.0],
            [1.2, 0.8],
        ]
    )
    y = np.array([0, 0, 0, 1, 1, 1])
    return X, y


# This data is for checking the SAMME multi-class path.
def _multiclass_data() -> tuple[np.ndarray, np.ndarray]:
    # Three groups are placed on one line.
    X = np.array([[0.0], [0.1], [0.2], [1.0], [1.1], [1.2], [2.0], [2.1], [2.2]])
    y = np.array(["A", "A", "A", "B", "B", "B", "C", "C", "C"])
    return X, y


# A DecisionStump must be only one split deep.
def test_decision_stump_uses_depth_one() -> None:
    # Fit the stump on easy binary data.
    X, y = _binary_data()
    stump = DecisionStump(random_state=3).fit(X, y)

    # Depth can be zero if no split is needed, but not more than one.
    assert stump.depth <= 1
    assert stump.predict(X).shape == y.shape


# Basic AdaBoost fit and predict should work.
def test_adaboost_fit_predict_binary_dataset() -> None:
    # Use a clean binary dataset.
    X, y = _binary_data()

    # Train several stumps with the same seed.
    model = AdaBoostClassifier(n_estimators=10, random_state=7).fit(X, y)
    predictions = model.predict(X)

    # The boosted model should learn this simple pattern.
    assert predictions.shape == y.shape
    assert np.mean(predictions == y) >= 0.9
    assert len(model.estimator_weights) >= 1
    assert len(model.estimator_errors) == len(model.estimator_weights)


# Probability rows must be valid distributions.
def test_predict_proba_rows_sum_to_one() -> None:
    # Fit on the small binary data.
    X, y = _binary_data()
    model = AdaBoostClassifier(n_estimators=5, random_state=2).fit(X, y)

    # Softmax probabilities should have one row per sample.
    probabilities = model.predict_proba(X)
    assert probabilities.shape == (len(X), 2)
    assert np.allclose(probabilities.sum(axis=1), 1.0)


# Same random_state should make the result repeatable.
def test_random_state_is_deterministic() -> None:
    # Train two models with the same settings.
    X, y = _binary_data()
    first = AdaBoostClassifier(n_estimators=8, random_state=42).fit(X, y)
    second = AdaBoostClassifier(n_estimators=8, random_state=42).fit(X, y)

    # Predictions, probabilities and alpha values should match.
    assert np.array_equal(first.predict(X), second.predict(X))
    assert np.allclose(first.predict_proba(X), second.predict_proba(X))
    assert np.allclose(first.estimator_weights, second.estimator_weights)


# staged_predict should show the ensemble after every round.
def test_staged_predict_returns_one_prediction_per_round() -> None:
    # Fit a model that may stop early if it becomes perfect.
    X, y = _binary_data()
    model = AdaBoostClassifier(n_estimators=6, random_state=9).fit(X, y)

    # The number of staged outputs must match fitted stump count.
    stages = list(model.staged_predict(X))
    assert len(stages) == len(model.estimator_weights)
    assert all(stage.shape == y.shape for stage in stages)


# Multi-class SAMME should keep all classes in probability output.
def test_multiclass_samme_path() -> None:
    # This checks that K > 2 does not break the formula.
    X, y = _multiclass_data()
    model = AdaBoostClassifier(n_estimators=12, random_state=4).fit(X, y)

    # All three classes should be present in the model output.
    probabilities = model.predict_proba(X)
    assert list(model.classes_) == ["A", "B", "C"]
    assert probabilities.shape == (len(X), 3)
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert np.mean(model.predict(X) == y) >= 0.7


# Perfect data should stop early instead of adding useless stumps.
def test_perfect_stump_can_stop_early() -> None:
    # One threshold is enough for this data.
    X, y = _binary_data()
    model = AdaBoostClassifier(n_estimators=20, random_state=5).fit(X, y)

    # If the first stump is perfect, the model may stop before 20 rounds.
    assert 1 <= len(model.estimator_weights) <= 20
    assert np.all(model.estimator_errors >= 0.0)


# Bad input values should raise clear errors.
def test_invalid_inputs_raise_clear_errors() -> None:
    # Constructor checks invalid settings.
    with pytest.raises(ValueError, match="n_estimators"):
        AdaBoostClassifier(n_estimators=0)
    with pytest.raises(ValueError, match="learning_rate"):
        AdaBoostClassifier(learning_rate=0.0)
    with pytest.raises(ValueError, match="criterion"):
        AdaBoostClassifier(criterion="bad")

    # Fit needs at least two classes.
    with pytest.raises(ValueError, match="at least two classes"):
        AdaBoostClassifier().fit([[0.0], [1.0]], [1, 1])


# A very bad first weak learner should be rejected.
def test_first_weak_learner_must_beat_random() -> None:
    # Constant features make the first stump no better than random here.
    X = np.ones((4, 1))
    y = np.array([0, 1, 0, 1])

    # With 50 percent error in binary case, boosting should not start.
    with pytest.raises(ValueError, match="weak learner"):
        AdaBoostClassifier(n_estimators=3).fit(X, y)
