import numpy as np
import pytest

from src.trees import DecisionTree


# Small 2D case, easy to check by eye.
def test_tiny_2d_dataset_with_depth_two() -> None:
    # I use four rows so the split result is clear.
    X = np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ]
    )
    y = np.array([0, 0, 1, 0])

    # Depth two is enough for this small example.
    tree = DecisionTree(max_depth=2, random_state=7).fit(X, y)

    # Basic outputs should all look correct.
    assert np.array_equal(tree.predict(X), y)
    assert tree.depth == 2
    assert tree.n_leaves == 3
    assert tree.predict_proba(X).shape == (4, 2)


# One feature should be enough for this dataset.
def test_separable_single_feature_dataset() -> None:
    # Low and high groups are far from each other.
    X = np.array([[0.0], [0.1], [0.2], [1.0], [1.1], [1.2]])
    y = np.array(["low", "low", "low", "high", "high", "high"])

    # I use entropy here, not only gini.
    tree = DecisionTree(max_depth=1, criterion="entropy").fit(X, y)

    # The split should be perfect on this simple data.
    assert np.array_equal(tree.predict(X), y)
    assert np.isclose(tree.feature_importances().sum(), 1.0)
    assert "x[0]" in repr(tree)


# One-class data should not create extra nodes.
def test_all_identical_labels_becomes_single_leaf() -> None:
    # All labels are same, so tree has nothing to learn.
    X = np.array([[0.0, 1.0], [1.0, 1.0], [2.0, 1.0]])
    y = np.array([1, 1, 1])

    # Fit should still work without errors.
    tree = DecisionTree(max_depth=None).fit(X, y)

    # Only root leaf is expected here.
    assert tree.depth == 0
    assert tree.n_leaves == 1
    assert np.array_equal(tree.predict(X), y)
    assert np.allclose(tree.predict_proba(X), np.ones((3, 1)))


# Same features cannot give a real threshold.
def test_identical_features_stop_without_split() -> None:
    # Every row has the same feature values.
    X = np.ones((4, 2))
    y = np.array([0, 1, 0, 1])

    # The tree should stop instead of making fake split.
    tree = DecisionTree().fit(X, y)

    # No split also means no important feature.
    assert tree.depth == 0
    assert tree.n_leaves == 1
    assert np.all(tree.feature_importances() == 0.0)


# max_depth=0 means root only.
def test_max_depth_zero_is_root_only() -> None:
    # Data is separable, but depth limit blocks split.
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0, 0, 1, 1])

    # This should create only one node.
    tree = DecisionTree(max_depth=0).fit(X, y)

    # Prediction is still from the root majority vote.
    assert tree.depth == 0
    assert tree.n_leaves == 1
    assert set(tree.predict(X)).issubset({0, 1})


# min_samples_split=1 should not make infinite recursion.
def test_min_samples_split_one_still_terminates() -> None:
    # This data has a clear border between classes.
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0, 0, 1, 1])

    # Other stop rules should still protect the tree.
    tree = DecisionTree(min_samples_split=1).fit(X, y)

    # It should still learn the simple pattern.
    assert np.array_equal(tree.predict(X), y)
    assert tree.n_leaves >= 2


# Sample weights must change the vote when needed.
def test_sample_weight_changes_leaf_prediction() -> None:
    # Same X values force the answer to come from class weights.
    X = np.array([[0.0], [0.0], [0.0]])
    y = np.array([0, 1, 1])

    # Class 0 gets a bigger weight in the second tree.
    unweighted = DecisionTree().fit(X, y)
    weighted = DecisionTree().fit(X, y, sample_weight=np.array([10.0, 1.0, 1.0]))

    # So the weighted prediction should flip to class 0.
    assert unweighted.predict([[0.0]])[0] == 1
    assert weighted.predict([[0.0]])[0] == 0


# Bad inputs should fail with clear errors.
def test_invalid_inputs_raise_clear_errors() -> None:
    # Criterion name is not valid here.
    with pytest.raises(ValueError, match="criterion"):
        DecisionTree(criterion="misclassification")

    # Total sample weight cannot be zero.
    with pytest.raises(ValueError, match="sample_weight"):
        DecisionTree().fit([[0.0]], [0], sample_weight=[0.0])


# Sklearn is used only as a rough check, not in our model.
def test_matches_sklearn_accuracy_on_simple_data() -> None:
    # If sklearn is missing, pytest skips this part.
    sklearn_tree = pytest.importorskip("sklearn.tree")

    # The rule is simple, based mostly on the first feature.
    rng = np.random.default_rng(42)
    X = rng.normal(size=(80, 2))
    y = (X[:, 0] + 0.25 * X[:, 1] > 0.0).astype(int)

    # Train both trees with the same max depth.
    ours = DecisionTree(max_depth=3, random_state=42).fit(X, y)
    reference = sklearn_tree.DecisionTreeClassifier(max_depth=3, random_state=42)
    reference.fit(X, y)

    # The accuracy should stay close on this easy case.
    ours_accuracy = np.mean(ours.predict(X) == y)
    reference_accuracy = np.mean(reference.predict(X) == y)
    assert abs(ours_accuracy - reference_accuracy) <= 0.02
