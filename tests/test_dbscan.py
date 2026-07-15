# Importing library for numerical operations (arrays, math)
import pytest
import numpy as np
from sklearn.cluster import DBSCAN as SklearnDBSCAN
from sklearn.metrics import adjusted_rand_score
from src.unsupervised.dbscan import DBSCAN

@pytest.fixture
def two_blobs_with_noise():
    # two dense clusters plus scattered points that shouldn't belong to either
    rng = np.random.default_rng(42)
    a = rng.normal([0, 0], 0.3, size=(30, 2))
    b = rng.normal([5, 5], 0.3, size=(30, 2))
    noise = rng.uniform(-3, 8, size=(10, 2))
    return np.vstack([a, b, noise])

def test_fit_returns_self(two_blobs_with_noise):
    d = DBSCAN(eps=0.6, min_samples=5)
    assert d.fit(two_blobs_with_noise) is d

def test_fit_predict_returns_labels(two_blobs_with_noise):
    d = DBSCAN(eps=0.6, min_samples=5)
    labels = d.fit_predict(two_blobs_with_noise)
    assert np.array_equal(labels, d.labels_)

def test_fit_predict_matches_fit(two_blobs_with_noise):
    # fit_predict() on one instance should agree with fit() + labels_ on another --
    # DBSCAN is fully deterministic, so this should be an exact match, not just a
    # high ARI
    labels = DBSCAN(eps=0.6, min_samples=5).fit_predict(two_blobs_with_noise)
    fitted = DBSCAN(eps=0.6, min_samples=5).fit(two_blobs_with_noise)
    assert np.array_equal(labels, fitted.labels_)

def test_output_shapes_and_types(two_blobs_with_noise):
    d = DBSCAN(eps=0.6, min_samples=5).fit(two_blobs_with_noise)
    assert d.labels_.shape == (70,)
    assert np.issubdtype(d.labels_.dtype, np.integer)
    assert np.all(d.labels_ >= -1)
    assert d.core_sample_indices_.ndim == 1
    assert np.issubdtype(d.core_sample_indices_.dtype, np.integer)
    assert np.all(
        (d.core_sample_indices_ >= 0) & (d.core_sample_indices_ < len(two_blobs_with_noise))
    )
    assert len(np.unique(d.core_sample_indices_)) == len(d.core_sample_indices_)

def test_matches_sklearn(two_blobs_with_noise):
    # ARI is permutation-invariant (sklearn may number clusters differently than
    # we do), so it's the right check for the labels themselves. Noise count and
    # the core point set, however, don't depend on cluster numbering at all, so
    # those are checked as exact matches.
    d = DBSCAN(eps=0.6, min_samples=5).fit(two_blobs_with_noise)
    skl = SklearnDBSCAN(eps=0.6, min_samples=5).fit(two_blobs_with_noise)
    assert adjusted_rand_score(d.labels_, skl.labels_) == 1.0
    assert np.sum(d.labels_ == -1) == np.sum(skl.labels_ == -1)
    assert set(d.core_sample_indices_.tolist()) == set(skl.core_sample_indices_.tolist())

def test_noise_label_is_minus_one(two_blobs_with_noise):
    # Noise points must use the label -1.
    d = DBSCAN(eps=0.6, min_samples=5).fit(two_blobs_with_noise)
    assert np.sum(d.labels_ == -1) > 0

def test_core_point_definition_includes_self():
    # 5 mutually close points, min_samples=5 -> a core point needs min_samples
    # neighbors INCLUDING itself, so all 5 should be core (verified against the
    # boundary: min_samples=6 with only 5 points means none can be core)
    pts = np.array([[0, 0], [0.1, 0], [0, 0.1], [0.1, 0.1], [0.05, 0.05]])
    d_core = DBSCAN(eps=0.2, min_samples=5).fit(pts)
    assert len(d_core.core_sample_indices_) == 5
    d_not_core = DBSCAN(eps=0.2, min_samples=6).fit(pts)
    assert len(d_not_core.core_sample_indices_) == 0
    assert np.all(d_not_core.labels_ == -1)

def test_border_points_join_cluster_without_being_core():
    # a line of 6 points spaced 0.1 apart, eps=0.15 -> interior points have 3
    # neighbors each (core, with min_samples=3), the two endpoints have only 2
    # neighbors (not core) but sit within eps of a core point, so they should
    # join the cluster as border points rather than being marked noise
    X = np.array([[i * 0.1, 0.0] for i in range(6)])
    d = DBSCAN(eps=0.15, min_samples=3).fit(X)
    assert np.all(d.labels_ == 0)
    assert set(d.core_sample_indices_.tolist()) == {1, 2, 3, 4}
    assert 0 not in d.core_sample_indices_
    assert 5 not in d.core_sample_indices_

def test_tiny_eps_marks_everything_noise(two_blobs_with_noise):
    d = DBSCAN(eps=1e-6, min_samples=5).fit(two_blobs_with_noise)
    assert np.all(d.labels_ == -1)
    assert len(d.core_sample_indices_) == 0

def test_huge_eps_merges_everything_into_one_cluster(two_blobs_with_noise):
    d = DBSCAN(eps=1000, min_samples=2).fit(two_blobs_with_noise)
    assert np.all(d.labels_ == 0)
    assert len(d.core_sample_indices_) == len(two_blobs_with_noise)

def test_min_samples_one_has_no_noise():
    # every point is within eps of itself, so with min_samples=1 every point
    # qualifies as its own core point -- nothing can ever be noise, and with
    # eps too small to reach any other point, each becomes its own cluster
    rng = np.random.default_rng(0)
    X = rng.normal(size=(10, 2))
    d = DBSCAN(eps=1e-6, min_samples=1).fit(X)
    assert np.all(d.labels_ != -1)
    assert len(d.core_sample_indices_) == len(X)
    assert len(np.unique(d.labels_)) == len(X)

def test_single_point_below_min_samples_is_noise():
    d = DBSCAN(eps=0.5, min_samples=2).fit(np.array([[0.0, 0.0]]))
    assert d.labels_[0] == -1

def test_single_point_with_min_samples_one_is_its_own_cluster():
    d = DBSCAN(eps=0.5, min_samples=1).fit(np.array([[0.0, 0.0]]))
    assert d.labels_[0] == 0

def test_duplicate_points_edge_case():
    X = np.tile([[1.0, 1.0]], (5, 1))
    d = DBSCAN(eps=0.1, min_samples=5).fit(X)
    assert np.all(d.labels_ == 0)
    assert len(d.core_sample_indices_) == 5

def test_fit_does_not_modify_input(two_blobs_with_noise):
    original = two_blobs_with_noise.copy()
    DBSCAN(eps=0.6, min_samples=5).fit(two_blobs_with_noise)
    assert np.array_equal(two_blobs_with_noise, original)

@pytest.mark.parametrize(
    "X",
    [
        np.array([1.0, 2.0, 3.0]),
        np.zeros((2, 3, 4)),
    ],
)
def test_non_2d_input_raises(X):
    with pytest.raises(ValueError):
        DBSCAN(eps=0.5, min_samples=5).fit(X)

def test_empty_input_raises():
    with pytest.raises(ValueError):
        DBSCAN(eps=0.5, min_samples=5).fit(np.empty((0, 3)))

def test_zero_feature_input_raises():
    with pytest.raises(ValueError):
        DBSCAN(eps=0.5, min_samples=5).fit(np.empty((5, 0)))

@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_non_finite_input_raises(two_blobs_with_noise, bad_value):
    bad = two_blobs_with_noise.copy()
    bad[0, 0] = bad_value
    with pytest.raises(ValueError):
        DBSCAN(eps=0.5, min_samples=5).fit(bad)

def test_eps_not_a_number_raises(two_blobs_with_noise):
    with pytest.raises(TypeError):
        DBSCAN(eps="0.5", min_samples=5).fit(two_blobs_with_noise)

@pytest.mark.parametrize("bad_eps", [np.nan, np.inf, -np.inf])
def test_eps_non_finite_raises(two_blobs_with_noise, bad_eps):
    with pytest.raises(ValueError):
        DBSCAN(eps=bad_eps, min_samples=5).fit(two_blobs_with_noise)

@pytest.mark.parametrize("bad_eps", [0, -1])
def test_eps_non_positive_raises(two_blobs_with_noise, bad_eps):
    with pytest.raises(ValueError):
        DBSCAN(eps=bad_eps, min_samples=5).fit(two_blobs_with_noise)

@pytest.mark.parametrize("bad_min_samples", [5.5, "5", None])
def test_min_samples_not_an_integer_raises(two_blobs_with_noise, bad_min_samples):
    with pytest.raises(TypeError):
        DBSCAN(eps=0.5, min_samples=bad_min_samples).fit(two_blobs_with_noise)

@pytest.mark.parametrize("bad_min_samples", [0, -1])
def test_min_samples_non_positive_raises(two_blobs_with_noise, bad_min_samples):
    with pytest.raises(ValueError):
        DBSCAN(eps=0.5, min_samples=bad_min_samples).fit(two_blobs_with_noise)

@pytest.mark.parametrize("bad_eps", [True, False])
def test_bool_eps_raises(two_blobs_with_noise, bad_eps):
    # True/False are technically ints in Python -- confirm they don't slip
    # through as eps=1.0 / eps=0.0
    with pytest.raises(TypeError):
        DBSCAN(eps=bad_eps, min_samples=5).fit(two_blobs_with_noise)

@pytest.mark.parametrize("bad_min_samples", [True, False])
def test_bool_min_samples_raises(two_blobs_with_noise, bad_min_samples):
    with pytest.raises(TypeError):
        DBSCAN(eps=0.5, min_samples=bad_min_samples).fit(two_blobs_with_noise)

def test_points_at_exactly_eps_are_neighbors():
    # distance exactly equal to eps should still count as a neighbor (the
    # implementation compares with <=, not <)
    X = np.array([[0.0, 0.0], [0.5, 0.0]])
    d = DBSCAN(eps=0.5, min_samples=2).fit(X)
    assert np.all(d.labels_ == 0)

def test_chain_connectivity_through_a_bridging_core_point():
    # the two end points are farther apart than eps, but the middle point is
    # within eps of both and is itself core, so all three should merge into
    # one cluster via chained reachability
    X = np.array([[0.0, 0.0], [0.4, 0.0], [0.8, 0.0]])
    d = DBSCAN(eps=0.5, min_samples=2).fit(X)
    assert np.all(d.labels_ == 0)
