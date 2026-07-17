import pytest
import numpy as np
from sklearn.cluster import DBSCAN as SklearnDBSCAN
from sklearn.metrics import adjusted_rand_score
from src.unsupervised.dbscan import DBSCAN

# two dense clusters plus scattered noise points
@pytest.fixture
def two_blobs_with_noise():
    rng = np.random.default_rng(42)
    a = rng.normal([0, 0], 0.3, size=(30, 2))
    b = rng.normal([5, 5], 0.3, size=(30, 2))
    noise = rng.uniform(-3, 8, size=(10, 2))
    return np.vstack([a, b, noise])

# fit() should return self
def test_fit_returns_self(two_blobs_with_noise):
    d = DBSCAN(eps=0.6, min_samples=5)
    assert d.fit(two_blobs_with_noise) is d

# fit_predict() should return labels_
def test_fit_predict_returns_labels(two_blobs_with_noise):
    d = DBSCAN(eps=0.6, min_samples=5)
    labels = d.fit_predict(two_blobs_with_noise)
    assert np.array_equal(labels, d.labels_)

# fit_predict() and fit()+labels_ should match exactly
def test_fit_predict_matches_fit(two_blobs_with_noise):
    labels = DBSCAN(eps=0.6, min_samples=5).fit_predict(two_blobs_with_noise)
    fitted = DBSCAN(eps=0.6, min_samples=5).fit(two_blobs_with_noise)
    assert np.array_equal(labels, fitted.labels_)

# check output shapes, dtypes, ranges
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

# compare labels, noise count, core points against sklearn
def test_matches_sklearn(two_blobs_with_noise):
    d = DBSCAN(eps=0.6, min_samples=5).fit(two_blobs_with_noise)
    skl = SklearnDBSCAN(eps=0.6, min_samples=5).fit(two_blobs_with_noise)
    assert adjusted_rand_score(d.labels_, skl.labels_) == 1.0
    assert np.sum(d.labels_ == -1) == np.sum(skl.labels_ == -1)
    assert set(d.core_sample_indices_.tolist()) == set(skl.core_sample_indices_.tolist())

# noise points should use label -1
def test_noise_label_is_minus_one(two_blobs_with_noise):
    d = DBSCAN(eps=0.6, min_samples=5).fit(two_blobs_with_noise)
    assert np.sum(d.labels_ == -1) > 0

# a core point's own min_samples neighbors include itself
def test_core_point_definition_includes_self():
    pts = np.array([[0, 0], [0.1, 0], [0, 0.1], [0.1, 0.1], [0.05, 0.05]])
    d_core = DBSCAN(eps=0.2, min_samples=5).fit(pts)
    assert len(d_core.core_sample_indices_) == 5
    d_not_core = DBSCAN(eps=0.2, min_samples=6).fit(pts)
    assert len(d_not_core.core_sample_indices_) == 0
    assert np.all(d_not_core.labels_ == -1)

# border points join a cluster without being core themselves
def test_border_points_join_cluster_without_being_core():
    X = np.array([[i * 0.1, 0.0] for i in range(6)])
    d = DBSCAN(eps=0.15, min_samples=3).fit(X)
    assert np.all(d.labels_ == 0)
    assert set(d.core_sample_indices_.tolist()) == {1, 2, 3, 4}
    assert 0 not in d.core_sample_indices_
    assert 5 not in d.core_sample_indices_

# tiny eps should mark everything as noise
def test_tiny_eps_marks_everything_noise(two_blobs_with_noise):
    d = DBSCAN(eps=1e-6, min_samples=5).fit(two_blobs_with_noise)
    assert np.all(d.labels_ == -1)
    assert len(d.core_sample_indices_) == 0

# huge eps should merge everything into one cluster
def test_huge_eps_merges_everything_into_one_cluster(two_blobs_with_noise):
    d = DBSCAN(eps=1000, min_samples=2).fit(two_blobs_with_noise)
    assert np.all(d.labels_ == 0)
    assert len(d.core_sample_indices_) == len(two_blobs_with_noise)

# min_samples=1 means no point can ever be noise
def test_min_samples_one_has_no_noise():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(10, 2))
    d = DBSCAN(eps=1e-6, min_samples=1).fit(X)
    assert np.all(d.labels_ != -1)
    assert len(d.core_sample_indices_) == len(X)
    assert len(np.unique(d.labels_)) == len(X)

# a single point below min_samples should be noise
def test_single_point_below_min_samples_is_noise():
    d = DBSCAN(eps=0.5, min_samples=2).fit(np.array([[0.0, 0.0]]))
    assert d.labels_[0] == -1

# a single point with min_samples=1 is its own cluster
def test_single_point_with_min_samples_one_is_its_own_cluster():
    d = DBSCAN(eps=0.5, min_samples=1).fit(np.array([[0.0, 0.0]]))
    assert d.labels_[0] == 0

# duplicate points shouldn't confuse clustering
def test_duplicate_points_edge_case():
    X = np.tile([[1.0, 1.0]], (5, 1))
    d = DBSCAN(eps=0.1, min_samples=5).fit(X)
    assert np.all(d.labels_ == 0)
    assert len(d.core_sample_indices_) == 5

# fit() should not mutate the input array
def test_fit_does_not_modify_input(two_blobs_with_noise):
    original = two_blobs_with_noise.copy()
    DBSCAN(eps=0.6, min_samples=5).fit(two_blobs_with_noise)
    assert np.array_equal(two_blobs_with_noise, original)

# non-2D input should raise
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

# empty input should raise
def test_empty_input_raises():
    with pytest.raises(ValueError):
        DBSCAN(eps=0.5, min_samples=5).fit(np.empty((0, 3)))

# zero-feature input should raise
def test_zero_feature_input_raises():
    with pytest.raises(ValueError):
        DBSCAN(eps=0.5, min_samples=5).fit(np.empty((5, 0)))

# NaN/Inf input should raise
@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_non_finite_input_raises(two_blobs_with_noise, bad_value):
    bad = two_blobs_with_noise.copy()
    bad[0, 0] = bad_value
    with pytest.raises(ValueError):
        DBSCAN(eps=0.5, min_samples=5).fit(bad)

# non-numeric eps should raise
def test_eps_not_a_number_raises(two_blobs_with_noise):
    with pytest.raises(TypeError):
        DBSCAN(eps="0.5", min_samples=5).fit(two_blobs_with_noise)

# non-finite eps should raise
@pytest.mark.parametrize("bad_eps", [np.nan, np.inf, -np.inf])
def test_eps_non_finite_raises(two_blobs_with_noise, bad_eps):
    with pytest.raises(ValueError):
        DBSCAN(eps=bad_eps, min_samples=5).fit(two_blobs_with_noise)

# eps <= 0 should raise
@pytest.mark.parametrize("bad_eps", [0, -1])
def test_eps_non_positive_raises(two_blobs_with_noise, bad_eps):
    with pytest.raises(ValueError):
        DBSCAN(eps=bad_eps, min_samples=5).fit(two_blobs_with_noise)

# non-integer min_samples should raise
@pytest.mark.parametrize("bad_min_samples", [5.5, "5", None])
def test_min_samples_not_an_integer_raises(two_blobs_with_noise, bad_min_samples):
    with pytest.raises(TypeError):
        DBSCAN(eps=0.5, min_samples=bad_min_samples).fit(two_blobs_with_noise)

# min_samples <= 0 should raise
@pytest.mark.parametrize("bad_min_samples", [0, -1])
def test_min_samples_non_positive_raises(two_blobs_with_noise, bad_min_samples):
    with pytest.raises(ValueError):
        DBSCAN(eps=0.5, min_samples=bad_min_samples).fit(two_blobs_with_noise)

# bool eps should raise, not be treated as 0/1
@pytest.mark.parametrize("bad_eps", [True, False])
def test_bool_eps_raises(two_blobs_with_noise, bad_eps):
    with pytest.raises(TypeError):
        DBSCAN(eps=bad_eps, min_samples=5).fit(two_blobs_with_noise)

# bool min_samples should raise, not be treated as 0/1
@pytest.mark.parametrize("bad_min_samples", [True, False])
def test_bool_min_samples_raises(two_blobs_with_noise, bad_min_samples):
    with pytest.raises(TypeError):
        DBSCAN(eps=0.5, min_samples=bad_min_samples).fit(two_blobs_with_noise)

# distance exactly equal to eps should count as a neighbor
def test_points_at_exactly_eps_are_neighbors():
    X = np.array([[0.0, 0.0], [0.5, 0.0]])
    d = DBSCAN(eps=0.5, min_samples=2).fit(X)
    assert np.all(d.labels_ == 0)

# points should chain-connect through a bridging core point
def test_chain_connectivity_through_a_bridging_core_point():
    X = np.array([[0.0, 0.0], [0.4, 0.0], [0.8, 0.0]])
    d = DBSCAN(eps=0.5, min_samples=2).fit(X)
    assert np.all(d.labels_ == 0)