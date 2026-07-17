import pytest
import numpy as np
from sklearn.cluster import KMeans as SklearnKMeans
from sklearn.metrics import adjusted_rand_score
from src.unsupervised.kmeans import KMeans

# 3 well-separated clusters for testing
@pytest.fixture
def three_blobs():
    rng = np.random.default_rng(42)
    a = rng.normal([0, 0], 0.5, size=(50, 2))
    b = rng.normal([5, 5], 0.5, size=(50, 2))
    c = rng.normal([0, 5], 0.5, size=(50, 2))
    return np.vstack([a, b, c])

# fit() should return self
def test_fit_returns_self(three_blobs):
    km = KMeans(n_clusters=3)
    assert km.fit(three_blobs) is km

# check output shapes, dtypes, ranges
def test_output_shapes(three_blobs):
    km = KMeans(n_clusters=3, random_state=42).fit(three_blobs)
    assert km.centroids_.shape == (3, 2)
    assert km.labels_.shape == (150,)
    assert np.issubdtype(km.labels_.dtype, np.integer)
    assert np.all((km.labels_ >= 0) & (km.labels_ < 3))
    assert np.isfinite(km.centroids_).all()
    assert np.isfinite(km.inertia_)
    assert km.inertia_ >= 0
    assert isinstance(km.n_iter_, (int, np.integer))
    assert 1 <= km.n_iter_ <= km.max_iter

# compare against sklearn single-run KMeans
def test_matches_sklearn_inertia(three_blobs):
    km = KMeans(n_clusters=3, random_state=42).fit(three_blobs)
    sklearn_km = SklearnKMeans(
        n_clusters=3, random_state=42, n_init=1, init="random",
    ).fit(three_blobs)
    assert adjusted_rand_score(km.labels_, sklearn_km.labels_) > 0.99
    assert np.isclose(km.inertia_, sklearn_km.inertia_, rtol=1e-4)

# inertia should match a manual sum of squared distances
def test_inertia_matches_manual_calculation(three_blobs):
    km = KMeans(n_clusters=3, random_state=42).fit(three_blobs)
    manual_inertia = np.sum((three_blobs - km.centroids_[km.labels_]) ** 2)
    assert np.isclose(km.inertia_, manual_inertia)

# each centroid should be the mean of its assigned points
def test_centroids_equal_assigned_cluster_means(three_blobs):
    km = KMeans(n_clusters=3, random_state=42).fit(three_blobs)
    for k in range(3):
        cluster_points = three_blobs[km.labels_ == k]
        assert len(cluster_points) > 0
        assert np.allclose(km.centroids_[k], cluster_points.mean(axis=0))

# same random_state should give identical results
def test_deterministic_with_same_random_state(three_blobs):
    km_a = KMeans(n_clusters=3, random_state=7).fit(three_blobs)
    km_b = KMeans(n_clusters=3, random_state=7).fit(three_blobs)
    assert np.array_equal(km_a.labels_, km_b.labels_)
    assert np.allclose(km_a.centroids_, km_b.centroids_)
    assert np.isclose(km_a.inertia_, km_b.inertia_)
    assert km_a.n_iter_ == km_b.n_iter_

# predict() on training data should match fit() labels
def test_predict_matches_fit_labels(three_blobs):
    km = KMeans(n_clusters=3, random_state=42).fit(three_blobs)
    assert np.array_equal(km.predict(three_blobs), km.labels_)

# predict() should assign each point to its nearest centroid
def test_predict_assigns_nearest_centroid(three_blobs):
    km = KMeans(n_clusters=3, random_state=42).fit(three_blobs)
    sample = three_blobs[:10]
    predicted = km.predict(sample)
    manual = np.array([
        np.argmin(np.sum((point - km.centroids_) ** 2, axis=1))
        for point in sample
    ])
    assert np.array_equal(predicted, manual)

# predict() before fit() should raise
def test_predict_before_fit_raises():
    km = KMeans(n_clusters=3)
    with pytest.raises(RuntimeError):
        km.predict(np.zeros((5, 2)))

# predict() with mismatched feature count should raise
def test_predict_wrong_number_of_features_raises(three_blobs):
    km = KMeans(n_clusters=3, random_state=42).fit(three_blobs)
    with pytest.raises(ValueError):
        km.predict(np.zeros((10, 3)))

# predict() on non-2D input should raise
def test_predict_non_2d_input_raises(three_blobs):
    km = KMeans(n_clusters=3, random_state=42).fit(three_blobs)
    with pytest.raises(ValueError):
        km.predict(np.array([1.0, 2.0]))

# predict() on NaN/Inf input should raise
@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_predict_non_finite_input_raises(three_blobs, bad_value):
    km = KMeans(n_clusters=3, random_state=42).fit(three_blobs)
    bad = three_blobs[:5].copy()
    bad[0, 0] = bad_value
    with pytest.raises(ValueError):
        km.predict(bad)

# n_clusters <= 0 should raise
@pytest.mark.parametrize("n_clusters", [0, -1])
def test_invalid_n_clusters_raises(three_blobs, n_clusters):
    with pytest.raises(ValueError):
        KMeans(n_clusters=n_clusters).fit(three_blobs)

# n_clusters greater than sample count should raise
def test_n_clusters_exceeds_samples_raises(three_blobs):
    with pytest.raises(ValueError):
        KMeans(n_clusters=1000).fit(three_blobs)

# bool n_clusters should raise, not be treated as int
def test_bool_n_clusters_raises(three_blobs):
    with pytest.raises(TypeError):
        KMeans(n_clusters=True).fit(three_blobs)

# various invalid n_clusters types should raise
@pytest.mark.parametrize(
    "n_clusters, expected_exception",
    [
        (2.5, TypeError),
        (False, ValueError),
        (None, TypeError),
        ("3", TypeError),
    ],
)
def test_invalid_n_clusters_type_raises(three_blobs, n_clusters, expected_exception):
    with pytest.raises(expected_exception):
        KMeans(n_clusters=n_clusters).fit(three_blobs)

# max_iter <= 0 should raise
@pytest.mark.parametrize("max_iter", [0, -1])
def test_invalid_max_iter_raises(three_blobs, max_iter):
    with pytest.raises(ValueError):
        KMeans(n_clusters=3, max_iter=max_iter).fit(three_blobs)

# non-integer max_iter should raise
def test_non_integer_max_iter_raises(three_blobs):
    with pytest.raises(TypeError):
        KMeans(n_clusters=3, max_iter=10.5).fit(three_blobs)

# negative tol should raise
def test_negative_tol_raises(three_blobs):
    with pytest.raises(ValueError):
        KMeans(n_clusters=3, tol=-1.0).fit(three_blobs)

# non-numeric tol should raise
def test_non_numeric_tol_raises(three_blobs):
    with pytest.raises(TypeError):
        KMeans(n_clusters=3, tol="0.001").fit(three_blobs)

# invalid random_state should raise
@pytest.mark.parametrize(
    "random_state, expected_exception",
    [(-5, ValueError), ("abc", TypeError)],
)
def test_invalid_random_state_raises(three_blobs, random_state, expected_exception):
    with pytest.raises(expected_exception):
        KMeans(n_clusters=3, random_state=random_state).fit(three_blobs)

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
        KMeans(n_clusters=1).fit(X)

# NaN/Inf input should raise
@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_non_finite_input_raises(three_blobs, bad_value):
    bad = three_blobs.copy()
    bad[0, 0] = bad_value
    with pytest.raises(ValueError):
        KMeans(n_clusters=3).fit(bad)

# empty input should raise
def test_empty_input_raises():
    with pytest.raises(ValueError):
        KMeans(n_clusters=1).fit(np.empty((0, 3)))

# zero-feature input should raise
def test_zero_feature_input_raises():
    with pytest.raises(ValueError):
        KMeans(n_clusters=1).fit(np.empty((5, 0)))

# n_clusters=1 should give one cluster at the overall mean
def test_single_cluster_edge_case(three_blobs):
    km = KMeans(n_clusters=1, random_state=0).fit(three_blobs)
    assert np.all(km.labels_ == 0)
    assert np.allclose(km.centroids_[0], three_blobs.mean(axis=0))
    manual_inertia = np.sum((three_blobs - three_blobs.mean(axis=0)) ** 2)
    assert np.isclose(km.inertia_, manual_inertia)

# n_clusters == n_samples should give zero inertia
def test_k_equals_n_samples_edge_case():
    X = np.array([[0.0, 0.0], [10.0, 10.0], [20.0, 0.0]])
    km = KMeans(n_clusters=3, random_state=0).fit(X)
    assert len(np.unique(km.labels_)) == 3
    assert set(map(tuple, np.round(km.centroids_, 6))) == set(map(tuple, X))
    assert np.isclose(km.inertia_, 0.0)

# duplicate points shouldn't break convergence
def test_duplicate_points_edge_case():
    X = np.array([[0.0, 0.0], [0.0, 0.0], [1.0, 1.0], [1.0, 1.0]])
    km = KMeans(n_clusters=2, random_state=0).fit(X)
    assert np.isfinite(km.centroids_).all()
    assert np.isclose(km.inertia_, 0.0)
    assert len(np.unique(km.labels_)) == 2

# fit() should not mutate the input array
def test_fit_does_not_modify_input(three_blobs):
    original = three_blobs.copy()
    KMeans(n_clusters=3, random_state=42).fit(three_blobs)
    assert np.array_equal(three_blobs, original)

# predict() should not mutate the input array
def test_predict_does_not_modify_input(three_blobs):
    km = KMeans(n_clusters=3, random_state=42).fit(three_blobs)
    sample = three_blobs[:5].copy()
    original = sample.copy()
    km.predict(sample)
    assert np.array_equal(sample, original)