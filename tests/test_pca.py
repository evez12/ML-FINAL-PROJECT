import pytest
import numpy as np
from src.unsupervised.pca import PCA
from sklearn.decomposition import PCA as SklearnPCA

# 3 clusters, 180 samples, 5 features
@pytest.fixture
def three_blobs():
    rng = np.random.default_rng(42)
    a = rng.normal([0, 0, 5, -2, 1], 0.6, size=(60, 5))
    b = rng.normal([6, 6, -3, 4, 0], 0.6, size=(60, 5))
    c = rng.normal([-6, 3, 2, -1, -4], 0.6, size=(60, 5))
    return np.vstack([a, b, c])

# fit() should return self
def test_fit_returns_self(three_blobs):
    pca = PCA(n_components=2)
    assert pca.fit(three_blobs) is pca

# check output shapes
def test_output_shapes(three_blobs):
    pca = PCA(n_components=2).fit(three_blobs)
    assert pca.components_.shape == (2, 5)
    assert pca.explained_variance_.shape == (2,)
    assert pca.explained_variance_ratio_.shape == (2,)
    projected = pca.transform(three_blobs)
    assert projected.shape == (180, 2)

# fit_transform() should match fit() then transform()
def test_fit_transform_matches_separate_calls(three_blobs):
    pca_a = PCA(n_components=2)
    combined = pca_a.fit_transform(three_blobs)
    pca_b = PCA(n_components=2).fit(three_blobs)
    separate = pca_b.transform(three_blobs)
    assert np.allclose(combined, separate)

# compare explained variance ratio against sklearn
def test_matches_sklearn_explained_variance_ratio(three_blobs):
    pca = PCA(n_components=2).fit(three_blobs)
    sklearn_pca = SklearnPCA(n_components=2).fit(three_blobs)
    assert np.allclose(
        pca.explained_variance_ratio_,
        sklearn_pca.explained_variance_ratio_,
        atol=1e-6)

# compare raw explained variance against sklearn
def test_matches_sklearn_explained_variance(three_blobs):
    pca = PCA(n_components=2).fit(three_blobs)
    sklearn_pca = SklearnPCA(n_components=2).fit(three_blobs)
    assert np.allclose(
        pca.explained_variance_,
        sklearn_pca.explained_variance_,
        atol=1e-6)

# keeping every component should recover 100% variance
def test_explained_variance_ratio_sums_close_to_one_with_all_components(three_blobs):
    n_features = three_blobs.shape[1]
    pca = PCA(n_components=n_features).fit(three_blobs)
    assert np.isclose(pca.explained_variance_ratio_.sum(), 1.0)

# projected training data should be centered at zero
def test_transformed_training_data_is_centered(three_blobs):
    projected = PCA(n_components=2).fit_transform(three_blobs)
    assert np.allclose(projected.mean(axis=0), 0.0, atol=1e-10)

# transform() before fit() should raise
def test_transform_before_fit_raises():
    pca = PCA(n_components=2)
    with pytest.raises(RuntimeError):
        pca.transform(np.zeros((5, 3)))

# transform() with mismatched feature count should raise
def test_transform_with_wrong_number_of_features_raises(three_blobs):
    pca = PCA(n_components=2).fit(three_blobs)
    with pytest.raises(ValueError):
        pca.transform(np.zeros((10, 4)))

# NaN/Inf input should raise
@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_non_finite_input_raises(three_blobs, bad_value):
    bad = three_blobs.copy()
    bad[0, 0] = bad_value
    with pytest.raises(ValueError):
        PCA(n_components=2).fit(bad)

# NaN/Inf input to transform() should raise
@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_transform_non_finite_input_raises(three_blobs, bad_value):
    pca = PCA(n_components=2).fit(three_blobs)
    bad = three_blobs[:5].copy()
    bad[0, 0] = bad_value
    with pytest.raises(ValueError):
        pca.transform(bad)

# n_components out of valid range should raise
@pytest.mark.parametrize("n_components", [0, -1, 10])
def test_invalid_n_components_raises(three_blobs, n_components):
    with pytest.raises(ValueError):
        PCA(n_components=n_components).fit(three_blobs)

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
        PCA(n_components=1).fit(X)

# single-feature dataset shouldn't crash
def test_single_feature_edge_case():
    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    pca = PCA(n_components=1).fit(X)
    assert pca.components_.shape == (1, 1)
    assert np.isclose(pca.explained_variance_ratio_.sum(), 1.0)

# zero-variance data shouldn't produce NaN
def test_identical_rows_edge_case():
    X = np.ones((10, 3))
    pca = PCA(n_components=2).fit(X)
    projected = pca.transform(X)
    assert projected.shape == (10, 2)
    assert np.allclose(projected, 0.0)
    assert np.all(np.isfinite(pca.explained_variance_))
    assert np.all(np.isfinite(pca.explained_variance_ratio_))
    assert np.allclose(pca.explained_variance_ratio_, 0.0)

# non-integer n_components should raise
@pytest.mark.parametrize("n_components", [1.5, "2", None])
def test_non_integer_n_components_raises(three_blobs, n_components):
    with pytest.raises((TypeError, ValueError)):
        PCA(n_components=n_components).fit(three_blobs)

# 0 or 1 samples should raise, not crash inside eigh()
@pytest.mark.parametrize(
    "X",
    [
        np.empty((0, 3)),
        np.array([[1.0, 2.0, 3.0]]),
    ],
)
def test_insufficient_samples_raises(X):
    with pytest.raises(ValueError):
        PCA(n_components=1).fit(X)