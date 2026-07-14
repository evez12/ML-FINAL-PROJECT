# Importing library for numerical operations (arrays, math)
import pytest
import numpy as np
from src.unsupervised.pca import PCA
from sklearn.decomposition import PCA as SklearnPCA

@pytest.fixture
def three_blobs():
    # generate synthetic data with 3 clusters, 180 samples, and 5 features
    rng = np.random.default_rng(42)
    a = rng.normal([0, 0, 5, -2, 1], 0.6, size=(60, 5))
    b = rng.normal([6, 6, -3, 4, 0], 0.6, size=(60, 5))
    c = rng.normal([-6, 3, 2, -1, -4], 0.6, size=(60, 5))
    return np.vstack([a, b, c])

def test_fit_returns_self(three_blobs):
    # fit() should return the object itself, so calls can be chained
    pca = PCA(n_components=2)
    assert pca.fit(three_blobs) is pca

def test_output_shapes(three_blobs):
    # check every output has the shape the interface promises
    pca = PCA(n_components=2).fit(three_blobs)
    assert pca.components_.shape == (2, 5)
    assert pca.explained_variance_.shape == (2,)
    assert pca.explained_variance_ratio_.shape == (2,)
    projected = pca.transform(three_blobs)
    assert projected.shape == (180, 2)

def test_fit_transform_matches_separate_calls(three_blobs):
    # fit_transform(X) should give the exact same result as fit(X) then transform(X)
    pca_a = PCA(n_components=2)
    combined = pca_a.fit_transform(three_blobs)
    pca_b = PCA(n_components=2).fit(three_blobs)
    separate = pca_b.transform(three_blobs)
    assert np.allclose(combined, separate)

def test_matches_sklearn_explained_variance_ratio(three_blobs):
    # the main correctness check: our numbers should match sklearn's
    pca = PCA(n_components=2).fit(three_blobs)
    sklearn_pca = SklearnPCA(n_components=2).fit(three_blobs)
    assert np.allclose(
        pca.explained_variance_ratio_,
        sklearn_pca.explained_variance_ratio_,
        atol=1e-6)

def test_matches_sklearn_explained_variance(three_blobs):
    # same check, but on the raw explained variance, not just the ratio
    pca = PCA(n_components=2).fit(three_blobs)
    sklearn_pca = SklearnPCA(n_components=2).fit(three_blobs)
    assert np.allclose(
        pca.explained_variance_,
        sklearn_pca.explained_variance_,
        atol=1e-6)

def test_explained_variance_ratio_sums_close_to_one_with_all_components(three_blobs):
    # if we keep every possible component, we should recover 100% of the variance
    n_features = three_blobs.shape[1]
    pca = PCA(n_components=n_features).fit(three_blobs)
    assert np.isclose(pca.explained_variance_ratio_.sum(), 1.0)

def test_transformed_training_data_is_centered(three_blobs):
    # projecting the training data onto the components should center it at zero
    projected = PCA(n_components=2).fit_transform(three_blobs)
    assert np.allclose(projected.mean(axis=0), 0.0, atol=1e-10)

def test_transform_before_fit_raises():
    # calling transform() before fit() should fail loudly, not silently misbehave
    pca = PCA(n_components=2)
    with pytest.raises(RuntimeError):
        pca.transform(np.zeros((5, 3)))

def test_transform_with_wrong_number_of_features_raises(three_blobs):
    # fitted on 5 features -> transforming 4-feature data should fail clearly
    pca = PCA(n_components=2).fit(three_blobs)
    with pytest.raises(ValueError):
        pca.transform(np.zeros((10, 4)))

@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_non_finite_input_raises(three_blobs, bad_value):
    # NaN/Inf should be rejected clearly, not silently poison the eigendecomposition
    bad = three_blobs.copy()
    bad[0, 0] = bad_value
    with pytest.raises(ValueError):
        PCA(n_components=2).fit(bad)

@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_transform_non_finite_input_raises(three_blobs, bad_value):
    pca = PCA(n_components=2).fit(three_blobs)
    bad = three_blobs[:5].copy()
    bad[0, 0] = bad_value
    with pytest.raises(ValueError):
        pca.transform(bad)

@pytest.mark.parametrize("n_components", [0, -1, 10])
def test_invalid_n_components_raises(three_blobs, n_components):
    # 0, negative, and more components than features (10 > 5) should all be rejected
    with pytest.raises(ValueError):
        PCA(n_components=n_components).fit(three_blobs)

@pytest.mark.parametrize(
    "X",
    [
        np.array([1.0, 2.0, 3.0]),
        np.zeros((2, 3, 4)),
    ],
)
def test_non_2d_input_raises(X):
    # 1D and 3D arrays should both be rejected, not silently misinterpreted
    with pytest.raises(ValueError):
        PCA(n_components=1).fit(X)

def test_single_feature_edge_case():
    # dataset with only 1 feature -- this used to crash before the np.atleast_2d fix
    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    pca = PCA(n_components=1).fit(X)
    assert pca.components_.shape == (1, 1)
    assert np.isclose(pca.explained_variance_ratio_.sum(), 1.0)

def test_identical_rows_edge_case():
    # every row is exactly the same -> zero variance, should not crash or produce NaN
    X = np.ones((10, 3))
    pca = PCA(n_components=2).fit(X)
    projected = pca.transform(X)
    assert projected.shape == (10, 2)
    assert np.allclose(projected, 0.0)
    assert np.all(np.isfinite(pca.explained_variance_))
    assert np.all(np.isfinite(pca.explained_variance_ratio_))
    assert np.allclose(pca.explained_variance_ratio_, 0.0)

@pytest.mark.parametrize("n_components", [1.5, "2", None])
def test_non_integer_n_components_raises(three_blobs, n_components):
    # n_components must be an int per the interface contract
    with pytest.raises((TypeError, ValueError)):
        PCA(n_components=n_components).fit(three_blobs)

@pytest.mark.parametrize(
    "X",
    [
        np.empty((0, 3)),
        np.array([[1.0, 2.0, 3.0]]),
    ],
)
def test_insufficient_samples_raises(X):
    # 0 or 1 samples -> variance is undefined, should raise a clear error
    # instead of crashing deep inside eigh()
    with pytest.raises(ValueError):
        PCA(n_components=1).fit(X)