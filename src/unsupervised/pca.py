# Importing library for numerical operations (arrays, math)
import numpy as np

class PCA:
    """PCA via eigendecomposition of the covariance matrix.
    """

    def __init__(self, n_components: int) -> None:
        self.n_components = n_components
        self.components_: np.ndarray = None
        self.explained_variance_: np.ndarray = None
        self.explained_variance_ratio_: np.ndarray = None
        self._center: np.ndarray = None

    def fit(self, X: np.ndarray) -> "PCA":
        X = np.asarray(X, dtype=float)
        n_samples, n_features = X.shape
        # need at least 2 samples -- variance is undefined for 0 or 1 sample,
        # and would otherwise crash later with a confusing LinAlgError
        if n_samples < 2:
            raise ValueError("X must contain at least 2 samples.")
        if not np.all(np.isfinite(X)):
            raise ValueError("X must not contain NaN or infinite values.")
        if not (1 <= self.n_components <= n_features):
            raise ValueError(f"n_components must be between 1 and {n_features}")

        # Center the data on zero (subtract the mean of each feature)
        # PCA only cares about how points spread out relative to each other, not where they sit on the axes
        self._center = X.mean(axis=0)
        centered = X - self._center

        # Covariance matrix: tells us how much each pair of features
        # Varies together (big value = they move together, ~0 = unrelated)
        # np.atleast_2d guards against the single-feature case, where
        # np.cov collapses to a 0-d scalar instead of a (1, 1) matrix
        cov_matrix = np.atleast_2d(np.cov(centered, rowvar=False))

        # Eigenvectors of the covariance matrix = the directions of most spread in the data. 
        # Eigenvalues = how much spread (variance) is along each direction. 
        # Cov matrix is symmetric, so we use eigh instead of eig -> more stable and always gives real numbers
        eigvals, eigvecs = np.linalg.eigh(cov_matrix)

        # Eigh sorts smallest -> largest, but we want the directions with the MOST variance first, so flip the order
        order = np.argsort(eigvals)[::-1]
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]

        # Keep only the top n_components directions
        self.components_ = eigvecs[:, :self.n_components].T
        self.explained_variance_ = eigvals[:self.n_components]

        # What fraction of the total variance in the data do these components capture
        # if every point is identical, total variance is 0 -> avoid a 0/0 NaN and
        # report 0% explained variance instead
        total_variance = eigvals.sum()
        if total_variance > 0:
            self.explained_variance_ratio_ = self.explained_variance_ / total_variance
        else:
            self.explained_variance_ratio_ = np.zeros_like(self.explained_variance_)

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.components_ is None:
            raise RuntimeError("call fit() before transform()")
        X = np.asarray(X, dtype=float)
        if not np.all(np.isfinite(X)):
            raise ValueError("X must not contain NaN or infinite values.")
        return (X - self._center) @ self.components_.T

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

if __name__ == "__main__":
    # quick manual test with fake data, just to see it runs and separates
    # the 3 groups visually before trying it on the real datasets
    rng = np.random.default_rng(42)
    a = rng.normal([0, 0, 5, -2, 1], 0.6, size=(60, 5))
    b = rng.normal([6, 6, -3, 4, 0], 0.6, size=(60, 5))
    c = rng.normal([-6, 3, 2, -1, -4], 0.6, size=(60, 5))
    X = np.vstack([a, b, c])
    labels = np.array([0]*60 + [1]*60 + [2]*60)

    pca = PCA(n_components=2).fit(X)
    X_proj = pca.transform(X)
    print("explained variance ratio:", pca.explained_variance_ratio_)
    print("total variance kept:", pca.explained_variance_ratio_.sum())

    import matplotlib.pyplot as plt
    plt.scatter(X_proj[:, 0], X_proj[:, 1], c=labels, cmap="Greens")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.show()