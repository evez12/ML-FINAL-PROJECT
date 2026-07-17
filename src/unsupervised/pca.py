import numpy as np
from typing import Optional

class PCA:
    """PCA via eigendecomposition of the covariance matrix."""

    def __init__(self, n_components: int) -> None:
        self.n_components = n_components
        self.components_: Optional[np.ndarray] = None
        self.explained_variance_: Optional[np.ndarray] = None
        self.explained_variance_ratio_: Optional[np.ndarray] = None
        self._center: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "PCA":
        X = np.asarray(X, dtype=float)
        n_samples, n_features = X.shape
        if n_samples < 2:
            raise ValueError("X must contain at least 2 samples.")
        if not np.all(np.isfinite(X)):
            raise ValueError("X must not contain NaN or infinite values.")
        if not (1 <= self.n_components <= n_features):
            raise ValueError(f"n_components must be between 1 and {n_features}")

        self._center = X.mean(axis=0)
        centered = X - self._center

        cov_matrix = np.atleast_2d(np.cov(centered, rowvar=False))

        eigvals, eigvecs = np.linalg.eigh(cov_matrix)

        order = np.argsort(eigvals)[::-1]
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]

        self.components_ = eigvecs[:, :self.n_components].T
        self.explained_variance_ = eigvals[:self.n_components]

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