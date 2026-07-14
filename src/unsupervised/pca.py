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
        if not (1 <= self.n_components <= n_features):
            raise ValueError(f"n_components must be between 1 and {n_features}")

        # center the data on zero (subtract the mean of each feature)
        # PCA only cares about how points spread out relative to each other,
        # not where they sit on the axes
        self._center = X.mean(axis=0)
        centered = X - self._center

        # covariance matrix: tells us how much each pair of features
        # varies together (big value = they move together, ~0 = unrelated)
        cov_matrix = np.cov(centered, rowvar=False)

        # eigenvectors of the covariance matrix = the directions of most
        # spread in the data. eigenvalues = how much spread (variance)
        # is along each direction. cov matrix is symmetric, so we use
        # eigh instead of eig -> more stable and always gives real numbers
        eigvals, eigvecs = np.linalg.eigh(cov_matrix)

        # eigh sorts smallest -> largest, but we want the directions with
        # the MOST variance first, so flip the order
        order = np.argsort(eigvals)[::-1]
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]

        # keep only the top n_components directions
        self.components_ = eigvecs[:, :self.n_components].T
        self.explained_variance_ = eigvals[:self.n_components]

        # what fraction of the total variance in the data do these
        # components capture (used for the scree plot later)
        self.explained_variance_ratio_ = self.explained_variance_ / eigvals.sum()

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        # project the data onto the components we kept -> fewer columns,
        # same number of rows
        if self.components_ is None:
            raise RuntimeError("call fit() before transform()")
        X = np.asarray(X, dtype=float)
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