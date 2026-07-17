import numpy as np
from typing import Optional

class KMeans:
    """K-Means clustering using Lloyd's algorithm."""

    def __init__(self, n_clusters: int, max_iter: int = 300, tol: float = 1e-4, random_state: Optional[int] = None) -> None:
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.centroids_: Optional[np.ndarray] = None
        self.labels_: Optional[np.ndarray] = None
        self.inertia_: Optional[float] = None
        self.n_iter_: Optional[int] = None

    def fit(self, X: np.ndarray) -> "KMeans":
        """Fit K-Means to X using Lloyd's algorithm."""
        X = np.asarray(X, dtype=float)

        if X.ndim != 2:
            raise ValueError("X must be a 2D array.")
        if X.shape[1] == 0:
            raise ValueError("X must contain at least one feature.")
        if not np.all(np.isfinite(X)):
            raise ValueError("X must not contain NaN or infinite values.")
        if self.n_clusters <= 0:
            raise ValueError("n_clusters must be greater than 0.")
        if self.n_clusters > X.shape[0]:
            raise ValueError("n_clusters cannot exceed the number of samples.")
        if self.max_iter <= 0:
            raise ValueError("max_iter must be greater than 0.")
        if self.tol < 0:
            raise ValueError("tol must not be negative.")

        n_samples, n_features = X.shape
        rng = np.random.default_rng(self.random_state)

        start_idx = rng.choice(n_samples, size=self.n_clusters, replace=False)
        centroids = X[start_idx].copy()

        for iteration in range(self.max_iter):
            distances = self._distances_to_centroids(X, centroids)
            labels = np.argmin(distances, axis=1)

            new_centroids = np.empty_like(centroids)
            for k in range(self.n_clusters):
                points_in_cluster = X[labels == k]
                if len(points_in_cluster) == 0:
                    new_centroids[k] = X[rng.integers(n_samples)]
                else:
                    new_centroids[k] = points_in_cluster.mean(axis=0)

            shift = np.linalg.norm(new_centroids - centroids)
            centroids = new_centroids
            if shift < self.tol:
                break

        distances = self._distances_to_centroids(X, centroids)
        labels = np.argmin(distances, axis=1)
        inertia = np.sum(np.min(distances, axis=1))

        self.centroids_ = centroids
        self.labels_ = labels
        self.inertia_ = inertia
        self.n_iter_ = iteration + 1

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.centroids_ is None:
            raise RuntimeError("call fit() before predict()")
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be a 2D array.")
        if X.shape[1] != self.centroids_.shape[1]:
            raise ValueError("X must have the same number of features as the data used to fit().")
        if not np.all(np.isfinite(X)):
            raise ValueError("X must not contain NaN or infinite values.")
        distances = self._distances_to_centroids(X, self.centroids_)
        return np.argmin(distances, axis=1)

    def _distances_to_centroids(self, X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        diff = X[:, np.newaxis, :] - centroids[np.newaxis, :, :]
        return np.sum(diff ** 2, axis=2)

if __name__ == "__main__":
    rng = np.random.default_rng(42)
    a = rng.normal([0, 0], 0.5, size=(50, 2))
    b = rng.normal([5, 5], 0.5, size=(50, 2))
    c = rng.normal([0, 5], 0.5, size=(50, 2))
    X = np.vstack([a, b, c])

    km = KMeans(n_clusters=3, random_state=42).fit(X)
    print("inertia:", km.inertia_)
    print("iterations to converge:", km.n_iter_)
    print("centroids:\n", km.centroids_)

    import matplotlib.pyplot as plt
    plt.scatter(X[:, 0], X[:, 1], c=km.labels_, cmap="Greens")
    plt.scatter(km.centroids_[:, 0], km.centroids_[:, 1], c="darkgreen", marker="x", s=100)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.show()