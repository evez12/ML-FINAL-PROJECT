# Importing library for numerical operations (arrays, math)
import numpy as np
from typing import Optional

class DBSCAN:
    """Density-based clustering (DBSCAN).
    """

    def __init__(self, eps: float, min_samples: int) -> None:
        self.eps = eps
        self.min_samples = min_samples
        self.labels_: Optional[np.ndarray] = None
        self.core_sample_indices_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "DBSCAN":
        """Run DBSCAN clustering on X.
        """
        X = np.asarray(X, dtype=float)

        if X.ndim != 2:
            raise ValueError("X must be a 2D array.")
        if X.shape[0] == 0:
            raise ValueError("X must contain at least one sample.")
        if X.shape[1] == 0:
            raise ValueError("X must contain at least one feature.")
        if not np.all(np.isfinite(X)):
            raise ValueError("X must not contain NaN or infinite values.")
        if not isinstance(self.eps, (int, float, np.integer, np.floating)):
            raise TypeError("eps must be a number.")
        if not np.isfinite(self.eps):
            raise ValueError("eps must be finite.")
        if self.eps <= 0:
            raise ValueError("eps must be greater than 0.")
        if not isinstance(self.min_samples, (int, np.integer)):
            raise TypeError("min_samples must be an integer.")
        if self.min_samples <= 0:
            raise ValueError("min_samples must be greater than 0.")

        n_samples = X.shape[0]

        # squared distance from every point to every other point, shape (n_samples, n_samples)
        # comparing squared distances against eps**2 skips a costly sqrt over the
        # whole matrix -- same neighbor relationships, cheaper to compute
        sq_dist_matrix = self._pairwise_squared_distances(X)
        neighbor_mask = sq_dist_matrix <= self.eps ** 2

        # a point is a "core point" if it has at least min_samples points
        # (including itself) within eps
        is_core = neighbor_mask.sum(axis=1) >= self.min_samples

        labels = np.full(n_samples, -1, dtype=int)  # -1 means noise, for now
        visited = np.zeros(n_samples, dtype=bool)
        cluster_id = 0

        for i in range(n_samples):
            if visited[i]:
                continue
            visited[i] = True

            if not is_core[i]:
                # not enough neighbors to start a cluster -> stays noise
                # (it may still get picked up below as a border point of
                # some other core point's cluster)
                continue

            # i is a core point -> start a new cluster and grow it outward
            labels[i] = cluster_id
            seeds = list(np.where(neighbor_mask[i])[0])

            # track what's already queued so a point can't get pushed onto
            # seeds more than once as different core points expand into it
            in_seeds = np.zeros(n_samples, dtype=bool)
            in_seeds[seeds] = True

            j = 0
            while j < len(seeds):
                q = seeds[j]
                if not visited[q]:
                    visited[q] = True
                    if is_core[q]:
                        # q is also a core point, its unqueued neighbors join the search too
                        q_neighbors = np.where(neighbor_mask[q])[0]
                        new_neighbors = q_neighbors[~in_seeds[q_neighbors]]
                        seeds.extend(new_neighbors.tolist())
                        in_seeds[new_neighbors] = True
                if labels[q] == -1:
                    # q hasn't been claimed by a cluster yet -> it becomes part
                    # of this cluster (either a core point itself or a border point)
                    labels[q] = cluster_id
                j += 1

            cluster_id += 1

        self.labels_ = labels
        self.core_sample_indices_ = np.where(is_core)[0]

        return self

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """Fit DBSCAN to X and return the resulting cluster labels."""
        self.fit(X)
        if self.labels_ is None:
            raise RuntimeError("DBSCAN did not produce labels.")
        return self.labels_

    def _pairwise_squared_distances(self, X: np.ndarray) -> np.ndarray:
        # squared euclidean distance between every pair of points, shape (n_samples, n_samples)
        # time and memory are both O(n^2) because the full distance matrix is
        # stored -- for large datasets a KD-tree/ball-tree would be needed instead
        diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
        return np.sum(diff ** 2, axis=2)

if __name__ == "__main__":
    # quick manual test: two dense blobs plus some scattered noise points
    rng = np.random.default_rng(42)
    a = rng.normal([0, 0], 0.3, size=(60, 2))
    b = rng.normal([5, 5], 0.3, size=(60, 2))
    noise = rng.uniform(-3, 8, size=(10, 2))
    X = np.vstack([a, b, noise])

    labels = DBSCAN(eps=0.6, min_samples=5).fit_predict(X)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = np.sum(labels == -1)
    print("clusters found:", n_clusters)
    print("noise points:", n_noise)

    import matplotlib.pyplot as plt
    noise_mask = labels == -1
    plt.scatter(X[~noise_mask, 0], X[~noise_mask, 1], c=labels[~noise_mask], cmap="Greens")
    plt.scatter(X[noise_mask, 0], X[noise_mask, 1], marker="x", c="gray", label="Noise")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend()
    plt.show()