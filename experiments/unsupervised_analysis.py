# Importing library for numerical operations (arrays, math)
import sys
import numpy as np
from pathlib import Path
from collections.abc import Iterable

# make sure this script can find src/ no matter where it's run from
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.unsupervised.kmeans import KMeans

def elbow_method(X: np.ndarray, k_values: Iterable[int], n_init: int = 10, random_state: int = 42) -> tuple[list[int], list[float]]:
    """Return the best K-Means inertia (lowest of n_init restarts) for each k."""
    X = np.asarray(X, dtype=float)
    k_values = list(k_values)

    if X.ndim != 2:
        raise ValueError("X must be a 2D array.")
    if X.shape[0] == 0:
        raise ValueError("X must contain at least one sample.")
    if X.shape[1] == 0:
        raise ValueError("X must contain at least one feature.")
    if not np.all(np.isfinite(X)):
        raise ValueError("X must not contain NaN or infinite values.")
    if not k_values:
        raise ValueError("k_values must contain at least one value.")
    if not all(isinstance(k, (int, np.integer)) for k in k_values):
        raise TypeError("Every k value must be an integer.")
    if not isinstance(n_init, (int, np.integer)):
        raise TypeError("n_init must be an integer.")
    if n_init <= 0:
        raise ValueError("n_init must be greater than 0.")
    if any(k <= 0 for k in k_values):
        raise ValueError("Every k value must be greater than 0.")
    if any(k > X.shape[0] for k in k_values):
        raise ValueError("A k value cannot exceed the number of samples.")

    inertias: list[float] = []

    for k in k_values:
        # try several random starting points and keep the best (lowest inertia) run,
        # since a single run can land in a bad local minimum
        run_inertias: list[float] = []
        for init_idx in range(n_init):
            seed = random_state + init_idx
            km = KMeans(n_clusters=k, random_state=seed).fit(X)
            if km.inertia_ is None:
                raise RuntimeError("K-Means did not calculate inertia.")
            run_inertias.append(km.inertia_)
        inertias.append(min(run_inertias))

    return k_values, inertias

def plot_elbow(k_values, inertias):
    import matplotlib.pyplot as plt
    plt.plot(k_values, inertias, marker="o", color="green")
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Inertia")
    plt.title("Elbow Method")
    plt.xticks(list(k_values))
    plt.show()

def k_distance_values(X: np.ndarray, min_samples: int) -> np.ndarray:
    """Return sorted distances to each point's min_samples-th neighbor.

    Used to pick a good eps for DBSCAN (look for the "knee" in the plot).
    DBSCAN counts the point itself as one of the min_samples, so we use
    index min_samples - 1 (index 0 in the sorted distances is the point
    itself, at distance 0).
    """
    X = np.asarray(X, dtype=float)

    if X.ndim != 2:
        raise ValueError("X must be a 2D array.")
    if X.shape[0] == 0:
        raise ValueError("X must contain at least one sample.")
    if X.shape[1] == 0:
        raise ValueError("X must contain at least one feature.")
    if not isinstance(min_samples, (int, np.integer)):
        raise TypeError("min_samples must be an integer.")
    if min_samples <= 1:
        raise ValueError("min_samples must be greater than 1.")
    if min_samples > X.shape[0]:
        raise ValueError("min_samples cannot exceed the number of samples.")

    diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
    squared_distances = np.sum(diff ** 2, axis=2)
    sorted_squared_distances = np.sort(squared_distances, axis=1)

    neighbor_index = min_samples - 1
    kth_distances = np.sqrt(sorted_squared_distances[:, neighbor_index])

    return np.sort(kth_distances)

def plot_k_distance(kth_distances: np.ndarray, min_samples: int):
    import matplotlib.pyplot as plt
    plt.plot(range(len(kth_distances)), kth_distances, color="green")
    plt.xlabel("Points, sorted by distance")
    plt.ylabel(f"Distance to {min_samples}-th nearest neighbor")
    plt.title("K-distance Plot (for choosing DBSCAN eps)")
    plt.show()

if __name__ == "__main__":
    # quick manual test with fake 3-cluster data
    rng = np.random.default_rng(42)
    a = rng.normal([0, 0], 0.4, size=(50, 2))
    b = rng.normal([5, 5], 0.4, size=(50, 2))
    c = rng.normal([0, 5], 0.4, size=(50, 2))
    X = np.vstack([a, b, c])

    k_values, inertias = elbow_method(X, range(1, 8))
    print("inertias per k:", [round(i, 2) for i in inertias])
    plot_elbow(k_values, inertias)

    min_samples = 5
    kdist = k_distance_values(X, min_samples=min_samples)
    plot_k_distance(kdist, min_samples=min_samples)