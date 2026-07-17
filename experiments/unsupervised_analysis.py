# Importing library for numerical operations (arrays, math)
import sys
import numpy as np
from pathlib import Path
from typing import Optional
from collections.abc import Iterable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.unsupervised.kmeans import KMeans

def _save_and_show(save_path: Optional[str]) -> None:
    """Save the current figure to disk (if a path was given), show it, then
    close it. Closing matters here since the pipeline creates ~18 figures in
    one run -- without closing, matplotlib keeps every one in memory.
    """
    import matplotlib.pyplot as plt
    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()

def plot_scree(explained_variance_ratio: np.ndarray, save_path: Optional[str] = None,
                n_components_selected: Optional[int] = None) -> None:
    """Plot cumulative explained variance vs number of components kept (scree plot).

    Helps decide how many principal components to keep -- look for where the
    curve crosses a target like 90%, or where it starts to flatten out.

    If n_components_selected is given, a vertical line marks that count on the
    plot with the exact variance it captures (e.g. "7 PCs retained (90.4%
    variance)") -- makes the report figure self-explanatory without needing
    the console output alongside it.

    If save_path is given, the figure is also saved to disk (e.g. into
    figures/unsupervised/) so it can be reused in the report without rerunning
    the pipeline.
    """
    import matplotlib.pyplot as plt
    cumulative_variance = np.cumsum(explained_variance_ratio)
    n_components = range(1, len(cumulative_variance) + 1)
    plt.plot(n_components, cumulative_variance, marker="o", color="green")
    plt.axhline(0.9, color="gray", linestyle="--", label="90% variance")
    if n_components_selected is not None:
        captured = cumulative_variance[n_components_selected - 1]
        plt.axvline(n_components_selected, color="darkgreen", linestyle=":",
                    label=f"{n_components_selected} PCs retained ({captured:.1%} variance)")
    plt.xlabel("Number of components")
    plt.ylabel("Cumulative explained variance")
    plt.title("Scree Plot")
    # forcing every tick label works fine for small feature counts, but with
    # 50+ components (e.g. Covertype) the labels overlap into unreadable mush --
    # thin them out to every 5th tick once there are more than 20 components
    if len(n_components) > 20:
        plt.xticks(range(5, len(n_components) + 1, 5))
    else:
        plt.xticks(list(n_components))
    plt.legend()
    _save_and_show(save_path)

def plot_pca_scatter(X_2d: np.ndarray, labels: np.ndarray, title: str, save_path: Optional[str] = None) -> None:
    """Scatter plot of 2D-PCA-projected data, colored by the given labels.

    Works for true class labels, K-Means cluster labels, or DBSCAN cluster
    labels -- pass in whichever one you want to visualize. DBSCAN's noise
    points (label -1) are drawn separately in gray, since they aren't part
    of any real cluster.

    If save_path is given, the figure is also saved to disk (e.g. into
    figures/unsupervised/) so it can be reused in the report without rerunning
    the pipeline.
    """
    import matplotlib.pyplot as plt
    noise_mask = labels == -1
    unique_labels = np.unique(labels[~noise_mask])

    # letting matplotlib normalize a small number of classes across the full
    # Greens colormap washes the lowest class out to near-white -- pick evenly
    # spaced, clearly visible shades instead, keeping the green theme
    shades = plt.cm.Greens(np.linspace(0.4, 1.0, len(unique_labels)))
    color_lookup = {label: shades[i] for i, label in enumerate(unique_labels)}
    point_colors = np.array([color_lookup[label] for label in labels[~noise_mask]])

    if np.any(noise_mask):
        plt.scatter(X_2d[~noise_mask, 0], X_2d[~noise_mask, 1], c=point_colors)
        plt.scatter(X_2d[noise_mask, 0], X_2d[noise_mask, 1], marker="x", c="gray", label="Noise")
        plt.legend()
    else:
        plt.scatter(X_2d[:, 0], X_2d[:, 1], c=point_colors)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(title)
    _save_and_show(save_path)

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

def plot_elbow(k_values, inertias, save_path: Optional[str] = None, k_used: Optional[int] = None):
    """Plot inertia vs k. If k_used is given, marks it with a vertical line.

    k_used is meant for the k that the final K-Means fit actually ran with --
    labeled explicitly as "not elbow-selected" when it wasn't chosen from this
    curve (e.g. when it's set to the known true class count instead), so the
    plot doesn't misrepresent it as the result of elbow-based model selection.
    """
    import matplotlib.pyplot as plt
    plt.plot(k_values, inertias, marker="o", color="green")
    if k_used is not None:
        plt.axvline(k_used, color="darkgreen", linestyle=":",
                    label=f"k used = {k_used} (true class count, not elbow-selected)")
        plt.legend()
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Inertia")
    plt.title("Elbow Method")
    plt.xticks(list(k_values))
    _save_and_show(save_path)

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

def plot_k_distance(kth_distances: np.ndarray, min_samples: int, save_path: Optional[str] = None,
                     eps: Optional[float] = None):
    """Plot sorted k-distances. If eps is given, marks the chosen eps with a
    horizontal line at that distance -- this is a real selection (from
    find_knee), unlike plot_elbow's k_used caveat.
    """
    import matplotlib.pyplot as plt
    plt.plot(range(len(kth_distances)), kth_distances, color="green")
    if eps is not None:
        plt.axhline(eps, color="darkgreen", linestyle=":", label=f"eps = {eps:.4f} (knee)")
        plt.legend()
    plt.xlabel("Points, sorted by distance")
    plt.ylabel(f"Distance to {min_samples}-th nearest neighbor")
    plt.title("K-distance Plot (for choosing DBSCAN eps)")
    _save_and_show(save_path)

if __name__ == "__main__":
    # This file is also the script run_all.py calls to produce the unsupervised
    # section's required output. The real pipeline (loading the three real
    # datasets, standardizing, running PCA/K-Means/DBSCAN, saving all figures)
    # lives in run_unsupervised.py -- delegate to it here so both
    # `python experiments/run_unsupervised.py` and `python experiments/run_all.py`
    # produce the same real results, instead of this file only running a small
    # synthetic-data smoke test.
    from experiments.run_unsupervised import run_pipeline

    results = [run_pipeline(name) for name in ["wdbc", "adult", "covtype"]]
    print(f"\n{'='*10} SUMMARY {'='*10}")
    for r in results:
        print(r)
