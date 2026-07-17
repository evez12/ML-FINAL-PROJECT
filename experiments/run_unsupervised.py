import gc
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import adjusted_rand_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.unsupervised.pca import PCA
from src.unsupervised.kmeans import KMeans
from src.unsupervised.dbscan import DBSCAN
from experiments.unsupervised_analysis import (
    plot_scree,
    plot_pca_scatter,
    elbow_method,
    plot_elbow,
    k_distance_values,
    plot_k_distance,
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures" / "unsupervised"
RANDOM_STATE = 42
KMEANS_LARGE_THRESHOLD = 50000
KMEANS_SAMPLE_SIZE = 3000
DBSCAN_LARGE_THRESHOLD = 5000
DBSCAN_SAMPLE_SIZE = 2000
MIN_SAMPLES = 5

def load_dataset(name: str) -> tuple[np.ndarray, np.ndarray]:
    """Load a preprocessed dataset's train split."""
    X_path = DATA_DIR / f"{name}_X_train_processed.csv"
    y_path = DATA_DIR / f"{name}_y_train_processed.csv"

    if not X_path.exists():
        raise FileNotFoundError(f"Feature file not found: {X_path}")
    if not y_path.exists():
        raise FileNotFoundError(f"Label file not found: {y_path}")

    X = pd.read_csv(X_path).astype(float).to_numpy()
    y = pd.read_csv(y_path).to_numpy().ravel()

    if X.shape[0] != y.shape[0]:
        raise ValueError(f"X and y have different row counts: {X.shape[0]} vs {y.shape[0]}")
    if X.shape[0] == 0 or X.shape[1] == 0:
        raise ValueError(f"Dataset {name!r} is empty.")
    if not np.all(np.isfinite(X)):
        raise ValueError(f"Dataset {name!r} contains NaN or infinite values.")

    return X, y

def standardize(X: np.ndarray) -> np.ndarray:
    """Zero mean, unit variance per feature."""
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1
    return (X - mean) / std

def subsample(X: np.ndarray, y: np.ndarray, n: int, random_state: int) -> tuple[np.ndarray, np.ndarray]:
    """Random subsample of n rows."""
    rng = np.random.default_rng(random_state)
    n = min(n, X.shape[0])
    idx = rng.choice(X.shape[0], size=n, replace=False)
    return X[idx], y[idx]

def find_knee(values: np.ndarray) -> int:
    """Index of the knee in a sorted ascending curve."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 1:
        raise ValueError("values must be a one-dimensional array.")
    if len(values) < 2:
        raise ValueError("At least two values are required to find a knee.")
    if not np.all(np.isfinite(values)):
        raise ValueError("values must contain only finite numbers.")

    n = len(values)
    x_norm = np.arange(n) / (n - 1)
    y_range = values.max() - values.min()
    y_norm = (values - values.min()) / y_range if y_range > 0 else np.zeros(n)
    line_vec = np.array([x_norm[-1] - x_norm[0], y_norm[-1] - y_norm[0]])
    line_vec = line_vec / np.linalg.norm(line_vec)
    point_vecs = np.stack([x_norm - x_norm[0], y_norm - y_norm[0]], axis=1)
    projection_lengths = point_vecs @ line_vec
    projections = np.outer(projection_lengths, line_vec)
    distances_from_line = np.linalg.norm(point_vecs - projections, axis=1)
    return int(np.argmax(distances_from_line))

def report_eps_candidates(X_dbscan: np.ndarray, kdist: np.ndarray, base_eps: float, min_samples: int) -> None:
    """Print cluster count and noise fraction for eps near the knee."""
    print("eps sensitivity check (cluster count / noise fraction, no label info used):")
    for multiplier in [0.8, 0.9, 1.0, 1.1, 1.2]:
        eps = base_eps * multiplier
        if eps <= 0:
            continue
        labels = DBSCAN(eps=eps, min_samples=min_samples).fit(X_dbscan).labels_
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        noise_fraction = np.mean(labels == -1)
        marker = " <- knee" if multiplier == 1.0 else ""
        print(f"  eps={eps:.4f} ({multiplier}x knee): {n_clusters} clusters, "
              f"{noise_fraction:.2%} noise{marker}")

def run_pipeline(name: str) -> dict:
    """Run PCA, K-Means, DBSCAN on one dataset."""
    print(f"\n{'='*10} {name.upper()} {'='*10}")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    X, y = load_dataset(name)
    X = standardize(X)
    n_classes = len(np.unique(y))
    print(f"Loaded {X.shape[0]} samples, {X.shape[1]} features, {n_classes} true classes")

    if min(X.shape) < 2:
        raise ValueError("PCA scatter plot requires at least two samples and two features.")

    max_components = min(X.shape[0], X.shape[1])
    pca_full = PCA(n_components=max_components).fit(X)
    n_components_90 = int(np.searchsorted(np.cumsum(pca_full.explained_variance_ratio_), 0.9) + 1)
    print(f"Components needed for >=90% variance: {n_components_90}/{max_components}")
    plot_scree(pca_full.explained_variance_ratio_, save_path=FIGURES_DIR / f"{name}_scree.png",
               n_components_selected=n_components_90)

    pca_2d = PCA(n_components=2).fit(X)

    if X.shape[0] > KMEANS_LARGE_THRESHOLD:
        X_plot, y_plot = subsample(X, y, KMEANS_SAMPLE_SIZE, RANDOM_STATE)
    else:
        X_plot, y_plot = X, y
    X_2d_plot = pca_2d.transform(X_plot)
    plot_pca_scatter(X_2d_plot, y_plot, f"{name}: True Class Labels",
                      save_path=FIGURES_DIR / f"{name}_true_labels.png")

    X_elbow, _ = subsample(X, y, KMEANS_SAMPLE_SIZE, RANDOM_STATE)
    k_values, inertias = elbow_method(X_elbow, range(2, 11), n_init=10, random_state=RANDOM_STATE)
    plot_elbow(k_values, inertias, save_path=FIGURES_DIR / f"{name}_elbow.png", k_used=n_classes)

    if X.shape[0] > KMEANS_LARGE_THRESHOLD:
        X_kmeans, y_kmeans = subsample(X, y, KMEANS_SAMPLE_SIZE, RANDOM_STATE)
        print(f"Subsampled to {X_kmeans.shape[0]} points for the final K-Means fit (dataset too large in full)")
    else:
        X_kmeans, y_kmeans = X, y
    X_2d_kmeans = pca_2d.transform(X_kmeans)

    km = KMeans(n_clusters=n_classes, random_state=RANDOM_STATE).fit(X_kmeans)
    ari_kmeans = adjusted_rand_score(y_kmeans, km.labels_)
    print(f"K-Means (k={n_classes}, matching true class count) ARI vs true labels: {ari_kmeans:.4f}")
    plot_pca_scatter(X_2d_kmeans, km.labels_, f"{name}: K-Means Cluster Labels",
                      save_path=FIGURES_DIR / f"{name}_kmeans_labels.png")

    del X_elbow, X_kmeans, X_2d_kmeans, X_plot, X_2d_plot
    gc.collect()

    if X.shape[0] > DBSCAN_LARGE_THRESHOLD:
        X_dbscan, y_dbscan = subsample(X, y, DBSCAN_SAMPLE_SIZE, RANDOM_STATE)
        print(f"Subsampled to {X_dbscan.shape[0]} points for DBSCAN (O(n^2) memory cost)")
    else:
        X_dbscan, y_dbscan = X, y
    X_2d_dbscan = pca_2d.transform(X_dbscan)

    kdist = k_distance_values(X_dbscan, min_samples=MIN_SAMPLES)
    if len(kdist) < 2:
        raise ValueError("Not enough k-distance values to estimate DBSCAN eps.")

    eps = float(kdist[find_knee(kdist)])
    if eps <= 0 or not np.isfinite(eps):
        raise ValueError(f"Invalid automatically selected eps value: {eps}")
    plot_k_distance(kdist, min_samples=MIN_SAMPLES, save_path=FIGURES_DIR / f"{name}_k_distance.png", eps=eps)
    report_eps_candidates(X_dbscan, kdist, eps, MIN_SAMPLES)

    db = DBSCAN(eps=eps, min_samples=MIN_SAMPLES).fit(X_dbscan)
    labels = db.labels_
    noise_fraction = float(np.mean(labels == -1))
    n_clusters_found = len(set(labels)) - (1 if -1 in labels else 0)
    ari_dbscan = adjusted_rand_score(y_dbscan, labels)
    print(f"DBSCAN (eps={eps:.4f} from k-distance knee, min_samples={MIN_SAMPLES}) "
          f"found {n_clusters_found} clusters; ARI vs true labels: {ari_dbscan:.4f}; "
          f"noise fraction: {noise_fraction:.2%}")
    plot_pca_scatter(X_2d_dbscan, labels, f"{name}: DBSCAN Cluster Labels",
                      save_path=FIGURES_DIR / f"{name}_dbscan_labels.png")

    return {
        "dataset": name,
        "n_samples": X.shape[0],
        "n_features": X.shape[1],
        "n_classes": n_classes,
        "n_components_90": n_components_90,
        "kmeans_sample_size": len(y_kmeans) if X.shape[0] > KMEANS_LARGE_THRESHOLD else X.shape[0],
        "ari_kmeans": ari_kmeans,
        "dbscan_sample_size": X_dbscan.shape[0],
        "dbscan_n_clusters": n_clusters_found,
        "ari_dbscan": ari_dbscan,
        "dbscan_noise_fraction": noise_fraction,
        "dbscan_eps": eps,
    }

if __name__ == "__main__":
    results = [run_pipeline(name) for name in ["wdbc", "adult", "covtype"]]
    print(f"\n{'='*10} SUMMARY {'='*10}")
    for r in results:
        print(r)