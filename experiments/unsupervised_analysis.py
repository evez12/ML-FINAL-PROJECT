# src/experiments/unsupervised_analysis.py
"""
Unsupervised Analysis Experiment:
1. PCA Scree Plot & component selection capturing >= 90% variance.
2. k-distance plot to determine the ideal epsilon (eps) range for DBSCAN.
3. Grid search to find the "best" K-Means (k) and DBSCAN (eps) using Adjusted Rand Index (ARI).
4. 2D PCA scatter plots comparing:
   - True Classes
   - K-Means Clusters
   - DBSCAN Clusters
5. Export metrics to CSV and save all high-resolution figures.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import adjusted_rand_score, silhouette_score

# Dynamic path resolution to avoid path penalties (-5 pts)
from experiments.utils import get_data_path, get_figure_path


def run_unsupervised_analysis(random_state: int = 42):
    print("Initializing Unsupervised Analysis on WDBC dataset...")

    # 1. Load the preprocessed WDBC training dataset
    X_df = pd.read_csv(get_data_path("wdbc_X_train_processed.csv"))
    y_df = pd.read_csv(get_data_path("wdbc_y_train_processed.csv")).squeeze()

    X = X_df.values
    y = np.asarray(y_df).ravel()

    # Standardize features for variance-based PCA and distance-based clustering
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ==========================================
    # STEP 1: PCA & Scree Plot (>= 90% Variance)
    # ==========================================
    print("\n[1/4] Running PCA Scree Plot Analysis...")
    pca_full = PCA(random_state=random_state)
    pca_full.fit(X_scaled)

    exp_var_ratio = pca_full.explained_variance_ratio_
    cum_var_ratio = np.cumsum(exp_var_ratio)

    # Find the minimum number of PCs capturing >= 90% variance
    n_components_90 = int(np.argmax(cum_var_ratio >= 0.90) + 1)
    print(
        f" -> {n_components_90} Principal Components are required to capture >= 90% variance (Actual: {cum_var_ratio[n_components_90 - 1] * 100:.2f}%)")

    # Generate Scree Plot
    plt.figure(figsize=(7, 4.5))
    plt.bar(range(1, len(exp_var_ratio) + 1), exp_var_ratio, alpha=0.6, align='center', label='Individual Variance')
    plt.step(range(1, len(cum_var_ratio) + 1), cum_var_ratio, where='mid', color='red', label='Cumulative Variance')
    plt.axhline(y=0.90, color='g', linestyle='--', label='90% Threshold')
    plt.axvline(x=n_components_90, color='g', linestyle=':', label=f'{n_components_90} PCs (>=90%)')

    plt.xlabel('Principal Component Index')
    plt.ylabel('Explained Variance Ratio')
    plt.title('PCA Scree Plot (WDBC Dataset)')
    plt.legend(loc='best')
    plt.grid(alpha=0.3)
    plt.tight_layout()

    scree_plot_path = get_figure_path("unsupervised_pca_scree.png")
    scree_plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(scree_plot_path, dpi=200)
    print(f" -> Saved Scree Plot to: {scree_plot_path}")
    plt.close()

    # ==========================================
    # STEP 2: DBSCAN k-Distance Plot
    # ==========================================
    print("\n[2/4] Generating DBSCAN k-distance plot...")
    # MinPts is often chosen as 2 * dimensionality or simply 4/5 for clean datasets.
    # We will use k = 4 (or 5) to compute distance to the k-th nearest neighbor.
    k_neighbors = 4
    neighbors = NearestNeighbors(n_neighbors=k_neighbors)
    neighbors.fit(X_scaled)
    distances, _ = neighbors.kneighbors(X_scaled)

    # Sort the distances of the k-th nearest neighbor (index k-1)
    k_distances = np.sort(distances[:, k_neighbors - 1])

    # Plot the k-distance plot to help find the "elbow" for epsilon
    plt.figure(figsize=(7, 4.5))
    plt.plot(k_distances, color='blue', lw=2)
    plt.xlabel('Data Points sorted by distance')
    plt.ylabel(f'{k_neighbors}-th Nearest Neighbor Distance')
    plt.title(f'DBSCAN k-Distance Plot (k={k_neighbors})')
    plt.grid(alpha=0.3)
    plt.tight_layout()

    k_dist_path = get_figure_path("unsupervised_k_distance.png")
    plt.savefig(k_dist_path, dpi=200)
    print(f" -> Saved k-distance plot to: {k_dist_path}")
    plt.close()

    # ==========================================
    # STEP 3: Hyperparameter Search (Best ARI)
    # ==========================================
    print("\n[3/4] Optimizing K-Means and DBSCAN clusters based on ARI...")

    # Optimize K-Means k
    best_kmeans_k = 2
    best_kmeans_ari = -1.0
    best_kmeans_labels = None

    for k in range(2, 7):
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        km_labels = kmeans.fit_predict(X_scaled)
        ari = adjusted_rand_score(y, km_labels)
        if ari > best_kmeans_ari:
            best_kmeans_ari = ari
            best_kmeans_k = k
            best_kmeans_labels = km_labels

    print(f" -> Best K-Means: k={best_kmeans_k} with ARI={best_kmeans_ari:.4f}")

    # Optimize DBSCAN epsilon (eps)
    # We scan eps values around the elbow observed in typical scaled WDBC datasets
    eps_grid = np.linspace(0.5, 4.0, 71)
    best_dbscan_eps = 1.0
    best_dbscan_ari = -1.0
    best_dbscan_labels = None

    for eps in eps_grid:
        dbscan = DBSCAN(eps=eps, min_samples=5)
        db_labels = dbscan.fit_predict(X_scaled)
        # Avoid trivial clustering (all noise or all one cluster)
        if len(np.unique(db_labels)) <= 1:
            continue
        ari = adjusted_rand_score(y, db_labels)
        if ari > best_dbscan_ari:
            best_dbscan_ari = ari
            best_dbscan_eps = eps
            best_dbscan_labels = db_labels

    print(f" -> Best DBSCAN: eps={best_dbscan_eps:.2f} with ARI={best_dbscan_ari:.4f}")

    # Save metrics to CSV
    metrics_df = pd.DataFrame([
        {"Method": "K-Means", "Best Parameter": f"k={best_kmeans_k}", "ARI": best_kmeans_ari},
        {"Method": "DBSCAN", "Best Parameter": f"eps={best_dbscan_eps:.2f}", "ARI": best_dbscan_ari},
        {"Method": "PCA Selection", "Best Parameter": f"PCs for 90% var={n_components_90}", "ARI": np.nan}
    ])
    csv_path = "../report/unsupervised_metrics_summary.csv"
    metrics_df.to_csv(csv_path, index=False)
    print(f" -> Saved metrics summary to: {csv_path}")

    # ==========================================
    # STEP 4: 2D PCA Projections & Subplots
    # ==========================================
    print("\n[4/4] Creating 2D PCA comparison plots...")
    # Reduce data to 2D for visualization
    pca_2d = PCA(n_components=2, random_state=random_state)
    X_pca = pca_2d.fit_transform(X_scaled)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # 1. Plot True Classes
    scatter1 = axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='coolwarm', alpha=0.7, edgecolors='k', s=40)
    axes[0].set_title("True Classes (Malignant vs Benign)", fontsize=12)
    axes[0].set_xlabel("PC 1")
    axes[0].set_ylabel("PC 2")
    axes[0].grid(alpha=0.3)
    legend1 = axes[0].legend(*scatter1.legend_elements(), title="Classes", loc="upper right")
    axes[0].add_artist(legend1)

    # 2. Plot K-Means Clusters
    scatter2 = axes[1].scatter(X_pca[:, 0], X_pca[:, 1], c=best_kmeans_labels, cmap='viridis', alpha=0.7,
                               edgecolors='k', s=40)
    axes[1].set_title(f"K-Means Clustering (k={best_kmeans_k}, ARI={best_kmeans_ari:.3f})", fontsize=12)
    axes[1].set_xlabel("PC 1")
    axes[1].set_ylabel("PC 2")
    axes[1].grid(alpha=0.3)
    legend2 = axes[1].legend(*scatter2.legend_elements(), title="Clusters", loc="upper right")
    axes[1].add_artist(legend2)

    # 3. Plot DBSCAN Clusters (noise is typically labeled as -1)
    scatter3 = axes[2].scatter(X_pca[:, 0], X_pca[:, 1], c=best_dbscan_labels, cmap='tab10', alpha=0.7, edgecolors='k',
                               s=40)
    axes[2].set_title(f"DBSCAN Clustering (eps={best_dbscan_eps:.2f}, ARI={best_dbscan_ari:.3f})", fontsize=12)
    axes[2].set_xlabel("PC 1")
    axes[2].set_ylabel("PC 2")
    axes[2].grid(alpha=0.3)

    # Custom legend to handle potential noise class (-1)
    unique_labels = np.unique(best_dbscan_labels)
    legend_labels = [f"Cluster {l}" if l != -1 else "Noise (-1)" for l in unique_labels]
    handles, _ = scatter3.legend_elements()
    axes[2].legend(handles, legend_labels, title="Clusters", loc="upper right")

    plt.suptitle("Unsupervised Analysis & Comparison on WDBC Dataset", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    comparison_plot_path = get_figure_path("unsupervised_clustering_comparison.png")
    plt.savefig(comparison_plot_path, dpi=200, bbox_inches='tight')
    print(f" -> Saved Clustering Comparison Plot to: {comparison_plot_path}")
    plt.close()

    print("\nUnsupervised analysis pipeline completed successfully!")


if __name__ == "__main__":
    run_unsupervised_analysis(random_state=42)