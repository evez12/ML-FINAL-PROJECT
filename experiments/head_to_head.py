# Experiment 4: Head-to-head (100 estimators, 5-fold CV)
"""
Run a head-to-head comparison between:
 - single DecisionTree (project implementation)
 - AdaBoost (project implementation)
 - RandomForest (project implementation)
 - sklearn.ensemble.RandomForestClassifier (reference)

For reproducibility and style this file follows the experiments/scaling.py and
experiments/rf_scaling.py conventions: functions accept already-loaded arrays
and save CSV + plots to report/figures by default.

Metrics per fold: accuracy, macro F1, AUC-ROC. 5-fold Stratified CV. Report mean
± std and produce box plots for each metric.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple

from sklearn.ensemble import RandomForestClassifier as SklearnRF
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from src.trees import DecisionTree
from src.boosting.adaboost import AdaBoostClassifier
from src.bagging.random_forest import RandomForestClassifier as ProjectRF
from utils import get_figure_path
from utils import get_data_path

def _ensure_arrays(X, y) -> Tuple[np.ndarray, np.ndarray]:
    X_np = X.values if hasattr(X, "values") else np.asarray(X)
    y_np = np.asarray(y).ravel()
    return X_np, y_np

def run_head_to_head_arrays(
    X,
    y,
    dataset: str = "dataset",
    out_dir: str = "../report/figures",
    n_estimators: int = 100,
    cv: int = 5,
    random_state: int = 42,
    show_plot: bool = False,
):
    """Run 5-fold CV comparing models and save CSV + boxplots.

    Returns: (results_df, plot_paths)
    results_df columns: model, fold, accuracy, f1_macro, auc_roc
    """

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_dir = out_dir.parent
    results_dir.mkdir(parents=True, exist_ok=True)

    X_np, y_np = _ensure_arrays(X, y)

    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)

    models = {
        "DecisionTree": lambda: DecisionTree(max_depth=None, random_state=random_state),
        "AdaBoost": lambda: AdaBoostClassifier(n_estimators=n_estimators, learning_rate=1.0, criterion="gini", random_state=random_state),
        "ProjectRandomForest": lambda: ProjectRF(n_estimators=n_estimators, max_depth=None, oob_score=False, random_state=random_state, n_jobs=1),
        "SklearnRandomForest": lambda: SklearnRF(n_estimators=n_estimators, max_depth=None, random_state=random_state, n_jobs=1),
    }

    records = []
    fold_idx = 0
    print(f"Running head-to-head on {dataset}: {cv}-fold CV, n_estimators={n_estimators}")
    for train_idx, test_idx in skf.split(X_np, y_np):
        fold_idx += 1
        X_train, X_test = X_np[train_idx], X_np[test_idx]
        y_train, y_test = y_np[train_idx], y_np[test_idx]

        for name, maker in models.items():
            clf = maker()
            # Fit. Custom implementations expect numpy arrays.
            clf.fit(X_train, y_train)

            y_pred = clf.predict(X_test)

            # Try to get probabilities for AUC; if not available, set NaN.
            auc = np.nan
            try:
                if hasattr(clf, "predict_proba"):
                    probs = clf.predict_proba(X_test)
                    # multiclass handling: sklearn's roc_auc_score supports multi_class='ovr'
                    if probs.ndim == 2 and probs.shape[1] > 2:
                        auc = float(roc_auc_score(y_test, probs, multi_class="ovr", average="macro"))
                    else:
                        # binary or two-class
                        # for binary roc_auc_score expects either proba for positive class or the 2-column matrix
                        if probs.ndim == 2 and probs.shape[1] == 2:
                            auc = float(roc_auc_score(y_test, probs[:, 1]))
                        else:
                            auc = float(roc_auc_score(y_test, probs))
                elif hasattr(clf, "decision_function"):
                    dec = clf.decision_function(X_test)
                    auc = float(roc_auc_score(y_test, dec))
            except Exception:
                auc = np.nan

            acc = float(accuracy_score(y_test, y_pred))
            f1m = float(f1_score(y_test, y_pred, average="macro"))

            records.append(
                {
                    "model": name,
                    "fold": int(fold_idx),
                    "accuracy": acc,
                    "f1_macro": f1m,
                    "auc_roc": auc,
                }
            )

            print(f" fold={fold_idx} model={name:20s} acc={acc:.4f} f1_macro={f1m:.4f} auc_roc={np.nan if np.isnan(auc) else f'{auc:.4f}'}")

    df = pd.DataFrame.from_records(records)

    results_csv = results_dir / f"head_to_head_{dataset}_cv{cv}_n{n_estimators}.csv"
    df.to_csv(results_csv, index=False)
    print(f"Saved results CSV to {results_csv}")

    # Summary table: mean ± std for each metric and model
    summary = df.groupby("model")[ ["accuracy", "f1_macro", "auc_roc"] ].agg(["mean", "std"])
    # flatten columns
    summary.columns = ["_" . join(col).strip() for col in summary.columns.values]
    summary_csv = results_dir / f"head_to_head_summary_{dataset}_cv{cv}_n{n_estimators}.csv"
    summary.to_csv(summary_csv)
    print(f"Saved summary CSV to {summary_csv}")

    # Box plots for each metric
    metrics = ["accuracy", "f1_macro", "auc_roc"]
    plot_paths = []
    for i, metric in enumerate(metrics, start=1):
        plt.figure(figsize=(6, 4))
        # Prepare data in model order
        groups = [df.loc[df.model == m, metric].dropna().values for m in models.keys()]
        # Older matplotlib versions may not accept the `labels` kwarg on boxplot.
        plt.boxplot(groups, showmeans=True)
        plt.ylabel(metric)
        plt.title(f"Head-to-head: {metric} (5-fold CV) on {dataset}")
        plt.grid(alpha=0.3)
        # Set x-tick labels explicitly (boxplot places boxes at 1..N)
        plt.xticks(range(1, len(models) + 1), list(models.keys()), rotation=25)
        plt.tight_layout()
        p = out_dir / f"head_to_head_{dataset}_{metric}_cv{cv}_n{n_estimators}.png"
        plt.savefig(p, dpi=200)
        plot_paths.append(str(p))
        print(f"Saved plot to {p}")
        if show_plot:
            plt.show()
        plt.close()

    return df, summary, plot_paths

if __name__ == "__main__":
    # Load same preprocessed datasets used by other experiments
    import pandas as _pd

    adult_X_train = pd.read_csv(get_data_path("adult_X_train_processed.csv"))
    adult_y_train = pd.read_csv(get_data_path("adult_y_train_processed.csv")).squeeze()

    covtype_X_train = pd.read_csv(get_data_path("covtype_X_train_processed.csv"))
    covtype_y_train = pd.read_csv(get_data_path("covtype_y_train_processed.csv")).squeeze()

    wdbc_X_train = pd.read_csv(get_data_path("wdbc_X_train_processed.csv"))
    wdbc_y_train = pd.read_csv(get_data_path("wdbc_y_train_processed.csv")).squeeze()

    # Example: run on the WDBC and Adult datasets (small/medium). Adjust as needed.
    run_head_to_head_arrays(wdbc_X_train, wdbc_y_train, dataset="wdbc", n_estimators=100, cv=5, random_state=42)
    run_head_to_head_arrays(adult_X_train, adult_y_train, dataset="adult", n_estimators=100, cv=5, random_state=42)
    run_head_to_head_arrays(covtype_X_train, covtype_y_train, dataset="covtype", n_estimators=100, cv=5, random_state=42)