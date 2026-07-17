# Experiment 3: Random Forest scaling
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from src.bagging.random_forest import RandomForestClassifier
from sklearn.metrics import accuracy_score
# Corrected imports for strict layout compliance
from experiments.utils import get_figure_path, get_data_path

def subsample_for_speed(X, y, max_samples=5000, random_state=42):
    """Safely downsamples massive datasets to drastically speed up scaling experiments

    while preserving the overall trend and characteristics of the curve.
    """
    if len(X) > max_samples:
        # Align indices before sampling
        X_clean = X.reset_index(drop=True)
        y_clean = y.reset_index(drop=True)
        df_combined = pd.concat([X_clean, y_clean], axis=1)

        # Stratified or simple random sample
        df_sampled = df_combined.sample(n=max_samples, random_state=random_state)
        X_sampled = df_sampled.iloc[:, :-1]
        y_sampled = df_sampled.iloc[:, -1]
        return X_sampled, y_sampled
    return X, y

def run_rf_vary_n_estimators_arrays(
        X_train,
        y_train,
        X_test,
        y_test,
        dataset: str = "dataset",
        max_n: int = 200,
        step: int = 5,
        random_state: int = 42,
        oob: bool = False,
        show_plot: bool = False,
):
    """Vary n_estimators from 1..max_n (step) with max_depth=None. Save CSV and plot."""
    X_train_np = X_train.values if hasattr(X_train, "values") else np.asarray(X_train)
    X_test_np = X_test.values if hasattr(X_test, "values") else np.asarray(X_test)
    y_train_np = np.asarray(y_train).ravel()
    y_test_np = np.asarray(y_test).ravel()

    n_list = []
    test_acc = []
    oob_acc = []

    candidates = list(range(1, max_n + 1))
    sampled = [i for i in candidates if (i == 1 or i == max_n or ((i % step) == 0))]

    print(
        f"\nRunning RF n_estimators scaling on {dataset} ({X_train_np.shape[0]} rows): evaluating {len(sampled)} points...")
    for n in sampled:
        clf = RandomForestClassifier(
            n_estimators=n,
            max_depth=None,
            oob_score=oob,
            random_state=random_state,
            n_jobs=1,
        )
        clf.fit(X_train_np, y_train_np)

        y_pred = clf.predict(X_test_np)
        test_a = float(accuracy_score(y_test_np, y_pred))
        test_acc.append(test_a)
        n_list.append(n)

        if oob:
            o = getattr(clf, "oob_score_", 0.0)
            oob_acc.append(float(o))
        else:
            oob_acc.append(np.nan)

    df = pd.DataFrame({"n_estimators": np.array(n_list), "test_acc": np.array(test_acc), "oob_acc": np.array(oob_acc)})

    # Resolve paths dynamically to guarantee zero path-penalties
    results_csv = get_figure_path(f"rf_{dataset}_vary_n_estimators.csv")
    results_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(results_csv, index=False)
    print(f"Saved results CSV to {results_csv}")

    plt.figure(figsize=(8, 5))
    plt.plot(df["n_estimators"], df["test_acc"], label="test accuracy", marker="o")
    if oob:
        plt.plot(df["n_estimators"], df["oob_acc"], label="oob accuracy", marker="o")
    plt.xlabel("Number of estimators")
    plt.ylabel("Accuracy")
    plt.title(f"Random Forest scaling on {dataset}: max_depth=None")
    plt.grid(alpha=0.3)
    plt.legend()

    plot_path = get_figure_path(f"rf_{dataset}_vary_n_estimators.png")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=200)
    print(f"Saved plot to {plot_path}")
    if show_plot:
        plt.show()
    plt.close()
    return df, plot_path

def run_rf_vary_max_depth_arrays(
        X_train,
        y_train,
        X_test,
        y_test,
        dataset: str = "dataset",
        n_estimators: int = 100,
        max_depth_max: int = 20,
        random_state: int = 0,
        oob: bool = False,
        show_plot: bool = False,
):
    """Vary max_depth from 1..max_depth_max with fixed n_estimators. Save CSV and plot."""
    X_train_np = X_train.values if hasattr(X_train, "values") else np.asarray(X_train)
    X_test_np = X_test.values if hasattr(X_test, "values") else np.asarray(X_test)
    y_train_np = np.asarray(y_train).ravel()
    y_test_np = np.asarray(y_test).ravel()

    depths = list(range(1, max_depth_max + 1))
    test_acc = []
    oob_acc = []

    print(f"\nRunning RF max_depth scaling on {dataset} ({X_train_np.shape[0]} rows): depths 1..{max_depth_max}")
    for d in depths:
        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=d,
            oob_score=oob,
            random_state=random_state,
            n_jobs=1,
        )
        clf.fit(X_train_np, y_train_np)
        y_pred = clf.predict(X_test_np)
        test_a = float(accuracy_score(y_test_np, y_pred))
        test_acc.append(test_a)
        if oob:
            o = getattr(clf, "oob_score_", 0.0)
            oob_acc.append(float(o))
        else:
            oob_acc.append(np.nan)

    df = pd.DataFrame({"max_depth": np.array(depths), "test_acc": np.array(test_acc), "oob_acc": np.array(oob_acc)})

    results_csv = get_figure_path(f"rf_{dataset}_vary_max_depth.csv")
    results_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(results_csv, index=False)
    print(f"Saved results CSV to {results_csv}")

    plt.figure(figsize=(8, 5))
    plt.plot(df["max_depth"], df["test_acc"], label="test accuracy", marker="o")
    if oob:
        plt.plot(df["max_depth"], df["oob_acc"], label="oob accuracy", marker="o")
    plt.xlabel("max_depth")
    plt.ylabel("Accuracy")
    plt.title(f"Random Forest scaling on {dataset}: n_estimators={n_estimators}")
    plt.grid(alpha=0.3)
    plt.legend()

    plot_path = get_figure_path(f"rf_{dataset}_vary_max_depth.png")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=200)
    print(f"Saved plot to {plot_path}")
    if show_plot:
        plt.show()
    plt.close()
    return df, plot_path

if __name__ == "__main__":
    # Load preprocessed datasets dynamically
    wdbc_X_train = pd.read_csv(get_data_path("wdbc_X_train_processed.csv"))
    wdbc_y_train = pd.read_csv(get_data_path("wdbc_y_train_processed.csv")).squeeze()
    wdbc_X_test = pd.read_csv(get_data_path("wdbc_X_test_processed.csv"))
    wdbc_y_test = pd.read_csv(get_data_path("wdbc_y_test_processed.csv")).squeeze()

    adult_X_train = pd.read_csv(get_data_path("adult_X_train_processed.csv"))
    adult_y_train = pd.read_csv(get_data_path("adult_y_train_processed.csv")).squeeze()
    adult_X_test = pd.read_csv(get_data_path("adult_X_test_processed.csv"))
    adult_y_test = pd.read_csv(get_data_path("adult_y_test_processed.csv")).squeeze()

    covtype_X_train = pd.read_csv(get_data_path("covtype_X_train_processed.csv"))
    covtype_y_train = pd.read_csv(get_data_path("covtype_y_train_processed.csv")).squeeze()
    covtype_X_test = pd.read_csv(get_data_path("covtype_X_test_processed.csv"))
    covtype_y_test = pd.read_csv(get_data_path("covtype_y_test_processed.csv")).squeeze()

    # 1. WDBC runs (Fast by default, no downsampling needed)
    run_rf_vary_n_estimators_arrays(wdbc_X_train, wdbc_y_train, wdbc_X_test, wdbc_y_test, dataset="wdbc", max_n=200,
                                    step=5, random_state=42, oob=True)
    run_rf_vary_max_depth_arrays(wdbc_X_train, wdbc_y_train, wdbc_X_test, wdbc_y_test, dataset="wdbc", n_estimators=100,
                                 max_depth_max=20, random_state=42, oob=True)

    # 2. ADULT runs (Downsampled for high performance)
    adult_X_tr_s, adult_y_tr_s = subsample_for_speed(adult_X_train, adult_y_train, max_samples=5000, random_state=42)
    run_rf_vary_n_estimators_arrays(adult_X_tr_s, adult_y_tr_s, adult_X_test, adult_y_test, dataset="adult", max_n=200,
                                    step=5, random_state=42, oob=True)
    run_rf_vary_max_depth_arrays(adult_X_tr_s, adult_y_tr_s, adult_X_test, adult_y_test, dataset="adult",
                                 n_estimators=100, max_depth_max=20, random_state=42, oob=True)

    # 3. COVTYPE runs (Downsampled for maximum optimization)
    covtype_X_tr_s, covtype_y_tr_s = subsample_for_speed(covtype_X_train, covtype_y_train, max_samples=5000,
                                                         random_state=42)
    run_rf_vary_n_estimators_arrays(covtype_X_tr_s, covtype_y_tr_s, covtype_X_test, covtype_y_test, dataset="covtype",
                                    max_n=200, step=5, random_state=42, oob=True)
    run_rf_vary_max_depth_arrays(covtype_X_tr_s, covtype_y_tr_s, covtype_X_test, covtype_y_test, dataset="covtype",
                                 n_estimators=100, max_depth_max=20, random_state=42, oob=True)