import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from src.boosting.adaboost import AdaBoostClassifier
from src.bagging.random_forest import RandomForestClassifier
from sklearn.metrics import accuracy_score
from utils import get_figure_path
from utils import get_data_path

def _as_array(y):
    return np.asarray(y).ravel()

def _flip_labels(y, eta, rng):
    """Flip fraction eta of labels in y using RNG. For multiclass, sample a different label uniformly."""
    y = y.copy()
    n = len(y)
    k = int(round(eta * n))
    if k <= 0:
        return y
    idx = rng.choice(n, size=k, replace=False)
    classes = np.unique(y)
    for i in idx:
        other = classes[classes != y[i]]
        if len(other) == 0:
            continue
        y[i] = rng.choice(other)
    return y

def run_noise_robustness_arrays(
    X_train,
    y_train,
    X_test,
    y_test,
    dataset: str = "dataset",
    out_dir: str = "../figures",
    n_estimators: int = 100,
    etas=(0.05, 0.10, 0.20),
    random_state: int = 42,
    show_plot: bool = False,
):
    """Train AdaBoost and RandomForest on training sets corrupted with label noise (etas).
    Evaluate both on the clean test set and save CSV + plot per dataset.
    Uses the project's custom AdaBoostClassifier and RandomForestClassifier.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_dir = out_dir.parent
    results_dir.mkdir(parents=True, exist_ok=True)

    X_train_np = X_train.values if hasattr(X_train, "values") else np.asarray(X_train)
    X_test_np = X_test.values if hasattr(X_test, "values") else np.asarray(X_test)
    y_train_np = _as_array(y_train)
    y_test_np = _as_array(y_test)

    rng_master = np.random.RandomState(random_state)

    eta_list = list(etas)
    ada_accs = []
    rf_accs = []

    print(f"Running noise robustness for {dataset}: etas={eta_list}, n_estimators={n_estimators}")
    for eta in eta_list:
        rng = np.random.RandomState(rng_master.randint(0, 2 ** 31 - 1))
        y_corrupted = _flip_labels(y_train_np, eta, rng)

        # AdaBoost
        ada = AdaBoostClassifier(n_estimators=n_estimators, random_state=random_state)
        ada.fit(X_train_np, y_corrupted)
        y_pred_ada = ada.predict(X_test_np)
        ada_acc = float(accuracy_score(y_test_np, y_pred_ada))

        # Random Forest
        rf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state, n_jobs=1)
        rf.fit(X_train_np, y_corrupted)
        y_pred_rf = rf.predict(X_test_np)
        rf_acc = float(accuracy_score(y_test_np, y_pred_rf))

        print(f" eta={eta:.2f}  ada_acc={ada_acc:.4f}  rf_acc={rf_acc:.4f}")

        ada_accs.append(ada_acc)
        rf_accs.append(rf_acc)

    df = pd.DataFrame({"eta": np.array(eta_list), "ada_acc": np.array(ada_accs), "rf_acc": np.array(rf_accs)})
    results_csv = results_dir / f"noise_robustness_{dataset}.csv"
    df.to_csv(results_csv, index=False)
    print(f"Saved results CSV to {results_csv}")

    plt.figure(figsize=(7, 4.5))
    plt.plot(df["eta"], df["ada_acc"], label="AdaBoost (custom)", marker="o")
    plt.plot(df["eta"], df["rf_acc"], label="Random Forest (custom)", marker="o")
    plt.xlabel("Label noise fraction (eta)")
    plt.ylabel("Accuracy on clean test set")
    plt.title(f"Noise robustness on {dataset}: n_estimators={n_estimators}")
    plt.grid(alpha=0.3)
    plt.legend()

    plot_path = out_dir / f"noise_robustness_{dataset}.png"
    plt.tight_layout()
    plt.savefig(plot_path, dpi=200)
    print(f"Saved plot to {plot_path}")
    if show_plot:
        plt.show()
    plt.close()

    return df, plot_path

if __name__ == "__main__":
    # Load preprocessed datasets
    adult_X_train = pd.read_csv(get_data_path("adult_X_train_processed.csv"))
    adult_y_train = pd.read_csv(get_data_path("adult_y_train_processed.csv")).squeeze()
    adult_X_test = pd.read_csv(get_data_path("adult_X_test_processed.csv"))
    adult_y_test = pd.read_csv(get_data_path("adult_y_test_processed.csv")).squeeze()

    covtype_X_train = pd.read_csv(get_data_path("covtype_X_train_processed.csv"))
    covtype_y_train = pd.read_csv(get_data_path("covtype_y_train_processed.csv")).squeeze()
    covtype_X_test = pd.read_csv(get_data_path("covtype_X_test_processed.csv"))
    covtype_y_test = pd.read_csv(get_data_path("covtype_y_test_processed.csv")).squeeze()

    wdbc_X_train = pd.read_csv(get_data_path("wdbc_X_train_processed.csv"))
    wdbc_y_train = pd.read_csv(get_data_path("wdbc_y_train_processed.csv")).squeeze()
    wdbc_X_test = pd.read_csv(get_data_path("wdbc_X_test_processed.csv"))
    wdbc_y_test = pd.read_csv(get_data_path("wdbc_y_test_processed.csv")).squeeze()

    # Run for each dataset and save results/plots
    run_noise_robustness_arrays(adult_X_train, adult_y_train, adult_X_test, adult_y_test, dataset="adult", n_estimators=100, etas=(0.05, 0.10, 0.20), random_state=42)
    run_noise_robustness_arrays(wdbc_X_train, wdbc_y_train, wdbc_X_test, wdbc_y_test, dataset="wdbc", n_estimators=100, etas=(0.05, 0.10, 0.20), random_state=42)
    run_noise_robustness_arrays(covtype_X_train, covtype_y_train, covtype_X_test, covtype_y_test, dataset="covtype", n_estimators=100, etas=(0.05, 0.10, 0.20), random_state=42)