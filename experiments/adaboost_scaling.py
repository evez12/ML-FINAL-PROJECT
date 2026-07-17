# AdaBoost scaling experiment (rewritten in the style of experiments/scaling.py)

import pandas as pd
import numpy as np

from pathlib import Path
import matplotlib.pyplot as plt

from src.boosting.adaboost import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from utils import get_figure_path
from utils import get_data_path

def run_adaboost_scaling_arrays(
    X_train,
    y_train,
    X_test,
    y_test,
    dataset: str = "dataset",
    out_dir: str = "../report/figures",
    max_estimators: int = 200,
    step: int = 5,
    random_state: int = 42,
    show_plot: bool = False,
):
    """Run AdaBoost scaling given already-loaded train/test arrays (pandas or numpy).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_dir = out_dir.parent
    results_dir.mkdir(parents=True, exist_ok=True)

    # Convert to numpy
    X_train_np = X_train.values if hasattr(X_train, "values") else np.asarray(X_train)
    X_test_np = X_test.values if hasattr(X_test, "values") else np.asarray(X_test)
    y_train_np = np.asarray(y_train).ravel()
    y_test_np = np.asarray(y_test).ravel()

    ada = AdaBoostClassifier(n_estimators=max_estimators, learning_rate=1.0, criterion="gini", random_state=random_state)

    print(f"Fitting custom AdaBoost with {max_estimators} estimators (DecisionStump) on {dataset}:")
    ada.fit(X_train_np, y_train_np)

    train_acc = []
    test_acc = []
    n_list = []

    # staged_predict yields predictions after each estimator
    for i, (y_pred_train, y_pred_test) in enumerate(zip(ada.staged_predict(X_train_np), ada.staged_predict(X_test_np)), start=1):
        if i > max_estimators:
            break
        if (i % step) != 0 and i != 1 and i != max_estimators:
            continue
        acc_t = accuracy_score(y_train_np, y_pred_train)
        acc_test = accuracy_score(y_test_np, y_pred_test)
        train_acc.append(acc_t)
        test_acc.append(acc_test)
        n_list.append(i)

    n_arr = np.array(n_list)
    train_acc = np.array(train_acc)
    test_acc = np.array(test_acc)

    df = pd.DataFrame({"n_estimators": n_arr, "train_acc": train_acc, "test_acc": test_acc})
    results_csv = results_dir / f"adaboost_{dataset}_scaling.csv"
    df.to_csv(results_csv, index=False)
    print(f"Saved results CSV to {results_csv}")
    
    plt.figure(figsize=(8, 5))
    plt.plot(n_arr, 1 - train_acc, label="train error", marker="o")
    plt.plot(n_arr, 1 - test_acc, label="test error", marker="o")
    plt.xlabel("Number of estimators")
    plt.ylabel("Error (1 - accuracy)")
    plt.title(f"AdaBoost scaling on {dataset}: max={max_estimators}, step={step}")
    plt.grid(alpha=0.3)
    plt.legend()

    plot_path = out_dir / f"adaboost_{dataset}_scaling.png"
    plt.tight_layout()
    plt.savefig(plot_path, dpi=200)
    print(f"Saved plot to {plot_path}")

    if show_plot:
        plt.show()
    plt.close()

    return df, plot_path


if __name__ == "__main__":
    # Load datasets similar to experiments/scaling.py
    adult_X_train = pd.read_csv(get_data_path("adult_X_train_processed.csv"))
    adult_y_train = pd.read_csv(get_data_path("adult_y_train_processed.csv"))
    adult_X_test = pd.read_csv(get_data_path("adult_X_test_processed.csv"))
    adult_y_test = pd.read_csv(get_data_path("adult_y_test_processed.csv"))

    covtype_X_train = pd.read_csv(get_data_path("covtype_X_train_processed.csv"))
    covtype_y_train = pd.read_csv(get_data_path("covtype_y_train_processed.csv"))
    covtype_X_test = pd.read_csv(get_data_path("covtype_X_test_processed.csv"))
    covtype_y_test = pd.read_csv(get_data_path("covtype_y_test_processed.csv"))

    wdbc_X_train = pd.read_csv(get_data_path("wdbc_X_train_processed.csv"))
    wdbc_y_train = pd.read_csv(get_data_path("wdbc_y_train_processed.csv"))
    wdbc_X_test = pd.read_csv(get_data_path("wdbc_X_test_processed.csv"))
    wdbc_y_test = pd.read_csv(get_data_path("wdbc_y_test_processed.csv"))

    # Run experiments in the same style as scaling.py
    run_adaboost_scaling_arrays(adult_X_train, adult_y_train, adult_X_test, adult_y_test, dataset="adult")
    run_adaboost_scaling_arrays(wdbc_X_train, wdbc_y_train, wdbc_X_test, wdbc_y_test, dataset="wdbc")
    run_adaboost_scaling_arrays(covtype_X_train, covtype_y_train, covtype_X_test, covtype_y_test, dataset="covtype")
