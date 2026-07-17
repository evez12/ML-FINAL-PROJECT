import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

from sklearn.utils import resample
from sklearn.metrics import accuracy_score

# custom implementations
from src.boosting.adaboost import AdaBoostClassifier
from src.bagging.random_forest import RandomForestClassifier
from utils import get_figure_path
from utils import get_data_path


def _as_array(y):
    if hasattr(y, "squeeze"):
        return np.asarray(y.squeeze())
    return np.asarray(y).ravel()


def balance_by_undersample(X, y, random_state=0):
    df = pd.concat([pd.DataFrame(X).reset_index(drop=True), pd.Series(y).reset_index(drop=True)], axis=1)
    df.columns = list(range(df.shape[1] - 1)) + ["y"]
    counts = df["y"].value_counts()
    minc = counts.min()
    parts = [group.sample(minc, random_state=random_state) for _, group in df.groupby("y")]
    bal = pd.concat(parts).sample(frac=1, random_state=random_state).reset_index(drop=True)
    Xb = bal.drop(columns=["y"])
    yb = bal["y"].values
    return Xb, yb


def bias_variance_decomposition(model_constructor, X_train, y_train, X_test, y_test, B=100, random_state=0):
    """Run B bootstrap replicates, return (bias_sq, variance, brier_total, accuracy_mean).

    Uses probability-based squared-error decomposition:
      bias^2 = mean_x (mean_p(x) - y(x))^2
      variance = mean_x mean_b (p_b(x) - mean_p(x))^2
    where p_b(x) are predicted positive-class probabilities from each bootstrap model.
    """
    rng = np.random.RandomState(random_state)
    X_train_np = X_train.values if hasattr(X_train, "values") else np.asarray(X_train)
    X_test_np = X_test.values if hasattr(X_test, "values") else np.asarray(X_test)
    y_train_np = np.asarray(y_train).ravel()
    y_test_np = np.asarray(y_test).ravel()

    n_test = X_test_np.shape[0]
    probs = np.zeros((B, n_test), dtype=float)
    accs = []

    for b in range(B):
        # bootstrap sample from balanced training set
        idx = rng.randint(0, X_train_np.shape[0], size=X_train_np.shape[0])
        Xb = X_train_np[idx]
        yb = y_train_np[idx]

        clf = model_constructor(random_state=(random_state + b))
        try:
            clf.fit(Xb, yb)
        except Exception:
            clf.fit(Xb, yb.ravel())

        # get predicted probability for positive class if available
        try:
            p = clf.predict_proba(X_test_np)
            # if binary, take column 1, else try to align classes (assume positive is last)
            if p.ndim == 2 and p.shape[1] >= 2:
                ppos = p[:, 1]
            else:
                # fallback: use predict (0/1)
                preds = clf.predict(X_test_np)
                ppos = np.asarray(preds).ravel().astype(float)
        except Exception:
            preds = clf.predict(X_test_np)
            ppos = np.asarray(preds).ravel().astype(float)

        probs[b, :] = ppos
        accs.append(float(accuracy_score(y_test_np, (ppos >= 0.5).astype(int))))

    # mean probability per sample
    p_mean = probs.mean(axis=0)
    # bias^2 per sample
    bias_sq_per_sample = (p_mean - y_test_np) ** 2
    bias_sq = float(bias_sq_per_sample.mean())
    # variance per sample
    var_per_sample = probs.var(axis=0, ddof=0)
    variance = float(var_per_sample.mean())
    # total Brier (expected squared error)
    brier_per_sample = ((probs - y_test_np[None, :]) ** 2).mean(axis=0)
    brier_total = float(brier_per_sample.mean())

    return {"bias_sq": bias_sq, "variance": variance, "brier": brier_total, "acc_mean": float(np.mean(accs))}

def run_on_wdbc(B=100, out_dir="../figures", random_state=42):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # load processed wdbc dataset
    X_train = pd.read_csv(get_data_path("wdbc_X_train_processed.csv"))
    y_train = pd.read_csv(get_data_path("wdbc_y_train_processed.csv")).squeeze()
    X_test = pd.read_csv(get_data_path("wdbc_X_test_processed.csv"))
    y_test = pd.read_csv(get_data_path("wdbc_y_test_processed.csv")).squeeze()

    # balance the training set for a fair balanced-binary experiment
    Xb, yb = balance_by_undersample(X_train, y_train, random_state=random_state)

    results = {}

    # model constructors that accept random_state kwarg
    def make_rf(random_state=0):
        return RandomForestClassifier(n_estimators=50, max_depth=None, random_state=random_state)

    def make_ada(random_state=0):
        return AdaBoostClassifier(n_estimators=50, learning_rate=1.0, random_state=random_state)

    models = {"RandomForest": make_rf, "AdaBoost": make_ada}

    for name, ctor in models.items():
        print(f"Running bias-variance for {name} (B={B})...")
        res = bias_variance_decomposition(ctor, Xb, yb, X_test, y_test, B=B, random_state=random_state)
        print(
            f" {name}: bias^2={res['bias_sq']:.6f}, variance={res['variance']:.6f}, brier={res['brier']:.6f}, acc_mean={res['acc_mean']:.4f}")
        results[name] = res

    # save CSV
    rows = []
    for name, r in results.items():
        rows.append({"model": name, "bias_sq": r["bias_sq"], "variance": r["variance"], "brier": r["brier"],
                     "acc_mean": r["acc_mean"]})
    df = pd.DataFrame(rows)
    csv_path = "../report/outputs/bias_variance_wdbc.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved results to {csv_path}")

    # plot grouped bars for bias^2 and variance per model
    labels = list(results.keys())
    bias_vals = [results[m]["bias_sq"] for m in labels]
    var_vals = [results[m]["variance"] for m in labels]

    x = np.arange(len(labels))
    width = 0.35

    plt.figure(figsize=(6, 4))
    plt.bar(x - width / 2, bias_vals, width, label="bias^2")
    plt.bar(x + width / 2, var_vals, width, label="variance")
    plt.xticks(x, labels)
    plt.ylabel("Value (squared-error)")
    plt.title(f"Bias^2 vs Variance on balanced wdbc (B={B})")
    plt.legend()
    plt.grid(alpha=0.2, axis="y")
    plot_path = out_dir / "bias_variance_wdbc.png"
    plt.tight_layout()
    plt.savefig(plot_path, dpi=200)
    print(f"Saved plot to {plot_path}")
    plt.close()

    return df, plot_path

if __name__ == "__main__":
    run_on_wdbc(B=100, out_dir="../figures", random_state=42)