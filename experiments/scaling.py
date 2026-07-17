# Experiment 1: Baseline single tree vs sklearn reference – placeholder stub
import pandas as pd
import numpy as np

from src.trees.decision_tree import DecisionTree
from sklearn.tree import DecisionTreeClassifier as SKTree
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize
from utils import get_figure_path
from utils import get_data_path

def _as_array(y):
    if hasattr(y, "squeeze"):
        arr = y.squeeze()
    else:
        arr = y
    return np.asarray(arr).ravel()

def _align_proba(proba, model_classes, classes):
    n_classes = len(classes)
    aligned = np.zeros((proba.shape[0], n_classes), dtype=float)
    for i, cls in enumerate(classes):
        try:
            idx = int(np.where(model_classes == cls)[0][0])
            aligned[:, i] = proba[:, idx]
        except Exception:
            pass
    return aligned

def train_and_report(X_train, y_train, X_test, y_test, name: str = "dataset"):
    y_train = _as_array(y_train)
    y_test = _as_array(y_test)

    my_tree = DecisionTree(max_depth=None, min_samples_split=2, criterion="gini", random_state=42)
    my_tree.fit(X_train.values if hasattr(X_train, 'values') else X_train, y_train)
    my_pred = my_tree.predict(X_test.values if hasattr(X_test, 'values') else X_test)
    my_proba = my_tree.predict_proba(X_test.values if hasattr(X_test, 'values') else X_test)

    stump = DecisionTree(max_depth=1, min_samples_split=2, criterion="gini", random_state=42)
    stump.fit(X_train.values if hasattr(X_train, 'values') else X_train, y_train)
    stump_pred = stump.predict(X_test.values if hasattr(X_test, 'values') else X_test)
    stump_proba = stump.predict_proba(X_test.values if hasattr(X_test, 'values') else X_test)

    sk = SKTree(criterion="gini", max_depth=None, min_samples_split=2, random_state=42)
    sk.fit(X_train, y_train)
    sk_pred = sk.predict(X_test)
    sk_proba = sk.predict_proba(X_test)

    classes = np.unique(np.concatenate([np.unique(y_train), np.unique(y_test)]))

    my_proba_al = _align_proba(my_proba, my_tree.classes_, classes)
    sk_proba_al = _align_proba(sk_proba, sk.classes_, classes)
    stump_proba_al = _align_proba(stump_proba, stump.classes_, classes)

    y_test_bin = label_binarize(y_test, classes=classes)

    def compute_auc(y_bin, proba_al):
        if y_bin.shape[1] == 1:
            return roc_auc_score(y_bin.ravel(), proba_al[:, 1])
        if y_bin.shape[1] == 2:
            return roc_auc_score(y_bin[:, 1], proba_al[:, 1])
        return roc_auc_score(y_bin, proba_al, multi_class="ovr", average="macro")

    results = {}
    for label, y_pred, proba_al in [
        ("my_tree", my_pred, my_proba_al),
        ("stump", stump_pred, stump_proba_al),
        ("sklearn", sk_pred, sk_proba_al),
    ]:
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="macro")
        try:
            auc = compute_auc(y_test_bin, proba_al)
        except Exception:
            auc = float("nan")
        results[label] = {"accuracy": acc, "f1_macro": f1, "auc": auc}

    print(f"\nResults for {name}:")
    for label, metrics in results.items():
        print(f" {label}: accuracy={metrics['accuracy']:.4f}, f1_macro={metrics['f1_macro']:.4f}, auc={metrics['auc'] if np.isfinite(metrics['auc']) else 'nan'}")

    for metric in ["accuracy", "f1_macro", "auc"]:
        v_my = results["my_tree"][metric]
        v_sk = results["sklearn"][metric]
        if not (np.isnan(v_my) or np.isnan(v_sk)):
            diff = abs(v_my - v_sk)
            ok = diff <= 0.02
            print(f"  Compare {metric}: my={v_my:.4f} sk={v_sk:.4f} diff={diff:.4f} -> {'OK' if ok else 'MISMATCH'}")
        else:
            print(f"  Compare {metric}: cannot compare (nan)")

if __name__ == "__main__":
    # Load datasets
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

    train_and_report(adult_X_train, adult_y_train, adult_X_test, adult_y_test, name="adult")
    train_and_report(covtype_X_train, covtype_y_train, covtype_X_test, covtype_y_test, name="covtype")
    train_and_report(wdbc_X_train, wdbc_y_train, wdbc_X_test, wdbc_y_test, name="wdbc")