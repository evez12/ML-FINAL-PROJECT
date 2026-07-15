# P2 AdaBoost Package

This folder contains the P2 part of the final machine learning project.

## What P2 owns

- `DecisionStump`, a depth-1 wrapper around the P1 `DecisionTree`.
- `AdaBoostClassifier`, implemented from scratch with NumPy.
- Weighted stump training through P1 `DecisionTree.fit(..., sample_weight=...)`.
- SAMME estimator weight and sample weight update logic.
- `predict`, `predict_proba`, `estimator_weights`, `estimator_errors`, and `staged_predict`.
- Unit tests for normal behavior, edge cases, deterministic seeds, probabilities, staged prediction, and multi-class SAMME.
- Report, slides, contribution record, dataset results, and full-flow explanation.

## Integration note

P2 imports `DecisionTree` from `src.trees`. It must be integrated after the P1
Decision Tree module is present in the main project. P2 does not duplicate tree
split logic.

## Dataset results

The model was evaluated on the selected project datasets:

| Dataset | Accuracy | Macro F1 | Config |
|---|---:|---:|---|
| Breast Cancer Wisconsin (WDBC) | 0.9469 | 0.9432 | `n_estimators=25, learning_rate=0.8` |
| Adult Income | 0.8489 | 0.7612 | `n_estimators=25, learning_rate=0.8` |
| Covertype subset | 0.6470 | 0.2715 | `n_estimators=25, learning_rate=0.8` |

## Files to copy into the main project

```text
src/boosting/adaboost.py
src/boosting/__init__.py
tests/test_adaboost.py
```

## Run tests

```bash
python -m pytest tests/test_adaboost.py
```

## Commit message

```bash
git add src/boosting/adaboost.py src/boosting/__init__.py tests/test_adaboost.py
git commit -m "Implement P2 AdaBoost classifier"
```
