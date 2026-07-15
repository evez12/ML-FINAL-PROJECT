# P2 Selected Dataset Results

These results were produced from the dataset files supplied in `Downloads`.

| Dataset | Train | Test | Features | Classes | Accuracy | Macro F1 | OOB | Config |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Breast Cancer Wisconsin (WDBC) | 456 | 113 | 30 | 2 | 0.9469 | 0.9432 |  | `AdaBoostClassifier(n_estimators=25, learning_rate=0.8, criterion='gini')` |
| Adult Income | 32561 | 16281 | 108 | 2 | 0.8489 | 0.7612 |  | `AdaBoostClassifier(n_estimators=25, learning_rate=0.8, criterion='gini')` |
| Covertype | 4000 | 1000 | 54 | 7 | 0.6470 | 0.2715 |  | `AdaBoostClassifier(n_estimators=25, learning_rate=0.8, criterion='gini')` |

Notes:
- WDBC uses the full file with a deterministic stratified 80/20 split.
- Adult Income uses the official `adult.data` and `adult.test` split.
- Covertype uses a deterministic stratified 5000-row subset because the full file is very large for from-scratch tree training.
