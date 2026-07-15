# P3 Selected Dataset Results

These results were produced from the dataset files supplied in `Downloads`.

| Dataset | Train | Test | Features | Classes | Accuracy | Macro F1 | OOB | Config |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Breast Cancer Wisconsin (WDBC) | 456 | 113 | 30 | 2 | 0.9469 | 0.9426 | 0.9452 | `RandomForestClassifier(n_estimators=15, max_depth=8, max_features='sqrt', min_samples_split=10, oob_score=True)` |
| Adult Income | 32561 | 16281 | 108 | 2 | 0.8529 | 0.7675 | 0.8455 | `RandomForestClassifier(n_estimators=15, max_depth=8, max_features='sqrt', min_samples_split=10, oob_score=True)` |
| Covertype | 4000 | 1000 | 54 | 7 | 0.6850 | 0.3307 | 0.6772 | `RandomForestClassifier(n_estimators=15, max_depth=8, max_features='sqrt', min_samples_split=10, oob_score=True)` |

Notes:
- WDBC uses the full file with a deterministic stratified 80/20 split.
- Adult Income uses the official `adult.data` and `adult.test` split.
- Covertype uses a deterministic stratified 5000-row subset because the full file is very large for from-scratch tree training.
