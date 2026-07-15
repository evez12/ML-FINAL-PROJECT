# P3 Package - Random Forest Classifier

Owner placeholder: `P3_MEMBER_NAME`

This folder contains only the final P3-owned materials.  Copy these files into
the main `ml_final_project` repository after P1's Decision Tree has been merged.

## Selected datasets

The final experiments should use:

- `Breast Cancer Wisconsin (WDBC)`
- `Adult Income`
- `Covertype`

P3 unit tests use small synthetic datasets only to verify Random Forest logic.
The three real datasets above are for the experiment and final analysis stage.

## Real dataset results

Results produced from the supplied files are saved in:

- `results/P3_Selected_Dataset_Results.csv`
- `results/P3_Selected_Dataset_Results.md`

Summary:

- WDBC: accuracy `0.9469`, macro-F1 `0.9426`, OOB `0.9452`
- Adult Income: accuracy `0.8529`, macro-F1 `0.7675`, OOB `0.8455`
- Covertype subset: accuracy `0.6850`, macro-F1 `0.3307`, OOB `0.6772`

## Final files

Code and tests:

- `src/bagging/random_forest.py`
- `src/bagging/__init__.py`
- `tests/test_random_forest.py`

Report:

- `report/P3_RandomForest_Report.tex`
- `report/P3_RandomForest_Report.pdf`

Slides:

- `slides/P3_Defense_Slides.tex`
- `slides/P3_Defense_Slides.pdf`

Contribution:

- `contribution/P3_Contribution_Record.tex`
- `contribution/P3_Contribution_Record.pdf`

Explanation:

- `P3_FULL_FLOW_EXPLANATION.md`
- `P3_FULL_FLOW_EXPLANATION.pdf`

Results:

- `results/P3_Selected_Dataset_Results.csv`
- `results/P3_Selected_Dataset_Results.md`

## Dependency rule

P3 depends on P1's `src/trees/decision_tree.py` and `src/trees/__init__.py`.
Do not duplicate tree logic inside the Random Forest module.

## Test command

```bash
python -m pytest tests/test_random_forest.py
```
