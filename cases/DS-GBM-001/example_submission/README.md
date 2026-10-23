# DS-GBM-001 reference submission

`model.py` (code only): `fit()` selects among four HistGradientBoosting settings on a holdout with early stopping and
refits the best one; `predict()` returns class-1 probabilities or regression values. The evaluator trains it on each
held-out dataset. Design report: `report/index.html`; references: `references.yaml`.
