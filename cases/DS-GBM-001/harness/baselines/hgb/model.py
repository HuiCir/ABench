"""DS-GBM-001 baseline run by the evaluator: scikit-learn HistGradientBoosting with its default settings
(a histogram gradient-boosted tree learner in the line of XGBoost and LightGBM), one model per dataset."""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor


def fit(x, y, task, budget_seconds, seed):
    if task == "classification":
        return HistGradientBoostingClassifier(random_state=seed).fit(x, y)
    return HistGradientBoostingRegressor(random_state=seed).fit(x, y)


def predict(model, x):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x)[:, list(model.classes_).index(1)]
    return model.predict(x)
