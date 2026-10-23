"""DS-GBM-001 reference submission: a budget-aware gradient-boosted tree learner.

fit() holds out 20% of the training rows, tries a small grid of HistGradientBoosting settings (learning rate,
leaf count, L2 regularisation) with early stopping on that holdout, in order, until half the time budget is used,
then refits the best setting on all rows with the number of iterations found by early stopping scaled by 1.25.
Classification uses log-loss on the holdout, regression squared error.
"""
from __future__ import annotations

import time

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

GRID = [dict(learning_rate=0.1, max_leaf_nodes=31, l2_regularization=0.0),
        dict(learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0),
        dict(learning_rate=0.05, max_leaf_nodes=63, l2_regularization=1.0),
        dict(learning_rate=0.03, max_leaf_nodes=31, l2_regularization=10.0, min_samples_leaf=40)]


def _model(task, seed, **kw):
    cls = HistGradientBoostingClassifier if task == "classification" else HistGradientBoostingRegressor
    return cls(random_state=seed, **kw)


def _loss(task, m, x, y):
    if task == "classification":
        p = np.clip(m.predict_proba(x)[:, list(m.classes_).index(1)], 1e-6, 1 - 1e-6)
        return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
    return float(np.mean((m.predict(x) - y) ** 2))


def fit(x, y, task, budget_seconds, seed):
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    cut = int(0.8 * len(y))
    tr, va = idx[:cut], idx[cut:]
    best = None
    for kw in GRID:
        if best is not None and time.perf_counter() - t0 > 0.5 * budget_seconds:
            break
        m = _model(task, seed, max_iter=2000, early_stopping=True, validation_fraction=0.15, n_iter_no_change=30, **kw)
        m.fit(x[tr], y[tr])
        loss = _loss(task, m, x[va], y[va])
        if best is None or loss < best[0]:
            best = (loss, kw, m.n_iter_)
    _, kw, n_iter = best
    return _model(task, seed, max_iter=max(10, int(1.25 * n_iter)), early_stopping=False, **kw).fit(x, y)


def predict(model, x):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x)[:, list(model.classes_).index(1)]
    return model.predict(x)
