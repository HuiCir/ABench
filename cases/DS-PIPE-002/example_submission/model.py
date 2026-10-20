"""DS-PIPE-002 reference submission, inference side: one fitted histogram gradient-boosted model per dataset.

train.py fits and pickles them into checkpoint/d<id>.pkl; load() unpickles all six (the evaluation environment has
the same scikit-learn version as the sandbox).
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np


def load(checkpoint_dir):
    models = {}
    for p in sorted(Path(checkpoint_dir).glob("d*.pkl")):
        with open(p, "rb") as f:
            models[int(p.stem[1:])] = pickle.load(f)
    return models


def predict_proba(model, dataset, X):
    return model[int(dataset)].predict_proba(np.asarray(X, dtype=np.float32))
