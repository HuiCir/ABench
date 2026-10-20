"""TC-FT-001 reference submission, model: hashed n-gram tf-idf features and a linear classifier loaded from checkpoint/.

load() reads checkpoint/linear.npz (idf, weights, intercepts written by train.py); predict() hashes unigrams and
bigrams of each document into 2^20 buckets, applies 1 + log(tf) and idf, L2-normalises and takes the argmax score.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer

N_FEATURES = 2 ** 20


def vectorizer() -> HashingVectorizer:
    return HashingVectorizer(ngram_range=(1, 2), n_features=N_FEATURES, alternate_sign=False, lowercase=True,
                             token_pattern=r"(?u)\b\w+\b", norm=None, dtype=np.float32)


def load(checkpoint_dir: str):
    with np.load(Path(checkpoint_dir) / "linear.npz", allow_pickle=False) as z:
        return {k: z[k] for k in z.files} | {"vec": vectorizer()}


def predict(model, texts):
    x = model["vec"].transform(texts).tocsr()
    x.data = 1.0 + np.log(x.data)
    x = x.multiply(model["idf"][None, :]).tocsr()
    norm = np.sqrt(np.asarray(x.multiply(x).sum(1)).ravel())
    norm[norm == 0] = 1.0
    scores = np.asarray(x @ model["coef"].T) / norm[:, None] + model["intercept"]
    return model["classes"][scores.argmax(1)].astype(np.int64)
