"""TF-IDF features of word 1-2-grams and character 3-5-grams (within word boundaries), stored without pickle."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def vectorizers():
    from sklearn.feature_extraction.text import TfidfVectorizer
    return (TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=1, sublinear_tf=True, lowercase=True),
            TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True, lowercase=True))


def save(vecs, path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for i, v in enumerate(vecs):
        (path / f"vocab{i}.json").write_text(json.dumps({k: int(x) for k, x in v.vocabulary_.items()}), encoding="utf-8")
        np.save(path / f"idf{i}.npy", v.idf_.astype(np.float32))


def load(path: Path):
    vecs = vectorizers()
    for i, v in enumerate(vecs):
        v.vocabulary = json.loads((path / f"vocab{i}.json").read_text(encoding="utf-8"))
        v.fit(["x"])                               # sets up the fixed vocabulary; idf replaced below
        v.idf_ = np.load(path / f"idf{i}.npy").astype(np.float64)
    return vecs


def transform(vecs, texts):
    from scipy.sparse import hstack
    return hstack([v.transform(texts) for v in vecs]).tocsr()
