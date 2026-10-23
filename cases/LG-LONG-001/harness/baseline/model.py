"""LG-LONG-001 reference submission, inference side: tf-idf over word unigrams and bigrams of the whole case, with one
L2-regularised logistic regression per article (NumPy/SciPy only; LexGLUE's linear baseline family). The whole document
is used, so there is no length limit. train.py builds the vocabulary, fits the models and a global decision threshold,
and saves checkpoint/linear.npz. It does not use the provided legal-bert model, which is left to the entries.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import scipy.sparse as sp

TOKEN = re.compile(r"[a-z][a-z0-9]+")


def terms(doc: str) -> list[str]:
    w = TOKEN.findall(doc.lower())
    return w + [a + " " + b for a, b in zip(w, w[1:])]


def tfidf(docs, vocab: dict, idf: np.ndarray) -> sp.csr_matrix:
    rows, cols, vals = [], [], []
    for i, d in enumerate(docs):
        counts: dict = {}
        for t in terms(d):
            j = vocab.get(t)
            if j is not None:
                counts[j] = counts.get(j, 0) + 1
        if counts:
            js = np.fromiter(counts.keys(), np.int64)
            v = (1 + np.log(np.fromiter(counts.values(), np.float64))) * idf[js]
            v /= np.linalg.norm(v)
            rows.append(np.full(len(js), i))
            cols.append(js)
            vals.append(v)
    if not rows:
        return sp.csr_matrix((len(docs), len(idf)))
    return sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(len(docs), len(idf)))


def load(checkpoint_dir, models_dir=None, device="cpu"):
    with np.load(Path(checkpoint_dir) / "linear.npz", allow_pickle=False) as z:
        vocab = {t: i for i, t in enumerate(z["vocab"].tolist())}
        return {"vocab": vocab, "idf": z["idf"], "W": z["W"], "b": z["b"], "threshold": float(z["threshold"])}


def predict(model, texts):
    X = tfidf(texts, model["vocab"], model["idf"])
    s = X @ model["W"] + model["b"]
    return (1 / (1 + np.exp(-s)) >= model["threshold"]).astype(np.int8)
