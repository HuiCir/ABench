"""LG-HOLD-002 reference submission, inference side: lexical overlap features with a learned linear scorer (NumPy only).

Each (citing prompt, candidate holding) pair is described by tf-idf cosine similarities of word unigrams and bigrams
and by length features; a linear scorer over these features is trained with a softmax over the five candidates
(conditional logit) by full-batch gradient descent (train.py; the idf table and weights are saved to
checkpoint/lexical.npz). It does not use the provided roberta-base model, which is left to the entries.
"""
from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import numpy as np

TOKEN = re.compile(r"[a-z0-9]+")


def tok(s):
    w = TOKEN.findall(s.lower())
    return w, [a + "_" + b for a, b in zip(w, w[1:])]


def _vec(tokens, idf):
    c = Counter(t for t in tokens if t in idf)
    v = {t: (1 + np.log(n)) * idf[t] for t, n in c.items()}
    norm = np.sqrt(sum(x * x for x in v.values())) or 1.0
    return {t: x / norm for t, x in v.items()}


def _cos(a, b):
    if len(a) > len(b):
        a, b = b, a
    return sum(x * b.get(t, 0.0) for t, x in a.items())


def features(m, prompts, holdings):
    F = np.zeros((len(prompts), 5, 6), np.float64)
    for i, (p, hs) in enumerate(zip(prompts, holdings)):
        pu, pb = tok(p)
        vu, vb = _vec(pu, m["idf"]), _vec(pb, m["idf"])
        for j, h in enumerate(hs):
            hu, hb = tok(h)
            F[i, j] = [_cos(vu, _vec(hu, m["idf"])), _cos(vb, _vec(hb, m["idf"])),
                       np.log1p(len(hu)), len(set(hu) & set(pu)) / (len(set(hu)) + 1), 1.0 if "not" in hu else 0.0, 1.0]
    F[..., :5] -= F[..., :5].mean(1, keepdims=True)          # only differences between candidates matter
    return F


def load(checkpoint_dir, models_dir=None, device="cpu"):
    with np.load(Path(checkpoint_dir) / "lexical.npz", allow_pickle=False) as z:
        return {"idf": dict(zip(z["terms"].tolist(), z["idf"].tolist())), "w": z["w"]}


def predict(model, prompts, holdings):
    return features(model, prompts, holdings) @ model["w"]
