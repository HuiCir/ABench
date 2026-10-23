"""Reference pipeline for X-DATA-007: exact deduplication, MinHash near-duplicate removal, then a quality filter — a
logistic regression on hashed word 1-2-grams separating the 5,000 target examples from random pool documents —
keeping the documents the classifier rates highest until the token budget is reached (each document at most once)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src import dedup  # noqa: E402


def _unpack(u, o, idx=None):
    b = np.asarray(u, np.uint8).tobytes()
    rows = range(len(o) - 1) if idx is None else idx
    return [b[o[i]:o[i + 1]].decode("utf-8", "replace") for i in rows]


def select(pool_dir, budget_tokens, workdir):
    from sklearn.feature_extraction.text import HashingVectorizer
    from sklearn.linear_model import LogisticRegression
    pool_dir = Path(pool_dir)
    with np.load(pool_dir / "pool_text.npz", allow_pickle=False) as z:
        texts = _unpack(z["text_utf8"], z["text_offsets"])
    with np.load(pool_dir / "pool_index.npz", allow_pickle=False) as z:
        off = z["token_offsets"]
    with np.load(pool_dir / "target_examples.npz", allow_pickle=False) as z:
        target = _unpack(z["text_utf8"], z["text_offsets"])
    n_tok = np.diff(off)
    keep = dedup.exact_keep(texts)
    idx = np.flatnonzero(keep)
    keep_near = dedup.near_keep(dedup.minhash([texts[i] for i in idx]))
    idx = idx[keep_near]
    hv = HashingVectorizer(n_features=2 ** 20, ngram_range=(1, 2), alternate_sign=False, norm="l2")
    rng = np.random.default_rng(0)
    neg = rng.choice(idx, min(len(idx), 4 * len(target)), replace=False)
    X = hv.transform(target + [texts[i] for i in neg])
    y = np.r_[np.ones(len(target)), np.zeros(len(neg))]
    clf = LogisticRegression(C=1.0, max_iter=200).fit(X, y)
    scores = np.concatenate([clf.decision_function(hv.transform([texts[i] for i in idx[s:s + 20000]]))
                             for s in range(0, len(idx), 20000)])
    order = idx[np.argsort(-scores)]
    cum = np.cumsum(n_tok[order])
    return order[:int(np.searchsorted(cum, budget_tokens)) + 1]
