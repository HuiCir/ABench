"""LG-LONG-001 reference submission, training program (run in the sandbox; the evaluator only loads the checkpoint).

  python3 train.py --data "$ABENCH_DATA_DIR" --out "$ABENCH_SUBMISSION_DIR/checkpoint" [--include-dev] [--quick]

Vocabulary: the 100,000 most frequent word unigrams and bigrams by document frequency (at least 3 training cases);
sublinear tf-idf with L2 normalisation. One logistic regression per article (L2 penalty 1e-4 per example, L-BFGS,
at most 200 iterations), then one decision threshold shared by all articles, chosen on 10% of the training cases
to maximise micro-F1 with the LexGLUE "no violation" label. By default the public validation split is left out so that
selfcheck.py gives an unbiased dev estimate; --include-dev adds it to the training cases.
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from model import terms, tfidf

if os.environ.get("ABENCH_VIEW_DIR"):
    sys.path.insert(0, str(Path(os.environ["ABENCH_VIEW_DIR"]) / "interface"))
from textpack import unpack  # noqa: E402

N_VOCAB, MIN_DF, L2 = 100_000, 3, 1e-4


def fit_logistic(X, y, l2, iters):
    n, d = X.shape

    def f(wb):
        w, b = wb[:-1], wb[-1]
        s = X @ w + b
        p = 1 / (1 + np.exp(-s))
        loss = np.mean(np.logaddexp(0, s) - y * s) + 0.5 * l2 * w @ w
        g = (p - y) / n
        return loss, np.concatenate([X.T @ g + l2 * w, [g.sum()]])
    r = minimize(f, np.zeros(d + 1), jac=True, method="L-BFGS-B", options={"maxiter": iters})
    return r.x[:-1], r.x[-1]


def micro_f1_none(y, p):
    y = np.concatenate([y, (y.sum(1, keepdims=True) == 0)], 1).astype(int)
    p = np.concatenate([p, (p.sum(1, keepdims=True) == 0)], 1).astype(int)
    tp, fp, fn = (y & p).sum(), ((1 - y) & p).sum(), (y & (1 - p)).sum()
    return 2 * tp / max(1, 2 * tp + fp + fn)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--include-dev", action="store_true")
    ap.add_argument("--quick", action="store_true", help="small vocabulary, few iterations (smoke test)")
    a = ap.parse_args()
    with np.load(Path(a.data) / "ecthr_a.npz", allow_pickle=False) as z:
        docs = unpack(z["train_text_utf8"], z["train_text_offsets"])
        y = z["train_y"].astype(np.float64)
        if a.include_dev:
            docs += unpack(z["val_text_utf8"], z["val_text_offsets"])
            y = np.concatenate([y, z["val_y"].astype(np.float64)])
    rng = np.random.default_rng(a.seed)
    perm = rng.permutation(len(docs))
    cut = int(0.9 * len(docs))
    fit_i, sel_i = perm[:cut], perm[cut:]
    df = Counter()
    for i in fit_i:
        df.update(set(terms(docs[i])))
    n_vocab = 5000 if a.quick else N_VOCAB
    vocab_list = [t for t, c in df.most_common(n_vocab) if c >= (1 if a.quick else MIN_DF)]
    vocab = {t: i for i, t in enumerate(vocab_list)}
    idf = np.log((1 + len(fit_i)) / (1 + np.asarray([df[t] for t in vocab_list], np.float64))) + 1
    X = tfidf([docs[i] for i in fit_i], vocab, idf)
    iters = 20 if a.quick else 200
    W = np.zeros((len(vocab_list), y.shape[1]))
    b = np.zeros(y.shape[1])
    for j in range(y.shape[1]):
        W[:, j], b[j] = fit_logistic(X, y[fit_i, j], L2, iters)
    Xs = tfidf([docs[i] for i in sel_i], vocab, idf)
    P = 1 / (1 + np.exp(-(Xs @ W + b)))
    best = max(np.arange(0.1, 0.71, 0.05), key=lambda t: micro_f1_none(y[sel_i].astype(int), (P >= t).astype(int)))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "linear.npz", vocab=np.asarray(vocab_list), idf=idf.astype(np.float32),
                        W=W.astype(np.float32), b=b.astype(np.float32), threshold=np.float64(best))


if __name__ == "__main__":
    main()
