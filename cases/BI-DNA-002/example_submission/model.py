"""BI-DNA-002 reference submission, inference side: k-mer spectrum features (k = 1..6) with a ridge classifier (NumPy).

Each sequence is represented by the normalised counts of all k-mers for k = 1..6 (5,460 features, N and other
symbols break k-mers), log-scaled and standardised; a ridge regression to ±1 labels is solved per dataset with the
ridge strength chosen on a seeded 10% split of the training data (train.py; checkpoint/<dataset>.npz).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

KS = (1, 2, 3, 4, 5, 6)
CODE = np.full(256, -1, np.int64)
for i, c in enumerate(b"ACGT"):
    CODE[c] = i
    CODE[c + 32] = i                                      # lower case


def spectrum(seqs) -> np.ndarray:
    dims = [4 ** k for k in KS]
    base = np.concatenate([[0], np.cumsum(dims)])
    out = np.zeros((len(seqs), base[-1]), np.float32)
    for r, s in enumerate(seqs):
        a = CODE[np.frombuffer(s.encode("ascii", "replace"), np.uint8)]
        for j, k in enumerate(KS):
            if len(a) < k:
                continue
            win = np.lib.stride_tricks.sliding_window_view(a, k)
            ok = (win >= 0).all(1)
            idx = (win[ok] * (4 ** np.arange(k - 1, -1, -1))).sum(1)
            cnt = np.bincount(idx, minlength=dims[j]).astype(np.float32)
            out[r, base[j]:base[j + 1]] = cnt / max(1.0, cnt.sum())
    return np.log1p(1000.0 * out)


def load(checkpoint_dir, device="cpu"):
    model = {}
    for p in sorted(Path(checkpoint_dir).glob("*.npz")):
        with np.load(p, allow_pickle=False) as z:
            model[p.stem] = {k: z[k] for k in z.files}
    return model


def predict(model, name, seqs):
    m = model[name]
    F = (spectrum(seqs).astype(np.float64) - m["mu"]) / m["sd"]
    z = np.concatenate([F, np.ones((len(F), 1))], 1) @ m["w"]
    p = 1.0 / (1.0 + np.exp(-2.0 * z))
    return np.stack([1 - p, p], 1)
