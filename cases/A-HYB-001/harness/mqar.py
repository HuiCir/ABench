"""A-HYB-001 copy of the A-LIN-002 multi-query associative recall (MQAR) generator, after Zoology (Arora et al. 2023).

A sequence of length T starts with K key-value pairs  k1 v1 k2 v2 ... kK vK  (keys drawn without replacement from
[1, V/2), values from [V/2, V)). Each of the K keys then appears once more at a random position of the remaining
T - 2K positions; every other remaining position holds the filler token 0. The model reads the sequence left to right
and, at the position of each repeated key, must predict the next token: that key's value. Targets are -100 elsewhere.
"""
from __future__ import annotations

import numpy as np

VOCAB = 8192
IGNORE = -100
TRAIN_GRID = [(64, 4), (64, 8), (128, 8), (128, 16), (256, 16), (256, 32), (256, 64)]   # (T, K) used for training


def batch(rng: np.random.Generator, n: int, T: int, K: int) -> tuple[np.ndarray, np.ndarray]:
    if 4 * K > T:
        raise ValueError("need T >= 4K")
    half = VOCAB // 2
    x = np.zeros((n, T), np.int64)
    y = np.full((n, T), IGNORE, np.int64)
    for r in range(n):
        keys = rng.choice(np.arange(1, half), size=K, replace=False)
        vals = rng.integers(half, VOCAB, size=K)
        x[r, 0:2 * K:2], x[r, 1:2 * K:2] = keys, vals
        pos = 2 * K + rng.choice(T - 2 * K, size=K, replace=False)
        x[r, pos] = keys
        y[r, pos] = vals
    return x, y
