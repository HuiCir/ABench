"""Reference submission (inference): user and item factors from implicit ALS (train.py); all items scored by inner product,
excluding each user's training items."""
from __future__ import annotations

from pathlib import Path

import numpy as np


def load(checkpoint_dir: str, device: str = "cpu"):
    with np.load(Path(checkpoint_dir) / "factors.npz", allow_pickle=False) as z:
        return {"U": z["U"].astype(np.float32), "V": z["V"].astype(np.float32)}


def recommend(model, train: dict, users: np.ndarray, k: int = 20, chunk: int = 2048) -> np.ndarray:
    U, V = model["U"], model["V"]
    ip, it = train["indptr"], train["items"]
    out = np.empty((len(users), k), np.int64)
    for s in range(0, len(users), chunk):
        us = users[s:s + chunk]
        sc = U[us] @ V.T
        for j, u in enumerate(us):
            sc[j, it[ip[u]:ip[u + 1]]] = -np.inf
        top = np.argpartition(-sc, k, axis=1)[:, :k]
        order = np.argsort(-np.take_along_axis(sc, top, 1), axis=1)
        out[s:s + len(us)] = np.take_along_axis(top, order, 1)
    return out
