"""RS-SEQ-001 reference submission, inference side: first-order item transitions blended with time-decayed popularity.

Not the anchor (SASRec); a transparent, fast baseline. train.py counts the transitions and the popularity and saves
them to checkpoint/markov.npz. For each user the next item is scored by
  s(j) = log(1 + T[last, j]) + beta * log(1 + recent_popularity(j)),
where `last` is the final item of the history passed to recommend(); items already in that history are excluded.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def load(checkpoint_dir, device="cpu"):
    with np.load(Path(checkpoint_dir) / "markov.npz", allow_pickle=False) as z:
        return {"T": z["T"], "pop": z["pop"]}


def recommend(model, histories, users, k=10):
    T, pop = model["T"], model["pop"]
    items, off = histories["items"].astype(np.int64), histories["offsets"]
    out = np.zeros((len(users), k), dtype=np.int64)
    for r, u in enumerate(users):
        hist = items[off[u]:off[u + 1]]
        s = (T[hist[-1]] + pop) if len(hist) else pop.copy()
        s[0] = -np.inf
        s[hist] = -np.inf
        top = np.argpartition(-s, k)[:k]
        out[r] = top[np.argsort(-s[top], kind="stable")]
    return out
