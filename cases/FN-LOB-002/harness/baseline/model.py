"""FN-LOB-002 reference submission, inference side: hand-crafted order-book features with a ridge classifier (NumPy).

Features of a window: the last snapshot (40 values), the mean and the change of the snapshot over the last 10 and 100
events, and order-flow imbalance proxies at the top 3 levels. Classes are fitted one-vs-rest by ridge regression with
class-balanced row weights, which counteracts the large share of the stationary class (train.py fits them and saves
checkpoint/ridge.npz).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

WINDOW = 100


def feats(w: np.ndarray) -> np.ndarray:
    last = w[:, -1, :]
    f = [last, w[:, -10:, :].mean(1), last - w[:, -10, :], last - w[:, 0, :]]
    ask_v, bid_v = w[:, :, 1:12:4], w[:, :, 3:12:4]               # volumes at levels 1-3
    imb = (bid_v - ask_v) / (np.abs(bid_v) + np.abs(ask_v) + 1e-3)
    f += [imb[:, -1, :], imb[:, -10:, :].mean(1)]
    out = np.concatenate(f, 1)
    return np.concatenate([out, np.ones((len(out), 1), np.float32)], 1)


def load(checkpoint_dir, device="cpu"):
    with np.load(Path(checkpoint_dir) / "ridge.npz", allow_pickle=False) as z:
        return {k: z[k] for k in ("W", "mu", "sd")}


def predict(model, x):
    f = (feats(np.asarray(x, np.float32)).astype(np.float64) - model["mu"]) / model["sd"]
    s = f @ model["W"]
    e = np.exp(s - s.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)
