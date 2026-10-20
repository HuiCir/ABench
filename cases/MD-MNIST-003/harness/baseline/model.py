"""MD-MNIST-003 reference submission, inference side: random ReLU features with a ridge classifier (NumPy).

Each image is flattened, standardised per pixel, and mapped through a fixed random projection followed by ReLU
(4,096 features, seeded); one-vs-rest ridge regression to one-hot labels is solved in closed form, with the ridge
strength chosen on a validation split. One model per dataset; train.py fits them and saves checkpoint/<name>.npz.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

N_RANDOM = 4096


def flat(x):
    x = np.asarray(x, np.float32) / 255.0
    return x.reshape(len(x), -1)


def phi(m, x):
    z = (flat(x) - m["mu"]) / m["sd"]
    h = np.maximum(z @ m["R"] + m["b"], 0.0)
    return np.concatenate([h, np.ones((len(h), 1), np.float32)], 1)


def load(checkpoint_dir, device="cpu"):
    model = {}
    for name in ("blood", "organa"):
        with np.load(Path(checkpoint_dir) / f"{name}.npz", allow_pickle=False) as z:
            model[name] = {k: z[k] for k in z.files}
    return model


def predict(model, name, x):
    s = phi(model[name], x) @ model[name]["W"]
    e = np.exp(s - s.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)
