"""MD-MNIST-003: BloodMNIST and OrganAMNIST (MedMNIST v2/v3 on Zenodo, CC BY 4.0), 28×28, official splits.

DermaMNIST is excluded on purpose (CC BY-NC 4.0). The held-out set is the official test split of each dataset.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from prepare.common import Frozen, check

SOURCES = [
    {"name": "bloodmnist.npz", "url": "https://zenodo.org/records/10519652/files/bloodmnist.npz",
     "md5": "7053d0359d879ad8a5505303e11de1dc", "license": "CC BY 4.0"},
    {"name": "organamnist.npz", "url": "https://zenodo.org/records/10519652/files/organamnist.npz",
     "md5": "68e3f8846a6bd62f0c9bf841c0d9eacc", "license": "CC BY 4.0"},
]
NAMES = {"blood": "bloodmnist.npz", "organa": "organamnist.npz"}


def build(raw: Path) -> Frozen:
    md5 = {s["name"]: s["md5"] for s in SOURCES}
    public, hidden, sums = {}, {}, {}
    for d, fn in NAMES.items():
        sums[fn] = check(raw / fn, md5=md5[fn])
        with np.load(raw / fn, allow_pickle=False) as z:
            public[f"{d}.npz"] = {"x_train": z["train_images"], "y_train": z["train_labels"].reshape(-1).astype(np.int64),
                                  "x_val": z["val_images"], "y_val": z["val_labels"].reshape(-1).astype(np.int64)}
            hidden[f"x_test_{d}"] = z["test_images"]
            hidden[f"y_test_{d}"] = z["test_labels"].reshape(-1).astype(np.int64)
    return Frozen(public=public, hidden=hidden, sources_sha256=sums)


def synthetic(n: int) -> Frozen:
    rng = np.random.default_rng(0)
    public, hidden = {}, {}
    for d, k, shape in (("blood", 8, (28, 28, 3)), ("organa", 11, (28, 28))):
        proto = rng.integers(0, 256, (k,) + shape)

        def make(m):
            y = rng.integers(0, k, m)
            x = np.clip(proto[y] * 0.5 + rng.normal(100, 50, (m,) + shape), 0, 255).astype(np.uint8)
            return x, y.astype(np.int64)
        (xa, ya), (xb, yb), (xc, yc) = make(n), make(n // 5), make(n // 4)
        public[f"{d}.npz"] = {"x_train": xa, "y_train": ya, "x_val": xb, "y_val": yb}
        hidden[f"x_test_{d}"], hidden[f"y_test_{d}"] = xc, yc
    return Frozen(public=public, hidden=hidden, sources_sha256={"synthetic": f"seed0-n{n}"})
