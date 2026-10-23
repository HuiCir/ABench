"""U-KAN-009: function-fitting tasks and Fashion-MNIST (Xiao et al. 2017, MIT licence, from the authors' GitHub
repository).

Functions: inputs uniform on [-1, 1]^d; 2,000 training, 500 validation and 5,000 test points per function, targets
standardised with the training mean and standard deviation, no noise. The five dev functions are public (all three
splits, in dev_functions.npz). The twelve held-out functions and their formulas are not in the sandbox; their points
are drawn with a secret seed (ABENCH_U_KAN_SEED), so even a guessed formula gives no test points.
Images: the 60,000 Fashion-MNIST training images are public; the 10,000 test images are held out.
"""
from __future__ import annotations

import gzip
import os
from pathlib import Path

import numpy as np

from prepare.common import Frozen, check

COMMIT = "b2617bb6d3ffa2e429640350f613e3291e10b141"
FILES = {"train-images-idx3-ubyte.gz": "3aede38d61863908ad78613f6a32ed271626dd12800ba2636569512369268a84",
         "train-labels-idx1-ubyte.gz": "a04f17134ac03560a47e3764e11b92fc97de4d1bfaf8ba1a3aa29af54cc90845",
         "t10k-images-idx3-ubyte.gz": "346e55b948d973a97e58d2351dde16a484bd415d4595297633bb08f03db6a073",
         "t10k-labels-idx1-ubyte.gz": "67da17c76eaffca5446c3361aaab5c3cd6d1c2608764d35dfb1850b086bf8dd5"}
SOURCES = [{"name": f"zalandoresearch/fashion-mnist data/fashion/{f}",
            "url": f"https://github.com/zalandoresearch/fashion-mnist/blob/{COMMIT}/data/fashion/{f}",
            "sha256": s, "license": "MIT", "note": f"RAW/fashion/{f}"} for f, s in FILES.items()]
SECRET_SEED_ENV = "ABENCH_U_KAN_SEED"
N_TRAIN, N_VAL, N_TEST = 2_000, 500, 5_000
PI = np.pi

DEV = {   # public: name -> (input dimension, function of x with shape (N, d))
    "d1": (2, lambda x: np.sin(PI * x[:, 0]) + x[:, 1] ** 2),
    "d2": (2, lambda x: x[:, 0] * x[:, 1]),
    "d3": (2, lambda x: np.exp(-(x[:, 0] ** 2 + x[:, 1] ** 2))),
    "d4": (2, lambda x: np.abs(x[:, 0]) + 0.5 * x[:, 1]),
    "d5": (2, lambda x: np.sin(4 * x[:, 0]) * np.cos(3 * x[:, 1])),
}
HELD = [  # held-out functions f00..f11 (never in the sandbox)
    (2, lambda x: np.exp(np.sin(PI * x[:, 0]) + x[:, 1] ** 2)),                          # KAN paper, toy example
    (3, lambda x: x[:, 0] * x[:, 1] * x[:, 2]),                                            # multiplicative
    (3, lambda x: np.sin(PI * x[:, 0]) * np.cos(PI * x[:, 1]) + 0.5 * x[:, 2]),
    (8, lambda x: np.exp(0.25 * np.sum(np.sin(PI * x / 2) ** 2, axis=1))),                 # KAN paper, many inputs
    (3, lambda x: np.sqrt(np.sum(x ** 2, axis=1))),                                        # distance
    (2, lambda x: np.tanh(5 * (x[:, 0] ** 4 + x[:, 1] ** 4 - 0.5))),                       # sharp transition
    (3, lambda x: np.exp(-((x[:, 0] - x[:, 1]) ** 2) / (2 * (1.5 + 0.5 * x[:, 2]) ** 2))
     / (np.sqrt(2 * PI) * (1.5 + 0.5 * x[:, 2]))),                                         # Feynman I.6.2b, scaled
    (2, lambda x: 1 / (1 + 25 * x[:, 0] ** 2) + x[:, 1]),                                  # Runge
    (4, lambda x: np.maximum(0, x[:, 0] - x[:, 1]) + np.maximum(0, x[:, 2] + x[:, 3])),   # piecewise linear
    (2, lambda x: np.sin(8 * x[:, 0]) * x[:, 1]),                                          # oscillating
    (2, lambda x: (x[:, 0] + x[:, 1]) / (1 + x[:, 0] * x[:, 1] / 2)),                      # velocity-addition form
    (3, lambda x: np.log(2 + x[:, 0] + x[:, 1] ** 2) * x[:, 2]),
]


def sample(d: int, f, rng: np.random.Generator) -> dict:
    n = N_TRAIN + N_VAL + N_TEST
    x = rng.uniform(-1, 1, size=(n, d))
    y = f(x)
    mu, sd = y[:N_TRAIN].mean(), y[:N_TRAIN].std()
    y = (y - mu) / (sd if sd > 0 else 1.0)
    a, b = N_TRAIN, N_TRAIN + N_VAL
    return {"x_train": x[:a].astype(np.float32), "y_train": y[:a].astype(np.float32),
            "x_val": x[a:b].astype(np.float32), "y_val": y[a:b].astype(np.float32),
            "x_test": x[b:].astype(np.float32), "y_test": y[b:].astype(np.float32)}


def _idx(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as fh:
        b = fh.read()
    if int.from_bytes(b[2:3], "big") != 8:
        raise SystemExit(f"{path.name}: not an unsigned-byte idx file")
    nd = b[3]
    shape = [int.from_bytes(b[4 + 4 * i:8 + 4 * i], "big") for i in range(nd)]
    return np.frombuffer(b, np.uint8, offset=4 + 4 * nd).reshape(shape)


def _freeze(xtr, ytr, xte, yte, seed: int, sums: dict) -> Frozen:
    rng_dev = np.random.default_rng(20240430)
    dev = {}
    for name, (d, f) in DEV.items():
        for k, v in sample(d, f, rng_dev).items():
            dev[f"{name}_{k}"] = v
    rng = np.random.default_rng(seed)
    hidden = {}
    for j, (d, f) in enumerate(HELD):
        for k, v in sample(d, f, rng).items():
            hidden[f"f{j:02d}_{k}"] = v
    hidden["img_x_test"] = xte.reshape(len(xte), -1)
    hidden["img_y_test"] = yte.astype(np.int64)
    public = {"dev_functions.npz": dev,
              "fashion_train.npz": {"x": xtr.reshape(len(xtr), -1), "y": ytr.astype(np.int64)}}
    return Frozen(public=public, hidden=hidden, sources_sha256=sums,
                  notes={"functions": len(HELD), "points": [N_TRAIN, N_VAL, N_TEST]})


def build(raw: Path) -> Frozen:
    seed = int(os.environ.get(SECRET_SEED_ENV, "0"))
    if not seed:
        raise SystemExit(f"set {SECRET_SEED_ENV} to a secret integer (sampling of the held-out functions)")
    sums = {f: check(raw / "fashion" / f, sha256=s) for f, s in FILES.items()}
    xtr, ytr = _idx(raw / "fashion" / "train-images-idx3-ubyte.gz"), _idx(raw / "fashion" / "train-labels-idx1-ubyte.gz")
    xte, yte = _idx(raw / "fashion" / "t10k-images-idx3-ubyte.gz"), _idx(raw / "fashion" / "t10k-labels-idx1-ubyte.gz")
    assert xtr.shape == (60000, 28, 28) and xte.shape == (10000, 28, 28) and set(np.unique(ytr)) == set(range(10))
    return _freeze(xtr, ytr, xte, yte, seed, sums)


def synthetic(n: int) -> Frozen:
    """Random images with label-dependent means (n training images, n // 4 test images)."""
    rng = np.random.default_rng(1)
    ytr, yte = rng.integers(0, 10, n), rng.integers(0, 10, n // 4)
    proto = rng.integers(0, 256, size=(10, 784))
    xtr = np.clip(proto[ytr] + rng.normal(0, 60, (n, 784)), 0, 255).astype(np.uint8)
    xte = np.clip(proto[yte] + rng.normal(0, 60, (n // 4, 784)), 0, 255).astype(np.uint8)
    return _freeze(xtr, ytr, xte, yte, 12345, {"synthetic": f"seed1-n{n}"})
