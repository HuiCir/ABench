"""IC-AUG-003: CIFAR-10 and CIFAR-10.1 v6, read and checked by tools/prepare_ic_spd.py (the data of IC-SPD-001).
Public: the 50,000 training images with a 5,000-image dev split; held out: the 10,000 test images and the 2,000
CIFAR-10.1 images."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from prepare.common import Frozen

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import prepare_ic_spd as ic  # noqa: E402

SOURCES = [
    {"name": "cifar-10-python.tar.gz", "url": "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz",
     "md5": ic.CIFAR10_MD5, "license": "no explicit license on the dataset page (to_verify)"},
    {"name": "cifar10.1_v6_data.npy", "sha256": ic.CIFAR101_SHA256["cifar10.1_v6_data.npy"],
     "url": "https://github.com/modestyachts/CIFAR-10.1/raw/d9982abb0bfc4846b8d13a11e66b887d946205d0/datasets/cifar10.1_v6_data.npy",
     "license": "code MIT; images follow Tiny Images terms (to_verify)"},
    {"name": "cifar10.1_v6_labels.npy", "sha256": ic.CIFAR101_SHA256["cifar10.1_v6_labels.npy"],
     "url": "https://github.com/modestyachts/CIFAR-10.1/raw/d9982abb0bfc4846b8d13a11e66b887d946205d0/datasets/cifar10.1_v6_labels.npy",
     "license": "code MIT; images follow Tiny Images terms (to_verify)"},
]


def _freeze(parts) -> Frozen:
    xtr, ytr, xte, yte, x101, y101, sums = parts
    dev = ic.dev_split(ytr, ic.DEV_PER_CLASS)
    return Frozen(public={"train.npz": {"x": xtr, "y": ytr, "dev_idx": dev}},
                  hidden={"x_test": xte, "y_test": yte, "x_101": x101, "y_101": y101}, sources_sha256=sums)


def build(raw: Path) -> Frozen:
    return _freeze(ic.load_raw(raw))


def synthetic(n: int) -> Frozen:
    return _freeze(ic.synthetic(n))
