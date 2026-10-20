"""T-OPT-002: CIFAR-10 (Krizhevsky 2009). Public: the 50,000 training images with a 5,000-image dev split;
held-out: the 10,000 official test images. Reads the same archive as IC-SPD-001 (tools/prepare_ic_spd.py)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from prepare.common import Frozen, check

TOOLS = Path(__file__).resolve().parents[1]
SOURCES = [{"name": "cifar-10-python.tar.gz", "url": "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz",
            "md5": "c58f30108f718f92721af3b95e74349a", "mirror": "https://huggingface.co/datasets/uoft-cs/cifar10",
            "license": "no explicit license on the dataset page (to_verify)"}]


def _ic():
    sys.path.insert(0, str(TOOLS))
    import prepare_ic_spd
    return prepare_ic_spd


def build(raw: Path) -> Frozen:
    ic = _ic()
    import tarfile
    tar = raw / SOURCES[0]["name"]
    sums = {tar.name: check(tar, md5=SOURCES[0]["md5"])}
    with tarfile.open(tar, "r:gz") as tf:
        parts = [ic._batch(tf, f"data_batch_{i}") for i in range(1, 6)]
        xte, yte = ic._batch(tf, "test_batch")
    xtr, ytr = np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts])
    return Frozen(public={"train.npz": {"x": xtr, "y": ytr, "dev_idx": ic.dev_split(ytr, ic.DEV_PER_CLASS)}},
                  hidden={"x": xte, "y": yte}, sources_sha256=sums)


def synthetic(n: int) -> Frozen:
    ic = _ic()
    xtr, ytr, xte, yte, _, _, src = ic.synthetic(n)
    return Frozen(public={"train.npz": {"x": xtr, "y": ytr, "dev_idx": ic.dev_split(ytr, ic.DEV_PER_CLASS)}},
                  hidden={"x": xte, "y": yte}, sources_sha256=src)
