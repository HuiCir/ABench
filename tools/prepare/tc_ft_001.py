"""TC-FT-001: Yelp Review Full (5 star ratings) -> public train.npz (with a dev split) and the held-out official test
split. Texts are stored packed (UTF-8 bytes + offsets, see cases/TC-FT-001/harness/textpack.py)."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from prepare.common import Frozen, check, dev_indices, text_pack

REV = "c1f9ee939b7d05667af864ee1cb066393154bf85"
BASE = f"https://huggingface.co/datasets/Yelp/yelp_review_full/resolve/{REV}/yelp_review_full"
LICENSE = "other: Yelp Dataset Agreement (non-commercial, academic use); operator confirms ABench use before release"
SOURCES = [
    {"name": "train-00000-of-00001.parquet", "url": f"{BASE}/train-00000-of-00001.parquet", "license": LICENSE},
    {"name": "test-00000-of-00001.parquet", "url": f"{BASE}/test-00000-of-00001.parquet", "license": LICENSE},
]
N_TRAIN, N_TEST, N_CLASSES = 650_000, 50_000, 5
DEV_FRAC, DEV_SEED = 0.02, 20260701


def _read(p: Path):
    import pandas as pd                       # operator environment only
    df = pd.read_parquet(p)
    return list(df["text"]), df["label"].to_numpy(np.int64)


def _split(texts, y) -> dict:
    utf8, off = text_pack(texts)
    return {"text_utf8": utf8, "text_offsets": off, "y": y}


def build(raw: Path) -> Frozen:
    sums = {s["name"]: check(raw / s["name"]) for s in SOURCES}
    xtr, ytr = _read(raw / SOURCES[0]["name"])
    xte, yte = _read(raw / SOURCES[1]["name"])
    assert len(ytr) == N_TRAIN and len(yte) == N_TEST, (len(ytr), len(yte))
    assert set(np.unique(ytr)) == set(range(N_CLASSES)) and set(np.unique(yte)) == set(range(N_CLASSES))
    dev = dev_indices(len(ytr), DEV_FRAC, DEV_SEED, ytr)
    return Frozen(public={"train.npz": dict(_split(xtr, ytr), dev_idx=dev)}, hidden=_split(xte, yte), sources_sha256=sums)


def synthetic(n: int) -> Frozen:
    rng = np.random.default_rng(0)
    vocab = [[f"w{c}_{i}" for i in range(30)] for c in range(N_CLASSES)] + [[f"common{i}" for i in range(200)]]

    def make(m):
        y = rng.integers(0, N_CLASSES, m)
        docs = [" ".join(list(rng.choice(vocab[c], 4)) + list(rng.choice(vocab[-1], int(rng.integers(8, 40)))))
                + (" é" if i % 7 == 0 else "") for i, c in enumerate(y)]
        return docs, y.astype(np.int64)
    xtr, ytr = make(n)
    xte, yte = make(max(200, n // 5))
    return Frozen(public={"train.npz": dict(_split(xtr, ytr), dev_idx=dev_indices(n, DEV_FRAC * 2.5, DEV_SEED, ytr))},
                  hidden=_split(xte, yte), sources_sha256={"synthetic": f"seed0-n{n}"})
