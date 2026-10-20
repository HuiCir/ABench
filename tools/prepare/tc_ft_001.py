"""TC-FT-001: AG News -> public train.npz (with a dev split) and the held-out official test split."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from prepare.common import Frozen, check, dev_indices, text_array

REV = "eb185aade064a813bc0b7f42de02595523103ca4"
BASE = f"https://huggingface.co/datasets/fancyzhx/ag_news/resolve/{REV}/data"
SOURCES = [
    {"name": "train-00000-of-00001.parquet", "url": f"{BASE}/train-00000-of-00001.parquet",
     "license": "unknown (research / non-commercial per dataset card)"},
    {"name": "test-00000-of-00001.parquet", "url": f"{BASE}/test-00000-of-00001.parquet",
     "license": "unknown (research / non-commercial per dataset card)"},
]
DEV_FRAC, DEV_SEED = 0.05, 20260701


def _read(p: Path):
    import pandas as pd                       # operator environment only
    df = pd.read_parquet(p)
    return text_array(df["text"]), df["label"].to_numpy(np.int64)


def build(raw: Path) -> Frozen:
    sums = {s["name"]: check(raw / s["name"]) for s in SOURCES}
    xtr, ytr = _read(raw / SOURCES[0]["name"])
    xte, yte = _read(raw / SOURCES[1]["name"])
    assert len(ytr) == 120000 and len(yte) == 7600, (len(ytr), len(yte))
    dev = dev_indices(len(ytr), DEV_FRAC, DEV_SEED, ytr)
    return Frozen(public={"train.npz": {"text": xtr, "y": ytr, "dev_idx": dev}},
                  hidden={"text": xte, "y": yte}, sources_sha256=sums)


def synthetic(n: int) -> Frozen:
    rng = np.random.default_rng(0)
    vocab = [[f"w{c}_{i}" for i in range(30)] for c in range(4)] + [[f"common{i}" for i in range(200)]]

    def make(m):
        y = rng.integers(0, 4, m)
        docs = [" ".join(list(rng.choice(vocab[c], 4)) + list(rng.choice(vocab[4], 12))) for c in y]
        return text_array(docs), y.astype(np.int64)
    xtr, ytr = make(n)
    xte, yte = make(max(200, n // 5))
    return Frozen(public={"train.npz": {"text": xtr, "y": ytr, "dev_idx": dev_indices(n, DEV_FRAC, DEV_SEED, ytr)}},
                  hidden={"text": xte, "y": yte}, sources_sha256={"synthetic": f"seed0-n{n}"})
