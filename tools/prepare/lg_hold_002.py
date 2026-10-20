"""LG-HOLD-002: CaseHOLD (Zheng et al. 2021), config "all": public train + val, held-out test; frozen roberta-base.

CSV layout (from the dataset's loading script casehold.py at the pinned revision): column 0 example_id, column 1
citing prompt, columns 2-6 the five candidate holdings, column 12 the index of the correct holding; header row skipped.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

from prepare.common import Frozen, check, text_array

REV = "acc4532a67e0966e9ed7ec4ca543e8532f983b0c"
ROBERTA_REV = "e2da8e2f811d1448a5b465c236feacd80ffbac7b"
SOURCES = [{"name": f"data/all/{s}.csv", "url": f"https://huggingface.co/datasets/casehold/casehold/resolve/{REV}/data/all/{s}.csv",
            "license": "not stated on the dataset card or loading script; reglab/casehold repository is Apache-2.0"}
           for s in ("train", "val", "test")] + [
           {"name": "roberta-base/", "url": f"https://huggingface.co/FacebookAI/roberta-base/tree/{ROBERTA_REV}",
            "files": ["config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"],
            "license": "MIT"}]


def _read(p: Path):
    csv.field_size_limit(sys.maxsize)
    with open(p, encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f))[1:]
    prompt = text_array(r[1] for r in rows)
    hold = np.asarray([[str(x) for x in r[2:7]] for r in rows], dtype=np.str_)
    y = np.asarray([int(r[12]) for r in rows], np.int64)
    assert set(np.unique(y)) <= set(range(5))
    return prompt, hold, y


def build(raw: Path) -> Frozen:
    sums = {f"{s}.csv": check(raw / "casehold" / f"{s}.csv") for s in ("train", "val", "test")}
    tp, th, ty = _read(raw / "casehold" / "train.csv")
    vp, vh, vy = _read(raw / "casehold" / "val.csv")
    sp, sh, sy = _read(raw / "casehold" / "test.csv")
    rb = raw / "roberta-base"
    for f in SOURCES[-1]["files"]:
        sums[f"roberta-base/{f}"] = check(rb / f)
    return Frozen(public={"train.npz": {"prompt": tp, "holdings": th, "y": ty, "val_prompt": vp, "val_holdings": vh, "val_y": vy}},
                  hidden={"prompt": sp, "holdings": sh, "y": sy}, sources_sha256=sums, models={"roberta-base": rb})


def synthetic(n: int) -> Frozen:
    rng = np.random.default_rng(0)
    words = [f"w{i}" for i in range(400)]

    def make(m):
        P, H, Y = [], [], []
        for _ in range(m):
            topic = list(rng.choice(words, 6))
            y = int(rng.integers(0, 5))
            hs = [" ".join(rng.choice(words, 8)) for _ in range(5)]
            hs[y] = " ".join(topic[:4] + list(rng.choice(words, 4)))
            P.append(" ".join(topic + list(rng.choice(words, 20)))); H.append(hs); Y.append(y)
        return text_array(P), np.asarray(H, dtype=np.str_), np.asarray(Y, np.int64)
    a, b, c = make(n), make(n // 8), make(n // 8)
    return Frozen(public={"train.npz": {"prompt": a[0], "holdings": a[1], "y": a[2], "val_prompt": b[0],
                                        "val_holdings": b[1], "val_y": b[2]}},
                  hidden={"prompt": c[0], "holdings": c[1], "y": c[2]}, sources_sha256={"synthetic": f"seed0-n{n}"})
