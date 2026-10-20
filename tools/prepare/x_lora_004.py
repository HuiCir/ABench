"""X-LORA-004: GLUE RTE, MRPC, CoLA and SST-2 (nyu-mll/glue); public training splits, held-out validation splits
(GLUE test labels are not public); frozen roberta-base (MIT).

GLUE licences differ by task (the card lists "other"): RTE and MRPC follow their original sources, CoLA and SST-2
are distributed for research; the benchmark uses them for non-commercial research evaluation only.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from prepare.common import Frozen, check, dev_indices, text_array

REV = "bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c"
ROBERTA_REV = "e2da8e2f811d1448a5b465c236feacd80ffbac7b"
TASKS = {"rte": ("sentence1", "sentence2"), "mrpc": ("sentence1", "sentence2"), "cola": ("sentence", None),
         "sst2": ("sentence", None)}
SOURCES = [{"name": f"{t}/{s}-00000-of-00001.parquet",
            "url": f"https://huggingface.co/datasets/nyu-mll/glue/resolve/{REV}/{t}/{s}-00000-of-00001.parquet",
            "license": "GLUE: per-task terms ('other' on the dataset card)"} for t in TASKS for s in ("train", "validation")] + [
           {"name": "roberta-base/", "url": f"https://huggingface.co/FacebookAI/roberta-base/tree/{ROBERTA_REV}",
            "files": ["config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"],
            "license": "MIT"}]
DEV_FRAC, DEV_SEED = 0.1, 20260701


def build(raw: Path) -> Frozen:
    import pandas as pd
    pub, hid, sums = {}, {}, {}
    for t, (a, b) in TASKS.items():
        for s, store in (("train", pub), ("validation", hid)):
            p = raw / "glue" / t / f"{s}-00000-of-00001.parquet"
            sums[f"{t}/{p.name}"] = check(p)
            df = pd.read_parquet(p)
            store[f"{t}_{a}"] = text_array(df[a])
            if b:
                store[f"{t}_{b}"] = text_array(df[b])
            store[f"{t}_label"] = df["label"].to_numpy(np.int64)
        pub[f"{t}_dev_idx"] = dev_indices(len(pub[f"{t}_label"]), DEV_FRAC, DEV_SEED, pub[f"{t}_label"])
    rb = raw / "roberta-base"
    for f in SOURCES[-1]["files"]:
        sums[f"roberta-base/{f}"] = check(rb / f)
    return Frozen(public={"glue_train.npz": pub}, hidden=hid, sources_sha256=sums, models={"roberta-base": rb})


def synthetic(n: int) -> Frozen:
    rng = np.random.default_rng(0)
    pub, hid = {}, {}
    for t, (a, b) in TASKS.items():
        for store, m in ((pub, n), (hid, n // 4)):
            y = rng.integers(0, 2, m)
            store[f"{t}_{a}"] = text_array([f"word{int(v)} filler {i}" for i, v in enumerate(y)])
            if b:
                store[f"{t}_{b}"] = text_array([f"other{int(v)}" for v in y])
            store[f"{t}_label"] = y.astype(np.int64)
        pub[f"{t}_dev_idx"] = dev_indices(n, DEV_FRAC, DEV_SEED, pub[f"{t}_label"])
    return Frozen(public={"glue_train.npz": pub}, hidden=hid, sources_sha256={"synthetic": f"n{n}"})
