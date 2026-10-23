"""X-LORA-004: GLUE SST-2, CoLA, MNLI and QNLI (nyu-mll/glue); public training splits, held-out validation splits
(MNLI: validation_matched; GLUE test labels are not public); frozen roberta-base (MIT). Texts are stored packed
(UTF-8 bytes + offsets, cases/X-LORA-004/harness/textpack.py): MNLI has 392,702 training pairs.

RTE and MRPC were replaced by MNLI and QNLI (2026-10-21 audit: their validation sets, 277 and 408 examples, are so
small that noise exceeds method differences). GLUE licences differ by task (the card lists "other"): SST-2 and CoLA are
distributed for research; MNLI is mostly OANC/OANC-derived text plus fiction (see its paper); QNLI derives from SQuAD
(CC BY-SA 4.0). The benchmark uses them for non-commercial research evaluation only.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from prepare.common import Frozen, check, dev_indices, text_pack

REV = "bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c"
ROBERTA_REV = "e2da8e2f811d1448a5b465c236feacd80ffbac7b"
TASKS = {"sst2": ("sentence", None), "cola": ("sentence", None), "mnli": ("premise", "hypothesis"),
         "qnli": ("question", "sentence")}
VAL = {"sst2": "validation", "cola": "validation", "mnli": "validation_matched", "qnli": "validation"}
SIZES = {"sst2": (67349, 872), "cola": (8551, 1043), "mnli": (392702, 9815), "qnli": (104743, 5463)}
SOURCES = [{"name": f"{t}/{s}-00000-of-00001.parquet",
            "url": f"https://huggingface.co/datasets/nyu-mll/glue/resolve/{REV}/{t}/{s}-00000-of-00001.parquet",
            "license": "GLUE: per-task terms ('other' on the dataset card)"} for t in TASKS for s in ("train", VAL[t])] + [
           {"name": "roberta-base/", "url": f"https://huggingface.co/FacebookAI/roberta-base/tree/{ROBERTA_REV}",
            "files": ["config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"],
            "license": "MIT"}]
DEV_FRAC, DEV_SEED = 0.1, 20260701


def build(raw: Path) -> Frozen:
    import pandas as pd
    pub, hid, sums = {}, {}, {}
    for t, (a, b) in TASKS.items():
        for i, (s, store) in enumerate((("train", pub), (VAL[t], hid))):
            p = raw / "glue" / t / f"{s}-00000-of-00001.parquet"
            sums[f"{t}/{p.name}"] = check(p)
            df = pd.read_parquet(p)
            assert len(df) == SIZES[t][i], (t, s, len(df))
            for c in (a, b):
                if c:
                    store[f"{t}_{c}_utf8"], store[f"{t}_{c}_offsets"] = text_pack(df[c])
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
        k = 3 if t == "mnli" else 2
        for store, m in ((pub, n), (hid, n // 4)):
            y = rng.integers(0, k, m)
            store[f"{t}_{a}_utf8"], store[f"{t}_{a}_offsets"] = text_pack([f"word{int(v)} filler {i}" for i, v in enumerate(y)])
            if b:
                store[f"{t}_{b}_utf8"], store[f"{t}_{b}_offsets"] = text_pack([f"other{int(v)}" for v in y])
            store[f"{t}_label"] = y.astype(np.int64)
        pub[f"{t}_dev_idx"] = dev_indices(n, DEV_FRAC, DEV_SEED, pub[f"{t}_label"])
    return Frozen(public={"glue_train.npz": pub}, hidden=hid, sources_sha256={"synthetic": f"n{n}"})
