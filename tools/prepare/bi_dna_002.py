"""BI-DNA-002: three human datasets of Genomic Benchmarks (Grešová et al. 2023), official train/test splits.

Read from the Hugging Face mirrors published by the benchmark authors (katarinagresova/Genomic_Benchmarks_<name>),
pinned to the revisions below; columns `seq` (string) and `label` (int64).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from prepare.common import Frozen, check, dev_indices, text_array

DATASETS = {  # name: (revision, train file, test file, train rows, test rows) as listed on 2026-10-19
    "human_nontata_promoters": ("e0003669df0a180c8570ebe091ed91f20fa080ec",
                                "data/train-00000-of-00001-aeb55728756a44ae.parquet",
                                "data/test-00000-of-00001-d7614d9116c62f44.parquet", 27097, 9034),
    "human_enhancers_cohn": ("49ef8d0d80087ece7441096afcef0123b0781ce0",
                             "data/train-00000-of-00001-308a8a054210be5f.parquet",
                             "data/test-00000-of-00001-b9d4d53093c06044.parquet", 20843, 6948),
    "human_ocr_ensembl": ("3c07e4be6425be11073baa2daacf6c143c64e4ff",
                          "data/train-00000-of-00001-3ce5711c171d3e15.parquet",
                          "data/test-00000-of-00001-dcaf3e2a1561c12e.parquet", 139804, 34952),
}
SOURCES = [{"name": f"{d}/{f}", "url": f"https://huggingface.co/datasets/katarinagresova/Genomic_Benchmarks_{d}/resolve/{rev}/{f}",
            "license": "upstream repository ML-Bioinfo-CEITEC/genomic_benchmarks is Apache-2.0; the HF mirror states none"}
           for d, (rev, a, b, _, _) in DATASETS.items() for f in (a, b)]
DEV_FRAC, DEV_SEED = 0.1, 20260701


def build(raw: Path) -> Frozen:
    import pandas as pd                       # operator environment only
    public, hidden, sums = {}, {}, {}
    for d, (_, ftr, fte, ntr, nte) in DATASETS.items():
        ptr, pte = raw / d / ftr, raw / d / fte
        sums[f"{d}/{ftr}"], sums[f"{d}/{fte}"] = check(ptr), check(pte)
        tr, te = pd.read_parquet(ptr), pd.read_parquet(pte)
        assert (len(tr), len(te)) == (ntr, nte), (d, len(tr), len(te))
        ytr = tr["label"].to_numpy(np.int64)
        assert set(np.unique(ytr)) == {0, 1}, (d, np.unique(ytr))           # all three are binary
        public[f"{d}.npz"] = {"seq": text_array(tr["seq"]), "y": ytr, "dev_idx": dev_indices(len(ytr), DEV_FRAC, DEV_SEED, ytr)}
        hidden[f"seq_{d}"], hidden[f"y_{d}"] = text_array(te["seq"]), te["label"].to_numpy(np.int64)
    return Frozen(public=public, hidden=hidden, sources_sha256=sums)


def synthetic(n: int) -> Frozen:
    rng = np.random.default_rng(0)
    public, hidden = {}, {}
    for i, d in enumerate(DATASETS):
        motif = "".join(rng.choice(list("ACGT"), 8))

        def make(m):
            y = rng.integers(0, 2, m)
            seqs = []
            for c in y:
                s = list(rng.choice(list("ACGT"), 200 + 50 * i))
                if c:
                    p = int(rng.integers(0, len(s) - 8))
                    s[p:p + 8] = motif
                seqs.append("".join(s))
            return text_array(seqs), y.astype(np.int64)
        xs, ys = make(n)
        xt, yt = make(n // 4)
        public[f"{d}.npz"] = {"seq": xs, "y": ys, "dev_idx": dev_indices(n, DEV_FRAC, DEV_SEED, ys)}
        hidden[f"seq_{d}"], hidden[f"y_{d}"] = xt, yt
    return Frozen(public=public, hidden=hidden, sources_sha256={"synthetic": f"seed0-n{n}"})
