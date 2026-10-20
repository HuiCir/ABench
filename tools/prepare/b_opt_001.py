"""B-OPT-001: FineWeb-10B GPT-2 token shards (kjj0/fineweb10B-gpt2, MIT). Public: train shards 1-2 verbatim.
Held-out: the first 4,194,305 tokens of fineweb_val_000000.bin (4,096 windows of 1,024 plus one target token)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from prepare.common import Frozen, check

REV = "889765ea1f903759787add96995d81171b632d0c"
FILES = ["fineweb_train_000001.bin", "fineweb_train_000002.bin", "fineweb_val_000000.bin"]
SOURCES = [{"name": f, "url": f"https://huggingface.co/datasets/kjj0/fineweb10B-gpt2/resolve/{REV}/{f}", "license": "MIT"}
           for f in FILES]
N_HELD = 4 * 1024 * 1024 + 1
HARNESS = Path(__file__).resolve().parents[2] / "cases" / "B-OPT-001" / "harness"


def build(raw: Path) -> Frozen:
    sys.path.insert(0, str(HARNESS))
    from token_data import read_shard
    sums = {f: check(raw / f) for f in FILES}
    val = np.asarray(read_shard(raw / FILES[2])[:N_HELD], np.uint16)
    return Frozen(public={}, hidden={"tokens": val}, sources_sha256=sums,
                  raw_files={f: raw / f for f in FILES[:2]})


def synthetic(n: int) -> Frozen:
    import tempfile
    sys.path.insert(0, str(HARNESS))
    from token_data import write_shard
    rng = np.random.default_rng(0)
    d = Path(tempfile.mkdtemp(prefix="bopt-syn-"))
    for f in FILES[:2]:
        write_shard(d / f, rng.integers(0, 50257, n * 1024).astype(np.uint16))
    return Frozen(public={}, hidden={"tokens": rng.integers(0, 50257, 65537).astype(np.uint16)},
                  sources_sha256={"synthetic": f"seed0-n{n}"}, raw_files={f: d / f for f in FILES[:2]})
