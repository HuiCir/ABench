"""PRO-SPD-004: FineWeb-10B GPT-2 token shards (kjj0/fineweb10B-gpt2, MIT), the data of the modded-nanogpt speedrun.
Public: all 103 training shards (about 10.3B tokens) verbatim. Held-out: the first 10,485,761 tokens of
fineweb_val_000000.bin (10,485,760 predicted tokens, the speedrun's validation set)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from prepare.common import Frozen, check

REV = "889765ea1f903759787add96995d81171b632d0c"
TRAIN = [f"fineweb_train_{i:06d}.bin" for i in range(1, 104)]
VAL = "fineweb_val_000000.bin"
SOURCES = [{"name": f, "url": f"https://huggingface.co/datasets/kjj0/fineweb10B-gpt2/resolve/{REV}/{f}", "license": "MIT"}
           for f in TRAIN + [VAL]]
N_HELD = 10_485_760 + 1
HARNESS = Path(__file__).resolve().parents[2] / "cases" / "PRO-SPD-004" / "harness"


def build(raw: Path) -> Frozen:
    sys.path.insert(0, str(HARNESS))
    from token_data import read_shard
    sums = {f: check(raw / f) for f in TRAIN + [VAL]}
    val = np.asarray(read_shard(raw / VAL)[:N_HELD], np.uint16)
    return Frozen(public={}, hidden={"tokens": val}, sources_sha256=sums, raw_files={f: raw / f for f in TRAIN})


def synthetic(n: int) -> Frozen:
    import tempfile
    sys.path.insert(0, str(HARNESS))
    from token_data import write_shard
    rng = np.random.default_rng(0)
    d = Path(tempfile.mkdtemp(prefix="pro-syn-"))
    for f in TRAIN[:2]:
        write_shard(d / f, rng.integers(0, 50257, n * 1024).astype(np.uint16))
    return Frozen(public={}, hidden={"tokens": rng.integers(0, 50257, 65537).astype(np.uint16)},
                  sources_sha256={"synthetic": f"n{n}"}, raw_files={f: d / f for f in TRAIN[:2]})
