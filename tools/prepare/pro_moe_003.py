"""PRO-MOE-003: the PRO-SPD-004 data (FineWeb-10B GPT-2 shards, kjj0/fineweb10B-gpt2 @889765ea, MIT): all 103
training shards public; hidden the first 10,485,761 tokens of fineweb_val_000000.bin and the resume number drawn
with the secret seed ABENCH_PRO_MOE_SEED."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from prepare import pro_spd_004 as SPD
from prepare.common import Frozen

SECRET_SEED_ENV = "ABENCH_PRO_MOE_SEED"
SOURCES = SPD.SOURCES


def build(raw: Path) -> Frozen:
    seed = int(os.environ.get(SECRET_SEED_ENV, "0"))
    if not seed:
        raise SystemExit(f"set {SECRET_SEED_ENV} to a secret integer (resume point)")
    fz = SPD.build(raw)
    fz.hidden["resume_u"] = np.asarray([np.random.default_rng(seed).random()])
    return fz


def synthetic(n: int) -> Frozen:
    fz = SPD.synthetic(n)
    fz.hidden["resume_u"] = np.asarray([0.25])
    return fz
