"""K-FUS-002: 20 KernelBench Level-2 problem files (MIT) and secret weight and input seeds — see prepare/kernelbench.py."""
from __future__ import annotations

from pathlib import Path

from prepare import kernelbench as KB
from prepare.common import Frozen

SECRET_SEED_ENV = "ABENCH_K_FUS_SEED"          # escrowed by tools/prepare/run.py next to the hidden-set key
SOURCES = KB.sources("level2")


def build(raw: Path) -> Frozen:
    return KB.build_level(raw, "level2", SECRET_SEED_ENV)


def synthetic(n: int) -> Frozen:
    return KB.synthetic_level("level2", n)
