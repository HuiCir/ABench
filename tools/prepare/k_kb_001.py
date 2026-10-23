"""K-KB-001: 30 KernelBench Level-1 problem files (MIT) and secret weight and input seeds — see prepare/kernelbench.py."""
from __future__ import annotations

from pathlib import Path

from prepare import kernelbench as KB
from prepare.common import Frozen

SECRET_SEED_ENV = "ABENCH_K_KB_SEED"          # escrowed by tools/prepare/run.py next to the hidden-set key
SOURCES = KB.sources("level1")


def build(raw: Path) -> Frozen:
    return KB.build_level(raw, "level1", SECRET_SEED_ENV)


def synthetic(n: int) -> Frozen:
    return KB.synthetic_level("level1", n)
