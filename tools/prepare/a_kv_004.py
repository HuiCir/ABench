"""A-KV-004: SmolLM2-135M (Apache-2.0) and FineWeb-Edu tokens (ODC-BY) — see prepare/smol_data.py."""
from __future__ import annotations

import tempfile
from pathlib import Path

from prepare import smol_data
from prepare.common import Frozen

SOURCES = smol_data.SOURCES


def build(raw: Path) -> Frozen:
    files, hidden, sums = smol_data.build_text(raw, raw / "a_kv_004_tokens")
    return Frozen(public={}, hidden=hidden, sources_sha256=sums, raw_files=files,
                  models={"SmolLM2-135M": raw / "SmolLM2-135M"})


def synthetic(n: int) -> Frozen:
    files, hidden = smol_data.synthetic_text(n, Path(tempfile.mkdtemp(prefix="akv-syn-")))
    return Frozen(public={}, hidden=hidden, sources_sha256={"synthetic": f"n{n}"}, raw_files=files)
