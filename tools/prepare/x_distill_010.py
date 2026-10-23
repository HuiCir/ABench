"""X-DISTILL-010: FineWeb-Edu text and the student SmolLM2-135M from prepare/smol_data.py, plus the teacher
SmolLM2-1.7B (HuggingFaceTB/SmolLM2-1.7B @effd688a, Apache-2.0; same tokenizer). Public: the 120M-token training shard;
hidden: the 4,194,305 held-out tokens."""
from __future__ import annotations

import tempfile
from pathlib import Path

from prepare import smol_data
from prepare.common import Frozen, check

TEACHER_REV = "effd688a12921b4cc83e3312b6feb579f70f9c71"
TEACHER_FILES = ["config.json", "generation_config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json",
                 "special_tokens_map.json"]
SOURCES = smol_data.SOURCES + [{"name": f"SmolLM2-1.7B/{f}", "url": f"https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B/resolve/{TEACHER_REV}/{f}",
                                "license": "Apache-2.0"} for f in TEACHER_FILES]


def build(raw: Path) -> Frozen:
    files, hidden, sums = smol_data.build_text(raw, raw / "x_distill_010_tokens")
    for f in TEACHER_FILES:
        sums[f"SmolLM2-1.7B/{f}"] = check(raw / "SmolLM2-1.7B" / f)
    return Frozen(public={}, hidden=hidden, sources_sha256=sums, raw_files=files,
                  models={"SmolLM2-135M": raw / "SmolLM2-135M", "SmolLM2-1.7B": raw / "SmolLM2-1.7B"})


def synthetic(n: int) -> Frozen:
    files, hidden = smol_data.synthetic_text(n, Path(tempfile.mkdtemp(prefix="xdist-syn-")))
    return Frozen(public={}, hidden=hidden, sources_sha256={"synthetic": f"n{n}"}, raw_files=files)
