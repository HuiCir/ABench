"""PRO-OPD-002: the PRO-ARL-001 prompts and held-out problems (DAPO-Math-17k; AIME 2025 and 2026 from MathArena),
the student Qwen3-1.7B-Base (@ea980cb0, Apache-2.0) and the teacher Qwen3-8B (@b968826d, Apache-2.0).
Secret seed ABENCH_PRO_OPD_SEED (sampling seed, resume point)."""
from __future__ import annotations

import os
from pathlib import Path

from prepare import pro_arl_001 as ARL
from prepare.common import Frozen, check

SECRET_SEED_ENV = "ABENCH_PRO_OPD_SEED"
STUDENT, S_REV = "Qwen3-1.7B-Base", "ea980cb0a6c2ae4b936e82123acc929f1cec04c1"
TEACHER, T_REV = "Qwen3-8B", "b968826d9c46dd6066d109eabc6255188de91218"
S_FILES = ["config.json", "generation_config.json", "merges.txt", "model.safetensors", "tokenizer.json",
           "tokenizer_config.json", "vocab.json", "LICENSE"]
T_FILES = ["config.json", "generation_config.json", "merges.txt", "model.safetensors.index.json", "tokenizer.json",
           "tokenizer_config.json", "vocab.json", "LICENSE"] + [f"model-0000{i}-of-00005.safetensors" for i in range(1, 6)]
SOURCES = [s for s in ARL.SOURCES if not s["name"].startswith(ARL.MODEL)] + [
    {"name": f"{STUDENT}/", "url": f"https://huggingface.co/Qwen/{STUDENT}/tree/{S_REV}", "files": S_FILES,
     "license": "Apache-2.0", "note": f"RAW/{STUDENT}/"},
    {"name": f"{TEACHER}/", "url": f"https://huggingface.co/Qwen/{TEACHER}/tree/{T_REV}", "files": T_FILES,
     "license": "Apache-2.0", "note": f"RAW/{TEACHER}/"}]


def build(raw: Path) -> Frozen:
    import pandas as pd
    seed = int(os.environ.get(SECRET_SEED_ENV, "0"))
    if not seed:
        raise SystemExit(f"set {SECRET_SEED_ENV} to a secret integer (sampling seed, resume point)")
    sums = {"dapo-math-17k.parquet": check(raw / "dapo-math-17k.parquet")}
    sums.update({f"{STUDENT}/{f}": check(raw / STUDENT / f) for f in S_FILES})
    sums.update({f"{TEACHER}/{f}": check(raw / TEACHER / f) for f in T_FILES})
    items = ARL.dapo_items(pd.read_parquet(raw / "dapo-math-17k.parquet"))
    aime = []
    for k in ARL.AIME:
        sums[f"{k}.parquet"] = check(raw / f"{k}.parquet")
        df = pd.read_parquet(raw / f"{k}.parquet")
        assert len(df) == 30, (k, len(df))
        aime += [{"problem": str(r["problem"]), "answer": int(r["answer"]), "set": k} for _, r in df.iterrows()]
    return ARL._freeze(items, aime, seed, sums, {"dapo_math_17k.parquet": raw / "dapo-math-17k.parquet"},
                       {STUDENT: raw / STUDENT, TEACHER: raw / TEACHER})


def synthetic(n: int) -> Frozen:
    return ARL.synthetic(n)
