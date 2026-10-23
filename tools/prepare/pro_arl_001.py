"""PRO-ARL-001: training prompts from DAPO-Math-17k (BytedTsinghua-SIA @65877096, Apache-2.0), starting weights
Qwen3-4B-Base (@906bfd4b, Apache-2.0), held-out problems AIME 2025 and AIME 2026 as released by MathArena
(MathArena/aime_2025 @c94da77e, MathArena/aime_2026 @d2de22f3; CC-BY-NC-SA 4.0; 30 problems each, integer answers).

Public: dapo_math_17k.parquet verbatim; train.jsonl with {problem, answer} after removing DAPO's instruction
wrapper and duplicates; dev.json with 60 of them set aside (public seed). Hidden: the 60 AIME problems and answers,
the sampling seed and the resume number (secret seed ABENCH_PRO_ARL_SEED).
Needs pandas and pyarrow in the operator environment.
"""
from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path

import numpy as np

from prepare.common import Frozen, check

SECRET_SEED_ENV = "ABENCH_PRO_ARL_SEED"
DAPO_REV, M_REV = "65877096c24ffa7abc4e4fa5edb95cf3413a5674", "906bfd4b4dc7f14ee4320094d8b41684abff8539"
AIME = {"aime_2025": "c94da77eb22bbd6439e62a323bec18493a421302", "aime_2026": "d2de22f3c656b4f56cf8981212186377d1e23bc3"}
MODEL = "Qwen3-4B-Base"
MODEL_FILES = ["config.json", "generation_config.json", "merges.txt", "model-00001-of-00003.safetensors",
               "model-00002-of-00003.safetensors", "model-00003-of-00003.safetensors", "model.safetensors.index.json",
               "tokenizer.json", "tokenizer_config.json", "vocab.json", "LICENSE"]
SOURCES = [{"name": "dapo-math-17k.parquet", "license": "Apache-2.0", "note": "RAW/dapo-math-17k.parquet",
            "url": f"https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k/resolve/{DAPO_REV}/data/dapo-math-17k.parquet"},
           {"name": f"{MODEL}/", "url": f"https://huggingface.co/Qwen/{MODEL}/tree/{M_REV}", "files": MODEL_FILES,
            "license": "Apache-2.0", "note": f"RAW/{MODEL}/"}] + [
          {"name": f"{k}.parquet", "license": "CC-BY-NC-SA-4.0", "note": f"RAW/{k}.parquet",
           "url": f"https://huggingface.co/datasets/MathArena/{k}/resolve/{rev}/data/train-00000-of-00001.parquet"}
          for k, rev in AIME.items()]
N_DEV, DEV_SEED = 60, 20250318
WRAP_HEAD = re.compile(r"^.*?(?:step by step\.).*?\n\n", re.S)
WRAP_TAIL = re.compile(r"\n\nRemember to put your answer.*$", re.S)


def unwrap(prompt: str) -> str:
    """The problem statement inside DAPO's instruction wrapper (the wrapper asks for an "Answer:" line)."""
    s = WRAP_TAIL.sub("", prompt)
    s2 = WRAP_HEAD.sub("", s, count=1)
    return (s2 if s2 != s else s).strip()


def dapo_items(df) -> list:
    seen, items = set(), []
    for _, r in df.iterrows():
        content = r["prompt"][0]["content"] if hasattr(r["prompt"], "__len__") else str(r["prompt"])
        prob = unwrap(str(content))
        ans = str(r["reward_model"]["ground_truth"]).strip()
        if prob in seen or not re.fullmatch(r"-?\d+", ans):
            continue
        seen.add(prob)
        items.append({"problem": prob, "answer": int(ans)})
    return items


def _freeze(items: list, aime: list, seed: int, sums: dict, raw_files: dict, models: dict) -> Frozen:
    rng = np.random.default_rng(DEV_SEED)
    perm = rng.permutation(len(items))
    dev = [items[i] for i in perm[:N_DEV]]
    train = [items[i] for i in perm[N_DEV:]]
    tmp = Path(tempfile.mkdtemp(prefix="proarl-"))
    (tmp / "train.jsonl").write_text("".join(json.dumps(x) + "\n" for x in train))
    (tmp / "dev.json").write_text(json.dumps(dev))
    s = np.random.default_rng(seed)
    hidden = {"problem": np.asarray([a["problem"] for a in aime], dtype=np.str_),
              "answer": np.asarray([a["answer"] for a in aime], np.int64),
              "set": np.asarray([a["set"] for a in aime], dtype=np.str_),
              "resume_u": np.asarray([s.random()]), "sample_seed": np.asarray([int(s.integers(1, 2**31 - 1))], np.int64)}
    return Frozen(public={}, hidden=hidden, sources_sha256=sums, models=models,
                  raw_files={**raw_files, "train.jsonl": tmp / "train.jsonl", "dev.json": tmp / "dev.json"},
                  notes={"n_train": len(train), "n_dev": len(dev)})


def build(raw: Path) -> Frozen:
    import pandas as pd
    seed = int(os.environ.get(SECRET_SEED_ENV, "0"))
    if not seed:
        raise SystemExit(f"set {SECRET_SEED_ENV} to a secret integer (sampling seed, resume point)")
    sums = {"dapo-math-17k.parquet": check(raw / "dapo-math-17k.parquet")}
    sums.update({f"{MODEL}/{f}": check(raw / MODEL / f) for f in MODEL_FILES})
    items = dapo_items(pd.read_parquet(raw / "dapo-math-17k.parquet"))
    if not 17000 <= len(items) <= 18500:
        raise SystemExit(f"{len(items)} unique DAPO problems; expected about 17,900")
    aime = []
    for k in AIME:
        sums[f"{k}.parquet"] = check(raw / f"{k}.parquet")
        df = pd.read_parquet(raw / f"{k}.parquet")
        assert len(df) == 30, (k, len(df))
        aime += [{"problem": str(r["problem"]), "answer": int(r["answer"]), "set": k} for _, r in df.iterrows()]
    return _freeze(items, aime, seed, sums, {"dapo_math_17k.parquet": raw / "dapo-math-17k.parquet"}, {MODEL: raw / MODEL})


def synthetic(n: int) -> Frozen:
    rng = np.random.default_rng(0)
    items = [{"problem": f"What is {a} + {b}?", "answer": int(a + b)} for a, b in rng.integers(0, 500, (max(n, 80), 2))]
    aime = [{"problem": f"Compute {a} * {b}.", "answer": int(a * b), "set": s}
            for s in AIME for a, b in rng.integers(1, 30, (30, 2))]
    return _freeze(items, aime, 12345, {"synthetic": f"n{n}"}, {}, {})
