"""X-DPO-006: UltraFeedback preference pairs, binarised (HuggingFaceH4/ultrafeedback_binarized @3949bf5f, MIT), the
starting model SmolLM2-360M-Instruct (@a10cc151, Apache-2.0) and the evaluator-only reward model
Skywork-Reward-V2-Qwen3-0.6B (@8c14a4e9, Apache-2.0; never in a sandbox).

Public: the train_prefs pairs (prompt, chosen and rejected assistant answers) and 500 pairs of test_prefs as the dev
split. Hidden: 1,000 other test_prefs pairs, chosen with ABENCH_X_DPO_SEED.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from prepare.common import Frozen, check, text_pack

REV = "3949bf5f8c17c394422ccfab0c31ea9c20bdeb85"
BASE_REV = "a10cc1512eabd3dde888204e902eca88bddb4951"
RM_REV = "8c14a4e9e6321deaf572544339b16b8d6bbe8886"
SECRET_SEED_ENV = "ABENCH_X_DPO_SEED"
FILES = ["data/train_prefs-00000-of-00001.parquet", "data/test_prefs-00000-of-00001.parquet"]
BASE_FILES = ["config.json", "generation_config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json",
              "special_tokens_map.json", "vocab.json", "merges.txt"]
RM_FILES = ["config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
            "vocab.json", "merges.txt", "added_tokens.json", "chat_template.jinja"]
N_DEV, N_HELD = 500, 1000
SOURCES = ([{"name": f"ultrafeedback_binarized/{f}", "url": f"https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized/resolve/{REV}/{f}",
             "license": "MIT"} for f in FILES]
           + [{"name": f"SmolLM2-360M-Instruct/{f}", "url": f"https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct/resolve/{BASE_REV}/{f}",
               "license": "Apache-2.0"} for f in BASE_FILES]
           + [{"name": f"Skywork-Reward-V2-Qwen3-0.6B/{f}", "url": f"https://huggingface.co/Skywork/Skywork-Reward-V2-Qwen3-0.6B/resolve/{RM_REV}/{f}",
               "license": "Apache-2.0", "note": "evaluator only"} for f in RM_FILES])


def _answer(msgs) -> str:
    return next(m["content"] for m in reversed(msgs) if m["role"] == "assistant")


def _pack(rows, prefix=""):
    out = {}
    for k in ("prompt", "chosen", "rejected"):
        out[f"{prefix}{k}_utf8"], out[f"{prefix}{k}_offsets"] = text_pack([r[k] for r in rows])
    return out


def _freeze(train, test, seed) -> Frozen:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(test))
    dev = [test[i] for i in sorted(perm[:N_DEV])]
    held = [test[i] for i in sorted(perm[N_DEV:N_DEV + N_HELD])]
    return Frozen(public={"prefs_public.npz": {**_pack(train, "train_"), **_pack(dev, "dev_")}}, hidden=_pack(held))


def build(raw: Path) -> Frozen:
    import pyarrow.parquet as pq
    seed = int(os.environ.get(SECRET_SEED_ENV, "0"))
    if not seed:
        raise SystemExit(f"set {SECRET_SEED_ENV} to a secret integer (dev/held-out split of test_prefs)")
    d = raw / "ultrafeedback_binarized"
    sums = {f: check(d / f) for f in FILES}
    rows = lambda f: [{"prompt": r["prompt"], "chosen": _answer(r["chosen"]), "rejected": _answer(r["rejected"])}
                      for r in pq.read_table(d / f, columns=["prompt", "chosen", "rejected"]).to_pylist()]
    fz = _freeze(rows(FILES[0]), rows(FILES[1]), seed)
    for f in BASE_FILES:
        sums[f"SmolLM2-360M-Instruct/{f}"] = check(raw / "SmolLM2-360M-Instruct" / f)
    for f in RM_FILES:
        sums[f"{'Skywork-Reward-V2-Qwen3-0.6B'}/{f}"] = check(raw / "Skywork-Reward-V2-Qwen3-0.6B" / f)
    fz.sources_sha256 = sums
    fz.models = {"SmolLM2-360M-Instruct": raw / "SmolLM2-360M-Instruct"}
    fz.judge_models = {"Skywork-Reward-V2-Qwen3-0.6B": raw / "Skywork-Reward-V2-Qwen3-0.6B"}
    return fz


def synthetic(n: int) -> Frozen:
    row = lambda i: {"prompt": f"Name a colour ({i}).", "chosen": "Blue.", "rejected": "I cannot answer that."}
    fz = _freeze([row(i) for i in range(max(20, n))], [row(i) for i in range(N_DEV + 12)], 1)
    fz.sources_sha256 = {"synthetic": f"n{n}"}
    return fz
