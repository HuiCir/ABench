"""PRO-SPEC-006: target Qwen3-8B (@b968826d, Apache-2.0); draft training data UltraChat-200k (HuggingFaceH4
@8049631c, MIT; train_sft files verbatim, public); prompts of three kinds, as in Spec-Bench-style evaluations:
MT-Bench first turns (lm-sys/FastChat @b494d0c6 fastchat/llm_judge/data/mt_bench/question.jsonl, Apache-2.0),
GSM8K (openai/gsm8k main @740312ad, MIT) and HumanEval (openai/openai_humaneval @7dce6050, MIT).

Hidden (secret seed ABENCH_PRO_SPEC_SEED): 50 of the 80 MT-Bench questions, 80 GSM8K test problems and 80 of the
164 HumanEval prompts; public dev.npz: the other 30 MT-Bench questions, 30 GSM8K training problems and 30 other
HumanEval prompts. Every prompt is the user turn in Qwen3's chat template with thinking disabled, as token ids;
check_idx: 60 hidden prompts whose answers are checked; resume_u: the resumed checkpoint.
Needs pandas and transformers in the operator environment.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from prepare.common import Frozen, check

SECRET_SEED_ENV = "ABENCH_PRO_SPEC_SEED"
FC_REV, GSM_REV, HE_REV, UC_REV = ("b494d0c6b4e7935f1764f8439e75da3e66beccc7", "740312add88f781978c0658806c59bc2815b9866",
                                   "7dce6050a7d6d172f3cc5c32aa97f52fa1a2e544", "8049631c405ae6576f93f445c6b8166f76f5505a")
UC_FILES = ["train_sft-00000-of-00003-a3ecf92756993583.parquet", "train_sft-00001-of-00003-0a1804bcb6ae68c6.parquet",
            "train_sft-00002-of-00003-ee46ed25cfae92c6.parquet"]
SOURCES = ([{"name": "mt_bench_question.jsonl", "license": "Apache-2.0",
             "url": f"https://raw.githubusercontent.com/lm-sys/FastChat/{FC_REV}/fastchat/llm_judge/data/mt_bench/question.jsonl"},
            {"name": "gsm8k_test.parquet", "license": "MIT",
             "url": f"https://huggingface.co/datasets/openai/gsm8k/resolve/{GSM_REV}/main/test-00000-of-00001.parquet"},
            {"name": "gsm8k_train.parquet", "license": "MIT",
             "url": f"https://huggingface.co/datasets/openai/gsm8k/resolve/{GSM_REV}/main/train-00000-of-00001.parquet"},
            {"name": "humaneval.parquet", "license": "MIT",
             "url": f"https://huggingface.co/datasets/openai/openai_humaneval/resolve/{HE_REV}/openai_humaneval/test-00000-of-00001.parquet"},
            {"name": "Qwen3-8B/", "license": "Apache-2.0", "url": "https://huggingface.co/Qwen/Qwen3-8B (as PRO-OPD-002)"}]
           + [{"name": f"ultrachat_200k/{f}", "license": "MIT",
               "url": f"https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k/resolve/{UC_REV}/data/{f}"} for f in UC_FILES])
N_HID = {"mtbench": 50, "gsm8k": 80, "humaneval": 80}
N_DEV, N_CHECK = 30, 60


def pack(chats: list, encode) -> tuple:
    ids = [encode(c) for c in chats]
    return (np.concatenate([np.asarray(i, np.int32) for i in ids]),
            np.concatenate([[0], np.cumsum([len(i) for i in ids])]).astype(np.int64))


def _freeze(sets: dict, dev_gsm: list, encode, seed: int, sums: dict, models: dict, raw_files: dict) -> Frozen:
    rng = np.random.default_rng(seed)
    hid_p, hid_s, dev_p, dev_s = [], [], [], []
    for si, name in enumerate(("mtbench", "gsm8k", "humaneval")):
        items = sets[name]
        perm = rng.permutation(len(items))
        h = [items[i] for i in perm[:N_HID[name]]]
        d = [items[i] for i in perm[N_HID[name]:N_HID[name] + N_DEV]] if name != "gsm8k" else dev_gsm[:N_DEV]
        hid_p += h
        hid_s += [si] * len(h)
        dev_p += d
        dev_s += [si] * len(d)
    hi, ho = pack(hid_p, encode)
    di, do = pack(dev_p, encode)
    hidden = {"ids": hi, "offsets": ho, "set_id": np.asarray(hid_s, np.int64),
              "check_idx": np.sort(rng.choice(len(hid_p), N_CHECK, replace=False)), "resume_u": np.asarray([rng.random()])}
    dev = {"ids": di, "offsets": do, "set_id": np.asarray(dev_s, np.int64), "check_idx": np.arange(15),
           "resume_u": np.asarray([0.5])}
    return Frozen(public={"dev.npz": dev}, hidden=hidden, sources_sha256=sums, models=models, raw_files=raw_files)


def build(raw: Path) -> Frozen:
    import pandas as pd
    from transformers import AutoTokenizer
    seed = int(os.environ.get(SECRET_SEED_ENV, "0"))
    if not seed:
        raise SystemExit(f"set {SECRET_SEED_ENV} to a secret integer (prompt selection, checked answers)")
    names = ["mt_bench_question.jsonl", "gsm8k_test.parquet", "gsm8k_train.parquet", "humaneval.parquet"]
    sums = {n: check(raw / n) for n in names}
    sums.update({f"ultrachat_200k/{f}": check(raw / "ultrachat_200k" / f) for f in UC_FILES})
    tok = AutoTokenizer.from_pretrained(str(raw / "Qwen3-8B"))
    encode = lambda u: tok.apply_chat_template([{"role": "user", "content": u}], tokenize=True,     # noqa: E731
                                               add_generation_prompt=True, enable_thinking=False)
    mt = [json.loads(x)["turns"][0] for x in (raw / "mt_bench_question.jsonl").read_text().splitlines() if x.strip()]
    assert len(mt) == 80, len(mt)
    sets = {"mtbench": mt, "gsm8k": pd.read_parquet(raw / "gsm8k_test.parquet")["question"].tolist(),
            "humaneval": ["Complete the following Python function.\n\n" + p for p in pd.read_parquet(raw / "humaneval.parquet")["prompt"]]}
    dev_gsm = pd.read_parquet(raw / "gsm8k_train.parquet")["question"].tolist()[:N_DEV]
    return _freeze(sets, dev_gsm, encode, seed, sums, {"Qwen3-8B": raw / "Qwen3-8B"},
                   {f"ultrachat_200k/{f}": raw / "ultrachat_200k" / f for f in UC_FILES})


def synthetic(n: int) -> Frozen:
    sets = {"mtbench": [f"question {i}" for i in range(80)], "gsm8k": [f"problem {i}" for i in range(200)],
            "humaneval": [f"def f{i}():" for i in range(164)]}
    encode = lambda u: [ord(c) % 500 for c in u]                                                   # noqa: E731
    return _freeze(sets, [f"train problem {i}" for i in range(40)], encode, 12345, {"synthetic": f"n{n}"}, {}, {})
