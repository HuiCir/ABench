"""C-SPEC-001: prompts of five workloads, chat-formatted for Qwen2.5-1.5B-Instruct (target) and tokenised with its
tokenizer (Qwen2.5 chat template, transformers' apply_chat_template), and the target and draft models
(Qwen/Qwen2.5-1.5B-Instruct and Qwen/Qwen2.5-0.5B-Instruct, Apache-2.0).

  math       GSM8K questions (openai/gsm8k main, MIT)                       held out: test;      dev: train
  code       HumanEval prompts (openai/openai_humaneval, MIT)               held out: last 64;   dev: first 100
  summarize  CNN/DailyMail articles, truncated to 1,500 tokens (abisee/cnn_dailymail 3.0.0, Apache-2.0)
                                                                            held out: test;      dev: validation
  chat       Dolly instructions with their context (databricks/databricks-dolly-15k, CC BY-SA 3.0)
                                                                            held out / dev: disjoint random draws
  translate  CNN/DailyMail highlights to translate into German (expected low draft acceptance)
                                                                            held out: test;      dev: validation
32 held-out prompts per workload drawn with a secret seed; 16 dev prompts per workload (public).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from prepare.common import Frozen, check

REVS = {"gsm8k": "740312add88f781978c0658806c59bc2815b9866", "humaneval": "7dce6050a7d6d172f3cc5c32aa97f52fa1a2e544",
        "cnn": "96df5e686bee6baa90b8bee7c28b81fa3fa6223d", "dolly": "bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a"}
MODELS = {"Qwen2.5-1.5B-Instruct": "989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
          "Qwen2.5-0.5B-Instruct": "7ae557604adf67be50417f59c2c2f167def9a775"}
MODEL_FILES = ["config.json", "generation_config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json",
               "vocab.json", "merges.txt"]
FILES = {"gsm8k/main/test-00000-of-00001.parquet": ("openai/gsm8k", "gsm8k"),
         "gsm8k/main/train-00000-of-00001.parquet": ("openai/gsm8k", "gsm8k"),
         "humaneval/openai_humaneval/test-00000-of-00001.parquet": ("openai/openai_humaneval", "humaneval"),
         "cnn/3.0.0/test-00000-of-00001.parquet": ("abisee/cnn_dailymail", "cnn"),
         "cnn/3.0.0/validation-00000-of-00001.parquet": ("abisee/cnn_dailymail", "cnn"),
         "dolly/databricks-dolly-15k.jsonl": ("databricks/databricks-dolly-15k", "dolly")}
SOURCES = [{"name": n, "url": f"https://huggingface.co/datasets/{repo}/resolve/{REVS[k]}/{n.split('/', 1)[1]}",
            "license": {"gsm8k": "MIT", "humaneval": "MIT", "cnn": "Apache-2.0", "dolly": "CC BY-SA 3.0"}[k]}
           for n, (repo, k) in FILES.items()] + [
          {"name": f"{m}/", "url": f"https://huggingface.co/Qwen/{m}/tree/{r}", "files": MODEL_FILES, "license": "Apache-2.0"}
          for m, r in MODELS.items()]
SECRET_SEED_ENV = "ABENCH_C_SPEC_SEED"          # escrowed by tools/prepare/run.py next to the hidden-set key
WORKLOADS = ("math", "code", "summarize", "chat", "translate")
N_HELD, N_DEV, MAX_ARTICLE = 32, 16, 1500


def pack(seqs: list) -> tuple[np.ndarray, np.ndarray]:
    off = np.concatenate([[0], np.cumsum([len(s) for s in seqs])]).astype(np.int64)
    return np.concatenate([np.asarray(s, np.int32) for s in seqs]), off


def messages(raw: Path, tok) -> dict:
    """{workload: (held-out candidate texts, dev candidate texts)}"""
    import pandas as pd
    r = raw / "c_spec"
    g_te, g_tr = pd.read_parquet(r / "gsm8k/main/test-00000-of-00001.parquet"), pd.read_parquet(r / "gsm8k/main/train-00000-of-00001.parquet")
    he = pd.read_parquet(r / "humaneval/openai_humaneval/test-00000-of-00001.parquet")
    c_te, c_va = pd.read_parquet(r / "cnn/3.0.0/test-00000-of-00001.parquet"), pd.read_parquet(r / "cnn/3.0.0/validation-00000-of-00001.parquet")
    dolly = [json.loads(x) for x in (r / "dolly/databricks-dolly-15k.jsonl").read_text().splitlines() if x.strip()]

    def trunc(text):
        ids = tok(text, add_special_tokens=False)["input_ids"][:MAX_ARTICLE]
        return tok.decode(ids)
    math_ = lambda q: f"Solve the following problem step by step, then give the final answer.\n\n{q}"          # noqa: E731
    code = lambda p: f"Complete the following Python function. Reply with the full function only.\n\n{p}"      # noqa: E731
    summ = lambda a: f"Summarize the following article in three sentences.\n\n{trunc(a)}"                     # noqa: E731
    tran = lambda h: f"Translate the following text into German.\n\n{h}"                                     # noqa: E731
    chat = lambda d: d["instruction"] + (f"\n\nContext:\n{d['context']}" if d.get("context") else "")        # noqa: E731
    half = len(dolly) // 2
    return {"math": ([math_(q) for q in g_te["question"]], [math_(q) for q in g_tr["question"]]),
            "code": ([code(p) for p in he["prompt"][100:]], [code(p) for p in he["prompt"][:100]]),
            "summarize": ([summ(a) for a in c_te["article"][:2000]], [summ(a) for a in c_va["article"][:2000]]),
            "chat": ([chat(d) for d in dolly[:half]], [chat(d) for d in dolly[half:]]),
            "translate": ([tran(h) for h in c_te["highlights"][2000:4000]], [tran(h) for h in c_va["highlights"][2000:4000]])}


def build(raw: Path) -> Frozen:
    from transformers import AutoTokenizer                 # operator environment only
    seed = int(os.environ.get(SECRET_SEED_ENV, "0"))
    if not seed:
        raise SystemExit(f"set {SECRET_SEED_ENV} to a secret integer (choice of the held-out prompts)")
    sums = {n: check(raw / "c_spec" / n) for n in FILES}
    for m in MODELS:
        for f in MODEL_FILES:
            sums[f"{m}/{f}"] = check(raw / m / f)
    tok = AutoTokenizer.from_pretrained(str(raw / "Qwen2.5-1.5B-Instruct"), local_files_only=True)
    enc = lambda t: tok.apply_chat_template([{"role": "user", "content": t}], add_generation_prompt=True)   # noqa: E731
    rng, drng = np.random.default_rng(seed), np.random.default_rng(1)
    hidden, dev = {}, {}
    for w, (held, devc) in messages(raw, tok).items():
        hp = [enc(held[i]) for i in rng.choice(len(held), N_HELD, replace=False)]
        dp = [enc(devc[i]) for i in drng.choice(len(devc), N_DEV, replace=False)]
        hidden[f"{w}_tokens"], hidden[f"{w}_offsets"] = pack(hp)
        dev[f"{w}_tokens"], dev[f"{w}_offsets"] = pack(dp)
    return Frozen(public={"dev_prompts.npz": dev}, hidden=hidden, sources_sha256=sums,
                  models={m: raw / m for m in MODELS})


def synthetic(n: int) -> Frozen:
    rng = np.random.default_rng(0)
    hidden, dev = {}, {}
    for w in WORKLOADS:
        hidden[f"{w}_tokens"], hidden[f"{w}_offsets"] = pack([rng.integers(0, 1000, 20) for _ in range(4)])
        dev[f"{w}_tokens"], dev[f"{w}_offsets"] = pack([rng.integers(0, 1000, 20) for _ in range(4)])
    return Frozen(public={"dev_prompts.npz": dev}, hidden=hidden, sources_sha256={"synthetic": f"n{n}"})
