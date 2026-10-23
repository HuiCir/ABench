"""PRO-SRV-005: Qwen3-8B (@b968826d, Apache-2.0); request arrivals and lengths from the Azure LLM inference traces
of 2023 (github.com/Azure/AzurePublicDataset @79092101, data/AzureLLMInferenceTrace_conv.csv and _code.csv, CC-BY
4.0: TIMESTAMP, ContextTokens, GeneratedTokens); prompt contents from FineWeb documents (kjj0/fineweb10B-gpt2,
decoded with the GPT-2 tokenizer and re-tokenised with Qwen3's), cut or extended to each request's context length.

Each trace gives windows of N_REQ consecutive requests; context lengths are clipped to [16, 8,192] and generated
lengths to [1, 1,024]. Public dev.npz: one window per trace at a public offset, prompts from training documents.
Hidden: one window per trace at offsets drawn with the secret seed ABENCH_PRO_SRV_SEED from the part of the trace
after the dev windows, prompts from validation documents; check_idx, 100 request indices checked for greedy
consistency. Needs pandas and transformers in the operator environment.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

from prepare.common import ROOT, Frozen, check

SECRET_SEED_ENV = "ABENCH_PRO_SRV_SEED"
AZ_REV, M_REV, TOK_REV, GPT2_REV = ("790921015d50dd6aae7f7e47f39ba0e235ad6b08", "b968826d9c46dd6066d109eabc6255188de91218",
                                     "889765ea1f903759787add96995d81171b632d0c", "607a30d783dfa663caf39e06633721c8d4cfcd7e")
TRACES = {"conv": "AzureLLMInferenceTrace_conv.csv", "code": "AzureLLMInferenceTrace_code.csv"}
SOURCES = ([{"name": f, "url": f"https://raw.githubusercontent.com/Azure/AzurePublicDataset/{AZ_REV}/data/{f}",
             "license": "CC-BY-4.0"} for f in TRACES.values()]
           + [{"name": "Qwen3-8B/", "url": f"https://huggingface.co/Qwen/Qwen3-8B/tree/{M_REV}", "license": "Apache-2.0",
               "note": "RAW/Qwen3-8B/ (as PRO-OPD-002)"}]
           + [{"name": f, "url": f"https://huggingface.co/datasets/kjj0/fineweb10B-gpt2/resolve/{TOK_REV}/{f}",
               "license": "MIT"} for f in ("fineweb_train_000001.bin", "fineweb_val_000000.bin")]
           + [{"name": "gpt2/", "url": f"https://huggingface.co/openai-community/gpt2/tree/{GPT2_REV}", "license": "MIT",
               "files": ["tokenizer.json", "vocab.json", "merges.txt"]}])
N_REQ, N_CHECK, DEV_OFFSET = 2000, 100, 0
EOT = 50256


def docs(tokens: np.ndarray, gpt2, n: int, rng) -> list:
    """n FineWeb documents as text (decoded GPT-2 tokens)."""
    eot = np.flatnonzero(tokens == EOT)
    pick = rng.choice(len(eot) - 1, size=n, replace=len(eot) - 1 < n)
    return [gpt2.decode(tokens[eot[i] + 1:eot[i + 1]].tolist()) for i in pick]


def window(df, start: int, texts: list, qtok, rng) -> dict:
    w = df.iloc[start:start + N_REQ]
    ts = (w["TIMESTAMP"] - w["TIMESTAMP"].iloc[0]).dt.total_seconds().to_numpy()
    ctx = np.clip(w["ContextTokens"].to_numpy(), 16, 8192)
    gen = np.clip(w["GeneratedTokens"].to_numpy(), 1, 1024)
    ids, offs, k = [], [0], 0
    for c in ctx:
        toks = []
        while len(toks) < c:
            toks += qtok(texts[k % len(texts)])["input_ids"]
            k += 1
        ids += toks[:int(c)]
        offs.append(len(ids))
    return {"ids": np.asarray(ids, np.int32), "offsets": np.asarray(offs, np.int64), "arrival": ts.astype(np.float64),
            "max_new": gen.astype(np.int64)}


def _freeze(dfs: dict, train_texts: list, val_texts: list, qtok, seed: int, sums: dict, models: dict) -> Frozen:
    rng = np.random.default_rng(seed)
    dev, hid = {}, {}
    for t, df in dfs.items():
        for k, v in window(df, DEV_OFFSET, train_texts, qtok, rng).items():
            dev[f"{t}_{k}"] = v
        start = int(rng.integers(N_REQ, len(df) - N_REQ))
        for k, v in window(df, start, val_texts, qtok, rng).items():
            hid[f"{t}_{k}"] = v
    dev["check_idx"] = np.sort(np.random.default_rng(0).choice(N_REQ, N_CHECK, replace=False))
    hid["check_idx"] = np.sort(rng.choice(N_REQ, N_CHECK, replace=False))
    return Frozen(public={"dev.npz": dev}, hidden=hid, sources_sha256=sums, models=models)


def build(raw: Path) -> Frozen:
    import pandas as pd
    from transformers import AutoTokenizer
    seed = int(os.environ.get(SECRET_SEED_ENV, "0"))
    if not seed:
        raise SystemExit(f"set {SECRET_SEED_ENV} to a secret integer (trace windows, checked requests)")
    sys.path.insert(0, str(ROOT / "cases" / "PRO-SPD-004" / "harness"))
    from token_data import read_shard
    sums = {f: check(raw / f) for f in TRACES.values()}
    dfs = {t: pd.read_csv(raw / f, parse_dates=["TIMESTAMP"]) for t, f in TRACES.items()}
    gpt2 = AutoTokenizer.from_pretrained(str(raw / "gpt2"))
    qtok = AutoTokenizer.from_pretrained(str(raw / "Qwen3-8B"))
    rng = np.random.default_rng(seed + 1)
    tr = docs(np.asarray(read_shard(raw / "fineweb_train_000001.bin")), gpt2, 3000, np.random.default_rng(1))
    va = docs(np.asarray(read_shard(raw / "fineweb_val_000000.bin")), gpt2, 3000, rng)
    return _freeze(dfs, tr, va, qtok, seed, sums, {"Qwen3-8B": raw / "Qwen3-8B"})


def synthetic(n: int) -> Frozen:
    import pandas as pd
    rng = np.random.default_rng(0)
    m = max(n, 3 * N_REQ)
    df = pd.DataFrame({"TIMESTAMP": pd.to_datetime("2023-11-11") + pd.to_timedelta(np.cumsum(rng.exponential(0.2, m)), unit="s"),
                       "ContextTokens": rng.integers(16, 2000, m), "GeneratedTokens": rng.integers(1, 400, m)})
    qtok = lambda s: {"input_ids": [ord(c) % 1000 for c in s]}                                  # noqa: E731
    texts = ["synthetic document number %d " % i * 20 for i in range(50)]
    return _freeze({"conv": df, "code": df}, texts, texts, qtok, 12345, {"synthetic": f"n{n}"}, {})
