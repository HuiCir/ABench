"""X-SFT-005: supervised fine-tuning data smol-smoltalk (HuggingFaceTB/smol-smoltalk @f73fe857, Apache-2.0), the base
model SmolLM2-360M (@f8027fd0, Apache-2.0) and FineWeb-Edu text for forgetting (see prepare/smol_data.py).

Public: the first 100,000 conversations of the train split (JSON lists of {role, content}), 500 single-turn dev
requests taken after them, and 1,048,577 FineWeb-Edu dev tokens (from the public training shard of smol_data).
Hidden: 600 single-turn requests from the test split (first user turn of 20-600 characters), each followed by one to
three verifiable constraints (cases/X-SFT-005/harness/ifcheck.py) drawn with ABENCH_X_SFT_SEED, and 1,048,577 held-out
FineWeb-Edu tokens.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

from prepare import smol_data
from prepare.common import ROOT, Frozen, check, text_pack

REV = "f73fe857d519ff6ac5af2ea67c4d3834da7b8bcc"
BASE_REV = "f8027fd0eaeea54caa13c31d31b9fdc459c38b49"
SECRET_SEED_ENV = "ABENCH_X_SFT_SEED"
TRAIN = [f"data/train-0000{i}-of-00004.parquet" for i in range(4)]
TEST = "data/test-00000-of-00001.parquet"
BASE_FILES = ["config.json", "generation_config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json",
              "special_tokens_map.json", "vocab.json", "merges.txt"]
N_TRAIN, N_DEV, N_TEST, N_TOK = 100_000, 500, 600, 1_048_577
SOURCES = ([{"name": f"smol-smoltalk/{f}", "url": f"https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk/resolve/{REV}/{f}",
             "license": "Apache-2.0"} for f in TRAIN + [TEST]]
           + [{"name": f"SmolLM2-360M/{f}", "url": f"https://huggingface.co/HuggingFaceTB/SmolLM2-360M/resolve/{BASE_REV}/{f}",
               "license": "Apache-2.0"} for f in BASE_FILES]
           + [s for s in smol_data.SOURCES if "fineweb" in s["name"]])


def _ifcheck():
    h = ROOT / "cases" / "X-SFT-005" / "harness"
    if str(h) not in sys.path:
        sys.path.insert(0, str(h))
    import ifcheck
    return ifcheck


def _single_turn(conv) -> str | None:
    users = [m for m in conv if m["role"] == "user"]
    if len(users) != 1:
        return None
    t = users[0]["content"].strip()
    return t if 20 <= len(t) <= 600 else None


def _freeze(train_convs, dev_reqs, test_reqs, dev_tok, held_tok, seed) -> Frozen:
    ic = _ifcheck()
    rng = np.random.default_rng(seed)
    items = [ic.make(r, rng) for r in test_reqs]
    cu, co = text_pack([json.dumps(c, ensure_ascii=False) for c in train_convs])
    du, do = text_pack(dev_reqs)
    pu, po = text_pack([it["prompt"] for it in items])
    cons = np.asarray([json.dumps({"kinds": it["kinds"], "params": it["params"]}) for it in items], dtype=np.str_)
    return Frozen(public={"sft_public.npz": {"conv_utf8": cu, "conv_offsets": co, "dev_request_utf8": du,
                                            "dev_request_offsets": do, "dev_tokens": dev_tok}},
                  hidden={"prompt_utf8": pu, "prompt_offsets": po, "tokens": held_tok, "constraints": cons})


def build(raw: Path) -> Frozen:
    import pyarrow.parquet as pq
    seed = int(os.environ.get(SECRET_SEED_ENV, "0"))
    if not seed:
        raise SystemExit(f"set {SECRET_SEED_ENV} to a secret integer (constraints of the held-out prompts)")
    d = raw / "smol-smoltalk"
    sums = {f: check(d / f) for f in TRAIN + [TEST]}
    convs = []
    for f in TRAIN:
        convs += pq.read_table(d / f, columns=["messages"]).column("messages").to_pylist()
        if len(convs) > N_TRAIN + 20 * N_DEV:
            break
    train = convs[:N_TRAIN]
    dev = [r for r in (_single_turn(c) for c in convs[N_TRAIN:]) if r][:N_DEV]
    test_convs = pq.read_table(d / TEST, columns=["messages"]).column("messages").to_pylist()
    test = [r for r in (_single_turn(c) for c in test_convs) if r]
    test = [test[i] for i in sorted(np.random.default_rng(seed + 1).choice(len(test), min(N_TEST, len(test)), replace=False))]
    m = raw / "SmolLM2-360M"
    for f in BASE_FILES:
        sums[f"SmolLM2-360M/{f}"] = check(m / f)
    fw = raw / "fineweb-edu"
    dev_tok = smol_data.tokenize_parquet(fw / "000_00000.parquet", m / "tokenizer.json", N_TOK)
    held_tok = smol_data.tokenize_parquet(fw / smol_data.FWE_HELD, m / "tokenizer.json", N_TOK)
    fz = _freeze(train, dev, test, dev_tok, held_tok, seed)
    fz.sources_sha256 = sums
    fz.models = {"SmolLM2-360M": m}
    return fz


def synthetic(n: int) -> Frozen:
    rng = np.random.default_rng(n)
    req = lambda i: f"Write a short note number {i} about the weather."
    convs = [[{"role": "user", "content": req(i)}, {"role": "assistant", "content": "It is sunny."}] for i in range(max(20, n))]
    tok = rng.integers(0, 49152, 8 * 1025 + 1).astype(np.uint16)
    fz = _freeze(convs, [req(i) for i in range(20)], [req(i) for i in range(12)], tok, tok, 1)
    fz.sources_sha256 = {"synthetic": f"n{n}"}
    return fz
