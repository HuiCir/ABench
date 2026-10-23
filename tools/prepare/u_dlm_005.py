"""U-DLM-005: the frozen MDLM (kuleshov-group/mdlm-owt, Apache-2.0), GPT-2 small as the sandbox's self-check scorer
(MIT), GPT-2 Large as the evaluator-only scorer (MIT), and text windows from FineWeb-10B GPT-2 tokens
(kjj0/fineweb10B-gpt2, MIT; text ODC-BY), which share MDLM's GPT-2 tokenizer.

A window is 1,024 tokens of one document starting at the document's first token (the token before it is
<|endoftext|> and the window contains none). Public: 64 dev windows from train shard 1 (public seed). Hidden: 256
windows from the validation shard drawn with the secret seed ABENCH_U_DLM5_SEED; their first 256 tokens are the
prefixes given to samplers, the windows the references for entropy and perplexity.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

from prepare.common import Frozen, check

TOK_REV = "889765ea1f903759787add96995d81171b632d0c"
MDLM_REV, GPT2_REV, GPT2L_REV = ("d0958fa851335ece6c15260ce0025f030673c0fb", "607a30d783dfa663caf39e06633721c8d4cfcd7e",
                                 "32b71b12589c2f8d625668d2335a01cac3249519")
SECRET_SEED_ENV = "ABENCH_U_DLM5_SEED"
SHARDS = ["fineweb_train_000001.bin", "fineweb_val_000000.bin"]
SOURCES = [{"name": f, "url": f"https://huggingface.co/datasets/kjj0/fineweb10B-gpt2/resolve/{TOK_REV}/{f}", "license": "MIT"}
           for f in SHARDS] + [
          {"name": "mdlm-owt/", "url": f"https://huggingface.co/kuleshov-group/mdlm-owt/tree/{MDLM_REV}",
           "files": ["config.json", "configuration_mdlm.py", "modeling_mdlm.py", "model.safetensors"], "license": "Apache-2.0"},
          {"name": "gpt2/", "url": f"https://huggingface.co/openai-community/gpt2/tree/{GPT2_REV}",
           "files": ["config.json", "model.safetensors", "tokenizer.json", "vocab.json", "merges.txt"], "license": "MIT"},
          {"name": "gpt2-large/", "url": f"https://huggingface.co/openai-community/gpt2-large/tree/{GPT2L_REV}",
           "files": ["config.json", "model.safetensors", "tokenizer.json", "vocab.json", "merges.txt"], "license": "MIT"}]
EOT, WIN, N_DEV, N_HELD, DEV_SEED = 50256, 1024, 64, 256, 20240612


def windows(tokens: np.ndarray, n: int, rng) -> np.ndarray:
    """n windows of WIN tokens, each the start of a document of at least WIN tokens."""
    t = np.asarray(tokens)
    eot = np.flatnonzero(t == EOT)
    starts = eot[:-1][np.diff(eot) > WIN] + 1
    if len(starts) < n:
        raise SystemExit(f"only {len(starts)} documents of at least {WIN} tokens")
    pick = np.sort(rng.choice(starts, size=n, replace=False))
    return np.stack([t[s:s + WIN] for s in pick]).astype(np.int32)


def _freeze(train: np.ndarray, val: np.ndarray, seed: int, sums: dict) -> Frozen:
    dev = windows(train, N_DEV, np.random.default_rng(DEV_SEED))
    held = windows(val, N_HELD, np.random.default_rng(seed))
    return Frozen(public={"dev_text.npz": {"windows": dev}}, sources_sha256=sums,
                  hidden={"prefixes": held[:, :256], "references": held})


def build(raw: Path) -> Frozen:
    seed = int(os.environ.get(SECRET_SEED_ENV, "0"))
    if not seed:
        raise SystemExit(f"set {SECRET_SEED_ENV} to a secret integer (held-out windows)")
    from prepare.common import ROOT
    sys.path.insert(0, str(ROOT / "cases" / "B-OPT-001" / "harness"))
    from token_data import read_shard
    sums = {f: check(raw / f) for f in SHARDS}
    fz = _freeze(np.asarray(read_shard(raw / SHARDS[0])), np.asarray(read_shard(raw / SHARDS[1])), seed, sums)
    fz.models = {"mdlm-owt": raw / "mdlm-owt", "gpt2": raw / "gpt2"}
    fz.judge_models = {"gpt2-large": raw / "gpt2-large"}
    return fz


def synthetic(n: int) -> Frozen:
    """Documents of random tokens separated by <|endoftext|> (n documents per split)."""
    rng = np.random.default_rng(0)

    def stream(k):
        docs = [np.r_[rng.integers(0, EOT, int(rng.integers(WIN + 1, WIN + 400))), EOT] for _ in range(k)]
        return np.concatenate([[EOT]] + docs)
    return _freeze(stream(max(n, N_DEV + 8)), stream(max(n, N_HELD + 8)), 12345, {"synthetic": f"n{n}"})
