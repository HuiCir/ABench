"""C-QNT-003: Qwen2.5-0.5B (Apache-2.0) frozen in bf16, WikiText-2 raw (CC BY-SA 3.0 / GFDL) tokenized with the
model's tokenizer. Public: train split (calibration) and validation split (self-check); held-out: test split.
Each split's documents are joined with "\\n\\n" before tokenization, as in the GPTQ perplexity protocol.
Raw layout: Qwen2.5-0.5B/ (snapshot at QWEN_REV), wikitext-2-raw-v1/{train,validation,test}-00000-of-00001.parquet.
Needs transformers (tokenizer) and pandas in the operator environment."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from prepare.common import Frozen, check

QWEN_REV = "060db6499f32faf8b98477b0a26969ef7d8b9987"
WIKI_REV = "b08601e04326c79dfdd32d625aee71d232d685c3"
QWEN_FILES = ["config.json", "generation_config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json",
              "vocab.json", "merges.txt", "LICENSE"]
SOURCES = [{"name": "Qwen2.5-0.5B/", "url": f"https://huggingface.co/Qwen/Qwen2.5-0.5B/tree/{QWEN_REV}", "files": QWEN_FILES,
            "license": "Apache-2.0"}] + [
           {"name": f"wikitext-2-raw-v1/{s}-00000-of-00001.parquet",
            "url": f"https://huggingface.co/datasets/Salesforce/wikitext/resolve/{WIKI_REV}/wikitext-2-raw-v1/{s}-00000-of-00001.parquet",
            "license": "CC BY-SA 3.0 and GFDL"} for s in ("train", "validation", "test")]


def build(raw: Path) -> Frozen:
    import pandas as pd
    from transformers import AutoTokenizer
    sums = {f"Qwen2.5-0.5B/{f}": check(raw / "Qwen2.5-0.5B" / f) for f in QWEN_FILES}
    tok = AutoTokenizer.from_pretrained(str(raw / "Qwen2.5-0.5B"))
    ids = {}
    for s in ("train", "validation", "test"):
        p = raw / "wikitext-2-raw-v1" / f"{s}-00000-of-00001.parquet"
        sums[p.name] = check(p)
        text = "\n\n".join(pd.read_parquet(p)["text"].tolist())
        ids[s] = np.asarray(tok(text)["input_ids"], np.int32)
    return _with_npy(ids, sums, raw)


def _with_npy(ids, sums, raw) -> Frozen:
    import tempfile
    d = Path(tempfile.mkdtemp(prefix="cqnt-"))
    np.save(d / "calib_tokens.npy", ids["train"])
    np.save(d / "dev_tokens.npy", ids["validation"])
    return Frozen(public={}, hidden={"tokens": ids["test"]}, sources_sha256=sums,
                  models={"Qwen2.5-0.5B": raw / "Qwen2.5-0.5B"},
                  raw_files={"calib_tokens.npy": d / "calib_tokens.npy", "dev_tokens.npy": d / "dev_tokens.npy"},
                  notes={"n_tokens": {k: int(len(v)) for k, v in ids.items()}})


def synthetic(n: int) -> Frozen:
    rng = np.random.default_rng(0)
    ids = {s: rng.integers(0, 151643, n * 64).astype(np.int32) for s in ("train", "validation", "test")}
    import tempfile
    d = Path(tempfile.mkdtemp(prefix="cqnt-syn-"))
    np.save(d / "calib_tokens.npy", ids["train"])
    np.save(d / "dev_tokens.npy", ids["validation"])
    return Frozen(public={}, hidden={"tokens": ids["test"]}, sources_sha256={"synthetic": f"n{n}"},
                  raw_files={"calib_tokens.npy": d / "calib_tokens.npy", "dev_tokens.npy": d / "dev_tokens.npy"})
