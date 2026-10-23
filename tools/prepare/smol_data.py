"""Shared data of the SmolLM2 surgery cases (A-KV-004, U-TTT-008).

- Model: HuggingFaceTB/SmolLM2-135M at revision 93efa2f0 (Apache-2.0): config, weights and tokenizer files, placed
  under sandbox/models/<ID>/SmolLM2-135M.
- Text: FineWeb-Edu sample/10BT (HuggingFaceFW/fineweb-edu, revision 87f09149, ODC-BY), tokenised with the SmolLM2
  tokenizer, documents separated by <|endoftext|> (id 0): the first TRAIN_TOKENS tokens of 000_00000.parquet are the
  public training shard (llm.c shard format, uint16), the first HELD_TOKENS tokens of 013_00000.parquet the held-out set.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from prepare.common import check

SMOL_REV = "93efa2f097d58c2a74874c7e644dbc9b0cee75a2"
SMOL_FILES = ["config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]
FWE_REV = "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9"
FWE = {"000_00000.parquet": "b1ba7b2ce4cb5ea6ef42dca40263eabb85f37700d01693a68e9b30a31d78e871"}
FWE_HELD = "013_00000.parquet"                       # sha256 recorded at freeze
TRAIN_TOKENS, HELD_TOKENS = 120_000_000, 4_194_305
SOURCES = ([{"name": f"SmolLM2-135M/{f}", "url": f"https://huggingface.co/HuggingFaceTB/SmolLM2-135M/resolve/{SMOL_REV}/{f}",
             "license": "Apache-2.0"} for f in SMOL_FILES]
           + [{"name": f"fineweb-edu/{f}", "url": f"https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/resolve/{FWE_REV}/sample/10BT/{f}",
               "sha256": s, "license": "ODC-BY"} for f, s in FWE.items()]
           + [{"name": f"fineweb-edu/{FWE_HELD}",
               "url": f"https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/resolve/{FWE_REV}/sample/10BT/{FWE_HELD}",
               "license": "ODC-BY"}])
B_OPT_HARNESS = Path(__file__).resolve().parents[2] / "cases" / "B-OPT-001" / "harness"


def tokenize_parquet(p: Path, tok_path: Path, limit: int) -> np.ndarray:
    import pyarrow.parquet as pq                       # operator environment only
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(tok_path))
    out, n = [], 0
    f = pq.ParquetFile(p)
    for rg in range(f.num_row_groups):
        texts = f.read_row_group(rg, columns=["text"]).column("text").to_pylist()
        for enc in tok.encode_batch(texts):
            ids = np.asarray(enc.ids + [0], np.uint16)
            out.append(ids)
            n += len(ids)
        if n >= limit:
            break
    return np.concatenate(out)[:limit]


def build_text(raw: Path, tmp: Path) -> tuple[dict, dict, dict]:
    """(raw_files for the sandbox, hidden arrays, sha256 sums)."""
    sys.path.insert(0, str(B_OPT_HARNESS))
    from token_data import write_shard
    m = raw / "SmolLM2-135M"
    sums = {f"SmolLM2-135M/{f}": check(m / f) for f in SMOL_FILES}
    fw = raw / "fineweb-edu"
    for f, s in FWE.items():
        sums[f"fineweb-edu/{f}"] = check(fw / f, sha256=s)
    sums[f"fineweb-edu/{FWE_HELD}"] = check(fw / FWE_HELD)
    train = tokenize_parquet(fw / "000_00000.parquet", m / "tokenizer.json", TRAIN_TOKENS)
    held = tokenize_parquet(fw / FWE_HELD, m / "tokenizer.json", HELD_TOKENS)
    assert len(train) == TRAIN_TOKENS and len(held) == HELD_TOKENS
    tmp.mkdir(parents=True, exist_ok=True)
    write_shard(tmp / "fwedu_train_000001.bin", train)
    return {"fwedu_train_000001.bin": tmp / "fwedu_train_000001.bin"}, {"tokens": held}, sums


def synthetic_text(n: int, tmp: Path) -> tuple[dict, dict]:
    sys.path.insert(0, str(B_OPT_HARNESS))
    from token_data import write_shard
    rng = np.random.default_rng(0)
    tmp.mkdir(parents=True, exist_ok=True)
    write_shard(tmp / "fwedu_train_000001.bin", rng.integers(0, 49152, n * 4096).astype(np.uint16))
    return {"fwedu_train_000001.bin": tmp / "fwedu_train_000001.bin"}, {"tokens": rng.integers(0, 49152, 65537).astype(np.uint16)}
