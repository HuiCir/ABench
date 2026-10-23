"""X-DATA-007: a raw web pool from four FineWeb dumps (HuggingFaceFW/fineweb @9bb295dd, ODC-BY; deduplicated only
within each dump, so cross-dump duplicates remain), planted near-duplicates, and FineWeb-Edu (@87f09149, ODC-BY) as
the target distribution. Tokens are GPT-2 BPE (openai-community/gpt2 tokenizer.json @607a30d7, MIT), as in B-OPT-001.

Public: the pool (400,000 documents plus 16,000 planted near-copies, shuffled; text, GPT-2 tokens, offsets, dump),
5,000 FineWeb-Edu documents as target examples, and dev target/general token sets of 2,097,153 tokens.
Hidden: 2,097,153 FineWeb-Edu tokens (another file), 2,097,153 tokens of a later FineWeb dump, and the cluster of
every planted copy and its original (ABENCH_X_DATA_SEED draws the originals, the edits and the order).
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import numpy as np

from prepare.common import ROOT, Frozen, check, text_pack

FW_REV = "9bb295ddab0e05d785b879661af7260fed5140fc"
FWE_REV = "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9"
GPT2_REV = "607a30d783dfa663caf39e06633721c8d4cfcd7e"
SECRET_SEED_ENV = "ABENCH_X_DATA_SEED"
POOL_DUMPS = ["CC-MAIN-2013-20", "CC-MAIN-2023-14", "CC-MAIN-2023-50", "CC-MAIN-2024-18"]
DEV_GENERAL, HELD_GENERAL = "CC-MAIN-2024-26", "CC-MAIN-2024-51"
PER_DUMP, PLANT_FRAC, N_TOK, N_TARGET_DOCS = 100_000, 0.04, 2_097_153, 5_000
SOURCES = ([{"name": f"fineweb/data/{d}/000_00000.parquet", "url": f"https://huggingface.co/datasets/HuggingFaceFW/fineweb/resolve/{FW_REV}/data/{d}/000_00000.parquet",
             "license": "ODC-BY"} for d in POOL_DUMPS + [DEV_GENERAL, HELD_GENERAL]]
           + [{"name": f"fineweb-edu/sample/10BT/{f}", "url": f"https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/resolve/{FWE_REV}/sample/10BT/{f}",
               "license": "ODC-BY"} for f in ("000_00000.parquet", "013_00000.parquet")]
           + [{"name": "gpt2/tokenizer.json", "url": f"https://huggingface.co/openai-community/gpt2/resolve/{GPT2_REV}/tokenizer.json", "license": "MIT"}])
BOILER = ["Share this:", "Posted by admin", "Click here to subscribe to our newsletter.", "All rights reserved.",
          "Related posts", "Leave a comment", "Read more"]


def near_copy(text: str, rng) -> str:
    """A near-duplicate: about 2% of words deleted, 1% of adjacent pairs swapped, boilerplate lines added."""
    w = text.split(" ")
    w = [x for x in w if rng.random() > 0.02]
    for i in range(len(w) - 1):
        if rng.random() < 0.01:
            w[i], w[i + 1] = w[i + 1], w[i]
    out = " ".join(w)
    if rng.random() < 0.7:
        out = BOILER[int(rng.integers(0, len(BOILER)))] + "\n" + out
    if rng.random() < 0.7:
        out = out + "\n" + BOILER[int(rng.integers(0, len(BOILER)))]
    return out


def _encode(tok, texts) -> list:
    return [np.asarray(e.ids + [50256], np.uint16) for e in tok.encode_batch(texts)]       # <|endoftext|> after each


def _write_shard(tokens, path):
    sys.path.insert(0, str(ROOT / "cases" / "B-OPT-001" / "harness"))
    from token_data import write_shard
    write_shard(path, tokens)


def _freeze(pool_texts, pool_dump, target_texts, dev_t, dev_g, held_t, held_g, tok_fn, seed, tmp: Path) -> Frozen:
    rng = np.random.default_rng(seed)
    n = len(pool_texts)
    orig = rng.choice(n, int(round(PLANT_FRAC * n)), replace=False)
    texts = list(pool_texts) + [near_copy(pool_texts[i], rng) for i in orig]
    dump = np.concatenate([pool_dump, pool_dump[orig]]).astype(np.int64)
    cluster = -np.ones(len(texts), np.int64)
    cluster[orig] = np.arange(len(orig))
    cluster[n:] = np.arange(len(orig))
    order = rng.permutation(len(texts))
    texts = [texts[i] for i in order]
    dump, cluster = dump[order], cluster[order]
    toks = tok_fn(texts)
    off = np.zeros(len(toks) + 1, np.int64)
    np.cumsum([len(t) for t in toks], out=off[1:])
    tmp.mkdir(parents=True, exist_ok=True)
    _write_shard(np.concatenate(toks), tmp / "pool_tokens.bin")
    u, o = text_pack(texts)
    tu, to = text_pack(target_texts)
    return Frozen(public={"pool_text.npz": {"text_utf8": u, "text_offsets": o},
                          "pool_index.npz": {"token_offsets": off, "dump": dump},
                          "target_examples.npz": {"text_utf8": tu, "text_offsets": to},
                          "dev_text.npz": {"target_tokens": dev_t, "general_tokens": dev_g}},
                  hidden={"target_tokens": held_t, "general_tokens": held_g, "cluster": cluster},
                  raw_files={"pool_tokens.bin": tmp / "pool_tokens.bin"},
                  notes={"documents": len(texts), "planted": int(len(orig)), "pool_tokens": int(off[-1])})


def build(raw: Path) -> Frozen:
    import pyarrow.parquet as pq
    from tokenizers import Tokenizer
    seed = int(os.environ.get(SECRET_SEED_ENV, "0"))
    if not seed:
        raise SystemExit(f"set {SECRET_SEED_ENV} to a secret integer (planted duplicates and pool order)")
    tok = Tokenizer.from_file(str(raw / "gpt2" / "tokenizer.json"))
    sums = {"gpt2/tokenizer.json": check(raw / "gpt2" / "tokenizer.json")}
    def docs(p, k):
        sums[str(p.relative_to(raw))] = check(p)
        f, out = pq.ParquetFile(p), []
        for rg in range(f.num_row_groups):
            out += f.read_row_group(rg, columns=["text"]).column("text").to_pylist()
            if len(out) >= k:
                break
        return out[:k]
    def tokens(p, k):
        out, n = [], 0
        for t in _encode(tok, docs(p, 20_000)):
            out.append(t)
            n += len(t)
            if n >= k:
                break
        return np.concatenate(out)[:k]
    fw, fe = raw / "fineweb" / "data", raw / "fineweb-edu"
    pool, dump = [], []
    for i, d in enumerate(POOL_DUMPS):
        t = docs(fw / d / "000_00000.parquet", PER_DUMP)
        pool += t
        dump += [i] * len(t)
    edu0 = docs(fe / "000_00000.parquet", N_TARGET_DOCS + 20_000)
    target_examples, dev_target_docs = edu0[:N_TARGET_DOCS], edu0[N_TARGET_DOCS:]
    dev_t = np.concatenate(_encode(tok, dev_target_docs))[:N_TOK]
    fz = _freeze(pool, np.asarray(dump), target_examples, dev_t, tokens(fw / DEV_GENERAL / "000_00000.parquet", N_TOK),
                 tokens(fe / "013_00000.parquet", N_TOK), tokens(fw / HELD_GENERAL / "000_00000.parquet", N_TOK),
                 lambda texts: _encode(tok, texts), seed, raw / "x_data_007_tokens")
    fz.sources_sha256 = sums
    return fz


def synthetic(n: int) -> Frozen:
    """Plumbing only: random 'documents' over a small vocabulary; byte values stand in for tokens."""
    rng = np.random.default_rng(n)
    words = [f"w{i}" for i in range(200)]
    pool = [" ".join(rng.choice(words, int(rng.integers(20, 60)))) for _ in range(max(50, n))]
    tgt = [" ".join(rng.choice(words[:50], 40)) for _ in range(20)]
    tok_fn = lambda texts: [np.frombuffer(t.encode("utf-8"), np.uint8).astype(np.uint16) for t in texts]
    small = rng.integers(0, 255, 4097).astype(np.uint16)
    return _freeze(pool, np.zeros(len(pool), np.int64), tgt, small, small, small, small, tok_fn, 1,
                   Path(tempfile.mkdtemp(prefix="xdata-syn-")))
