"""U-TTT-008: SmolLM2-135M (Apache-2.0) and FineWeb-Edu tokens (ODC-BY) as in A-KV-004 (prepare/smol_data.py), plus
long-context material from PG19 (Rae et al. 2020, deepmind/pg19 on Hugging Face at revision 4d28bd77, Apache-2.0;
books from the storage bucket its loading script names):
- the first 8,193 tokens of each test book that is at least that long (at least 90 of the 100), tokenised with the SmolLM2 tokenizer;
- passkey prompts: book text from the 50 validation books as filler, the sentence "The pass key is NNNNN. Remember it.
  NNNNN is the pass key." inserted at a random depth, then "What is the pass key? The pass key is NNNNN"; 30 prompts
  each of 2,048, 4,096 and 8,192 tokens, keys and depths from a secret seed.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np

from prepare import smol_data
from prepare.common import Frozen, check

PG19_REV = "4d28bd77e66947ad3835cf78ed7aaeb4dd87ad8b"
BUCKET = "https://storage.googleapis.com/deepmind-gutenberg/"
SOURCES = smol_data.SOURCES + [
    {"name": "pg19/test_files.txt", "url": f"https://huggingface.co/datasets/deepmind/pg19/resolve/{PG19_REV}/data/test_files.txt",
     "license": "Apache-2.0"},
    {"name": "pg19/validation_files.txt",
     "url": f"https://huggingface.co/datasets/deepmind/pg19/resolve/{PG19_REV}/data/validation_files.txt", "license": "Apache-2.0"},
    {"name": "pg19/<split>/<book>.txt", "url": BUCKET + "<path listed in the file lists>", "license": "Apache-2.0 (PG19)"}]
SECRET_SEED_ENV = "ABENCH_U_TTT_SEED"          # escrowed by tools/prepare/run.py next to the hidden-set key
BOOK_TOKENS, PK_LENGTHS, PK_N = 8193, (2048, 4096, 8192), 30


def _books(raw: Path, split: str, sums: dict) -> list[str]:
    lst = raw / "pg19" / f"{split}_files.txt"
    sums[f"pg19/{lst.name}"] = check(lst)
    out = []
    for rel in lst.read_text().split():
        p = raw / "pg19" / rel
        sums[f"pg19/{rel}"] = check(p)
        out.append(p.read_text(encoding="utf-8", errors="replace"))
    return out


def passkeys(tok, filler: list, rng: np.random.Generator) -> tuple[dict, int]:
    fill = np.concatenate([np.asarray(tok.encode(t).ids, np.int64) for t in filler])
    out, ans_len = {}, None
    for L in PK_LENGTHS:
        rows = []
        for _ in range(PK_N):
            key = str(int(rng.integers(10000, 100000)))
            needle = np.asarray(tok.encode(f" The pass key is {key}. Remember it. {key} is the pass key. ").ids)
            q = np.asarray(tok.encode(" What is the pass key? The pass key is").ids)
            a = np.asarray(tok.encode(f" {key}").ids)
            ans_len = len(a) if ans_len is None else ans_len
            assert len(a) == ans_len, (key, len(a), ans_len)
            n_fill = L - len(needle) - len(q) - len(a)
            o = int(rng.integers(0, len(fill) - n_fill))
            hay = fill[o:o + n_fill]
            d = int(rng.integers(0, n_fill))
            rows.append(np.concatenate([hay[:d], needle, hay[d:], q, a]))
        out[f"pk_{L}"] = np.stack(rows).astype(np.int32)
    return out, ans_len


def build(raw: Path) -> Frozen:
    from tokenizers import Tokenizer                    # operator environment only
    seed = int(os.environ.get(SECRET_SEED_ENV, "0"))
    if not seed:
        raise SystemExit(f"set {SECRET_SEED_ENV} to a secret integer (passkeys and their depths)")
    files, hidden, sums = smol_data.build_text(raw, raw / "u_ttt_008_tokens")
    tok = Tokenizer.from_file(str(raw / "SmolLM2-135M" / "tokenizer.json"))
    test = _books(raw, "test", sums)
    books = [np.asarray(tok.encode(b).ids, np.int64) for b in test]
    books = [b for b in books if len(b) >= BOOK_TOKENS]
    assert len(books) >= 90, len(books)                 # PG19 books average about 69,000 words
    hidden["books"] = np.stack([b[:BOOK_TOKENS] for b in books]).astype(np.int32)
    pk, n_ans = passkeys(tok, _books(raw, "validation", sums), np.random.default_rng(seed))
    hidden.update(pk)
    hidden["pk_answer_len"] = np.asarray([n_ans])
    return Frozen(public={}, hidden=hidden, sources_sha256=sums, raw_files=files,
                  models={"SmolLM2-135M": raw / "SmolLM2-135M"})


def synthetic(n: int) -> Frozen:
    files, hidden = smol_data.synthetic_text(n, Path(tempfile.mkdtemp(prefix="uttt-syn-")))
    rng = np.random.default_rng(0)
    hidden["books"] = rng.integers(0, 49152, (4, BOOK_TOKENS)).astype(np.int32)
    for L in PK_LENGTHS:
        hidden[f"pk_{L}"] = rng.integers(0, 49152, (4, L)).astype(np.int32)
    hidden["pk_answer_len"] = np.asarray([3])
    return Frozen(public={}, hidden=hidden, sources_sha256={"synthetic": f"n{n}"}, raw_files=files)
