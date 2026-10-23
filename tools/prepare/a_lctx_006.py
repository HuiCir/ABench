"""A-LCTX-006: FineWeb-10B GPT-2 token shards (kjj0/fineweb10B-gpt2, MIT; text ODC-BY) for training and the
short-context loss, and PG-19 books (Rae et al. 2020, github.com/google-deepmind/pg19, Apache-2.0; files from the
deepmind-gutenberg bucket) tokenised with the GPT-2 tokenizer (openai-community/gpt2, MIT) for the long-context loss.

Public: train shards 1-2 verbatim; pg19_dev.npz with one 16,385-token window from each validation book that is long
enough (public seed). Hidden: the first 4,194,305 tokens of fineweb_val_000000.bin; one window per test book (offset
drawn with the secret seed ABENCH_A_LCTX_SEED); 64 copy-test sequences of 8,192 and 32 of 16,384 tokens built from
the FineWeb validation tokens with the same seed (harness/longctx.make_copy).
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import numpy as np

from prepare import b_opt_001 as B
from prepare.common import ROOT, Frozen, check

SECRET_SEED_ENV = "ABENCH_A_LCTX_SEED"
PG19_COMMIT, GPT2_REV = "b0e1f2925359688af19f8784c23c3deeda52d3ea", "607a30d783dfa663caf39e06633721c8d4cfcd7e"
SOURCES = B.SOURCES + [
    {"name": "PG-19 validation and test books", "url": "https://storage.googleapis.com/deepmind-gutenberg/",
     "license": "Apache-2.0 (github.com/google-deepmind/pg19 @b0e1f292)", "note": "RAW/pg19/validation/*.txt, RAW/pg19/test/*.txt"},
    {"name": "gpt2 tokenizer", "url": f"https://huggingface.co/openai-community/gpt2/tree/{GPT2_REV}",
     "files": ["tokenizer.json", "vocab.json", "merges.txt"], "license": "MIT", "note": "RAW/gpt2/"}]
WIN, N_COPY, DEV_SEED = 16385, {8192: 64, 16384: 32}, 20230120


def _longctx():
    p = ROOT / "cases" / "A-LCTX-006" / "harness" / "longctx.py"
    spec = importlib.util.spec_from_file_location("longctx_case", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def book_windows(books: list, rng) -> np.ndarray:
    """One WIN-token window per book with at least WIN tokens, at a random offset."""
    out = []
    for toks in books:
        if len(toks) >= WIN:
            o = int(rng.integers(0, len(toks) - WIN + 1))
            out.append(np.asarray(toks[o:o + WIN], np.int32))
    return np.stack(out)


def _freeze(fz: Frozen, val_books: list, test_books: list, seed: int) -> Frozen:
    LC = _longctx()
    rng = np.random.default_rng(seed)
    fz.public = {**fz.public, "pg19_dev.npz": {"windows": book_windows(val_books, np.random.default_rng(DEV_SEED))}}
    hidden = dict(fz.hidden)
    hidden["pg19_windows"] = book_windows(test_books, rng)
    stream = np.asarray(hidden["tokens"]).astype(np.int64)
    for L, n in N_COPY.items():
        hidden[f"copy_x_{L}"], hidden[f"copy_start_{L}"] = LC.make_copy(stream, L, n, rng)
    fz.hidden = hidden
    return fz


def _tokenize(raw: Path, split: str) -> tuple[list, dict]:
    from transformers import AutoTokenizer                    # operator environment
    tok = AutoTokenizer.from_pretrained(str(raw / "gpt2"))
    files = sorted((raw / "pg19" / split).glob("*.txt"))
    sums = {f"pg19/{split}/{f.name}": check(f) for f in files}
    return [tok(f.read_text(encoding="utf-8"))["input_ids"] for f in files], sums


def build(raw: Path) -> Frozen:
    seed = int(os.environ.get(SECRET_SEED_ENV, "0"))
    if not seed:
        raise SystemExit(f"set {SECRET_SEED_ENV} to a secret integer (book offsets and copy tests)")
    fz = B.build(raw)
    val, s1 = _tokenize(raw, "validation")
    test, s2 = _tokenize(raw, "test")
    assert (len(val), len(test)) == (50, 100), (len(val), len(test))
    fz.sources_sha256 = {**fz.sources_sha256, **s1, **s2}
    return _freeze(fz, val, test, seed)


def synthetic(n: int) -> Frozen:
    rng = np.random.default_rng(0)
    books = [rng.integers(0, 50257, WIN + int(rng.integers(0, 2000))) for _ in range(6)]
    return _freeze(B.synthetic(max(n, 80)), books[:3], books[3:], 12345)
