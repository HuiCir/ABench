"""U-BYTE-007: FineWeb as UTF-8 bytes. The token shards of B-OPT-001 (kjj0/fineweb10B-gpt2, MIT; text ODC-BY) are
decoded with the GPT-2 byte-level vocabulary (openai-community/gpt2 vocab.json, MIT, git blob pinned below); the
document separator <|endoftext|> (50256) becomes the byte 0. Public: the decoded training shards 1-2 as raw uint8
files; held out: the first 4,194,304 tokens of fineweb_val_000000.bin decoded (about 19 MB; the first 8,388,608
bytes are scored)."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

from prepare import b_opt_001
from prepare.common import Frozen, check

VOCAB_REV = "607a30d783dfa663caf39e06633721c8d4cfcd7e"
VOCAB_BLOB = "1f1d9aaca301414e7f6c9396df506798ff4eb9a6"          # git blob sha1 of vocab.json (1,042,301 bytes)
SOURCES = b_opt_001.SOURCES + [{"name": "vocab.json",
                                "url": f"https://huggingface.co/openai-community/gpt2/resolve/{VOCAB_REV}/vocab.json",
                                "git_blob_sha1": VOCAB_BLOB, "license": "MIT"}]
EOT = 50256
N_HELD_TOKENS = 4 * 1024 * 1024


def git_blob_sha1(p: Path) -> str:
    data = p.read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def bytes_to_unicode() -> dict:
    """The GPT-2 byte <-> printable-unicode table (Radford et al. 2019, encoder.py)."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


def byte_table(vocab: dict) -> tuple[np.ndarray, np.ndarray]:
    """(flat bytes, offsets) such that token i decodes to flat[offsets[i]:offsets[i + 1]]."""
    dec = {v: k for k, v in bytes_to_unicode().items()}
    n = max(vocab.values()) + 1
    seqs = [b""] * n
    for s, i in vocab.items():
        seqs[i] = b"\0" if i == EOT else bytes(dec[c] for c in s)
    off = np.concatenate([[0], np.cumsum([len(x) for x in seqs])]).astype(np.int64)
    return np.frombuffer(b"".join(seqs), np.uint8), off


def decode(tokens: np.ndarray, flat: np.ndarray, off: np.ndarray, chunk: int = 10_000_000) -> np.ndarray:
    out = []
    for s in range(0, len(tokens), chunk):
        t = np.asarray(tokens[s:s + chunk], np.int64)
        L = off[t + 1] - off[t]
        starts = np.repeat(off[t], L)
        within = np.arange(int(L.sum())) - np.repeat(np.cumsum(L) - L, L)
        out.append(flat[starts + within])
    return np.concatenate(out)


def build(raw: Path) -> Frozen:
    sys.path.insert(0, str(b_opt_001.HARNESS))
    from token_data import read_shard
    sums = {f: check(raw / f) for f in b_opt_001.FILES}
    vp = raw / "vocab.json"
    if git_blob_sha1(vp) != VOCAB_BLOB:
        raise SystemExit("vocab.json: git blob sha1 mismatch")
    sums["vocab.json"] = check(vp)
    vocab = json.loads(vp.read_text(encoding="utf-8"))
    assert len(vocab) == 50257 and vocab["<|endoftext|>"] == EOT
    flat, off = byte_table(vocab)
    files = {}
    for i, f in enumerate(b_opt_001.FILES[:2], start=1):
        b = decode(read_shard(raw / f), flat, off)
        assert len(b) > 300_000_000, (f, len(b))
        dst = raw / f"bytes_train_{i:06d}.bin"
        b.tofile(dst)
        files[dst.name] = dst
    held = decode(read_shard(raw / b_opt_001.FILES[2])[:N_HELD_TOKENS], flat, off)
    return Frozen(public={}, hidden={"bytes": held}, sources_sha256=sums, raw_files=files)


def synthetic(n: int) -> Frozen:
    import tempfile
    rng = np.random.default_rng(0)
    words = [bytes(rng.integers(97, 123, int(rng.integers(1, 9)))) for _ in range(200)]

    def text(k):
        return np.frombuffer(b" ".join(words[i] for i in rng.integers(0, len(words), k)), np.uint8)
    d = Path(tempfile.mkdtemp(prefix="ubyte-syn-"))
    files = {}
    for i in (1, 2):
        p = d / f"bytes_train_{i:06d}.bin"
        text(n * 2000).tofile(p)
        files[p.name] = p
    return Frozen(public={}, hidden={"bytes": text(20000)}, sources_sha256={"synthetic": f"seed0-n{n}"}, raw_files=files)
