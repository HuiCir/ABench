"""Exact and near-duplicate removal: normalised-text hashes, then MinHash over word 5-shingles with banding
(20 bands × 5 rows ≈ a Jaccard threshold of 0.55); within a group of candidates the first document is kept."""
import hashlib
import re

import numpy as np

P = (1 << 61) - 1


def _norm(t: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", t.lower())).strip()


def exact_keep(texts) -> np.ndarray:
    seen, keep = set(), np.zeros(len(texts), bool)
    for i, t in enumerate(texts):
        h = hashlib.blake2b(_norm(t).encode(), digest_size=12).digest()
        if h not in seen:
            seen.add(h)
            keep[i] = True
    return keep


def minhash(texts, k: int = 100, shingle: int = 5, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    a = rng.integers(1, P, k, dtype=np.uint64)
    b = rng.integers(0, P, k, dtype=np.uint64)
    sig = np.full((len(texts), k), np.iinfo(np.uint64).max, np.uint64)
    for i, t in enumerate(texts):
        w = _norm(t).split()
        if len(w) < shingle:
            w = w + [""] * (shingle - len(w))
        hs = np.fromiter((int.from_bytes(hashlib.blake2b(" ".join(w[j:j + shingle]).encode(), digest_size=7).digest(), "little")
                          for j in range(len(w) - shingle + 1)), np.uint64)
        if len(hs) > 2000:                                # long documents: an evenly spaced sample of shingles
            hs = hs[np.linspace(0, len(hs) - 1, 2000).astype(int)]
        with np.errstate(over="ignore"):
            v = (hs[:, None] * a[None, :] + b[None, :]) % np.uint64(P)
        sig[i] = v.min(0)
    return sig


def near_keep(sig: np.ndarray, bands: int = 20, rows: int = 5) -> np.ndarray:
    keep = np.ones(len(sig), bool)
    for bnd in range(bands):
        buckets = {}
        for i in np.flatnonzero(keep):
            key = sig[i, bnd * rows:(bnd + 1) * rows].tobytes()
            if key in buckets:
                keep[i] = False
            else:
                buckets[key] = i
    return keep
