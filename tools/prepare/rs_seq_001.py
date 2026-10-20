"""RS-SEQ-001: MovieLens-1M -> per-user chronological sequences; the final item of each user is held out.

GroupLens terms: no redistribution without permission and no commercial use; the operator downloads the archive and
uses it only inside the benchmark (public part in the sandbox, held-out part encrypted). Cite Harper & Konstan 2015.
"""
from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np

from prepare.common import Frozen, check

SOURCES = [{"name": "ml-1m.zip", "url": "https://files.grouplens.org/datasets/movielens/ml-1m.zip",
            "checksum_url": "https://files.grouplens.org/datasets/movielens/ml-1m.zip.md5",
            "license": "GroupLens usage license: research use, no redistribution, no commercial use, citation required"}]


def _sequences(users: np.ndarray, movies: np.ndarray, ts: np.ndarray):
    uid = {u: i for i, u in enumerate(np.unique(users))}
    mid = {m: i + 1 for i, m in enumerate(np.unique(movies))}          # 0 is reserved for padding
    u = np.array([uid[x] for x in users], np.int64)
    m = np.array([mid[x] for x in movies], np.int64)
    order = np.lexsort((m, ts, u))                                       # by user, then time, then item id
    u, m, ts = u[order], m[order], ts[order]
    off = np.concatenate([[0], np.cumsum(np.bincount(u, minlength=len(uid)))]).astype(np.int64)
    return m, ts.astype(np.int64), off, len(mid)


def _freeze(m, ts, off, n_items, sums) -> Frozen:
    last = off[1:] - 1
    keep = np.ones(len(m), bool)
    keep[last] = False
    pub_off = (off - np.arange(len(off))).astype(np.int64)              # each user loses one item
    public = {"train.npz": {"items": m[keep].astype(np.int32), "times": ts[keep], "offsets": pub_off,
                            "n_items": np.asarray(n_items, np.int64)}}
    return Frozen(public=public, hidden={"target": m[last].astype(np.int64)}, sources_sha256=sums)


def build(raw: Path) -> Frozen:
    p = raw / "ml-1m.zip"
    sums = {p.name: check(p)}
    with zipfile.ZipFile(p) as z:
        lines = io.TextIOWrapper(z.open("ml-1m/ratings.dat"), encoding="latin-1").read().splitlines()
    arr = np.array([ln.split("::") for ln in lines if ln], dtype=np.int64)
    assert arr.shape == (1000209, 4), arr.shape
    m, ts, off, n_items = _sequences(arr[:, 0], arr[:, 1], arr[:, 3])
    assert len(off) - 1 == 6040
    return _freeze(m, ts, off, n_items, sums)


def synthetic(n: int) -> Frozen:
    """n users walking a noisy item chain, so that transitions are learnable."""
    rng = np.random.default_rng(0)
    n_items = 300
    users, movies, ts = [], [], []
    for u in range(n):
        cur, t = int(rng.integers(1, n_items + 1)), 0
        seen = set()
        for _ in range(int(rng.integers(8, 30))):
            while cur in seen:
                cur = cur % n_items + 1
            seen.add(cur)
            users.append(u); movies.append(cur); ts.append(t)
            t += int(rng.integers(1, 1000))
            cur = (cur + int(rng.choice([1, 2, 3, int(rng.integers(1, n_items))], p=[.5, .2, .2, .1]))) % n_items + 1
    m, ts, off, n_it = _sequences(np.array(users), np.array(movies), np.array(ts))
    return _freeze(m, ts, off, n_it, {"synthetic": f"seed0-n{n}"})
