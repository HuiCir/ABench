"""RS-GRAPH-002: Gowalla check-ins as split by NGCF/LightGCN (Data/gowalla of github.com/kuandeng/LightGCN at commit
b067ec05): 29,858 users, 40,981 items, 810,128 training and 217,242 test interactions (a random 80/20 split per user,
10-core filtered). Each line of train.txt / test.txt is "user item item ...". The training lists are public; the test
lists are held out.

Licence: the repository has no licence file; the check-ins come from the SNAP loc-gowalla dataset (Cho, Myers &
Leskovec 2011), published without an explicit licence. Both are to be confirmed before release (to_verify).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from prepare.common import Frozen, check

COMMIT = "b067ec05f16a24e0e1efe63cd82385f0c469af20"
BASE = f"https://github.com/kuandeng/LightGCN/raw/{COMMIT}/Data/gowalla"
SHA256 = {"train.txt": "7244696fdeb23911319cf7b689f2e15a3c4c92549c873a5677b44e3f13cc3e62",
          "test.txt": "099a2a73924e4b754dc6efd730d764f492321b53c38eb93298850c83acf57be3"}
LICENSE = ("no licence file in the repository; SNAP loc-gowalla has no explicit licence (to_verify before release)")
SOURCES = [{"name": n, "url": f"{BASE}/{n}", "sha256": s, "license": LICENSE} for n, s in SHA256.items()]
N_USERS, N_ITEMS, N_TRAIN, N_TEST = 29858, 40981, 810128, 217242


def read_lists(p: Path) -> list[np.ndarray]:
    rows = []
    for u, ln in enumerate(p.read_text().splitlines()):
        v = ln.split()
        if not v:
            continue
        assert int(v[0]) == u, (p.name, u, v[0])
        rows.append(np.array(sorted(int(x) for x in v[1:]), np.int64))
    return rows


def pad_lists(lists: list) -> np.ndarray:
    """Per-user item lists -> (U, max_len) int64 array padded with -1 (the layout of the held-out reference)."""
    out = np.full((len(lists), max(1, max(map(len, lists)))), -1, np.int64)
    for u, r in enumerate(lists):
        out[u, :len(r)] = r
    return out


def _freeze(train: list, test: list, n_items: int, sums: dict) -> Frozen:
    ip = np.concatenate([[0], np.cumsum([len(r) for r in train])]).astype(np.int64)
    public = {"train.npz": {"indptr": ip, "items": np.concatenate(train).astype(np.int32), "n_items": np.asarray(n_items)}}
    return Frozen(public=public, hidden={"test": pad_lists(test)}, sources_sha256=sums)


def build(raw: Path) -> Frozen:
    d = raw / "gowalla"
    sums = {n: check(d / n, sha256=s) for n, s in SHA256.items()}
    train, test = read_lists(d / "train.txt"), read_lists(d / "test.txt")
    assert len(train) == len(test) == N_USERS
    assert sum(map(len, train)) == N_TRAIN and sum(map(len, test)) == N_TEST
    assert max(int(r.max()) for r in train + test if len(r)) + 1 == N_ITEMS
    assert all(len(np.intersect1d(a, b)) == 0 for a, b in zip(train, test))
    return _freeze(train, test, N_ITEMS, sums)


def synthetic(n: int) -> Frozen:
    """n users with preferences over 8 latent clusters of 400 items; 80/20 split per user."""
    rng = np.random.default_rng(0)
    n_items, n_cl = 400, 8
    cl = rng.integers(0, n_cl, n_items)
    train, test = [], []
    for _ in range(n):
        pref = rng.dirichlet(np.ones(n_cl) * 0.3)
        p = pref[cl] / pref[cl].sum()
        row = rng.choice(n_items, size=int(rng.integers(10, 40)), replace=False, p=p)
        cut = int(0.8 * len(row))
        train.append(np.sort(row[:cut])); test.append(np.sort(row[cut:]))
    return _freeze(train, test, n_items, {"synthetic": f"seed0-n{n}"})
