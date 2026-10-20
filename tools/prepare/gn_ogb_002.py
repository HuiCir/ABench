"""GN-OGB-002: ogbn-arxiv (OGB, ODC-BY) -> public graph with test labels removed; held-out test labels.

Reads the official OGB archive (the same file NodePropPredDataset downloads) without needing the ogb package:
raw/edge.csv.gz, raw/node-feat.csv.gz, raw/node-label.csv.gz, raw/node_year.csv.gz and split/time/{train,valid,test}.csv.gz.
"""
from __future__ import annotations

import gzip
import io
import zipfile
from pathlib import Path

import numpy as np

from prepare.common import Frozen, check

SOURCES = [{"name": "arxiv.zip", "url": "http://snap.stanford.edu/ogb/data/nodeproppred/arxiv.zip",
            "registry": "https://raw.githubusercontent.com/snap-stanford/ogb/master/ogb/nodeproppred/master.csv",
            "license": "ODC-BY"}]


def _csv(z: zipfile.ZipFile, suffix: str, dtype) -> np.ndarray:
    names = [n for n in z.namelist() if n.endswith(suffix)]
    if len(names) != 1:
        raise SystemExit(f"arxiv.zip: expected one member ending in {suffix}, found {names}")
    with z.open(names[0]) as f:
        text = gzip.decompress(f.read()).decode("ascii")
    return np.loadtxt(io.StringIO(text), delimiter=",", dtype=dtype, ndmin=2)


def _freeze(x, edges, year, y, tr, va, te, sums) -> Frozen:
    y_pub = y.copy()
    y_pub[te] = -1
    public = {"graph.npz": {"x": x.astype(np.float32), "edge_index": edges.astype(np.int64), "year": year.astype(np.int64),
                            "y": y_pub.astype(np.int64), "train_idx": tr.astype(np.int64),
                            "valid_idx": va.astype(np.int64), "test_idx": te.astype(np.int64)}}
    return Frozen(public=public, hidden={"y_test": y[te].astype(np.int64)}, sources_sha256=sums)


def build(raw: Path) -> Frozen:
    p = raw / "arxiv.zip"
    sums = {p.name: check(p)}
    with zipfile.ZipFile(p) as z:
        edges = _csv(z, "raw/edge.csv.gz", np.int64).T
        x = _csv(z, "raw/node-feat.csv.gz", np.float32)
        y = _csv(z, "raw/node-label.csv.gz", np.int64).ravel()
        year = _csv(z, "raw/node_year.csv.gz", np.int64).ravel()
        tr, va, te = (_csv(z, f"split/time/{s}.csv.gz", np.int64).ravel() for s in ("train", "valid", "test"))
    assert x.shape == (169343, 128) and edges.shape == (2, 1166243), (x.shape, edges.shape)
    return _freeze(x, edges, year, y, tr, va, te, sums)


def synthetic(n: int) -> Frozen:
    rng = np.random.default_rng(0)
    k = 40
    y = rng.integers(0, k, n)
    centers = rng.normal(size=(k, 128))
    x = centers[y] * 0.4 + rng.normal(size=(n, 128))
    src = rng.integers(0, n, 6 * n)
    same = rng.random(6 * n) < 0.7
    dst = np.where(same, np.array([rng.choice(np.flatnonzero(y == y[s])) for s in src]), rng.integers(0, n, 6 * n))
    year = rng.integers(2010, 2021, n)
    order = np.argsort(year, kind="stable")
    tr, va, te = order[: int(.6 * n)], order[int(.6 * n): int(.75 * n)], order[int(.75 * n):]
    return _freeze(x, np.stack([src, dst]), year, y, np.sort(tr), np.sort(va), np.sort(te), {"synthetic": f"seed0-n{n}"})
