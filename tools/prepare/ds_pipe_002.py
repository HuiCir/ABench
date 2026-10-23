"""DS-PIPE-002: six binary numerical-feature datasets of the Grinsztajn et al. (2022) tabular benchmark on OpenML
(suite 337, numerical classification). jannis (45021) replaces electricity (44120), which is a time series whose
shuffled split leaks temporal neighbours (2026-10-21 audit). The jannis id is taken from the dataset card of
huggingface.co/datasets/inria-soda/tabular-benchmark (clf_num_jannis: OpenML d/41168 -> d/45021, 57,580 rows, 54
features); its licence, target column and suite membership are checked by the operator at freeze (build() fails
loudly on a wrong target column or a non-binary target).

Per dataset: rows are shuffled with a fixed seed; the held-out split takes min(30%, 5,000) rows (stratified), and the
public training split takes min(remaining, 10,000) rows, mirroring the benchmark's medium-size regime.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from prepare.common import Frozen, check, dev_indices

DATASETS = {  # OpenML data id: (name, target column, licence as listed by OpenML on 2026-10-19)
    44122: ("pol", "binaryClass", "Public"),
    44123: ("house_16H", "binaryClass", "Public"),
    44125: ("MagicTelescope", "class", "Public"),
    44126: ("bank-marketing", "Class", "Public"),
    44128: ("MiniBooNE", "signal", "CC0"),
    45021: ("jannis", "class", "to_verify (OpenML 41168 / 45021; ChaLearn AutoML challenge data)"),
}
SOURCES = [{"name": f"dataset_{d}.pq", "url": f"https://data.openml.org/datasets/{d // 10000:04d}/{d}/dataset_{d}.pq",
            "openml": f"https://www.openml.org/d/{d}", "dataset": v[0], "target": v[1], "license": v[2]}
           for d, v in DATASETS.items()]
SPLIT_SEED, DEV_FRAC, DEV_SEED = 337, 0.1, 20260701
MAX_TEST, MAX_TRAIN, TEST_FRAC = 5000, 10000, 0.3


def _split(x: np.ndarray, y: np.ndarray, did: int):
    rng = np.random.default_rng(SPLIT_SEED + did)
    n = len(y)
    n_test = min(int(TEST_FRAC * n), MAX_TEST)
    test = []
    for c in (0, 1):
        idx = np.flatnonzero(y == c)
        test.append(rng.choice(idx, size=int(round(n_test * len(idx) / n)), replace=False))
    test = np.concatenate(test)
    rest = np.setdiff1d(np.arange(n), test)
    train = rng.permutation(rest)[:MAX_TRAIN]
    return x[train], y[train], x[test], y[test]


def build(raw: Path) -> Frozen:
    import pandas as pd                       # operator environment only
    public, hidden, sums = {}, {}, {}
    for did, (name, target, _) in DATASETS.items():
        p = raw / f"dataset_{did}.pq"
        sums[p.name] = check(p)
        df = pd.read_parquet(p)
        yraw = df.pop(target)
        classes = sorted(yraw.astype(str).unique())
        assert len(classes) == 2, (name, classes)
        y = (yraw.astype(str) == classes[1]).to_numpy(np.int64)
        x = df.apply(pd.to_numeric, errors="raise").to_numpy(np.float32)
        xtr, ytr, xte, yte = _split(x, y, did)
        public[f"d{did}.npz"] = {"x": xtr, "y": ytr, "dev_idx": dev_indices(len(ytr), DEV_FRAC, DEV_SEED + did, ytr)}
        hidden[f"x_{did}"], hidden[f"y_{did}"] = xte, yte
    return Frozen(public=public, hidden=hidden, sources_sha256=sums,
                  notes={"label_encoding": "1 = lexicographically larger class value of the target column"})


def synthetic(n: int) -> Frozen:
    rng = np.random.default_rng(0)
    public, hidden = {}, {}
    for i, did in enumerate(DATASETS):
        d = 6 + 2 * i
        w = rng.normal(size=d)
        x = rng.normal(size=(n + n // 3, d)).astype(np.float32)
        y = ((x @ w + 0.5 * np.sin(3 * x[:, 0]) + rng.normal(0, 0.5, len(x))) > 0).astype(np.int64)
        xtr, ytr, xte, yte = x[:n], y[:n], x[n:], y[n:]
        public[f"d{did}.npz"] = {"x": xtr, "y": ytr, "dev_idx": dev_indices(n, DEV_FRAC, DEV_SEED + did, ytr)}
        hidden[f"x_{did}"], hidden[f"y_{did}"] = xte, yte
    return Frozen(public=public, hidden=hidden, sources_sha256={"synthetic": f"seed0-n{n}"})
