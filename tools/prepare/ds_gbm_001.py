"""DS-GBM-001: a tabular learner that the evaluator trains on 20 datasets it has never seen.

Pool: the Grinsztajn et al. (2022) tabular benchmark as published on Hugging Face (inria-soda/tabular-benchmark,
revision 8d0ff910, one CSV per dataset, all features numeric, target in the last column; classification targets are
binary 0/1). Datasets used by DS-PIPE-002 (pol, house_16H, MagicTelescope, bank-marketing, MiniBooNE, jannis),
electricity (a time series), compas-two-years (recidivism data) and credit (GiveMeSomeCredit, Kaggle competition
data) are excluded. 10 datasets are public dev data;
20 others are the held-out tasks: their training splits are given to the submission's fit() inside the evaluator and
their test splits are scored. Per dataset, rows are shuffled with a fixed seed; the test split takes
min(30%, 5,000) rows (stratified for classification), the training split min(rest, 10,000) rows, as in the
benchmark's medium-size regime and DS-PIPE-002.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from prepare.common import Frozen, check

REV = "8d0ff9103525b7e3579b180230fddb3186258301"
BASE = f"https://huggingface.co/datasets/inria-soda/tabular-benchmark/resolve/{REV}"
LICENSE = ("no licence on the dataset card (to_verify); each dataset is a re-upload of an OpenML dataset whose licence "
           "is listed there")
# (path in the repository, task)
DEV = [("clf_num/default-of-credit-card-clients.csv", "classification"), ("clf_num/heloc.csv", "classification"),
       ("clf_num/eye_movements.csv", "classification"), ("clf_num/california.csv", "classification"),
       ("reg_num/abalone.csv", "regression"), ("reg_num/cpu_act.csv", "regression"),
       ("reg_num/wine_quality.csv", "regression"), ("reg_num/house_sales.csv", "regression"),
       ("reg_num/elevators.csv", "regression"), ("reg_num/Bike_Sharing_Demand.csv", "regression")]
HIDDEN = [("clf_cat/albert.csv", "classification"), ("clf_num/covertype.csv", "classification"),
          ("clf_cat/road-safety.csv", "classification"),
          ("clf_num/Bioresponse.csv", "classification"), ("clf_num/Diabetes130US.csv", "classification"),
          ("clf_num/Higgs.csv", "classification"),
          ("reg_num/Ailerons.csv", "regression"), ("reg_num/Brazilian_houses.csv", "regression"),
          ("reg_num/MiamiHousing2016.csv", "regression"), ("reg_num/diamonds.csv", "regression"),
          ("reg_num/houses.csv", "regression"), ("reg_num/medical_charges.csv", "regression"),
          ("reg_num/sulfur.csv", "regression"), ("reg_num/superconduct.csv", "regression"),
          ("reg_num/yprop_4_1.csv", "regression"), ("reg_cat/SGEMM_GPU_kernel_performance.csv", "regression"),
          ("reg_cat/analcatdata_supreme.csv", "regression"), ("reg_cat/topo_2_1.csv", "regression"),
          ("reg_cat/particulate-matter-ukair-2017.csv", "regression"),
          ("reg_cat/visualizing_soil.csv", "regression")]
SOURCES = [{"name": p, "url": f"{BASE}/{p}", "task": t, "role": role, "license": LICENSE}
           for role, lst in (("dev", DEV), ("held-out", HIDDEN)) for p, t in lst]
SPLIT_SEED, MAX_TEST, MAX_TRAIN, TEST_FRAC = 2016, 5000, 10000, 0.3
TASK_CODE = {"classification": 0, "regression": 1}


def split(x: np.ndarray, y: np.ndarray, task: str, seed: int):
    rng = np.random.default_rng(seed)
    n = len(y)
    n_test = min(int(TEST_FRAC * n), MAX_TEST)
    if task == "classification":
        test = np.concatenate([rng.choice(np.flatnonzero(y == c), size=int(round(n_test * np.mean(y == c))), replace=False)
                               for c in (0, 1)])
    else:
        test = rng.choice(n, size=n_test, replace=False)
    rest = np.setdiff1d(np.arange(n), test)
    train = rng.permutation(rest)[:MAX_TRAIN]
    test = rng.permutation(test)
    return x[train], y[train], x[test], y[test]


def _read(p: Path, task: str):
    import pandas as pd                       # operator environment only
    df = pd.read_csv(p)
    y = df.pop(df.columns[-1])
    x = df.apply(pd.to_numeric, errors="raise").to_numpy(np.float32)
    if task == "classification":
        vals = sorted(y.unique())
        assert len(vals) == 2, (p.name, vals)
        y = (y == vals[1]).to_numpy(np.int64)
    else:
        y = pd.to_numeric(y, errors="raise").to_numpy(np.float64)
        assert np.isfinite(y).all(), p.name
    assert len(y) >= 1000 and x.shape[1] >= 2, (p.name, x.shape)
    return x, y


def _pack(name: str, task: str, xtr, ytr, xte, yte, out_pub: dict | None, hid: dict):
    t = np.asarray([TASK_CODE[task]], np.int64)
    if out_pub is not None:
        out_pub[f"{name}.npz"] = {"x_train": xtr, "y_train": ytr, "x_test": xte, "y_test": yte, "task": t}
    else:
        hid[f"{name}_x_train"], hid[f"{name}_y_train"], hid[f"{name}_task"] = xtr, ytr, t
        hid[f"{name}_x_test"], hid[f"{name}_y_test"] = xte, yte


def build(raw: Path) -> Frozen:
    public, hidden, sums = {}, {}, {}
    for i, (path, task) in enumerate(DEV + HIDDEN):
        p = raw / path
        sums[path] = check(p)
        x, y = _read(p, task)
        parts = split(x, y, task, SPLIT_SEED + i)
        if i < len(DEV):
            _pack(f"dev_{Path(path).stem}", task, *parts, public, hidden)
        else:
            _pack(f"h{i - len(DEV):02d}", task, *parts, None, hidden)     # anonymous names for held-out tasks
    names = {f"h{j:02d}": p for j, (p, _) in enumerate(HIDDEN)}
    return Frozen(public=public, hidden=hidden, sources_sha256=sums, notes={"held_out_names": names})


def synthetic(n: int) -> Frozen:
    rng = np.random.default_rng(0)
    public, hidden = {}, {}

    def make(task, d):
        x = rng.normal(size=(n + n // 3, d)).astype(np.float32)
        x[rng.random(x.shape) < 0.02] = np.nan
        w = rng.normal(size=d)
        f = np.nan_to_num(x) @ w + np.sin(2 * np.nan_to_num(x[:, 0]))
        y = (f + rng.normal(0, .5, len(f)) > 0).astype(np.int64) if task == "classification" else f + rng.normal(0, .3, len(f))
        return split(x, y, task, 1)
    for i in range(4):
        task = "classification" if i % 2 == 0 else "regression"
        _pack(f"dev_s{i}", task, *make(task, 5 + i), public, hidden)
    for j in range(len(HIDDEN)):                        # the same 20 held-out names as the real data
        task = "classification" if j % 2 else "regression"
        _pack(f"h{j:02d}", task, *make(task, 4 + j % 5), None, hidden)
    return Frozen(public=public, hidden=hidden, sources_sha256={"synthetic": f"seed0-n{n}"})
