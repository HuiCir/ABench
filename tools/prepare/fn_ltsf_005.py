"""FN-LTSF-005: long-horizon multivariate forecasting on the ETT series (Zhou et al. 2021, CC BY-ND 4.0) and the
hourly electricity-load panel (UCI ElectricityLoadDiagrams20112014, CC BY 4.0, in the 321-client hourly version of
Lai et al. 2018). Splits, origins and the linear reference are defined in tools/harness_kit/ltsf/ltsf.py.

Public, per dataset: the standardised series up to the end of the validation rows, the row borders and the training
mean and standard deviation. Hidden, per dataset: the histories of the test windows (given to predict() after the
timed phase), their 720-step targets and the reference's per-window errors (evaluator only).

ETT is under CC BY-ND 4.0: the frozen copy (rows split, values standardised) is an internal evaluation copy and must not
be published; the module rebuilds it from the original files. Nothing here is random, so there is no secret seed: the
test origins are public, their values are not.
"""
from __future__ import annotations

import gzip
import importlib.util
from pathlib import Path

import numpy as np

from prepare.common import ROOT, Frozen, check

ETT_COMMIT = "1d16c8f4f943005d613b5bc962e9eeb06058cf07"
LSTNET_COMMIT = "7f402f185cc2435b5e66aed13a3b560ed142e023"
ETT_SHA = {"ETTh1.csv": "f18de3ad269cef59bb07b5438d79bb3042d3be49bdeecf01c1cd6d29695ee066",
           "ETTh2.csv": "a3dc2c597b9218c7ce1cd55eb77b283fd459a1d09d753063f944967dd6b9218b",
           "ETTm1.csv": "6ce1759b1a18e3328421d5d75fadcb316c449fcd7cec32820c8dafda71986c9e",
           "ETTm2.csv": "db973ca252c6410a30d0469b13d696cf919648d0f3fd588c60f03fdbdbadd1fd"}
ELEC_SHA = "3c4c069588198c1fcc95cace7bb69c99922129edfd673b7286661dad20badefa"
SOURCES = [{"name": f"zhouhaoyi/ETDataset ETT-small/{f}", "url": f"https://github.com/zhouhaoyi/ETDataset/blob/{ETT_COMMIT}/ETT-small/{f}",
            "sha256": s, "license": "CC BY-ND 4.0", "note": f"RAW/ETT-small/{f}"} for f, s in ETT_SHA.items()] + \
          [{"name": "laiguokun/multivariate-time-series-data electricity/electricity.txt.gz",
            "url": f"https://github.com/laiguokun/multivariate-time-series-data/blob/{LSTNET_COMMIT}/electricity/electricity.txt.gz",
            "sha256": ELEC_SHA, "license": "CC BY 4.0 (UCI ElectricityLoadDiagrams20112014)",
            "note": "RAW/electricity/electricity.txt.gz"}]
FILES = {"etth1": "ETTh1.csv", "etth2": "ETTh2.csv", "ettm1": "ETTm1.csv", "ettm2": "ETTm2.csv"}
ROWS = {"etth1": 17420, "etth2": 17420, "ettm1": 69680, "ettm2": 69680, "electricity": 26304}


def _ltsf():
    p = ROOT / "tools" / "harness_kit" / "ltsf" / "ltsf.py"
    spec = importlib.util.spec_from_file_location("ltsf_kit", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _stages():
    """The case's own reference_errors(), so the frozen reference and the self-check use the same code."""
    import sys
    L = _ltsf()
    sys.modules.setdefault("ltsf", L)
    h = ROOT / "cases" / "FN-LTSF-005" / "harness"
    sys.path.insert(0, str(ROOT / "tools" / "harness_kit"))
    spec = importlib.util.spec_from_file_location("fn_ltsf_stages", h / "stages.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_raw(raw: Path) -> dict:
    out = {}
    for d, f in FILES.items():
        out[d] = np.loadtxt(raw / "ETT-small" / f, delimiter=",", skiprows=1, usecols=range(1, 8), dtype=np.float64)
    with gzip.open(raw / "electricity" / "electricity.txt.gz", "rt") as fh:
        out["electricity"] = np.loadtxt(fh, delimiter=",", dtype=np.float64)
    return out


def _freeze(data: dict, sums: dict) -> Frozen:
    L = _ltsf()
    S = _stages()
    public, hidden, notes = {}, {}, {"ridge_lambda": {}, "n_windows": {}}
    for d in L.DATASETS:
        raw = data[d]
        tr, va, te = L.borders(d, len(raw))
        s, mu, sd = L.standardise(raw[:te], tr)
        public[f"{d}.npz"] = {"series": s[:va], "train_end": np.asarray(tr, np.int64), "val_end": np.asarray(va, np.int64),
                              "mean": mu, "std": sd}
        orig = L.origins(va, te, L.STRIDE[d])
        r = S.reference_errors(s[:te], tr, va, orig, d)
        hidden[f"x_{d}"] = r["x"]
        hidden.update(r["reference"])
        hidden[f"origins_{d}"] = orig
        notes["ridge_lambda"][d] = r["lambda"]
        notes["n_windows"][d] = int(len(orig))
    return Frozen(public=public, hidden=hidden, sources_sha256=sums, notes=notes)


def build(raw: Path) -> Frozen:
    sums = {f"ETT-small/{f}": check(raw / "ETT-small" / f, sha256=s) for f, s in ETT_SHA.items()}
    sums["electricity/electricity.txt.gz"] = check(raw / "electricity" / "electricity.txt.gz", sha256=ELEC_SHA)
    data = load_raw(raw)
    L = _ltsf()
    for d in L.DATASETS:
        assert data[d].shape == (ROWS[d], L.CHANNELS[d]), (d, data[d].shape)
    return _freeze(data, sums)


def synthetic(n: int) -> Frozen:
    """Seasonal series with noise and a slow drift, at the real row counts (n: number of electricity channels)."""
    L = _ltsf()
    rng = np.random.default_rng(0)
    data = {}
    for d in L.DATASETS:
        rows = ROWS[d]
        c = L.CHANNELS[d] if d != "electricity" else max(2, min(n, L.CHANNELS[d]))
        t = np.arange(rows)[:, None]
        period = 96 if d.startswith("ettm") else 24
        phase = rng.uniform(0, 2 * np.pi, c)
        x = np.sin(2 * np.pi * t / period + phase) + 0.5 * np.sin(2 * np.pi * t / (7 * period) + phase)
        x = x + 0.3 * rng.standard_normal((rows, c)) + np.cumsum(rng.standard_normal((rows, c)) * 0.01, axis=0)
        data[d] = x
    return _freeze(data, {"synthetic": f"seed0-n{n}"})
