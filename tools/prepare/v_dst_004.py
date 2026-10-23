"""V-DST-004: CIFAR-10 training images (public, for training the student), pytorch-fid Inception weights and the FID reference
statistics of the CIFAR-10 training set, as in V-SMP-002; the frozen teacher google/ddpm-cifar10-32 (Apache-2.0)
is provided in the sandbox for distillation only (the evaluation stage that samples does not receive it).
Needs torch and pytorch-fid in the operator environment.
"""
from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

import numpy as np

from prepare.common import Frozen, check
from prepare.v_smp_002 import INCEPTION, REV, SOURCES as _V_SMP_SOURCES, _stats, _train_images  # noqa: F401

SECRET_SEED_ENV = "ABENCH_V_DST_SEED"          # escrowed by tools/prepare/run.py next to the hidden-set key
WITH_TEACHER = bool(1)
SOURCES = ([_V_SMP_SOURCES[0]] if WITH_TEACHER else []) + _V_SMP_SOURCES[1:]
N_BUDGETS = 3


def _labels(raw: Path) -> np.ndarray:
    import sys
    import tarfile
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import prepare_ic_spd as ic
    with tarfile.open(raw / "cifar-10-python.tar.gz", "r:gz") as tf:
        return np.concatenate([ic._batch(tf, f"data_batch_{i}")[1] for i in range(1, 6)])


def build(raw: Path) -> Frozen:
    sums = {INCEPTION: check(raw / INCEPTION)}
    if not sums[INCEPTION].startswith("6726825d"):
        raise SystemExit(f"{INCEPTION}: sha256 {sums[INCEPTION]} does not start with 6726825d")
    sums["cifar-10-python.tar.gz"] = check(raw / "cifar-10-python.tar.gz", md5=_V_SMP_SOURCES[2]["md5"])
    models = {}
    if WITH_TEACHER:
        for f in _V_SMP_SOURCES[0]["files"]:
            sums[f"ddpm-cifar10-32/{f}"] = check(raw / "ddpm-cifar10-32" / f)
        models["ddpm-cifar10-32"] = raw / "ddpm-cifar10-32"
    imgs = _train_images(raw)
    assert imgs.shape == (50000, 32, 32, 3), imgs.shape
    mu, sigma = _stats(imgs, raw / INCEPTION)
    seed = int(os.environ.get(SECRET_SEED_ENV, "0"))
    if not seed:
        raise SystemExit(f"set {SECRET_SEED_ENV} to a secret integer (sampling seeds)")
    inc_dir = Path(tempfile.mkdtemp(prefix="inception-"))
    shutil.copyfile(raw / INCEPTION, inc_dir / INCEPTION)
    models["inception"] = inc_dir
    rng = np.random.default_rng(seed)
    return Frozen(public={"cifar10_train.npz": {"x": imgs, "y": _labels(raw)},
                          "cifar10_train_inception_stats.npz": {"mu": mu, "sigma": sigma}},
                  hidden={"seeds": rng.integers(1, 2**31 - 1, N_BUDGETS).astype(np.int64), "ref_mu": mu, "ref_sigma": sigma},
                  sources_sha256=sums, models=models)


def synthetic(n: int) -> Frozen:
    rng = np.random.default_rng(0)
    f = rng.standard_normal((max(n, 300), 2048))
    mu, sigma = f.mean(0), np.cov(f, rowvar=False)
    return Frozen(public={"cifar10_train.npz": {"x": rng.integers(0, 256, (n, 32, 32, 3)).astype(np.uint8),
                                                "y": rng.integers(0, 10, n).astype(np.int64)},
                          "cifar10_train_inception_stats.npz": {"mu": mu, "sigma": sigma}},
                  hidden={"seeds": np.arange(1, N_BUDGETS + 1, dtype=np.int64), "ref_mu": mu, "ref_sigma": sigma},
                  sources_sha256={"synthetic": f"n{n}"})
