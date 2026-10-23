"""T-SPEC-003: the frozen model of V-SMP-002, google/ddpm-cifar10-32 (Apache-2.0), pytorch-fid Inception weights (Apache-2.0) and the FID
reference statistics of the CIFAR-10 training set, computed here with the same Inception network.

Raw directory layout:
  ddpm-cifar10-32/                       snapshot of the model repository at REV (config.json, diffusion_pytorch_model.safetensors, ...)
  pt_inception-2015-12-05-6726825d.pth   FID Inception weights released with pytorch-fid
  cifar-10-python.tar.gz                 CIFAR-10 (for the reference statistics)
Needs torch and pytorch-fid in the operator environment.
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

from prepare.common import Frozen, check

REV = "267b167dc01f0e4e61923ea244e8b988f84deb80"
INCEPTION = "pt_inception-2015-12-05-6726825d.pth"
SECRET_SEED_ENV = "ABENCH_T_SPEC_SEED"          # escrowed by tools/prepare/run.py next to the hidden-set key

SOURCES = [
    {"name": "ddpm-cifar10-32/", "url": f"https://huggingface.co/google/ddpm-cifar10-32/tree/{REV}", "license": "Apache-2.0",
     "files": ["config.json", "diffusion_pytorch_model.safetensors", "scheduler_config.json", "model_index.json"]},
    {"name": INCEPTION, "url": f"https://github.com/mseitzer/pytorch-fid/releases/download/fid_weights/{INCEPTION}",
     "license": "Apache-2.0 (pytorch-fid)"},
    {"name": "cifar-10-python.tar.gz", "url": "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz",
     "md5": "c58f30108f718f92721af3b95e74349a", "license": "no explicit license (to_verify)"},
]


def _train_images(raw: Path) -> np.ndarray:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import tarfile
    import prepare_ic_spd as ic
    with tarfile.open(raw / "cifar-10-python.tar.gz", "r:gz") as tf:
        return np.concatenate([ic._batch(tf, f"data_batch_{i}")[0] for i in range(1, 6)])


def _stats(imgs: np.ndarray, inception_file: Path) -> tuple[np.ndarray, np.ndarray]:
    import torch
    from pytorch_fid.inception import InceptionV3
    hub = Path(torch.hub.get_dir()) / "checkpoints"
    hub.mkdir(parents=True, exist_ok=True)
    if not (hub / INCEPTION).exists():
        shutil.copyfile(inception_file, hub / INCEPTION)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]]).to(dev).eval()
    feats = []
    with torch.no_grad():
        for i in range(0, len(imgs), 500):
            x = torch.from_numpy(imgs[i:i + 500]).to(dev).permute(0, 3, 1, 2).float() / 255.0
            feats.append(net(x)[0].squeeze(-1).squeeze(-1).double().cpu().numpy())
    f = np.concatenate(feats)
    return f.mean(0), np.cov(f, rowvar=False)


def build(raw: Path) -> Frozen:
    sums = {f"ddpm-cifar10-32/{f}": check(raw / "ddpm-cifar10-32" / f) for f in SOURCES[0]["files"]}
    sums[INCEPTION] = check(raw / INCEPTION)
    if not sums[INCEPTION].startswith("6726825d"):      # torch-hub naming: the suffix is the start of the file's sha256
        raise SystemExit(f"{INCEPTION}: sha256 {sums[INCEPTION]} does not start with 6726825d")
    sums["cifar-10-python.tar.gz"] = check(raw / "cifar-10-python.tar.gz", md5=SOURCES[2]["md5"])
    mu, sigma = _stats(_train_images(raw), raw / INCEPTION)
    seed = int(os.environ.get("ABENCH_T_SPEC_SEED", "0"))
    if not seed:
        raise SystemExit("set ABENCH_T_SPEC_SEED to a secret integer (sampling seeds)")
    rng = np.random.default_rng(seed)
    inc_dir = Path(tempfile.mkdtemp(prefix="inception-"))
    shutil.copyfile(raw / INCEPTION, inc_dir / INCEPTION)
    return Frozen(public={"cifar10_train_inception_stats.npz": {"mu": mu, "sigma": sigma}},
                  hidden={"seeds": rng.integers(1, 2**31 - 1, 2).astype(np.int64), "ref_mu": mu, "ref_sigma": sigma},
                  sources_sha256=sums, models={"ddpm-cifar10-32": raw / "ddpm-cifar10-32", "inception": inc_dir})


def synthetic(n: int) -> Frozen:
    rng = np.random.default_rng(0)
    f = rng.standard_normal((max(n, 300), 2048))
    mu, sigma = f.mean(0), np.cov(f, rowvar=False)
    return Frozen(public={"cifar10_train_inception_stats.npz": {"mu": mu, "sigma": sigma}},
                  hidden={"seeds": np.asarray([1, 2], np.int64), "ref_mu": mu, "ref_sigma": sigma},
                  sources_sha256={"synthetic": f"n{n}"})
