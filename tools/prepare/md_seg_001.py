"""MD-SEG-001: four tasks of the Medical Segmentation Decathlon (Antonelli et al. 2022; Simpson et al. 2019,
CC-BY-SA 4.0), from the tar files MONAI downloads (msd-for-monai S3 bucket, md5 as in MONAI's DecathlonDataset).

Only the labelled training volumes (imagesTr, labelsTr) are used. Every volume is resampled to its task's target
spacing (scipy.ndimage.zoom: images order 1, labels order 0) and stored with the slowest axis first; MRI channels are
z-scored per volume, CT is clipped to the 0.5-99.5 percentiles of the training foreground and z-scored with its mean
and standard deviation.

  Hippocampus (Task04, 260 volumes), Prostate (Task05, 32), Heart (Task02, 20): with the secret seed
  ABENCH_MD_SEG_SEED, 20% of each task's volumes are held out (52, 6, 4); the rest are public.
  Spleen (Task09, 41 CT volumes) is the unseen task: 31 training volumes and the configuration are given to the
  evaluator's training stage, 10 are held out; nothing of it is public.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import tarfile
from pathlib import Path

import numpy as np

from prepare.common import ROOT, Frozen, check

SECRET_SEED_ENV = "ABENCH_MD_SEG_SEED"
BUCKET = "https://msd-for-monai.s3-us-west-2.amazonaws.com"
MSD = {"hippocampus": ("Task04_Hippocampus", "9d24dba78a72977dbd1d2e110310f31b"),
       "prostate": ("Task05_Prostate", "35138f08b1efaef89d7424d2bcc928db"),
       "heart": ("Task02_Heart", "06ee59366e1e5124267b774dbd654057"),
       "unseen": ("Task09_Spleen", "410d4a301da4e5b2f6f86ec3ddba524e")}
UNSEEN = {"channels": 1, "classes": 2, "spacing": [3.0, 1.5, 1.5], "patch": [64, 128, 128], "batch": 2,
          "iterations": 2500, "nsd_mm": 2.0, "modality": "CT"}
SOURCES = [{"name": f"{n}.tar", "url": f"{BUCKET}/{n}.tar", "md5": m, "license": "CC-BY-SA 4.0", "note": f"RAW/{n}.tar"}
           for n, m in MSD.values()]
HOLD = 0.2
N_UNSEEN_TEST = 10


def seg3d():
    p = ROOT / "cases" / "MD-SEG-001" / "harness" / "seg3d.py"
    spec = importlib.util.spec_from_file_location("seg3d", p)
    m = importlib.util.module_from_spec(spec)
    sys.modules["seg3d"] = m
    spec.loader.exec_module(m)
    return m


def resample(a: np.ndarray, spacing, target, order: int) -> np.ndarray:
    from scipy.ndimage import zoom
    f = [s / t for s, t in zip(spacing, target)]
    return zoom(a, f, order=order, mode="nearest", grid_mode=False)


def zscore(x: np.ndarray) -> np.ndarray:
    return ((x - x.mean()) / max(float(x.std()), 1e-8)).astype(np.float16)


def read_task(tar: Path, name: str):
    """[(image (C, z, y, x) float32, label (z, y, x) uint8, spacing (z, y, x))] of the labelled volumes."""
    import nibabel as nib                                   # operator environment
    out = []
    with tarfile.open(tar) as tf:
        members = {m.name: m for m in tf.getmembers()}
        labels = sorted(n for n in members if f"{name}/labelsTr/" in n and n.endswith(".nii.gz")
                        and not Path(n).name.startswith("._"))
        for ln in labels:
            im_name = ln.replace("labelsTr", "imagesTr")
            img = nib.Nifti1Image.from_bytes(__import__("gzip").decompress(tf.extractfile(members[im_name]).read()))
            lab = nib.Nifti1Image.from_bytes(__import__("gzip").decompress(tf.extractfile(members[ln]).read()))
            a = np.asarray(img.dataobj, np.float32)
            a = a[..., None] if a.ndim == 3 else a
            sp = tuple(float(z) for z in img.header.get_zooms()[:3])[::-1]
            out.append((np.transpose(a, (3, 2, 1, 0)), np.transpose(np.asarray(lab.dataobj), (2, 1, 0)).astype(np.uint8), sp))
    return out


def preprocess(vols: list, cfg: dict, train_idx) -> list:
    tgt = cfg["spacing"]
    res = []
    for img, lab, sp in vols:
        im = np.stack([resample(c, sp, tgt, 1) for c in img])
        lb = resample(lab, sp, tgt, 0).astype(np.uint8)
        res.append((im, lb))
    if cfg["modality"] == "CT":
        fg = np.concatenate([res[i][0][0][res[i][1] > 0] for i in train_idx])
        lo, hi = np.percentile(fg, [0.5, 99.5])
        clipped = np.clip(fg, lo, hi)
        mu, sd = float(clipped.mean()), float(clipped.std())
        return [(((np.clip(im, lo, hi) - mu) / sd).astype(np.float16), lb) for im, lb in res]
    return [(np.stack([zscore(c) for c in im]), lb) for im, lb in res]


def _freeze(tasks: dict, seed: int, sums: dict) -> Frozen:
    S = seg3d()
    rng = np.random.default_rng(seed)
    public, hidden = {}, {}
    for t, vols in tasks.items():
        n = len(vols)
        perm = rng.permutation(n)
        k = N_UNSEEN_TEST if t == "unseen" else int(round(HOLD * n))
        te, tr = sorted(perm[:k].tolist()), sorted(perm[k:].tolist())
        cfg = UNSEEN if t == "unseen" else S.TASKS[t]
        pv = preprocess(vols, cfg, tr) if not isinstance(vols[0][2], str) else vols
        img_tr, seg_tr = S.pack([pv[i][0] for i in tr]), S.pack([pv[i][1] for i in tr])
        hidden[f"{t}_test_img"], hidden[f"{t}_test_img_shape"] = S.pack([pv[i][0] for i in te])
        hidden[f"{t}_test_seg"], hidden[f"{t}_test_seg_shape"] = S.pack([pv[i][1] for i in te])
        if t == "unseen":
            hidden.update({"unseen_train_img": img_tr[0], "unseen_train_img_shape": img_tr[1], "unseen_train_seg": seg_tr[0],
                           "unseen_train_seg_shape": seg_tr[1], "unseen_config": np.asarray([json.dumps(UNSEEN)])})
        else:
            public[f"{t}.npz"] = {"img": img_tr[0], "img_shape": img_tr[1], "seg": seg_tr[0], "seg_shape": seg_tr[1]}
    return Frozen(public=public, hidden=hidden, sources_sha256=sums)


def build(raw: Path) -> Frozen:
    seed = int(os.environ.get(SECRET_SEED_ENV, "0"))
    if not seed:
        raise SystemExit(f"set {SECRET_SEED_ENV} to a secret integer (held-out volumes)")
    sums, tasks = {}, {}
    for t, (name, md5) in MSD.items():
        p = raw / f"{name}.tar"
        sums[p.name] = check(p, md5=md5)
        tasks[t] = read_task(p, name)
    return _freeze(tasks, seed, sums)


def synthetic(n: int) -> Frozen:
    """Spheres of random size in noise (already at the target spacing; plumbing only)."""
    S = seg3d()
    rng = np.random.default_rng(0)
    tasks = {}
    for t in MSD:
        cfg = UNSEEN if t == "unseen" else S.TASKS[t]
        shape = [max(p // 2, 8) + int(rng.integers(0, 6)) for p in cfg["patch"]]
        vols = []
        for _ in range(max(n // 4, 6)):
            zz, yy, xx = np.meshgrid(*[np.arange(s) for s in shape], indexing="ij")
            c = [rng.uniform(0.3, 0.7) * s for s in shape]
            r = rng.uniform(0.15, 0.3) * min(shape)
            d = np.sqrt((zz - c[0]) ** 2 + (yy - c[1]) ** 2 + (xx - c[2]) ** 2)
            lab = ((d < r).astype(np.uint8) + (d < r / 2)).clip(0, cfg["classes"] - 1).astype(np.uint8)
            img = (lab[None] + rng.normal(0, 0.5, (cfg["channels"], *shape))).astype(np.float16)
            vols.append((img, lab, "synthetic"))
        tasks[t] = vols
    return _freeze(tasks, 12345, {"synthetic": f"n{n}"})
