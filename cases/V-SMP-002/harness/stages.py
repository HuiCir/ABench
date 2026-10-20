"""V-SMP-002 pipeline: training-free sampling from a frozen CIFAR-10 DDPM at small NFE, scored by FID.

Stage "sample" (imports sampler.py)
  setup   load the frozen UNet to the GPU (untimed)
  timed   import sampler.py
  finish  for each NFE in (5, 10, 20): N images from sample(denoiser, ...) in batches, each batch with its own
          seeded generator; the denoiser refuses calls beyond the NFE budget; images saved as uint8 for the next stage
Stage "fid" (does not import your code; receives only the image files)
  finish  Inception-v3 pool features (pytorch-fid weights) → mean and covariance per NFE; the evaluator computes FID
          against the CIFAR-10 training-set statistics
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

from abench_stage import Context, Stage

NFES = (5, 10, 20)
PARAMS = {"nfes": list(NFES), "n_samples": 10000, "batch": 500, "fid_reference": "CIFAR-10 training set (50,000 images)",
          "inception": "pytorch-fid InceptionV3 pool3 (pt_inception-2015-12-05-6726825d.pth)"}
HELD_OUT_KEYS = ["seeds"]                      # one base seed per NFE; batch b of run seed s uses base + 100003·s + b
PIPELINE = [
    {"stage": "sample", "timeout": 1200, "finish_cap": 900, "tail_slack": 20, "threads": 8, "held_out": True},
    {"stage": "fid", "timeout": 600, "finish_cap": 300, "tail_slack": 20, "threads": 8, "held_out": False,
     "carry": {"sample": [f"samples_{n}.npy" for n in NFES]}},
]
UNET_DIR, INCEPTION_FILE = "ddpm-cifar10-32", "pt_inception-2015-12-05-6726825d.pth"


def _overrides(ctx: Context) -> dict:
    p = ctx.public / "overrides.json" if ctx.public else None
    return json.loads(p.read_text()) if p and p.is_file() else {}


def _setup_sample(ctx: Context):
    from diffusers import UNet2DModel
    unet = UNet2DModel.from_pretrained(str(ctx.models / UNET_DIR)).to(ctx.device).eval()
    return {"unet": unet, "prm": dict(PARAMS, **_overrides(ctx))}


def _timed_sample(ctx: Context, st):
    st["mod"] = ctx.import_submission("sampler.py", required=("sample",))
    return st


def _finish_sample(ctx: Context, st):
    import torch
    from diffusion import Denoiser
    prm, mod = st["prm"], st["mod"]
    with np.load(ctx.post_dir / "held_out.npz", allow_pickle=False) as z:
        seeds = z["seeds"].astype(np.int64)
    used = []
    for nfe, base in zip(NFES, seeds):
        imgs, max_calls = [], 0
        for b in range(prm["n_samples"] // prm["batch"]):
            g = torch.Generator(device=ctx.device)
            g.manual_seed((int(base) + 100003 * int(ctx.seed) + b) % (2**63 - 1))
            den = Denoiser(st["unet"], nfe)
            x = mod.sample(den, prm["batch"], nfe, g, ctx.device)
            if tuple(x.shape) != (prm["batch"], 3, 32, 32) or not torch.isfinite(x).all():
                raise SystemExit(f"sample() returned {tuple(x.shape)} or non-finite values")
            max_calls = max(max_calls, den.calls)
            u8 = ((x.float().clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
            imgs.append(u8)
        np.save(ctx.work / f"samples_{nfe}.npy", np.concatenate(imgs))
        used.append(max_calls)
    return {"nfe_used": np.asarray(used, np.int64), "nfe_budget": np.asarray(NFES, np.int64)}


def _finish_fid(ctx: Context, _):
    import torch
    from pytorch_fid.inception import InceptionV3
    hub = Path(torch.hub.get_dir()) / "checkpoints"
    hub.mkdir(parents=True, exist_ok=True)
    if not (hub / INCEPTION_FILE).exists():
        shutil.copyfile(ctx.models / "inception" / INCEPTION_FILE, hub / INCEPTION_FILE)
    net = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]]).to(ctx.device).eval()
    out = {}
    for nfe in NFES:
        imgs = np.load(ctx.work / f"samples_{nfe}.npy")
        feats = []
        with torch.no_grad():
            for i in range(0, len(imgs), 500):
                x = torch.from_numpy(imgs[i:i + 500]).to(ctx.device).permute(0, 3, 1, 2).float() / 255.0
                feats.append(net(x)[0].squeeze(-1).squeeze(-1).double().cpu().numpy())
        f = np.concatenate(feats)
        out[f"mu_{nfe}"], out[f"sigma_{nfe}"] = f.mean(0), np.cov(f, rowvar=False)
    return out


STAGES = {"sample": Stage(setup=_setup_sample, timed=_timed_sample, finish=_finish_sample),
          "fid": Stage(timed=lambda ctx, s: s, finish=_finish_fid)}


def make_dev(public: Path, out: Path, quick: bool = False) -> dict:
    """Dev setting for selfcheck.py: public seeds, 2,000 samples per NFE (500 with --quick). FID from fewer samples is
    biased upwards; compare dev FIDs with each other, not with the held-out numbers."""
    (out / "public").mkdir(parents=True, exist_ok=True)
    (out / "public" / "overrides.json").write_text(json.dumps({"n_samples": 500 if quick else 2000}))
    np.savez(out / "held_out.npz", seeds=np.asarray([11, 22, 33], np.int64))
    with np.load(public / "cifar10_train_inception_stats.npz", allow_pickle=False) as z:
        ref = {"ref_mu": z["mu"], "ref_sigma": z["sigma"]}
    return {"public": out / "public", "held_out": out / "held_out.npz", "reference": ref, "time_scale": 1.0}
