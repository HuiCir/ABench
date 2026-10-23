"""V-DST-004 pipeline: images from your trained generator, scored by FID against CIFAR-10 (evaluator and self-check).

Stage "sample" (imports model.py; the teacher is not available, only checkpoint/)
  timed   load(checkpoint_dir, device) -> net, a torch.nn.Module with at most PARAMS["max_params"] parameters
  finish  for each step budget S in (1, 2, 4): N images from generate(net, n, S, generator, device) in batches, each batch
          with its own seeded generator; a forward hook counts calls of net during each generate() (at most S);
          images saved as uint8 for the next stage
Stage "fid" (does not import your code; receives only the image files)
  finish  Inception-v3 pool features (pytorch-fid weights) -> mean and covariance per budget; the evaluator computes
          FID against the CIFAR-10 training-set statistics
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

from abench_stage import Context, Stage

STEPS = (1, 2, 4)
PARAMS = {"steps": list(STEPS), "n_samples": 10000, "batch": 500, "max_params": 72_000_000,
          "fid_reference": "CIFAR-10 training set (50,000 images)",
          "inception": "pytorch-fid InceptionV3 pool3 (pt_inception-2015-12-05-6726825d.pth)"}
HELD_OUT_KEYS = ["seeds"]                      # one base seed per budget; batch b of run seed s uses base + 100003·s + b
PIPELINE = [
    {"stage": "sample", "timeout": 300 + 600, "finish_cap": 600, "tail_slack": 20, "threads": 8,
     "held_out": True, "models": False},
    {"stage": "fid", "timeout": 600, "finish_cap": 300, "tail_slack": 20, "threads": 8, "held_out": False,
     "carry": {"sample": [f"samples_{s}.npy" for s in STEPS]}},
]
INCEPTION_FILE = "pt_inception-2015-12-05-6726825d.pth"


def _overrides(ctx: Context) -> dict:
    p = ctx.public / "overrides.json" if ctx.public else None
    return json.loads(p.read_text()) if p and p.is_file() else {}


def _timed_sample(ctx: Context, _):
    import torch
    mod = ctx.import_submission("model.py", required=("load", "generate"))
    net = mod.load(str(ctx.submission / "checkpoint"), ctx.device)
    if not isinstance(net, torch.nn.Module):
        raise SystemExit("load() must return a torch.nn.Module")
    return {"mod": mod, "net": net, "prm": dict(PARAMS, **_overrides(ctx))}


def _finish_sample(ctx: Context, st):
    import torch
    prm, mod, net = st["prm"], st["mod"], st["net"]
    n_params = sum(p.numel() for p in net.parameters()) + sum(b.numel() for b in net.buffers())
    calls = [0]
    net.register_forward_hook(lambda *_: calls.__setitem__(0, calls[0] + 1))
    with np.load(ctx.post_dir / "held_out.npz", allow_pickle=False) as z:
        seeds = z["seeds"].astype(np.int64)
    used = []
    for steps, base in zip(STEPS, seeds):
        imgs, max_calls = [], 0
        for b in range(prm["n_samples"] // prm["batch"]):
            g = torch.Generator(device=ctx.device)
            g.manual_seed((int(base) + 100003 * int(ctx.seed) + b) % (2**63 - 1))
            calls[0] = 0
            with torch.no_grad():
                x = mod.generate(net, prm["batch"], steps, g, ctx.device)
            if tuple(x.shape) != (prm["batch"], 3, 32, 32) or not torch.isfinite(x).all():
                raise SystemExit(f"generate() returned {tuple(x.shape)} or non-finite values")
            max_calls = max(max_calls, calls[0])
            u8 = ((x.float().clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
            imgs.append(u8)
        np.save(ctx.work / f"samples_{steps}.npy", np.concatenate(imgs))
        used.append(max_calls)
    return {"calls_used": np.asarray(used, np.int64), "n_params": np.asarray([n_params], np.int64)}


def _finish_fid(ctx: Context, _):
    import torch
    from pytorch_fid.inception import InceptionV3
    hub = Path(torch.hub.get_dir()) / "checkpoints"
    hub.mkdir(parents=True, exist_ok=True)
    if not (hub / INCEPTION_FILE).exists():
        shutil.copyfile(ctx.models / "inception" / INCEPTION_FILE, hub / INCEPTION_FILE)
    net = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]]).to(ctx.device).eval()
    out = {}
    for steps in STEPS:
        imgs = np.load(ctx.work / f"samples_{steps}.npy")
        feats = []
        with torch.no_grad():
            for i in range(0, len(imgs), 500):
                x = torch.from_numpy(imgs[i:i + 500]).to(ctx.device).permute(0, 3, 1, 2).float() / 255.0
                feats.append(net(x)[0].squeeze(-1).squeeze(-1).double().cpu().numpy())
        f = np.concatenate(feats)
        out[f"mu_{steps}"], out[f"sigma_{steps}"] = f.mean(0), np.cov(f, rowvar=False)
    return out


STAGES = {"sample": Stage(timed=_timed_sample, finish=_finish_sample),
          "fid": Stage(timed=lambda ctx, s: s, finish=_finish_fid)}


def make_dev(public: Path, out: Path, quick: bool = False) -> dict:
    """Dev setting for selfcheck.py: public seeds, 2,000 samples per budget (500 with --quick). FID from fewer samples is
    biased upwards; compare dev FIDs with each other, not with the held-out numbers."""
    (out / "public").mkdir(parents=True, exist_ok=True)
    (out / "public" / "overrides.json").write_text(json.dumps({"n_samples": 500 if quick else 2000}))
    np.savez(out / "held_out.npz", seeds=np.asarray([11, 22, 33], np.int64))
    with np.load(public / "cifar10_train_inception_stats.npz", allow_pickle=False) as z:
        ref = {"ref_mu": z["mu"], "ref_sigma": z["sigma"]}
    return {"public": out / "public", "held_out": out / "held_out.npz", "reference": ref, "time_scale": 1.0}
