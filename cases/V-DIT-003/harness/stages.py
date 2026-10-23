"""V-DIT-003 pipeline: the evaluator trains your latent-diffusion backbone with the fixed recipe of ldm.py on Imagenette
latents, samples with DDIM and classifier-free guidance, and measures FID (evaluator and self-check use this file).

Stage "train" (imports backbone.py):
  setup   load the public latents (untimed)
  timed   net = make_backbone((4, 16, 16), 10); forward FLOPs per latent; ldm.train for ldm.STEPS steps
  finish  with the EMA weights: N_SAMPLES latents with guidance 1.5 and N_SAMPLES without, from the held-out seeds
          (500 per class), saved for the next stage; FLOPs, parameters, loss curve, divergence           (≤ 900 s)
Stage "fid" (does not import your code; receives only the latent files)
  finish  VAE decoding to 128 x 128 images, Inception-v3 pool features (pytorch-fid weights) -> mean and covariance
          per guidance setting; the evaluator computes FID against the Imagenette validation images
The evaluator trains DiT-S/2 (baselines/dit_s) through the same stages on the same seed.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

import ldm
from abench_stage import Context, Stage

N_SAMPLES, GUIDANCE = 5000, (1.5, 1.0)
INCEPTION_FILE = "pt_inception-2015-12-05-6726825d.pth"
PARAMS = {"steps": ldm.STEPS, "batch": ldm.BATCH, "lr": ldm.LR, "ema": ldm.EMA, "ddim_steps": ldm.DDIM_STEPS,
          "guidance": list(GUIDANCE), "n_samples": N_SAMPLES, "latent": list(ldm.LATENT), "max_flops_ratio": 1.10,
          "fid_reference": "Imagenette validation images (3,925), 128 x 128", "vae": "stabilityai/sd-vae-ft-ema"}
HELD_OUT_KEYS = ["seeds"]
TAGS = {1.5: "cfg15", 1.0: "cfg10"}
PIPELINE = [
    {"stage": "train", "timeout": 3600 + 900, "finish_cap": 900, "tail_slack": 20, "threads": 8,
     "held_out": True, "models": False},
    {"stage": "fid", "timeout": 900, "finish_cap": 600, "tail_slack": 20, "threads": 8, "held_out": False,
     "carry": {"train": [f"latents_{t}.npy" for t in TAGS.values()]}},
]


def _overrides(ctx: Context) -> dict:
    p = ctx.public / "overrides.json" if ctx.public else None        # written only by the self-check
    return json.loads(p.read_text()) if p and p.is_file() else {}


def _setup(ctx: Context):
    with np.load(ctx.public / "latents.npz", allow_pickle=False) as z:
        return {"lat": z["latents"], "y": z["labels"].astype(np.int64), "ov": _overrides(ctx)}


def _timed(ctx: Context, st):
    import torch
    mod = ctx.import_submission("backbone.py", required=("make_backbone",))
    torch.manual_seed(ctx.seed)
    net = mod.make_backbone(latent_shape=ldm.LATENT, num_classes=10).to(ctx.device)
    flops = ldm.count_flops(net, ctx.device)
    res = ldm.train(net, st["lat"], st["y"], ctx.seed, ctx.device, st["ov"].get("steps"))
    return {"net": net, "res": res, "flops": flops, "n": int(st["ov"].get("n_samples", N_SAMPLES))}


def _finish(ctx: Context, st):
    res = st["res"]
    out = {"flops": np.asarray([st["flops"]], np.int64), "diverged": np.asarray([res["diverged"]]),
           "params": np.asarray([sum(p.numel() for p in st["net"].parameters())], np.int64), "curve": res["curve"]}
    with np.load(ctx.post_dir / "held_out.npz", allow_pickle=False) as z:
        seeds = z["seeds"].astype(np.int64)
    labels = ldm.labels_for(st["n"])
    for (w, tag), s in zip(TAGS.items(), seeds):
        z = ldm.sample(res["ema"], labels, int(s) + 1000003 * ctx.seed, ctx.device, cfg=w)
        if not np.isfinite(z).all():
            z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
            out["nonfinite_samples"] = np.asarray([1])
        np.save(ctx.work / f"latents_{tag}.npy", z.astype(np.float32))
    return out


def _finish_fid(ctx: Context, _):
    import torch
    from diffusers import AutoencoderKL
    from pytorch_fid.inception import InceptionV3
    hub = Path(torch.hub.get_dir()) / "checkpoints"
    hub.mkdir(parents=True, exist_ok=True)
    if not (hub / INCEPTION_FILE).exists():
        shutil.copyfile(ctx.models / "inception" / INCEPTION_FILE, hub / INCEPTION_FILE)
    vae = AutoencoderKL.from_pretrained(str(ctx.models / "sd-vae-ft-ema"), local_files_only=True).to(ctx.device).eval()
    net = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]]).to(ctx.device).eval()
    out = {}
    for tag in TAGS.values():
        imgs = ldm.decode(vae, np.load(ctx.work / f"latents_{tag}.npy"), ctx.device)
        feats = []
        with torch.no_grad():
            for i in range(0, len(imgs), 500):
                x = torch.from_numpy(imgs[i:i + 500]).to(ctx.device).permute(0, 3, 1, 2).float() / 255.0
                feats.append(net(x)[0].squeeze(-1).squeeze(-1).double().cpu().numpy())
        f = np.concatenate(feats)
        out[f"mu_{tag}"], out[f"sigma_{tag}"] = f.mean(0), np.cov(f, rowvar=False)
    return out


STAGES = {"train": Stage(setup=_setup, timed=_timed, finish=_finish),
          "fid": Stage(timed=lambda ctx, s: s, finish=_finish_fid)}


def make_dev(public: Path, out: Path, quick: bool = False) -> dict:
    """Self-check: public seeds, 1,000 samples per setting against the training images' statistics (--quick: 2,000
    steps and 200 samples). FID from fewer samples is biased upwards: compare dev FIDs with each other only."""
    pub = out / "public"
    pub.mkdir(parents=True, exist_ok=True)
    (pub / "latents.npz").symlink_to((public / "latents.npz").resolve())
    (pub / "overrides.json").write_text(json.dumps({"n_samples": 200, "steps": 2000} if quick else {"n_samples": 1000}))
    np.savez(out / "held_out.npz", seeds=np.asarray([31, 41], np.int64))
    with np.load(public / "train_inception_stats.npz", allow_pickle=False) as z:
        ref = {"ref_mu": z["mu"], "ref_sigma": z["sigma"]}
    return {"public": pub, "held_out": out / "held_out.npz", "reference": ref, "time_scale": 1.0}
