"""V-DIT-003: Imagenette (fastai, Apache-2.0; images from ImageNet) at 128 x 128, encoded by the Stable Diffusion VAE
(stabilityai/sd-vae-ft-ema @f04b2c4b, MIT), and the FID Inception weights released with pytorch-fid (Apache-2.0).

Public: latents.npz with the posterior-mean latents x 0.18215 of the 9,469 training images and of their mirror
images, (N, 2, 4, 16, 16) float16, and their labels; train_inception_stats.npz (Inception pool statistics of the
128-pixel training images, for the self-check). Models: the VAE and the Inception weights. Hidden: the Inception
statistics of the 3,925 validation images at 128 pixels and two sampling seeds drawn from ABENCH_V_DIT_SEED.
Needs torch, diffusers, Pillow and pytorch-fid in the operator environment.
"""
from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

import numpy as np

from prepare import imagenette as IN
from prepare.common import Frozen, check
from prepare.v_smp_002 import INCEPTION, SOURCES as _V_SMP_SOURCES, _stats

SECRET_SEED_ENV = "ABENCH_V_DIT_SEED"
VAE_REV = "f04b2c4b98319346dad8c65879f680b1997b204a"
VAE_FILES = ["config.json", "diffusion_pytorch_model.safetensors"]
SOURCES = IN.SOURCES + [_V_SMP_SOURCES[1]] + [
    {"name": "sd-vae-ft-ema/", "url": f"https://huggingface.co/stabilityai/sd-vae-ft-ema/tree/{VAE_REV}",
     "files": VAE_FILES, "license": "MIT", "note": "RAW/sd-vae-ft-ema/"}]
SIDE, SCALE = 128, 0.18215


def resize(x: np.ndarray) -> np.ndarray:
    from PIL import Image
    return np.stack([np.asarray(Image.fromarray(a).resize((SIDE, SIDE), Image.BICUBIC), np.uint8) for a in x])


def encode(x: np.ndarray, vae_dir: Path) -> np.ndarray:
    """(N, 128, 128, 3) uint8 -> (N, 2, 4, 16, 16) float16: the image and its mirror."""
    import torch
    from diffusers import AutoencoderKL
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    vae = AutoencoderKL.from_pretrained(str(vae_dir), local_files_only=True).to(dev).eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(x), 64):
            b = torch.as_tensor(x[i:i + 64], device=dev).permute(0, 3, 1, 2).float() / 127.5 - 1
            z = [vae.encode(v).latent_dist.mean * SCALE for v in (b, torch.flip(b, dims=[3]))]
            out.append(torch.stack(z, 1).half().cpu().numpy())
    return np.concatenate(out)


def build(raw: Path) -> Frozen:
    seed = int(os.environ.get(SECRET_SEED_ENV, "0"))
    if not seed:
        raise SystemExit(f"set {SECRET_SEED_ENV} to a secret integer (sampling seeds)")
    pub, hid, sums = IN.load(raw)
    sums[INCEPTION] = check(raw / INCEPTION)
    if not sums[INCEPTION].startswith("6726825d"):
        raise SystemExit(f"{INCEPTION}: sha256 {sums[INCEPTION]} does not start with 6726825d")
    sums.update({f"sd-vae-ft-ema/{f}": check(raw / "sd-vae-ft-ema" / f) for f in VAE_FILES})
    xtr, xva = resize(pub["x"]), resize(hid["x_test"])
    lat = encode(xtr, raw / "sd-vae-ft-ema")
    mu_t, s_t = _stats(xtr, raw / INCEPTION)
    mu_v, s_v = _stats(xva, raw / INCEPTION)
    inc_dir = Path(tempfile.mkdtemp(prefix="inception-"))
    shutil.copyfile(raw / INCEPTION, inc_dir / INCEPTION)
    seeds = np.random.default_rng(seed).integers(1, 2**31 - 1, 2)
    return Frozen(public={"latents.npz": {"latents": lat, "labels": pub["y"]},
                          "train_inception_stats.npz": {"mu": mu_t, "sigma": s_t}},
                  hidden={"seeds": seeds.astype(np.int64), "ref_mu": mu_v, "ref_sigma": s_v}, sources_sha256=sums,
                  models={"sd-vae-ft-ema": raw / "sd-vae-ft-ema", "inception": inc_dir})


def synthetic(n: int) -> Frozen:
    rng = np.random.default_rng(0)
    y = np.repeat(np.arange(10), max(n // 10, 2))
    lat = (rng.normal(size=(len(y), 1, 4, 16, 16)) + y[:, None, None, None, None] / 10).astype(np.float16)
    lat = np.concatenate([lat, lat[..., ::-1]], 1)
    mu, a = rng.normal(size=2048), rng.normal(size=(2048, 64))
    return Frozen(public={"latents.npz": {"latents": lat, "labels": y}, "train_inception_stats.npz": {"mu": mu, "sigma": a @ a.T}},
                  hidden={"seeds": np.asarray([5, 6], np.int64), "ref_mu": mu + 0.1, "ref_sigma": a @ a.T},
                  sources_sha256={"synthetic": f"n{n}"})
