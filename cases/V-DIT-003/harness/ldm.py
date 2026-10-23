"""V-DIT-003 latent diffusion recipe (evaluator code, agent-visible): everything except the backbone is fixed.

Latents: Imagenette images centre-cropped and resized to 128 x 128, encoded by the Stable Diffusion VAE
(stabilityai/sd-vae-ft-ema, MIT; the posterior mean) and multiplied by 0.18215: arrays (4, 16, 16); each image is
stored as itself and mirrored.

Diffusion (as in DiT, Peebles & Xie 2023, without the learned variance): 1,000 steps, linear betas 1e-4 to 0.02, the
backbone predicts the noise, mean-squared error; the class label is replaced by the null class NULL = 10 with
probability 0.1. Training: AdamW (lr 1e-4, no weight decay, betas 0.9/0.999), batch 128, STEPS steps, EMA of the
weights with decay 0.9999 (used for sampling), bfloat16 autocast on CUDA.

Sampling: DDIM with 50 evenly spaced steps (eta 0), classifier-free guidance eps_u + w (eps_c - eps_u) with w = 1.5
(and w = 1 without guidance); the same number of images for every class; decoded by the VAE to uint8 images.

Cost: forward FLOPs of the backbone for one latent (torch.utils.flop_counter.FlopCounterMode, which counts matrix
products, convolutions and scaled_dot_product_attention), measured by the evaluator in training mode without
gradients.
"""
from __future__ import annotations

import copy
import math

import numpy as np

T, NULL, P_UNCOND, SCALE = 1000, 10, 0.1, 0.18215
STEPS, BATCH, LR, EMA = 30000, 128, 1e-4, 0.9999
DDIM_STEPS, CFG = 50, 1.5
LATENT = (4, 16, 16)


def alphas_cumprod() -> np.ndarray:
    betas = np.linspace(1e-4, 0.02, T, dtype=np.float64)
    return np.cumprod(1.0 - betas)


def ddim_timesteps(n: int = DDIM_STEPS) -> np.ndarray:
    """Descending timesteps used by DDIM (evenly spaced, as in the DDIM/DiT samplers)."""
    return np.arange(0, T, T // n)[::-1].copy()


def ddim_update(x, eps, a_t: float, a_prev: float):
    """One deterministic DDIM step from alpha-bar a_t to a_prev (works on numpy arrays and torch tensors)."""
    x0 = (x - math.sqrt(1 - a_t) * eps) / math.sqrt(a_t)
    return math.sqrt(a_prev) * x0 + math.sqrt(1 - a_prev) * eps


def labels_for(n: int, classes: int = 10) -> np.ndarray:
    return np.arange(n) % classes


def count_flops(net, device: str) -> int:
    """Forward FLOPs of the backbone for one latent (conditional, t = 500)."""
    import torch
    from torch.utils.flop_counter import FlopCounterMode
    x = torch.zeros(1, *LATENT, device=device)
    t = torch.full((1,), 500, device=device, dtype=torch.long)
    y = torch.zeros(1, device=device, dtype=torch.long)
    was = net.training
    net.train()                                  # avoids fused inference-only paths that the counter cannot see
    with torch.no_grad(), FlopCounterMode(display=False) as fc:
        net(x, t, y)
    net.train(was)
    return int(fc.get_total_flops())


def train(net, latents: np.ndarray, labels: np.ndarray, seed: int, device: str, steps: int | None = None) -> dict:
    """latents (N, 2, 4, 16, 16) float16 (image and its mirror), labels (N,). Returns {"ema", "curve", "diverged"}."""
    import torch
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    ab = torch.as_tensor(alphas_cumprod(), dtype=torch.float32, device=device)
    ema = copy.deepcopy(net).eval().requires_grad_(False)
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=0.0, betas=(0.9, 0.999))
    cuda = device.startswith("cuda")
    curve, diverged = [], 0
    n = int(steps or STEPS)
    net.train()
    for it in range(n):
        i = rng.integers(0, len(latents), BATCH)
        m = rng.integers(0, 2, BATCH)
        x0 = torch.as_tensor(latents[i, m].astype(np.float32), device=device)
        y = np.where(rng.random(BATCH) < P_UNCOND, NULL, labels[i])
        y = torch.as_tensor(y, device=device, dtype=torch.long)
        t = torch.as_tensor(rng.integers(0, T, BATCH), device=device, dtype=torch.long)
        noise = torch.randn_like(x0)
        a = ab[t].view(-1, 1, 1, 1)
        xt = a.sqrt() * x0 + (1 - a).sqrt() * noise
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=cuda):
            pred = net(xt, t, y)
        loss = ((pred.float() - noise) ** 2).mean()
        if not torch.isfinite(loss):
            diverged = 1
            break
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        with torch.no_grad():
            for pe, pn in zip(ema.parameters(), net.parameters()):
                pe.mul_(EMA).add_(pn.detach(), alpha=1 - EMA)
            for be, bn in zip(ema.buffers(), net.buffers()):
                be.copy_(bn)
        if it % 100 == 0:
            curve.append(float(loss))
    return {"ema": ema, "curve": np.asarray(curve), "diverged": diverged}


def sample(net, labels: np.ndarray, seed: int, device: str, cfg: float = CFG, batch: int = 250) -> np.ndarray:
    """DDIM samples (len(labels), 4, 16, 16) float32 from the EMA backbone; noise from torch.Generator(seed + b)."""
    import torch
    ab = alphas_cumprod()
    ts = ddim_timesteps()
    out = []
    cuda = device.startswith("cuda")
    for b in range(0, len(labels), batch):
        g = torch.Generator(device=device)
        g.manual_seed(int(seed) + b)
        y = torch.as_tensor(labels[b:b + batch], device=device, dtype=torch.long)
        x = torch.randn((len(y), *LATENT), generator=g, device=device)
        for k, t in enumerate(ts):
            a_t = float(ab[t])
            a_prev = float(ab[ts[k + 1]]) if k + 1 < len(ts) else 1.0
            tt = torch.full((len(y),), int(t), device=device, dtype=torch.long)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=cuda):
                if cfg == 1.0:
                    eps = net(x, tt, y).float()
                else:
                    e = net(torch.cat([x, x]), torch.cat([tt, tt]), torch.cat([y, torch.full_like(y, NULL)])).float()
                    ec, eu = e.chunk(2)
                    eps = eu + cfg * (ec - eu)
            x = ddim_update(x, eps, a_t, a_prev)
        out.append(x.float().cpu().numpy())
    return np.concatenate(out)


def decode(vae, z: np.ndarray, device: str, batch: int = 100) -> np.ndarray:
    """VAE decoding of latents to uint8 (N, 128, 128, 3)."""
    import torch
    imgs = []
    for i in range(0, len(z), batch):
        with torch.no_grad():
            x = vae.decode(torch.as_tensor(z[i:i + batch], device=device) / SCALE).sample
        imgs.append(((x.float().clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy())
    return np.concatenate(imgs)
