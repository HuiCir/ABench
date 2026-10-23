"""V-DST-004 reference submission, model code: a consistency model (Song et al. 2023) distilled from the frozen
CIFAR-10 DDPM teacher. The student has the teacher's architecture (diffusers UNet2DModel, about 35.7M parameters) and is
initialised from its weights; it predicts ε, and its consistency function is the implied clean image
f(x, t) = (x − sqrt(1 − ᾱ_t) ε̂(x, t)) / sqrt(ᾱ_t), clipped to [-1, 1]. Generation with S steps evaluates f at S
timesteps, re-noising the estimate in between (multistep consistency sampling).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

UNET_CONFIG = {"sample_size": 32, "in_channels": 3, "out_channels": 3, "layers_per_block": 2,
               "block_out_channels": [128, 256, 256, 256],
               "down_block_types": ["DownBlock2D", "AttnDownBlock2D", "DownBlock2D", "DownBlock2D"],
               "up_block_types": ["UpBlock2D", "UpBlock2D", "AttnUpBlock2D", "UpBlock2D"],
               "act_fn": "silu", "norm_eps": 1e-6, "norm_num_groups": 32, "flip_sin_to_cos": False, "freq_shift": 1,
               "downsample_padding": 0, "time_embedding_type": "positional"}
ALPHAS_CUMPROD = np.cumprod(1.0 - np.linspace(1e-4, 0.02, 1000, dtype=np.float64))


def build():
    from diffusers import UNet2DModel
    return UNet2DModel(**UNET_CONFIG)


def consistency(net, x, t, ac):
    """f(x, t): the student's estimate of the clean image; t a (B,) long tensor, ac the ᾱ table on the device."""
    a = ac[t].float()[:, None, None, None]
    eps = net(x, t).sample
    return ((x - (1 - a).sqrt() * eps) / a.sqrt()).clamp(-1, 1)


def schedule(steps: int) -> np.ndarray:
    return np.round(np.linspace(999, 0, steps + 1)[:-1]).astype(np.int64)        # 1: [999]; 2: [999, 500]; ...


def load(checkpoint_dir, device="cuda"):
    net = build()
    net.load_state_dict(torch.load(Path(checkpoint_dir) / "student.pt", map_location="cpu", weights_only=True), strict=True)
    return net.to(device).eval()


def generate(net, n: int, steps: int, generator: torch.Generator, device: str) -> torch.Tensor:
    ac = torch.tensor(ALPHAS_CUMPROD, dtype=torch.float64, device=device)
    ts = schedule(steps)
    x = torch.randn(n, 3, 32, 32, generator=generator, device=device)
    for i, t in enumerate(ts):
        tt = torch.full((n,), int(t), device=device, dtype=torch.long)
        if i > 0:
            a = ac[tt].float()[:, None, None, None]
            x = a.sqrt() * x0 + (1 - a).sqrt() * torch.randn(x0.shape, generator=generator, device=device)
        x0 = consistency(net, x, tt, ac)
    return x0
