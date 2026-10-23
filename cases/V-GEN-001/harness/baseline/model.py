"""V-GEN-001 reference submission, model code: the DDPM of Ho et al. (2020) for CIFAR-10 — a diffusers UNet2DModel with
the architecture of google/ddpm-cifar10-32 (about 35.7M parameters), ε-prediction, linear β schedule from 1e-4 to
0.02 over 1,000 steps — trained from scratch by train.py; images are generated with deterministic DDIM (Song et al.
2021) on uniformly spaced timesteps, one network evaluation per step.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

UNET_CONFIG = {"sample_size": 32, "in_channels": 3, "out_channels": 3, "layers_per_block": 2,
               "block_out_channels": [128, 256, 256, 256],
               "down_block_types": ["DownBlock2D", "AttnDownBlock2D", "DownBlock2D", "DownBlock2D"],
               "up_block_types": ["UpBlock2D", "UpBlock2D", "AttnUpBlock2D", "UpBlock2D"],
               "act_fn": "silu", "norm_eps": 1e-6, "norm_num_groups": 32, "flip_sin_to_cos": False, "freq_shift": 1,
               "downsample_padding": 0, "time_embedding_type": "positional", "dropout": 0.1}
BETAS = np.linspace(1e-4, 0.02, 1000, dtype=np.float64)
ALPHAS_CUMPROD = np.cumprod(1.0 - BETAS)


def build():
    from diffusers import UNet2DModel
    return UNet2DModel(**UNET_CONFIG)


def load(checkpoint_dir, device="cuda"):
    net = build()
    sd = torch.load(Path(checkpoint_dir) / "ema_unet.pt", map_location="cpu", weights_only=True)
    net.load_state_dict(sd, strict=True)
    return net.to(device).eval()


def generate(net, n: int, steps: int, generator: torch.Generator, device: str) -> torch.Tensor:
    ac = torch.tensor(ALPHAS_CUMPROD, dtype=torch.float64, device=device)
    ts = np.round(np.linspace(999, 0, steps)).astype(np.int64)
    x = torch.randn(n, 3, 32, 32, generator=generator, device=device, dtype=torch.float64)
    for i, t in enumerate(ts):
        a_t = ac[t]
        a_prev = ac[ts[i + 1]] if i + 1 < len(ts) else torch.tensor(1.0, dtype=torch.float64, device=device)
        eps = net(x.float(), torch.full((n,), float(t), device=device)).sample.double()
        x0 = ((x - (1 - a_t).sqrt() * eps) / a_t.sqrt()).clamp(-1, 1)
        x = a_prev.sqrt() * x0 + (1 - a_prev).sqrt() * eps
    return x.float().clamp(-1, 1)
