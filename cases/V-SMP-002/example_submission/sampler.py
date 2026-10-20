"""V-SMP-002 reference submission (identical to the evaluator's baseline): deterministic DDIM (Song et al. 2021,
arXiv:2010.02502) with uniformly spaced timesteps and x0 clipping. Uses exactly `nfe` network evaluations."""
from __future__ import annotations

import numpy as np
import torch


def sample(denoiser, n: int, nfe: int, generator: torch.Generator, device: str) -> torch.Tensor:
    ac = torch.tensor(denoiser.alphas_cumprod, dtype=torch.float64, device=device)
    ts = np.round(np.linspace(999, 0, nfe)).astype(np.int64)
    x = torch.randn(n, 3, 32, 32, generator=generator, device=device, dtype=torch.float64)
    for i, t in enumerate(ts):
        a_t = ac[t]
        a_prev = ac[ts[i + 1]] if i + 1 < len(ts) else torch.tensor(1.0, dtype=torch.float64, device=device)
        eps = denoiser(x.float(), int(t)).double()
        x0 = ((x - (1 - a_t).sqrt() * eps) / a_t.sqrt()).clamp(-1, 1)
        x = a_prev.sqrt() * x0 + (1 - a_prev).sqrt() * eps
    return x.float().clamp(-1, 1)
