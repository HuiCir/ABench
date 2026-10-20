"""V-SMP-002 frozen diffusion model and its noise schedule (used by the evaluator and passed to your sampler).

google/ddpm-cifar10-32 (Ho et al. 2020): a UNet2DModel trained with ε-prediction on CIFAR-10, linear β schedule from
1e-4 to 0.02 over 1,000 steps. The evaluator wraps it in a Denoiser that counts network evaluations (NFE).
"""
from __future__ import annotations

import numpy as np

N_TRAIN_STEPS = 1000
BETAS = np.linspace(1e-4, 0.02, N_TRAIN_STEPS, dtype=np.float64)
ALPHAS_CUMPROD = np.cumprod(1.0 - BETAS)            # ᾱ_t for t = 0 .. 999


class NFEExceeded(RuntimeError):
    pass


class Denoiser:
    """eps = denoiser(x, t): ε-prediction of the frozen UNet.

    x  float tensor (B, 3, 32, 32) in the model's data scale ([-1, 1] at t = 0)
    t  timesteps in [0, 999]: a Python number or a tensor of shape () or (B,); non-integer values are allowed (the
       time embedding is continuous) but the network was trained on integers
    Every call counts as one evaluation for the whole batch; calls beyond the budget raise NFEExceeded.
    """

    def __init__(self, unet, budget: int):
        self._unet = unet
        self.budget = int(budget)
        self.calls = 0
        self.alphas_cumprod = ALPHAS_CUMPROD.copy()

    def __call__(self, x, t):
        import torch
        if self.calls >= self.budget:
            raise NFEExceeded(f"more than {self.budget} network evaluations")
        self.calls += 1
        if not torch.is_tensor(t):
            t = torch.tensor(float(t), device=x.device)
        t = t.to(device=x.device, dtype=torch.float32)
        if t.ndim == 0:
            t = t.expand(x.shape[0])
        with torch.no_grad():
            return self._unet(x.float(), t).sample
