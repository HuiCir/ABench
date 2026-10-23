"""T-SPEC-003 frozen diffusion model and its noise schedule (the model of V-SMP-002), with a denoiser that counts
sequential rounds instead of single evaluations.

google/ddpm-cifar10-32 (Ho et al. 2020): a UNet2DModel trained with ε-prediction on CIFAR-10, linear β schedule from
1e-4 to 0.02 over 1,000 steps.
"""
from __future__ import annotations

import numpy as np

N_TRAIN_STEPS = 1000
BETAS = np.linspace(1e-4, 0.02, N_TRAIN_STEPS, dtype=np.float64)
ALPHAS_CUMPROD = np.cumprod(1.0 - BETAS)            # ᾱ_t for t = 0 .. 999


class BudgetExceeded(RuntimeError):
    pass


class ParallelDenoiser:
    """eps = denoiser(x, t): ε-prediction of the frozen UNet for up to `max_parallel` inputs per image in one round.

    x  float tensor (k·n, 3, 32, 32) in the model's data scale, k = 1 .. max_parallel, where n is the number of images
       the sampler was asked for (rows can be any states you like, e.g. k candidate points of each image's trajectory)
    t  timesteps in [0, 999]: a Python number or a tensor of shape () or (k·n,)
    Every call is one sequential round, whatever k is. Rounds beyond `rounds`, or more than max_parallel·n rows in one
    call, raise BudgetExceeded. `evaluations` counts rows / n over all calls.
    """

    def __init__(self, unet, n: int, rounds: int, max_parallel: int):
        self._unet = unet
        self.n, self.budget, self.max_parallel = int(n), int(rounds), int(max_parallel)
        self.rounds, self.evaluations = 0, 0.0
        self.alphas_cumprod = ALPHAS_CUMPROD.copy()

    def __call__(self, x, t):
        import torch
        if self.rounds >= self.budget:
            raise BudgetExceeded(f"more than {self.budget} rounds")
        if x.shape[0] > self.max_parallel * self.n:
            raise BudgetExceeded(f"{x.shape[0]} rows in one round; at most {self.max_parallel} x {self.n}")
        self.rounds += 1
        self.evaluations += x.shape[0] / self.n
        if not torch.is_tensor(t):
            t = torch.tensor(float(t), device=x.device)
        t = t.to(device=x.device, dtype=torch.float32)
        if t.ndim == 0:
            t = t.expand(x.shape[0])
        with torch.no_grad():
            return self._unet(x.float(), t).sample
