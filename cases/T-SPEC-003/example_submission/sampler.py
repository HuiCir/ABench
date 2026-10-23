"""T-SPEC-003 reference submission: windowed Picard iteration of DDIM (draft and verify in parallel rounds).

The DDIM update on a grid of N = 2R timesteps is affine in the state given the network's ε:
    x_{j+1} = c_j x_j + d_j ε(x_j, t_j),  c_j = sqrt(ᾱ_{j+1} / ᾱ_j),  d_j = sqrt(1 − ᾱ_{j+1}) − sqrt(ᾱ_{j+1}(1 − ᾱ_j) / ᾱ_j)
Each round evaluates ε in parallel at the current guesses of W = 4 consecutive states of the trajectory (the draft),
then recomputes those states by the recursion from the window start (the verification), and slides the window by 2.
Every state is evaluated twice, first as a draft and then refined, so R rounds cover 2R DDIM steps with 4 evaluations
per image per round; states are only approximately verified (no convergence test), and x0 is not clipped.
"""
from __future__ import annotations

import numpy as np
import torch

WINDOW, ADVANCE = 4, 2


def coefficients(ac: np.ndarray, rounds: int):
    n_steps = ADVANCE * rounds
    ts = np.round(np.linspace(999, 0, n_steps)).astype(np.int64)
    a = np.append(ac[ts], 1.0)                          # ᾱ at each grid point; the last state is the clean image
    c = np.sqrt(a[1:] / a[:-1])
    d = np.sqrt(1 - a[1:]) - np.sqrt(a[1:] * (1 - a[:-1]) / a[:-1])
    return ts, c, d


def sample(denoiser, n: int, rounds: int, generator: torch.Generator, device: str) -> torch.Tensor:
    ts, c, d = coefficients(np.asarray(denoiser.alphas_cumprod), rounds)
    N = len(ts)
    x0 = torch.randn(n, 3, 32, 32, generator=generator, device=device, dtype=torch.float64)
    X = [x0.clone() for _ in range(N + 1)]               # current guesses of the whole trajectory
    s = 0
    while s < N:
        idx = list(range(s, min(s + WINDOW, N)))
        rows = torch.cat([X[j].float() for j in idx])
        tt = torch.cat([torch.full((n,), float(ts[j]), device=device) for j in idx])
        eps = denoiser(rows, tt).double().split(n)
        for k, j in enumerate(idx):                      # verify: recompute the window from its start
            X[j + 1] = c[j] * X[j] + d[j] * eps[k]
        for j in range(idx[-1] + 2, N + 1):              # states beyond the window start from the newest guess
            X[j] = X[idx[-1] + 1].clone()
        s += ADVANCE
    return X[N].float().clamp(-1, 1)
