"""Reference submission: a position-dependent masking schedule that interpolates between MDLM and autoregression.

Position i (of T) is masked during its own time window [s_i, s_i + BETA] with s_i = (1 - BETA) · (1 - i / (T - 1)):
    alpha_i(t) = 1 - clamp((t - s_i) / BETA, 0, 1).
Later positions are masked earlier in the forward process, so the reverse process unmasks roughly from left to right,
with windows of positions unmasked together. BETA = 1 is MDLM's linear schedule; BETA -> 0 is autoregressive order.
The training loss is the bound itself under this schedule (dlm.mdlm_loss with alpha), which stays a valid
upper bound for any per-position schedule (Shi et al. 2024).
"""
from __future__ import annotations

import torch

import dlm

BETA = 0.5


def alpha(t, seq: int):
    i = torch.arange(seq, device=t.device, dtype=t.dtype)
    s = (1.0 - BETA) * (1.0 - i / max(seq - 1, 1))
    return 1.0 - ((t[:, None] - s[None, :]) / BETA).clamp(0.0, 1.0)


def loss(model, x):
    return dlm.mdlm_loss(model, x, alpha_fn=alpha)
