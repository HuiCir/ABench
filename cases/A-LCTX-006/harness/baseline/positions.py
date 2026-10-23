"""A-LCTX-006 reference: RoPE with YaRN scaling beyond the training length (Peng et al. 2023).

Up to train_len tokens this is plain RoPE (base 10,000). For a sequence of T > train_len tokens, with s = T / train_len:
each rotary frequency theta_i is interpolated by "NTK-by-parts": with r_i = train_len * theta_i / (2 pi) (how many
turns the dimension makes over the training length), gamma_i = clamp((r_i - 1) / (32 - 1), 0, 1), and
theta_i' = (1 - gamma_i) * theta_i / s + gamma_i * theta_i, so slow dimensions are interpolated and fast ones kept;
queries and keys are both multiplied by sqrt(1/t) = 0.1 ln(s) + 1 (YaRN's attention temperature). The extension phase at 4,096 tokens (s = 4) trains
the model under this scaling, as YaRN's fine-tuning does.
"""
import math

import torch
import torch.nn as nn

ALPHA, BETA = 1.0, 32.0


class YaRN(nn.Module):
    def __init__(self, head_dim: int, train_len: int, base: float = 10000.0):
        super().__init__()
        self.train_len = train_len
        self.register_buffer("theta", 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim)), persistent=False)

    def freqs(self, T: int):
        if T <= self.train_len:
            return self.theta, 1.0
        s = T / self.train_len
        r = self.train_len * self.theta / (2 * math.pi)
        gamma = ((r - ALPHA) / (BETA - ALPHA)).clamp(0.0, 1.0)
        return (1 - gamma) * self.theta / s + gamma * self.theta, 0.1 * math.log(s) + 1.0

    def forward(self, q, k):
        T = q.shape[-2]
        th, temp = self.freqs(T)
        f = torch.outer(torch.arange(T, device=q.device).float(), th.to(q.device))
        cos, sin = f.cos(), f.sin()

        def rot(x):
            x1, x2 = x[..., 0::2].float(), x[..., 1::2].float()
            return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).flatten(-2)
        return rot(q) * temp, rot(k) * temp


def make_positions(head_dim, n_heads, n_kv_heads, train_len):
    return YaRN(head_dim, train_len)
