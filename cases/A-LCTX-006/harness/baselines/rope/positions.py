"""Baseline run by the evaluator under the same conditions as every submission: rotary position embedding (Su et al.
2021) with base 10,000 and no change beyond the training length."""
import torch
import torch.nn as nn


class RoPE(nn.Module):
    def __init__(self, head_dim: int, base: float = 10000.0):
        super().__init__()
        self.register_buffer("inv", 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim)), persistent=False)

    def angles(self, T: int, device):
        return torch.outer(torch.arange(T, device=device).float(), self.inv.to(device))

    @staticmethod
    def rotate(x, cos, sin):
        x1, x2 = x[..., 0::2].float(), x[..., 1::2].float()
        return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).flatten(-2)

    def forward(self, q, k):
        f = self.angles(q.shape[-2], q.device)
        cos, sin = f.cos(), f.sin()
        return self.rotate(q, cos, sin), self.rotate(k, cos, sin)


def make_positions(head_dim, n_heads, n_kv_heads, train_len):
    return RoPE(head_dim)
