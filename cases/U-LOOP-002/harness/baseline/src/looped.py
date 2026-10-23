"""Recurrent-depth model of the reference submission (shared by model.py and train.py)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

import reason_tasks as T


class Layer(nn.Module):
    def __init__(self, d: int = 256, heads: int = 4, mlp: int = 1024):
        super().__init__()
        self.h = heads
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.o = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.fc1, self.fc2 = nn.Linear(d, mlp), nn.Linear(mlp, d)

    def forward(self, x):
        B, L, D = x.shape
        q, k, v = self.qkv(self.n1(x)).view(B, L, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(B, L, D)
        x = x + self.o(a)
        return x + self.fc2(F.gelu(self.fc1(self.n2(x))))


class LoopedModel(nn.Module):
    """embed -> repeat core(h, e) -> head on the answer token (position 0)."""

    def __init__(self, d: int = 256, heads: int = 4, mlp: int = 1024):
        super().__init__()
        self.d = d
        self.tok = nn.ModuleDict({t: nn.Embedding(T.VOCAB[t], d) for t in T.TASKS})
        self.pos = nn.ParameterDict({t: nn.Parameter(torch.randn(T.SEQ_LEN[t] + 1, d) * 0.02) for t in T.TASKS})
        self.ans = nn.ParameterDict({t: nn.Parameter(torch.randn(d) * 0.02) for t in T.TASKS})
        self.adapter = nn.Linear(2 * d, d)
        self.layer = Layer(d, heads, mlp)
        self.norm = nn.LayerNorm(d)
        self.heads = nn.ModuleDict({t: nn.Linear(d, T.N_CLASSES[t]) for t in T.TASKS})

    def embed(self, task: str, x: torch.Tensor) -> torch.Tensor:
        e = self.tok[task](x.long())
        a = self.ans[task].expand(len(x), 1, self.d)
        return torch.cat([a, e], 1) + self.pos[task][None]

    def core(self, h: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        return self.layer(self.adapter(torch.cat([h, e], -1)))

    def head(self, task: str, h: torch.Tensor) -> torch.Tensor:
        return self.heads[task](self.norm(h[:, 0]))


@torch.no_grad()
def measure_costs(net: LoopedModel, task: str, seq_len: int, device) -> tuple[float, float]:
    """FLOPs per problem of one core iteration and of one readout, counted as the evaluator counts them."""
    from torch.utils.flop_counter import FlopCounterMode
    x = torch.zeros(2, seq_len, dtype=torch.long, device=device)
    e = net.embed(task, x)
    with FlopCounterMode(display=False) as fc:
        h = net.core(e, e)
    step_c = fc.get_total_flops() / 2
    with FlopCounterMode(display=False) as fc:
        net.head(task, h)
    return float(step_c), float(fc.get_total_flops() / 2)
