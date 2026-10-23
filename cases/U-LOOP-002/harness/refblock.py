"""U-LOOP-002 reference layer: the unit of the test-time compute budgets (evaluator code; not part of your model).

A pre-norm transformer layer of width 256 with 4 attention heads (scaled dot-product attention, no mask) and a GELU
MLP of width 1,024. F_ref(task) is the number of floating-point operations torch's FlopCounterMode counts for one
application of this layer to one problem of the task's sequence length (33, 50 or 128 tokens).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RefLayer(nn.Module):
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
