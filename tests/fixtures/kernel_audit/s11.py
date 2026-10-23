# Problem: problem_sum.py
import torch
import torch.nn as nn


class ModelNew(nn.Module):
    """Reduce in fp32 over a transposed contiguous view so that the reduced dimension is innermost."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        y = x.transpose(self.dim, -1).contiguous().sum(dim=-1, keepdim=True)
        return y.transpose(self.dim, -1)
