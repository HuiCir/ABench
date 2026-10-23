"""L1_20_LeakyReLU: one-pass Triton elementwise kernel (src/ew.py)."""
import torch
import torch.nn as nn

from src.ew import apply


class ModelNew(nn.Module):
    def __init__(self, negative_slope: float = 0.01):
        super().__init__()
        self.negative_slope = negative_slope

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return apply(x, 1, a=self.negative_slope)
