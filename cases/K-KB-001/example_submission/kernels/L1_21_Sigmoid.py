"""L1_21_Sigmoid: one-pass Triton elementwise kernel (src/ew.py)."""
import torch
import torch.nn as nn

from src.ew import apply


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return apply(x, 2)
