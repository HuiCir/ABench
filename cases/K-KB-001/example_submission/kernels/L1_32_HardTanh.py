"""L1_32_HardTanh: one-pass Triton elementwise kernel (src/ew.py)."""
import torch
import torch.nn as nn

from src.ew import apply


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return apply(x, 4, a=-1.0, b=1.0)
