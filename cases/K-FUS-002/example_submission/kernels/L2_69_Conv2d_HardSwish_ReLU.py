"""L2_69: cuDNN convolution followed by hardswish and ReLU in one Triton pass (src/ew.py, OP 5)."""
import torch.nn as nn

from src.ew import apply


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)

    def forward(self, x):
        return apply(self.conv(x), 5)
