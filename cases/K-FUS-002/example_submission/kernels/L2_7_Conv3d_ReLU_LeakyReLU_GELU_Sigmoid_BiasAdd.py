"""L2_7: cuDNN convolution followed by the whole activation tail and the bias add in one Triton pass (src/ew.py, OP 7)."""
import torch
import torch.nn as nn

from src.ew import apply


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        return apply(self.conv(x), 7, a=0.01, bias=self.bias)
