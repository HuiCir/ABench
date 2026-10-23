"""L2_100: cuDNN transposed convolution followed by the clamp and the division in one Triton pass (src/ew.py, OP 6)."""
import torch.nn as nn

from src.ew import apply


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, min_value, divisor):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.min_value = min_value
        self.divisor = divisor

    def forward(self, x):
        return apply(self.conv_transpose(x), 6, a=self.min_value, b=self.divisor)
