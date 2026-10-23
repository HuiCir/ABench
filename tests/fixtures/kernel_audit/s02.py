# Problem: problem_relu.py
import torch
import torch.nn as nn


class ModelNew(nn.Module):
    """Skip the elementwise pass when the input has no negative entry: one reduction instead of a read and a write
    of every element through the activation."""

    def forward(self, x):
        if bool(x.min() >= 0):
            return x.clone()
        return torch.relu(x)
