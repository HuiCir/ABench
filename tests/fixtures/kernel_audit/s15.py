# Problem: problem_relu.py
import torch
import torch.nn as nn


class ModelNew(nn.Module):
    """In-place activation: no output allocation."""

    def forward(self, x):
        return torch.relu_(x)
