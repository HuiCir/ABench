# Problem: problem_rmsnorm.py
import torch
import torch.nn as nn


class ModelNew(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-5):
        super().__init__()
        self.num_features = num_features
        self.eps = eps

    def forward(self, x):
        # one fused reduction and one multiply instead of square, mean, add, sqrt and divide as separate passes
        return x * torch.rsqrt(x.pow(2).mean(dim=1, keepdim=True) + self.eps)
