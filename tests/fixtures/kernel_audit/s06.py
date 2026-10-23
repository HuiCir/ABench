# Problem: problem_matmul.py
import torch
import torch.nn as nn


class ModelNew(nn.Module):
    """Tensor-core product in half precision (fp32 accumulation), result returned in half precision cast to fp32."""

    def forward(self, A, B):
        return torch.matmul(A.half(), B.half()).float()
