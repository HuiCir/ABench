# Problem: problem_matmul.py
import torch
import torch.nn as nn


class ModelNew(nn.Module):
    """TF32 tensor cores for the product: fp32 range, 10-bit mantissa inputs, fp32 accumulation."""

    def forward(self, A, B):
        prev = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = True
        try:
            return torch.matmul(A, B)
        finally:
            torch.backends.cuda.matmul.allow_tf32 = prev
