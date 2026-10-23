"""One-pass elementwise Triton kernels: each reads its input once and writes its output once (fp32). No `from __future__ import annotations`: Triton
reads the `tl.constexpr` annotations of the kernel signature.

OP codes: 0 relu, 1 leaky_relu(a), 2 sigmoid, 3 hardsigmoid, 4 hardtanh(a, b), 5 relu(hardswish), 6 clamp(min=a) / b,
7 the K-FUS-002 problem 7 tail: sigmoid(gelu(leaky_relu(relu(x), a))) + bias[channel].
"""
from typing import Optional

import torch
import triton
import triton.language as tl

BLOCK = 4096


@triton.jit
def _ew(X, Y, BIAS, N, C, S, a, b, OP: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < N
    x = tl.load(X + offs, mask=m, other=0.0).to(tl.float32)
    if OP == 0:
        y = tl.maximum(x, 0.0)
    elif OP == 1:
        y = tl.where(x >= 0, x, x * a)
    elif OP == 2:
        y = 1.0 / (1.0 + tl.exp(-x))
    elif OP == 3:
        y = tl.minimum(tl.maximum(x + 3.0, 0.0), 6.0) / 6.0
    elif OP == 4:
        y = tl.minimum(tl.maximum(x, a), b)
    elif OP == 5:
        y = tl.maximum(x * tl.minimum(tl.maximum(x + 3.0, 0.0), 6.0) / 6.0, 0.0)
    elif OP == 6:
        y = tl.maximum(x, a) / b
    else:
        y = tl.maximum(x, 0.0)
        y = tl.where(y >= 0, y, y * a)
        y = 0.5 * y * (1.0 + tl.math.erf(y * 0.7071067811865476))
        y = 1.0 / (1.0 + tl.exp(-y))
        c = (offs // S) % C
        y = y + tl.load(BIAS + c, mask=m, other=0.0).to(tl.float32)
    tl.store(Y + offs, y, mask=m)


def apply(x: torch.Tensor, op: int, a: float = 0.0, b: float = 1.0, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    """y = OP(x) elementwise; for op 7, x is (B, C, *spatial) and bias has C elements."""
    x = x.contiguous()
    y = torch.empty_like(x)
    n = x.numel()
    if op == 7:
        C, S = x.shape[1], x[0, 0].numel()
        bias = bias.reshape(-1).contiguous()
    else:
        C, S, bias = 1, 1, x
    _ew[(triton.cdiv(n, BLOCK),)](x, y, bias, n, C, S, float(a), float(b), OP=op, BLOCK=BLOCK, num_warps=8)
    return y
