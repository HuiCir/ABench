"""Reference submission: the maximal-update parametrisation (μP) for Adam, in the form of Yang et al. (2022, Table 3),
with base width 128.

With r = width / base_width:
- hidden matrices (attention and MLP weights): init std σ_base / sqrt(r), Adam learning rate η / r;
- embedding (input and, tied, output): init std 0.02 and learning rate η, independent of width;
- output logits multiplied by 1 / r (the tied embedding acts as the readout, whose μP scaling is 1/width);
- gains of RMSNorm: learning rate η;
- attention scale 1 / head_dim instead of 1 / sqrt(head_dim) (head_dim is fixed at 64 here, so this only rescales
  the logits by 1/8, compensated by query-key initialisation at width 128).
σ_base is 0.02 for input matrices and 0.02 / sqrt(16) for residual output projections, as in the baseline at width 128,
so both parametrisations coincide at the tuning width except for the attention scale.
"""
from __future__ import annotations

import math


class MuP:
    @staticmethod
    def _hidden(name: str) -> bool:
        return ".attn." in name or ".mlp." in name

    @classmethod
    def init_std(cls, name, shape, width, base_width):
        r = width / base_width
        if name.endswith(("attn.o.weight", "mlp.down.weight")):
            return 0.02 / math.sqrt(2 * 8) / math.sqrt(r)
        if name.endswith(("attn.q.weight", "attn.k.weight")):
            return 0.02 * math.sqrt(8.0) / math.sqrt(r)           # offsets the 1/head_dim attention scale at r = 1
        if cls._hidden(name):
            return 0.02 / math.sqrt(r)
        return 0.02

    @classmethod
    def lr_scale(cls, name, shape, width, base_width):
        return base_width / width if cls._hidden(name) and len(shape) >= 2 else 1.0

    @staticmethod
    def output_multiplier(width, base_width):
        return base_width / width

    @staticmethod
    def attention_scale(head_dim, width, base_width):
        return 1.0 / head_dim


PARAM = MuP
