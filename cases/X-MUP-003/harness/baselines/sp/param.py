"""Baseline: standard parametrisation, as in GPT-2 and the B-OPT-001 model: every matrix initialised N(0, 0.02²)
regardless of width (residual output projections scaled by 1/sqrt(2 · layers)), one learning rate for all parameters,
output logits unscaled, attention scale 1/sqrt(head_dim)."""
from __future__ import annotations

import math


class StandardParam:
    @staticmethod
    def init_std(name, shape, width, base_width):
        if name.endswith(("attn.o.weight", "mlp.down.weight")):
            return 0.02 / math.sqrt(2 * 8)
        return 0.02

    @staticmethod
    def lr_scale(name, shape, width, base_width):
        return 1.0

    @staticmethod
    def output_multiplier(width, base_width):
        return 1.0


PARAM = StandardParam
