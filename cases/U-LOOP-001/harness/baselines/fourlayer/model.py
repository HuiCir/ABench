"""Baseline: the B-OPT-001 decoder with 4 layers of width 384 (the parameter budget of the case) and no loops; the
`loops` argument is accepted and ignored."""
from __future__ import annotations

from lm_model import LM


class FourLayer(LM):
    def forward(self, idx, loops: int = 1):
        return super().forward(idx)

    def loss(self, idx, targets, loops: int = 1):
        return super().loss(idx, targets)


def build(cfg):
    return FourLayer(cfg)
