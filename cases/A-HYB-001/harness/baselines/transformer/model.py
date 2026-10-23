"""Baseline: the B-OPT-001 decoder (lm_model.py: pre-RMSNorm, grouped-query attention with RoPE, SwiGLU MLP, tied
embeddings) at the sizes the evaluator passes; every layer uses full causal attention."""
from __future__ import annotations

from lm_model import LM


def build(cfg):
    return LM(cfg)
