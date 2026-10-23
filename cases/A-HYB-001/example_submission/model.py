"""Reference submission: a local-global attention hybrid.

Layers 3 and 7 (of 8) keep full causal attention; the other six use blocked local attention: the sequence is cut into
blocks of 256 tokens and each block attends causally to itself and the previous block (a window of 257 to 512 tokens).
Local attention is computed block by block, so its cost grows linearly with the sequence length. Everything else is
the baseline's (lm_model.py). Interleaving local and global layers follows Longformer (Beltagy et al. 2020) and
Gemma 2 (Gemma Team 2024); the two global layers keep exact retrieval over the whole context.
"""
from __future__ import annotations

import lm_model as base
from src.local_attention import LocalAttention

BLOCK = 256                     # every fourth layer (3 and 7 of 8) keeps global attention


def build(cfg):
    model = base.LM(cfg)
    for i, blk in enumerate(model.blocks):
        if i % 4 != 3:
            local = LocalAttention(cfg, BLOCK)
            local.load_state_dict(blk.attn.state_dict())
            blk.attn = local
    return model
