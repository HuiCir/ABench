"""Reference submission: TTT-Linear layers (Sun et al. 2024) in every odd layer of SmolLM2-135M (15 of 30).

Each replaced layer keeps the pretrained query, key, value and output projections, expanded to 9 heads of 64. Per
head, a linear inner model W (64 x 64, learned initial value W0 = 0) is trained online on the sequence with the
self-supervised loss ||W k_t - v_t||² (keys and queries L2-normalised); updates are taken in mini-batches of 16
tokens with gradients at the state of the mini-batch start (the dual form, src/ttt.py), and the output for token t is
W_t q_t, with W_t including the gradient steps of the tokens up to t. A learned per-head inner learning rate (initial
0.1) scales the steps; a learned per-head gate (initial 0.1) mixes the result into the residual output. The state is W
plus at most 15 buffered keys and values, independent of the context length.
"""
from __future__ import annotations

import smol_lm
from src.ttt import TTTLinear


def build(base_dir: str, device: str):
    model = smol_lm.load_base(base_dir, device)
    for i, layer in enumerate(model.layers):
        if i % 2 == 1:
            layer.mixer = TTTLinear(layer.mixer).to(device)
    return model
