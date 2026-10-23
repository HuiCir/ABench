"""Baseline: SmolLM2-135M converted to multi-query attention (one key-value head per layer) by mean-pooling the key and
value projections of its three key-value heads, as in GQA uptraining (Ainslie et al. 2023); the cache shrinks from
23,040 to 7,680 bytes per token."""
from __future__ import annotations

import torch

import smol_lm


def build(base_dir: str, device: str):
    model = smol_lm.load_base(base_dir, device)
    c = model.c
    for layer in model.layers:
        att = layer.mixer
        with torch.no_grad():
            for name in ("k_proj", "v_proj"):
                w = getattr(att, name).weight.view(c.n_kv_heads, c.head_dim, c.d_model).mean(0)
                lin = torch.nn.Linear(c.d_model, c.head_dim, bias=False).to(device)
                lin.weight.copy_(w)
                setattr(att, name, lin)
        att.c = smol_lm.Config(n_kv_heads=1)
    return model
