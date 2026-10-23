"""Reference submission: fine-grained experts with auxiliary-loss-free load balancing.

Changes relative to the baseline (top-2 of 8 experts of width 1,024 with the Switch balancing loss):
1. Fine-grained experts (Dai et al. 2024, DeepSeekMoE): 16 experts of width 512, top-4 routing. Total and active
   parameters are unchanged (16 · 512 = 8 · 1,024 and 4 · 512 = 2 · 1,024), while the number of expert combinations a
   token can use grows from 28 to 1,820.
2. Auxiliary-loss-free balancing (Wang et al. 2024): a per-expert bias, added to the router scores only for the top-k
   selection (not to the gate weights), is moved by ±BIAS_RATE after every training forward pass against the sign of
   the expert's load deviation. No balancing loss interferes with the language-modelling gradient.
3. Sigmoid router scores, normalised over the selected experts, as in DeepSeek-V3.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.experts import Experts, route_and_combine  # noqa: E402

N_EXPERTS, FF, K, BIAS_RATE = 16, 512, 4, 1e-3


class FineGrainedFree(nn.Module):
    def __init__(self, d_model: int, layer: int, n_layers: int):
        super().__init__()
        self.router = nn.Linear(d_model, N_EXPERTS, bias=False)
        nn.init.normal_(self.router.weight, std=0.02)
        self.experts = Experts(N_EXPERTS, d_model, FF, std=0.02 / (2 * n_layers) ** 0.5)
        nn.init.normal_(self.experts.w_gate, std=0.02)
        nn.init.normal_(self.experts.w_up, std=0.02)
        self.register_buffer("bias", torch.zeros(N_EXPERTS))

    def forward(self, x):
        B, T, d = x.shape
        h = x.reshape(-1, d)
        s = torch.sigmoid(self.router(h).float())
        _, idx = (s + self.bias).topk(K, dim=-1)                  # bias only steers the selection
        g = s.gather(1, idx)
        w = (g / g.sum(-1, keepdim=True)).to(h.dtype)
        y, counts = route_and_combine(self.experts, h, idx, w, N_EXPERTS)
        if self.training:
            with torch.no_grad():
                load = counts.float()
                self.bias += BIAS_RATE * torch.sign(load.mean() - load)
        return y.reshape(B, T, d), {"counts": counts}


FFN = FineGrainedFree
