"""Reference submission: dynamic patches that end at spaces (the space-patching scheme of Slagle 2024 and of the BLT
paper's comparison), in the baseline's local-global layout.

- A new patch starts after every space, newline or document boundary (byte 0), so a patch is a word with its trailing
  separator; the decision for byte j uses only byte j - 1, so patching stays causal.
- Patch embeddings are the mean of the local encoder states of the patch's bytes (byte_blocks.segment_mean), mapped
  to the global width; the global transformer runs over the patches of each sequence (padded to the longest row; causal
  attention never looks at the padding of later positions).
- Byte j receives the global state of the patches before its own, exactly as in the baseline.
With words of about 5.5 bytes the global transformer runs over fewer positions than with 4-byte patches.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

import lm_model as base
from byte_blocks import Blocks, segment_mean

LOCAL = base.Config(vocab_size=257, d_model=256, n_layers=1, n_heads=4, n_kv_heads=2, d_ff=704, max_seq=4096)
GLOBAL = base.Config(vocab_size=1, d_model=384, n_layers=8, n_heads=6, n_kv_heads=2, d_ff=1024, max_seq=4096)
BOS = 256
SEPARATORS = (32, 10, 0)


class SpacePatch(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb_enc = nn.Embedding(257, LOCAL.d_model)
        self.enc = Blocks(LOCAL, 1, window=256)
        self.to_patch = nn.Linear(LOCAL.d_model, GLOBAL.d_model, bias=False)
        self.bos = nn.Parameter(torch.zeros(GLOBAL.d_model))
        self.glob = Blocks(GLOBAL, 8)
        self.up = nn.Linear(GLOBAL.d_model, LOCAL.d_model, bias=False)
        self.emb_dec = nn.Embedding(257, LOCAL.d_model)
        self.dec = Blocks(LOCAL, 1, window=256)
        self.norm = base.RMSNorm(LOCAL.d_model)
        self.head = nn.Linear(LOCAL.d_model, 256, bias=False)
        for e in (self.emb_enc, self.emb_dec):
            nn.init.normal_(e.weight, std=0.02)
        for lin in (self.to_patch, self.up, self.head):
            nn.init.normal_(lin.weight, std=0.02)

    @staticmethod
    def segments(x: torch.Tensor) -> torch.Tensor:
        """Patch id of every byte: a patch starts at position 0 and after every separator byte."""
        sep = torch.zeros_like(x, dtype=torch.bool)
        for s in SEPARATORS:
            sep |= x == s
        start = torch.zeros_like(sep)
        start[:, 1:] = sep[:, :-1]
        return torch.cumsum(start.long(), 1)

    def forward(self, x):
        B, T = x.shape
        x = x.long()
        seg = self.segments(x)
        n = int(seg.max()) + 1
        h = self.enc(self.emb_enc(x))
        patches = self.to_patch(segment_mean(h, seg, n))
        g = self.glob(torch.cat([self.bos.expand(B, 1, -1), patches[:, :-1]], 1))      # g_s sees patches < s
        up = self.up(g).gather(1, seg[..., None].expand(-1, -1, LOCAL.d_model))
        prev = torch.cat([torch.full((B, 1), BOS, device=x.device, dtype=torch.long), x[:, :-1]], 1)
        d = self.dec(self.emb_dec(prev) + up)
        return self.head(self.norm(d))

    def loss(self, x):
        logits = self(x)
        return F.cross_entropy(logits.float().reshape(-1, 256), x.reshape(-1))


def build():
    return SpacePatch()
