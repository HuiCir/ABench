"""Reference submission: the baseline ViT with two changes for small data (Xiao et al. 2021; Beyer et al. 2022).

1. Convolutional stem: the 16 x 16 patchify convolution is replaced by four 3 x 3 convolutions with stride 2 (width
   d/8, d/4, d/2, d, with batch normalisation and ReLU) and a 1 x 1 convolution, which gives the same 10 x 10 token
   grid at 160 pixels with a locality prior and more stable optimisation.
2. Global average pooling of the tokens instead of a class token, and fixed 2-D sine-cosine position embeddings
   instead of learned ones.
Analytic FLOPs at 160 pixels: 1.27 G (small, 12 blocks) and 4.51 G (large, 11 blocks: with 12 blocks the convolutional
stem would bring it to 4.88 G, over the 4.8 G budget).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from src.vit_blocks import Block


def sincos_2d(n_side: int, d: int) -> torch.Tensor:
    y, x = torch.meshgrid(torch.arange(n_side), torch.arange(n_side), indexing="ij")
    q = d // 4
    omega = 1.0 / (10000 ** (torch.arange(q) / q))
    out = [f(c.flatten()[:, None].float() * omega[None]) for c in (x, y) for f in (torch.sin, torch.cos)]
    return torch.cat(out, 1)[None]


class ConvStemViT(nn.Module):
    def __init__(self, d: int, heads: int, depth: int, res: int = 160, num_classes: int = 10):
        super().__init__()
        chans = [3, d // 8, d // 4, d // 2, d]
        layers = []
        for a, b in zip(chans[:-1], chans[1:]):
            layers += [nn.Conv2d(a, b, 3, 2, 1, bias=False), nn.BatchNorm2d(b), nn.ReLU(inplace=True)]
        layers.append(nn.Conv2d(d, d, 1))
        self.stem = nn.Sequential(*layers)
        self.register_buffer("pos", sincos_2d(res // 16, d), persistent=False)
        self.blocks = nn.Sequential(*[Block(d, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x):
        x = self.stem(x).flatten(2).transpose(1, 2) + self.pos
        return self.head(self.norm(self.blocks(x)).mean(1))


def build(variant: str, num_classes: int = 10):
    if variant == "small":
        return ConvStemViT(192, 3, 12, num_classes=num_classes)
    return ConvStemViT(384, 6, 11, num_classes=num_classes)
