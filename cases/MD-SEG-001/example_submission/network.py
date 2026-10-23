"""Reference submission for MD-SEG-001: a U-Net with a residual encoder (after the nnU-Net ResEnc layouts of Isensee et
al. 2024, scaled down: 1, 2, 3, 4, 4, 4 residual blocks per encoder stage, features capped at 256, one convolution
block per decoder stage), planned
from the patch size and spacing like the baseline, with deep supervision."""
from __future__ import annotations

import torch
import torch.nn as nn

BLOCKS = (1, 2, 3, 4, 4, 4)


def plan(patch, spacing, max_stages: int = 6):
    size, sp = list(patch), list(spacing)
    strides, kernels = [[1, 1, 1]], [[3 if s <= 2 * min(sp) else 1 for s in sp]]
    while len(strides) < max_stages:
        st = [2 if (n >= 8 and s <= 2 * min(sp)) else 1 for n, s in zip(size, sp)]
        if st == [1, 1, 1]:
            break
        size = [n // f for n, f in zip(size, st)]
        sp = [s * f for s, f in zip(sp, st)]
        strides.append(st)
        kernels.append([3 if s <= 2 * min(sp) else 1 for s in sp])
    return strides, kernels


def conv(cin, cout, k, stride=1):
    return nn.Sequential(nn.Conv3d(cin, cout, k, stride, [x // 2 for x in k], bias=False),
                         nn.InstanceNorm3d(cout, affine=True))


class Res(nn.Module):
    def __init__(self, cin, cout, k, stride):
        super().__init__()
        self.a, self.b = conv(cin, cout, k, stride), conv(cout, cout, k)
        self.skip = nn.Identity() if (cin == cout and stride == [1, 1, 1]) else nn.Sequential(
            nn.AvgPool3d(stride, stride) if stride != [1, 1, 1] else nn.Identity(), conv(cin, cout, [1, 1, 1]))
        self.act = nn.LeakyReLU(0.01, inplace=True)

    def forward(self, x):
        return self.act(self.b(self.act(self.a(x))) + self.skip(x))


class ResEncUNet(nn.Module):
    def __init__(self, cin, ncls, patch, spacing, width: int = 32, cap: int = 256):
        super().__init__()
        strides, kernels = plan(patch, spacing)
        feats = [min(width * 2 ** i, cap) for i in range(len(strides))]
        self.stem = nn.Sequential(conv(cin, feats[0], kernels[0]), nn.LeakyReLU(0.01, inplace=True))
        self.enc = nn.ModuleList()
        c = feats[0]
        for i, (f, s, k) in enumerate(zip(feats, strides, kernels)):
            layers = [Res(c, f, k, s)] + [Res(f, f, k, [1, 1, 1]) for _ in range(BLOCKS[i] - 1)]
            self.enc.append(nn.Sequential(*layers))
            c = f
        self.up, self.dec, self.heads = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        for i in range(len(feats) - 1, 0, -1):
            self.up.append(nn.ConvTranspose3d(feats[i], feats[i - 1], strides[i], strides[i]))
            self.dec.append(nn.Sequential(conv(2 * feats[i - 1], feats[i - 1], kernels[i - 1]),
                                          nn.LeakyReLU(0.01, inplace=True)))
            self.heads.append(nn.Conv3d(feats[i - 1], ncls, 1))

    def forward(self, x):
        x = self.stem(x)
        skips = []
        for e in self.enc:
            x = e(x)
            skips.append(x)
        x = skips.pop()
        outs = []
        for up, dec, head in zip(self.up, self.dec, self.heads):
            x = dec(torch.cat([up(x), skips.pop()], 1))
            outs.append(head(x))
        outs = outs[::-1]
        return outs if self.training else outs[0]


def make_network(in_channels: int, num_classes: int, patch_size, spacing):
    return ResEncUNet(in_channels, num_classes, patch_size, spacing)
