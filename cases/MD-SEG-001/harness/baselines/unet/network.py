"""The baseline: an nnU-Net-style plain convolutional U-Net (Isensee et al. 2021) planned from the patch size and
spacing, with deep supervision.

Planning, as in nnU-Net's default planner (simplified): at each new stage an axis is downsampled by 2 if it is still at
least 8 voxels long and its current spacing is at most twice the smallest current spacing; the kernel along an axis
is 1 while its spacing is more than twice the smallest one, else 3. Features 32, 64, 128, 256, 320, 320 (capped at
320), at most 6 stages. Each stage: two (conv, instance norm, leaky ReLU 0.01) blocks."""
from __future__ import annotations

import torch
import torch.nn as nn


def plan(patch, spacing, max_stages: int = 6):
    size, sp = list(patch), list(spacing)
    strides, kernels = [[1, 1, 1]], []
    kernels.append([3 if s <= 2 * min(sp) else 1 for s in sp])
    while len(strides) < max_stages:
        st = [2 if (n >= 8 and s <= 2 * min(sp)) else 1 for n, s in zip(size, sp)]
        if st == [1, 1, 1]:
            break
        size = [n // f for n, f in zip(size, st)]
        sp = [s * f for s, f in zip(sp, st)]
        strides.append(st)
        kernels.append([3 if s <= 2 * min(sp) else 1 for s in sp])
    return strides, kernels


def block(cin, cout, k, stride):
    pad = [x // 2 for x in k]
    return nn.Sequential(nn.Conv3d(cin, cout, k, stride, pad), nn.InstanceNorm3d(cout, affine=True),
                         nn.LeakyReLU(0.01, inplace=True))


class UNet(nn.Module):
    def __init__(self, cin, ncls, patch, spacing, width: int = 32, cap: int = 320):
        super().__init__()
        strides, kernels = plan(patch, spacing)
        feats = [min(width * 2 ** i, cap) for i in range(len(strides))]
        self.enc = nn.ModuleList()
        c = cin
        for f, s, k in zip(feats, strides, kernels):
            self.enc.append(nn.Sequential(block(c, f, k, s), block(f, f, k, 1)))
            c = f
        self.up, self.dec, self.heads = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        for i in range(len(feats) - 1, 0, -1):
            self.up.append(nn.ConvTranspose3d(feats[i], feats[i - 1], strides[i], strides[i]))
            self.dec.append(nn.Sequential(block(2 * feats[i - 1], feats[i - 1], kernels[i - 1], 1),
                                          block(feats[i - 1], feats[i - 1], kernels[i - 1], 1)))
            self.heads.append(nn.Conv3d(feats[i - 1], ncls, 1))

    def forward(self, x):
        skips = []
        for e in self.enc:
            x = e(x)
            skips.append(x)
        x = skips.pop()
        outs = []
        for up, dec, head in zip(self.up, self.dec, self.heads):
            x = dec(torch.cat([up(x), skips.pop()], 1))
            outs.append(head(x))
        outs = outs[::-1]                      # full resolution first
        return outs if self.training else outs[0]


def make_network(in_channels: int, num_classes: int, patch_size, spacing):
    return UNet(in_channels, num_classes, patch_size, spacing)
