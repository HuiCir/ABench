"""Reference submission: ResNet-18 (small) and ResNet-50 (large) with two tricks from He et al. (2019, "Bag of
Tricks for Image Classification with Convolutional Neural Networks"):

1. ResNet-D shortcuts: a downsampling shortcut averages 2 x 2 pixels and then applies a 1 x 1 convolution with stride
   1, instead of a 1 x 1 convolution with stride 2 that ignores three quarters of the input.
2. Zero-initialised residual branches: the last batch-normalisation scale of every block starts at 0 (Goyal et al.
   2017), so each block starts as the identity.
The deep stem of ResNet-C is not used: at 128 pixels it would exceed the small budget (1.34 GFLOPs against 1.25).
Analytic FLOPs at 128 pixels: 1.18 G (small) and 2.67 G (large), the same as the baseline."""
from __future__ import annotations

import torch.nn as nn


def conv_bn(cin, cout, k, s=1, relu=True):
    layers = [nn.Conv2d(cin, cout, k, s, k // 2, bias=False), nn.BatchNorm2d(cout)]
    return nn.Sequential(*(layers + [nn.ReLU(inplace=True)] if relu else layers))


def shortcut(cin, cout, s):
    if s == 1 and cin == cout:
        return nn.Identity()
    pool = [nn.AvgPool2d(2, 2, ceil_mode=True, count_include_pad=False)] if s != 1 else []
    return nn.Sequential(*pool, conv_bn(cin, cout, 1, 1, relu=False))


class Basic(nn.Module):
    expansion = 1

    def __init__(self, cin, c, s):
        super().__init__()
        self.f = nn.Sequential(conv_bn(cin, c, 3, s), conv_bn(c, c, 3, 1, relu=False))
        self.sc = shortcut(cin, c, s)
        self.act = nn.ReLU(inplace=True)
        nn.init.zeros_(self.f[-1][1].weight)

    def forward(self, x):
        return self.act(self.f(x) + self.sc(x))


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, cin, c, s):
        super().__init__()
        self.f = nn.Sequential(conv_bn(cin, c, 1), conv_bn(c, c, 3, s), conv_bn(c, 4 * c, 1, relu=False))
        self.sc = shortcut(cin, 4 * c, s)
        self.act = nn.ReLU(inplace=True)
        nn.init.zeros_(self.f[-1][1].weight)

    def forward(self, x):
        return self.act(self.f(x) + self.sc(x))


class ResNet(nn.Module):
    def __init__(self, block, depths, num_classes):
        super().__init__()
        self.stem = nn.Sequential(conv_bn(3, 64, 7, 2), nn.MaxPool2d(3, 2, 1))
        layers, cin = [], 64
        for i, (c, n) in enumerate(zip((64, 128, 256, 512), depths)):
            for j in range(n):
                layers.append(block(cin, c, 2 if (j == 0 and i > 0) else 1))
                cin = c * block.expansion
        self.layers = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(cin, num_classes))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        for b in self.layers:                        # the loop above does not touch BN; re-zero the branch ends
            nn.init.zeros_(b.f[-1][1].weight)

    def forward(self, x):
        return self.head(self.layers(self.stem(x)))


def build(variant: str, num_classes: int = 10):
    if variant == "small":
        return ResNet(Basic, (2, 2, 2, 2), num_classes)
    return ResNet(Bottleneck, (3, 4, 6, 3), num_classes)
