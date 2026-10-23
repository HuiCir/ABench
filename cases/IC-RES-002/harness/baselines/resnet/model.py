"""Baseline: ResNet-18 (small) and ResNet-50 (large) of He et al. (2016), the torchvision layout (7x7 stem with max
pooling, batch normalisation, projection shortcuts), written out so that no vision library is needed."""
from __future__ import annotations

import torch.nn as nn


def conv_bn(cin, cout, k, s=1, relu=True):
    layers = [nn.Conv2d(cin, cout, k, s, k // 2, bias=False), nn.BatchNorm2d(cout)]
    return nn.Sequential(*(layers + [nn.ReLU(inplace=True)] if relu else layers))


class Basic(nn.Module):
    expansion = 1

    def __init__(self, cin, c, s):
        super().__init__()
        self.f = nn.Sequential(conv_bn(cin, c, 3, s), conv_bn(c, c, 3, 1, relu=False))
        self.sc = conv_bn(cin, c, 1, s, relu=False) if s != 1 or cin != c else nn.Identity()
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.f(x) + self.sc(x))


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, cin, c, s):
        super().__init__()
        self.f = nn.Sequential(conv_bn(cin, c, 1), conv_bn(c, c, 3, s), conv_bn(c, 4 * c, 1, relu=False))
        self.sc = conv_bn(cin, 4 * c, 1, s, relu=False) if s != 1 or cin != 4 * c else nn.Identity()
        self.act = nn.ReLU(inplace=True)

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

    def forward(self, x):
        return self.head(self.layers(self.stem(x)))


def build(variant: str, num_classes: int = 10):
    if variant == "small":
        return ResNet(Basic, (2, 2, 2, 2), num_classes)
    return ResNet(Bottleneck, (3, 4, 6, 3), num_classes)
