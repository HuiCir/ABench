"""A shared Fourier neural operator with per-system input and output layers and FiLM conditioning on the system's
parameters."""
from __future__ import annotations

import torch
import torch.nn as nn

SYSTEMS = {"turbulent_radiative": (4, 1), "active_matter": (11, 2), "gray_scott": (2, 2)}   # channels, parameters
HISTORY = 4


class SpectralConv(nn.Module):
    def __init__(self, width: int, modes: int):
        super().__init__()
        self.modes = modes
        s = 1 / (width * width)
        self.w1 = nn.Parameter(s * torch.randn(width, width, modes, modes, dtype=torch.cfloat))
        self.w2 = nn.Parameter(s * torch.randn(width, width, modes, modes, dtype=torch.cfloat))

    def forward(self, x):
        B, C, H, W = x.shape
        f = torch.fft.rfft2(x.float())
        out = torch.zeros(B, C, H, W // 2 + 1, dtype=torch.cfloat, device=x.device)
        m = self.modes
        out[:, :, :m, :m] = torch.einsum("bixy,ioxy->boxy", f[:, :, :m, :m], self.w1)
        out[:, :, -m:, :m] = torch.einsum("bixy,ioxy->boxy", f[:, :, -m:, :m], self.w2)
        return torch.fft.irfft2(out, s=(H, W))


class FNO(nn.Module):
    def __init__(self, width: int = 96, modes: int = 16, layers: int = 6):
        super().__init__()
        self.inp = nn.ModuleDict({s: nn.Conv2d(c * HISTORY + 2, width, 1) for s, (c, _) in SYSTEMS.items()})
        self.out = nn.ModuleDict({s: nn.Sequential(nn.Conv2d(width, 128, 1), nn.GELU(), nn.Conv2d(128, c, 1))
                                  for s, (c, _) in SYSTEMS.items()})
        self.film = nn.ModuleDict({s: nn.Linear(p, 2 * width * layers) for s, (_, p) in SYSTEMS.items()})
        self.spec = nn.ModuleList([SpectralConv(width, modes) for _ in range(layers)])
        self.point = nn.ModuleList([nn.Conv2d(width, width, 1) for _ in range(layers)])
        self.norm = nn.ModuleList([nn.GroupNorm(8, width) for _ in range(layers)])
        self.layers, self.width = layers, width

    def forward(self, system: str, hist, params):
        B, T, C, H, W = hist.shape
        gy, gx = torch.meshgrid(torch.linspace(0, 1, H, device=hist.device), torch.linspace(0, 1, W, device=hist.device),
                                indexing="ij")
        x = torch.cat([hist.reshape(B, T * C, H, W), gy.expand(B, 1, H, W), gx.expand(B, 1, H, W)], 1)
        x = self.inp[system](x)
        film = self.film[system](params).view(B, self.layers, 2, self.width)
        for i in range(self.layers):
            h = self.norm[i](self.spec[i](x) + self.point[i](x))
            h = h * (1 + film[:, i, 0, :, None, None]) + film[:, i, 1, :, None, None]
            x = x + nn.functional.gelu(h)
        return hist[:, -1] + self.out[system](x)              # predicts the change from the last frame
