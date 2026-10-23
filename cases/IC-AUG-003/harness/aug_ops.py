"""IC-AUG-003 augmentation primitives: the only operations an augmentation policy may use.

All functions act on a whole batch on the GPU: x is a float tensor (N, 3, 32, 32) with values in [0, 1]; every random
draw uses the torch.Generator `gen` passed in. Magnitudes are drawn per image, uniformly between 0 and the given
maximum, unless stated otherwise. Functions return a new tensor; they never modify x in place.

  bernoulli(x, p, gen)              -> bool mask (N,), True with probability p           (to apply an op to some images)
  where(mask, a, b)                 -> per-image choice between two augmented batches
  uniform(x, lo, hi, gen)           -> float (N,) per-image magnitudes
  choice(x, k, gen)                 -> int (N,) per-image index in 0..k-1                 (to pick one of k ops per image)
  flip_lr(x, gen, p=0.5)            mirror left-right
  translate(x, max_px, gen)         shift by up to max_px pixels in x and y (reflect padding)
  rotate(x, max_deg, gen)           rotate by an angle in [-max_deg, max_deg] (bilinear, reflect border)
  shear(x, max_shear, gen, axis)    shear along axis 0 (x) or 1 (y) by a factor in [-max_shear, max_shear]
  scale(x, max_rel, gen)            zoom by a factor in [1 - max_rel, 1 + max_rel] around the centre
  cutout(x, size, gen)              zero a size x size square at a random centre (may extend past the border)
  brightness(x, max_delta, gen)     add a value in [-max_delta, max_delta]
  contrast(x, max_rel, gen)         scale deviations from the image mean by a factor in [1 - max_rel, 1 + max_rel]
  saturation(x, max_rel, gen)       scale deviations from the grey image by a factor in [1 - max_rel, 1 + max_rel]
  grayscale(x)                      luminance replicated on three channels
  solarize(x, threshold)            invert pixels above threshold
  posterize(x, bits)                keep `bits` bits per channel (1..8)
  invert(x)                         1 - x
  sharpness(x, max_rel, gen)        blend with a 3x3 blurred image by a factor in [1 - max_rel, 1 + max_rel]
Results are clamped to [0, 1].
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def _u(n, lo, hi, gen, device):
    return lo + (hi - lo) * torch.rand(n, generator=gen, device=device)


def bernoulli(x, p, gen):
    return torch.rand(x.shape[0], generator=gen, device=x.device) < float(p)


def where(mask, a, b):
    return torch.where(mask.view(-1, 1, 1, 1), a, b)


def uniform(x, lo, hi, gen):
    return _u(x.shape[0], float(lo), float(hi), gen, x.device)


def choice(x, k, gen):
    return torch.randint(0, int(k), (x.shape[0],), generator=gen, device=x.device)


def _clamp(x):
    return x.clamp(0.0, 1.0)


def flip_lr(x, gen, p=0.5):
    return where(bernoulli(x, p, gen), x.flip(3), x)


def _affine(x, theta):
    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="reflection", align_corners=False)


def translate(x, max_px, gen):
    n = x.shape[0]
    t = torch.zeros(n, 2, 3, device=x.device)
    t[:, 0, 0] = t[:, 1, 1] = 1.0
    t[:, 0, 2] = torch.randint(-int(max_px), int(max_px) + 1, (n,), generator=gen, device=x.device).float() * 2 / 32
    t[:, 1, 2] = torch.randint(-int(max_px), int(max_px) + 1, (n,), generator=gen, device=x.device).float() * 2 / 32
    return _clamp(_affine(x, t))


def rotate(x, max_deg, gen):
    a = _u(x.shape[0], -float(max_deg), float(max_deg), gen, x.device) * math.pi / 180
    t = torch.zeros(x.shape[0], 2, 3, device=x.device)
    t[:, 0, 0], t[:, 0, 1], t[:, 1, 0], t[:, 1, 1] = a.cos(), -a.sin(), a.sin(), a.cos()
    return _clamp(_affine(x, t))


def shear(x, max_shear, gen, axis=0):
    s = _u(x.shape[0], -float(max_shear), float(max_shear), gen, x.device)
    t = torch.zeros(x.shape[0], 2, 3, device=x.device)
    t[:, 0, 0] = t[:, 1, 1] = 1.0
    t[:, 0 if axis == 0 else 1, 1 if axis == 0 else 0] = s
    return _clamp(_affine(x, t))


def scale(x, max_rel, gen):
    f = 1.0 / _u(x.shape[0], 1 - float(max_rel), 1 + float(max_rel), gen, x.device)
    t = torch.zeros(x.shape[0], 2, 3, device=x.device)
    t[:, 0, 0] = t[:, 1, 1] = f
    return _clamp(_affine(x, t))


def cutout(x, size, gen):
    n = x.shape[0]
    cy = torch.randint(0, 32, (n, 1, 1), generator=gen, device=x.device)
    cx = torch.randint(0, 32, (n, 1, 1), generator=gen, device=x.device)
    r = torch.arange(32, device=x.device)
    h = int(size) // 2
    dy, dx = r.view(1, 32, 1) - cy, r.view(1, 1, 32) - cx
    inside = (dy >= -h) & (dy < int(size) - h) & (dx >= -h) & (dx < int(size) - h)
    return x.masked_fill(inside.unsqueeze(1), 0.0)


def brightness(x, max_delta, gen):
    return _clamp(x + _u(x.shape[0], -float(max_delta), float(max_delta), gen, x.device).view(-1, 1, 1, 1))


def contrast(x, max_rel, gen):
    f = _u(x.shape[0], 1 - float(max_rel), 1 + float(max_rel), gen, x.device).view(-1, 1, 1, 1)
    m = x.mean(dim=(1, 2, 3), keepdim=True)
    return _clamp(m + f * (x - m))


def grayscale(x):
    g = (0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]).unsqueeze(1)
    return g.expand_as(x).contiguous()


def saturation(x, max_rel, gen):
    f = _u(x.shape[0], 1 - float(max_rel), 1 + float(max_rel), gen, x.device).view(-1, 1, 1, 1)
    g = grayscale(x)
    return _clamp(g + f * (x - g))


def solarize(x, threshold):
    return torch.where(x >= float(threshold), 1.0 - x, x)


def posterize(x, bits):
    q = 2 ** (8 - int(max(1, min(8, bits))))
    return torch.floor(x * 255 / q) * q / 255


def invert(x):
    return 1.0 - x


def sharpness(x, max_rel, gen):
    k = torch.tensor([[1, 1, 1], [1, 5, 1], [1, 1, 1]], dtype=x.dtype, device=x.device) / 13
    blur = F.conv2d(F.pad(x, (1, 1, 1, 1), mode="replicate"), k.expand(3, 1, 3, 3), groups=3)
    f = _u(x.shape[0], 1 - float(max_rel), 1 + float(max_rel), gen, x.device).view(-1, 1, 1, 1)
    return _clamp(blur + f * (x - blur))
