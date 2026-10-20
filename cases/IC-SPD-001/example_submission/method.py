"""IC-SPD-001 reference submission: ResNet-9-style network + one-cycle learning rate + a full data pipeline on the GPU.

This is a baseline demonstrating format and interface; it does not aim to be fastest. Design points (see report/index.html):
  - training data moved to GPU memory once; normalization, random crop (4-pixel reflect padding), horizontal flip and 8×8 cutout are all done in batches on the GPU;
  - channels_last + bfloat16 autocast, cudnn.benchmark;
  - SGD + Nesterov momentum, one-cycle learning rate with linear warmup then linear decay, label smoothing;
  - at inference, logits of the original and horizontally flipped images are averaged.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HP = dict(epochs=24, batch_size=512, lr=0.4, momentum=0.9, weight_decay=5e-4, label_smoothing=0.2,
          warmup_frac=0.2, pad=4, cutout=8, widths=(64, 128, 256, 512), head_scale=0.125)


def conv_bn(c_in: int, c_out: int, pool: bool = False) -> nn.Sequential:
    layers = [nn.Conv2d(c_in, c_out, 3, padding=1, bias=False)]
    if pool:
        layers.append(nn.MaxPool2d(2))
    layers += [nn.BatchNorm2d(c_out), nn.ReLU(inplace=True)]
    return nn.Sequential(*layers)


class Residual(nn.Module):
    def __init__(self, c: int):
        super().__init__()
        self.f = nn.Sequential(conv_bn(c, c), conv_bn(c, c))

    def forward(self, x):
        return x + self.f(x)


class ResNet9(nn.Module):
    def __init__(self, widths=HP["widths"], n_classes: int = 10, head_scale: float = HP["head_scale"]):
        super().__init__()
        w0, w1, w2, w3 = widths
        self.body = nn.Sequential(
            conv_bn(3, w0),
            conv_bn(w0, w1, pool=True), Residual(w1),
            conv_bn(w1, w2, pool=True),
            conv_bn(w2, w3, pool=True), Residual(w3),
            nn.AdaptiveMaxPool2d(1), nn.Flatten())
        self.head = nn.Linear(w3, n_classes, bias=False)
        self.head_scale = head_scale

    def forward(self, x):
        return self.head(self.body(x)) * self.head_scale


def _to_tensor(x: np.ndarray, device: str, mean=None, std=None):
    t = torch.from_numpy(np.ascontiguousarray(x)).to(device, non_blocking=True)
    t = t.permute(0, 3, 1, 2).float().div_(255.0)                         # NCHW, [0,1]
    if mean is None:
        mean = t.mean(dim=(0, 2, 3), keepdim=True)
        std = t.std(dim=(0, 2, 3), keepdim=True)
    t = ((t - mean) / std).contiguous(memory_format=torch.channels_last)
    return t, mean, std


def _augment(x: torch.Tensor, pad: int, cutout: int, gen: torch.Generator) -> torch.Tensor:
    """Whole-batch random crop (reflect padding) + horizontal flip + cutout, all on the GPU."""
    n = x.shape[0]
    xp = F.pad(x, (pad, pad, pad, pad), mode="reflect")
    dx = torch.randint(0, 2 * pad + 1, (n,), device=x.device, generator=gen)
    dy = torch.randint(0, 2 * pad + 1, (n,), device=x.device, generator=gen)
    # crop grouped by shift, avoiding a per-sample loop
    out = torch.empty_like(x)
    for i in range(2 * pad + 1):
        for j in range(2 * pad + 1):
            m = (dy == i) & (dx == j)
            if m.any():
                out[m] = xp[m, :, i:i + 32, j:j + 32]
    flip = torch.rand(n, device=x.device, generator=gen) < 0.5
    out[flip] = out[flip].flip(3)
    if cutout:
        cy = torch.randint(0, 32, (n, 1, 1), device=x.device, generator=gen)
        cx = torch.randint(0, 32, (n, 1, 1), device=x.device, generator=gen)
        r = torch.arange(32, device=x.device)
        h = cutout // 2                                                   # cutout×cutout square centered at (cy, cx)
        dy_, dx_ = r.view(1, 32, 1) - cy, r.view(1, 1, 32) - cx
        inside = (dy_ >= -h) & (dy_ < cutout - h) & (dx_ >= -h) & (dx_ < cutout - h)
        out = out.masked_fill(inside.unsqueeze(1), 0.0)
    return out


def train(data: dict, seed: int, device: str):
    torch.manual_seed(seed)
    np.random.seed(seed % (2 ** 32))
    use_cuda = device.startswith("cuda") and torch.cuda.is_available()
    dev = device if use_cuda else "cpu"
    if use_cuda:
        torch.backends.cudnn.benchmark = True
    gen = torch.Generator(device=dev)
    gen.manual_seed(seed)

    x, mean, std = _to_tensor(data["x"], dev)
    y = torch.from_numpy(np.asarray(data["y"], dtype=np.int64)).to(dev)
    model = ResNet9().to(dev).to(memory_format=torch.channels_last)

    bs, epochs = HP["batch_size"], HP["epochs"]
    steps_per_epoch = math.ceil(len(y) / bs)
    total = epochs * steps_per_epoch
    warm = int(HP["warmup_frac"] * total)
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        (no_decay if p.ndim <= 1 else decay).append(p)
    opt = torch.optim.SGD([{"params": decay, "weight_decay": HP["weight_decay"]},
                           {"params": no_decay, "weight_decay": 0.0}],
                          lr=HP["lr"], momentum=HP["momentum"], nesterov=True)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / max(1, warm) if s < warm else max(0.0, (total - s) / max(1, total - warm)))
    amp = torch.autocast("cuda", dtype=torch.bfloat16) if use_cuda else torch.autocast("cpu", enabled=False)

    model.train()
    for _ in range(epochs):
        xa = _augment(x, HP["pad"], HP["cutout"], gen)
        perm = torch.randperm(len(y), device=dev, generator=gen)
        for k in range(steps_per_epoch):
            idx = perm[k * bs:(k + 1) * bs]
            with amp:
                loss = F.cross_entropy(model(xa[idx]), y[idx], label_smoothing=HP["label_smoothing"])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
    model.eval()
    return {"model": model, "mean": mean, "std": std, "device": dev}


@torch.no_grad()
def predict(state, x, device: str):
    model, dev = state["model"], state["device"]
    xt, _, _ = _to_tensor(x, dev, state["mean"], state["std"])
    amp = torch.autocast("cuda", dtype=torch.bfloat16) if dev.startswith("cuda") else torch.autocast("cpu", enabled=False)
    outs = []
    with amp:
        for k in range(0, xt.shape[0], 2000):
            b = xt[k:k + 2000]
            outs.append((model(b) + model(b.flip(3))).float() / 2)          # horizontal-flip test-time augmentation
    return torch.cat(outs)
