"""IC-AUG-003 fixed training recipe (the evaluator's; your policy only changes the augmentation).

The ResNet-9-style network and one-cycle SGD recipe of the IC-SPD-001 reference submission, trained for 40 epochs:
batch 512, SGD with Nesterov momentum 0.9, peak lr 0.4 after 20% linear warm-up then linear decay, weight decay 5e-4
on weights, label smoothing 0.2, bf16 autocast, channels_last. At the start of every epoch the whole training set (in
[0, 1], on the GPU) is passed through policy.augment(x, aug_ops, gen, epoch, n_epochs) and then normalised with the
training-set channel statistics. At test time the logits of the image and of its mirror image are averaged.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HP = dict(epochs=40, batch_size=512, lr=0.4, momentum=0.9, weight_decay=5e-4, label_smoothing=0.2, warmup_frac=0.2,
          widths=(64, 128, 256, 512), head_scale=0.125)


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
        self.body = nn.Sequential(conv_bn(3, w0), conv_bn(w0, w1, pool=True), Residual(w1), conv_bn(w1, w2, pool=True),
                                  conv_bn(w2, w3, pool=True), Residual(w3), nn.AdaptiveMaxPool2d(1), nn.Flatten())
        self.head = nn.Linear(w3, n_classes, bias=False)
        self.head_scale = head_scale

    def forward(self, x):
        return self.head(self.body(x)) * self.head_scale


def to01(x_uint8: np.ndarray, device: str) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(x_uint8)).to(device).permute(0, 3, 1, 2).float().div_(255.0)


def train(x01: torch.Tensor, y: torch.Tensor, augment, ops, seed: int, device: str, epochs: int = HP["epochs"]):
    torch.manual_seed(seed)
    use_cuda = device.startswith("cuda") and torch.cuda.is_available()
    if use_cuda:
        torch.backends.cudnn.benchmark = True
    gen = torch.Generator(device=x01.device)
    gen.manual_seed(seed)
    mean = x01.mean(dim=(0, 2, 3), keepdim=True)
    std = x01.std(dim=(0, 2, 3), keepdim=True)
    model = ResNet9().to(x01.device).to(memory_format=torch.channels_last)
    bs = HP["batch_size"]
    steps_per_epoch = math.ceil(len(y) / bs)
    total = epochs * steps_per_epoch
    warm = int(HP["warmup_frac"] * total)
    decay = [p for p in model.parameters() if p.ndim > 1]
    no_decay = [p for p in model.parameters() if p.ndim <= 1]
    opt = torch.optim.SGD([{"params": decay, "weight_decay": HP["weight_decay"]}, {"params": no_decay, "weight_decay": 0.0}],
                          lr=HP["lr"], momentum=HP["momentum"], nesterov=True)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / max(1, warm) if s < warm else max(0.0, (total - s) / max(1, total - warm)))
    amp = torch.autocast("cuda", dtype=torch.bfloat16) if use_cuda else torch.autocast("cpu", enabled=False)
    model.train()
    for epoch in range(epochs):
        with torch.no_grad():
            xa = augment(x01, ops, gen, epoch, epochs)
            if tuple(xa.shape) != tuple(x01.shape) or not torch.isfinite(xa).all():
                raise SystemExit("augment() must return a finite tensor of the input's shape")
            xa = ((xa.clamp(0, 1) - mean) / std).contiguous(memory_format=torch.channels_last)
        perm = torch.randperm(len(y), device=x01.device, generator=gen)
        for k in range(steps_per_epoch):
            idx = perm[k * bs:(k + 1) * bs]
            with amp:
                loss = F.cross_entropy(model(xa[idx]), y[idx], label_smoothing=HP["label_smoothing"])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
    model.eval()
    return model, mean, std


@torch.no_grad()
def predict(model, mean, std, x01: torch.Tensor) -> torch.Tensor:
    use_cuda = x01.device.type == "cuda"
    amp = torch.autocast("cuda", dtype=torch.bfloat16) if use_cuda else torch.autocast("cpu", enabled=False)
    outs = []
    with amp:
        for k in range(0, x01.shape[0], 2000):
            b = ((x01[k:k + 2000] - mean) / std).contiguous(memory_format=torch.channels_last)
            outs.append(((model(b) + model(b.flip(3))).float() / 2).softmax(-1))
    return torch.cat(outs)
