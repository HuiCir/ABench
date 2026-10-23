"""Fixed image-classification training loop of the Imagenette cases (evaluator code; identical copy in each case that
uses it). The recipe is fixed per case so that differences come from the submitted backbone.

Images are uint8 arrays (N, 160, 160, 3): Imagenette's 160-pixel release, centre-cropped to a square. Training
augmentation on the GPU: random resized crop (area 35-100%, aspect 3/4-4/3) to the case resolution and horizontal
flip; evaluation: the whole square resized to the case resolution. Inputs are normalised with ImageNet statistics.

  train(model, x, y, recipe, seed, device)    one run of the case recipe; returns the loss curve and divergence
  predict(model, x, res, device)              logits (N, classes) for the evaluation view
  forward_flops(model, res, device)           FLOPs of one forward pass on one image (FlopCounterMode)
"""
from __future__ import annotations

import math
import time

import numpy as np

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def _to_tensor(xb, device):
    import torch
    t = torch.from_numpy(np.ascontiguousarray(xb)).to(device, non_blocking=True)
    return t.permute(0, 3, 1, 2).float().div_(255.0)


def _normalise(x):
    import torch
    m = torch.tensor(MEAN, device=x.device).view(1, 3, 1, 1)
    s = torch.tensor(STD, device=x.device).view(1, 3, 1, 1)
    return (x - m) / s


def augment(x, res: int, gen):
    """Random resized crop + flip as one affine resampling per image (x: float (B, 3, H, W) in [0, 1])."""
    import torch
    import torch.nn.functional as F
    B = x.shape[0]
    dev = x.device
    area = torch.empty(B, device=dev).uniform_(0.35, 1.0, generator=gen)
    logr = torch.empty(B, device=dev).uniform_(math.log(3 / 4), math.log(4 / 3), generator=gen)
    ratio = logr.exp()
    w = (area * ratio).sqrt().clamp(max=1.0)
    h = (area / ratio).sqrt().clamp(max=1.0)
    cx = (torch.rand(B, device=dev, generator=gen) * 2 - 1) * (1 - w)
    cy = (torch.rand(B, device=dev, generator=gen) * 2 - 1) * (1 - h)
    flip = torch.where(torch.rand(B, device=dev, generator=gen) < 0.5, -1.0, 1.0)
    theta = torch.zeros(B, 2, 3, device=dev)
    theta[:, 0, 0], theta[:, 0, 2] = w * flip, cx
    theta[:, 1, 1], theta[:, 1, 2] = h, cy
    grid = F.affine_grid(theta, (B, 3, res, res), align_corners=False)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="reflection", align_corners=False)


def eval_view(x, res: int):
    import torch.nn.functional as F
    return F.interpolate(x, size=(res, res), mode="bilinear", antialias=True, align_corners=False)


def schedule(step: int, total: int, warmup: int) -> float:
    if step < warmup:
        return (step + 1) / warmup
    t = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1 + math.cos(math.pi * min(1.0, t)))


def train(model, x: np.ndarray, y: np.ndarray, recipe: dict, seed: int, device: str) -> dict:
    import torch
    import torch.nn.functional as F
    r = recipe
    bs, epochs, res = int(r["batch"]), float(r["epochs"]), int(r["resolution"])
    n = len(x)
    steps = int(epochs * n // bs)
    warm = int(r["warmup_epochs"] * n // bs)
    decay = [p for p in model.parameters() if p.ndim > 1]
    other = [p for p in model.parameters() if p.ndim <= 1]
    groups = [{"params": decay, "weight_decay": r["weight_decay"]}, {"params": other, "weight_decay": 0.0}]
    if r["optimizer"] == "sgd":
        opt = torch.optim.SGD(groups, lr=r["lr"], momentum=0.9, nesterov=True)
    else:
        opt = torch.optim.AdamW(groups, lr=r["lr"], betas=(0.9, 0.999))
    gen = torch.Generator(device=device).manual_seed(seed)
    rng = np.random.default_rng(seed)
    model.train()
    curve, diverged = [], 0
    t0 = time.perf_counter()
    perm, pos = rng.permutation(n), 0
    for step in range(steps):
        if pos + bs > n:
            perm, pos = rng.permutation(n), 0
        idx = np.sort(perm[pos:pos + bs])
        pos += bs
        for g in opt.param_groups:
            g["lr"] = r["lr"] * schedule(step, steps, warm)
        xb = _normalise(augment(_to_tensor(x[idx], device), res, gen))
        yb = torch.from_numpy(y[idx]).to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
            logits = model(xb.contiguous(memory_format=torch.channels_last))
        loss = F.cross_entropy(logits.float(), yb, label_smoothing=r["label_smoothing"])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if r.get("grad_clip"):
            torch.nn.utils.clip_grad_norm_(model.parameters(), r["grad_clip"])
        opt.step()
        if step % max(1, steps // 50) == 0 or step == steps - 1:
            v = float(loss.detach())
            curve.append((step, v))
            if not math.isfinite(v):
                diverged = 1
                break
    return {"curve": np.asarray(curve, np.float64).reshape(-1, 2), "diverged": diverged,
            "seconds": time.perf_counter() - t0, "steps": steps}


def predict(model, x: np.ndarray, res: int, device: str, batch: int = 250) -> np.ndarray:
    import torch
    model.eval()
    out = []
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
        for i in range(0, len(x), batch):
            xb = _normalise(eval_view(_to_tensor(x[i:i + batch], device), res))
            out.append(model(xb.contiguous(memory_format=torch.channels_last)).float().cpu().numpy())
    model.train()
    return np.concatenate(out)


def forward_flops(model, res: int, device: str) -> float:
    import torch
    from torch.utils.flop_counter import FlopCounterMode
    model.eval()
    x = torch.zeros(1, 3, res, res, device=device)
    with torch.no_grad(), FlopCounterMode(display=False) as fc:
        model(x)
    model.train()
    return float(fc.get_total_flops())
