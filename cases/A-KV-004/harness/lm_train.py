"""Fixed training and evaluation loop of the micro language-model ladder cases (evaluator code; identical copy in each
case that uses it). The recipe is fixed so that differences come from the submitted component, not from tuning.

  train(model, stream, steps, prm, seed, device, lr)   AdamW, linear warm-up then cosine decay, bf16 autocast,
                                                        gradient clipping, optional gradient accumulation; returns the
                                                        loss curve, divergence and the number of loss spikes
  heldout_loss(model, tokens, prm, device)             mean next-token cross-entropy on held-out windows
  forward_flops_per_token(model, seq, device)          FLOPs of one forward pass per token (FlopCounterMode)
"""
from __future__ import annotations

import math
import time

import numpy as np

RECIPE = {"batch": 32, "accum": 1, "seq": 1024, "betas": (0.9, 0.95), "weight_decay": 0.1, "warmup": 150, "final_lr_fraction": 0.1,
          "grad_clip": 1.0, "eval_batch": 16, "spike_window": 100, "spike_ratio": 1.15,
          "precision": "bf16 autocast, fp32 weights and optimizer"}
DIVERGED_LOSS = math.log(50304)          # loss assigned to a run that diverged (uniform prediction)


def schedule(step: int, steps: int, warmup: int, lo: float) -> float:
    if step < warmup:
        return (step + 1) / warmup
    t = (step - warmup) / max(1, steps - warmup)
    return lo + (1 - lo) * 0.5 * (1 + math.cos(math.pi * min(1.0, t)))


def count_spikes(losses: np.ndarray, warmup: int, window: int, ratio: float) -> int:
    """Steps after warm-up whose training loss exceeds `ratio` times the median of the previous `window` steps; a
    spike that lasts several steps counts once."""
    n, spikes, inside = len(losses), 0, False
    for i in range(max(warmup, window), n):
        med = float(np.median(losses[i - window:i]))
        hi = losses[i] > ratio * med
        if hi and not inside:
            spikes += 1
        inside = hi
    return spikes


def param_groups(model, weight_decay: float):
    decay = [p for n, p in model.named_parameters() if p.requires_grad and p.ndim >= 2]
    other = [p for n, p in model.named_parameters() if p.requires_grad and p.ndim < 2]
    return [{"params": decay, "weight_decay": weight_decay}, {"params": other, "weight_decay": 0.0}]


def batches(stream: np.ndarray, rng: np.random.Generator, B: int, T: int, device: str):
    import torch
    off = rng.integers(0, len(stream) - T - 1, B)
    chunk = np.stack([stream[o:o + T + 1] for o in off]).astype(np.int64)
    x = torch.from_numpy(chunk[:, :-1]).to(device, non_blocking=True)
    y = torch.from_numpy(chunk[:, 1:]).to(device, non_blocking=True)
    return x, y


def train(model, stream: np.ndarray, steps: int, prm: dict, seed: int, device: str, lr: float, groups=None,
          loss_fn=None) -> dict:
    """Trains `model` in place. loss_fn(model, x, y) defaults to model.loss(x, y)."""
    import torch
    r = dict(RECIPE, **prm)
    opt = torch.optim.AdamW(groups if groups is not None else param_groups(model, r["weight_decay"]), lr=lr,
                            betas=tuple(r["betas"]), fused=device.startswith("cuda"))
    for g in opt.param_groups:
        g.setdefault("lr_scale", 1.0)
    base = [g["lr"] if "lr" in g else lr for g in opt.param_groups]
    rng = np.random.default_rng(seed)
    loss_fn = loss_fn or (lambda m, x, y: m.loss(x, y))
    losses = torch.zeros(steps, device=device)
    diverged = 0
    t0 = time.perf_counter()
    model.train()
    done = steps
    for step in range(steps):
        f = schedule(step, steps, r["warmup"], r["final_lr_fraction"])
        for g, b in zip(opt.param_groups, base):
            g["lr"] = b * f * g["lr_scale"]
        acc = int(r["accum"])
        loss = 0.0
        for _ in range(acc):                           # gradient accumulation over acc micro-batches
            x, y = batches(stream, rng, r["batch"] // acc, r["seq"], device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
                micro = loss_fn(model, x, y) / acc
            micro.backward()
            loss = loss + micro.detach()
        torch.nn.utils.clip_grad_norm_(model.parameters(), r["grad_clip"])
        opt.step()
        opt.zero_grad(set_to_none=True)
        losses[step] = loss.detach().float()
        if step % 50 == 49 or step == steps - 1:
            v = float(losses[step])
            if not math.isfinite(v) or v > 20:
                diverged, done = 1, step + 1
                break
    seconds = time.perf_counter() - t0
    curve = losses[:done].cpu().numpy().astype(np.float64)
    curve = np.where(np.isfinite(curve), curve, 1e9)
    spikes = count_spikes(curve, r["warmup"], r["spike_window"], r["spike_ratio"])
    return {"curve": curve, "diverged": diverged, "spikes": spikes, "seconds": seconds,
            "tokens_per_second": done * r["batch"] * r["seq"] / max(seconds, 1e-9)}


def heldout_loss(model, tokens: np.ndarray, prm: dict, device: str, loss_fn=None) -> float:
    import torch
    r = dict(RECIPE, **prm)
    T, bs = r["seq"], r["eval_batch"]
    toks = np.asarray(tokens).astype(np.int64)
    n = (len(toks) - 1) // T
    x_all = torch.from_numpy(toks[: n * T].reshape(n, T))
    y_all = torch.from_numpy(toks[1: n * T + 1].reshape(n, T))
    loss_fn = loss_fn or (lambda m, x, y: m.loss(x, y))
    total, count = 0.0, 0
    model.eval()
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
        for i in range(0, n, bs):
            x, y = x_all[i:i + bs].to(device), y_all[i:i + bs].to(device)
            loss = float(loss_fn(model, x, y))
            if not math.isfinite(loss):
                return DIVERGED_LOSS
            total += loss * y.numel()
            count += y.numel()
    model.train()
    return min(total / max(count, 1), DIVERGED_LOSS)


def forward_flops_per_token(model, seq: int, device: str, call=None) -> float:
    """FLOPs of one forward pass on a (1, seq) batch divided by seq, counted by torch's FlopCounterMode."""
    import torch
    from torch.utils.flop_counter import FlopCounterMode
    x = torch.zeros(1, seq, dtype=torch.long, device=device)
    call = call or (lambda m, x: m(x))
    model.eval()
    with torch.no_grad(), FlopCounterMode(display=False) as fc:
        call(model, x)
    model.train()
    return float(fc.get_total_flops()) / seq
