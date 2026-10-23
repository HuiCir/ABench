"""U-KAN-009: the fixed networks, budgets and training recipe in which a submitted layer replaces the MLP layer
(evaluator code, agent-visible).

A layer is built by the submission's make_layer(d_in, d_out, final) and maps (B, d_in) to (B, d_out); `final` marks the
output layer. Every task uses DEPTH = 3 layers, d_in -> h -> h -> d_out. The evaluator picks the hidden width h for the
submitted layer and, separately, for the baseline MLP layer (Linear followed by SiLU, no activation on the final
layer): the largest h whose network has at most the task's parameter BUDGET (counted with numel over all parameters,
buffers excluded). Both networks are trained with the same recipe, seeds and data, so the comparison is at equal
parameters; their training-step times are measured on the same GPU and the ratio is a gate (TIME_RATIO_MAX).

Recipe (RECIPE): Adam, learning rate chosen per network and task from LRS by the validation error after training
with the first seed, cosine decay to zero, no warm-up, gradient-norm clipping at 1.0, float32. Function tasks: mean
squared error on standardised targets, full training set as one batch of up to 2,000 points, FUNC_STEPS steps.
Image task: cross-entropy, batches of 256, IMG_EPOCHS epochs over 55,000 images.
"""
from __future__ import annotations

import time

import numpy as np

DEPTH = 3
BUDGET = {"function": 3_000, "image": 100_000}
LRS = (1e-3, 3e-3, 1e-2)
FUNC_STEPS = 3_000
IMG_EPOCHS = 10
IMG_BATCH = 256
CLIP = 1.0
TIME_RATIO_MAX = 3.0
WIDTH_MAX = 4096
RECIPE = {"depth": DEPTH, "budget": BUDGET, "lrs": LRS, "func_steps": FUNC_STEPS, "img_epochs": IMG_EPOCHS,
          "img_batch": IMG_BATCH, "clip": CLIP, "optimizer": "Adam, cosine decay to 0", "time_ratio_max": TIME_RATIO_MAX}


def mlp_layer(d_in: int, d_out: int, final: bool):
    import torch.nn as nn
    lin = nn.Linear(d_in, d_out)
    return lin if final else nn.Sequential(lin, nn.SiLU())


def build(make_layer, d_in: int, h: int, d_out: int):
    import torch.nn as nn
    dims = [d_in] + [h] * (DEPTH - 1) + [d_out]
    return nn.Sequential(*[make_layer(dims[i], dims[i + 1], i == DEPTH - 1) for i in range(DEPTH)])


def n_params(net) -> int:
    return int(sum(p.numel() for p in net.parameters()))


def pick_width(make_layer, d_in: int, d_out: int, budget: int) -> int:
    """Largest h in [1, WIDTH_MAX] whose network has at most `budget` parameters (bisection; parameter counts are
    assumed to grow with h). 0 if even h = 1 exceeds the budget."""
    import torch
    torch.manual_seed(0)
    if n_params(build(make_layer, d_in, 1, d_out)) > budget:
        return 0
    lo, hi = 1, WIDTH_MAX
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if n_params(build(make_layer, d_in, mid, d_out)) <= budget:
            lo = mid
        else:
            hi = mid - 1
    return lo


def _cosine(opt, lr0: float, step: int, total: int):
    lr = 0.5 * lr0 * (1 + np.cos(np.pi * step / max(total, 1)))
    for g in opt.param_groups:
        g["lr"] = lr


def _sync(device: str):
    import torch
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def train(net, x, y, kind: str, lr: float, seed: int, device: str) -> dict:
    """Train in place; returns the median step time in milliseconds (after 20 warm-up steps) and divergence."""
    import torch
    import torch.nn.functional as F
    torch.manual_seed(seed)
    net.to(device).train()
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    x = torch.as_tensor(x, device=device)
    y = torch.as_tensor(y, device=device)
    n = len(x)
    if kind == "function":
        total, batch = FUNC_STEPS, n
    else:
        total, batch = IMG_EPOCHS * (n // IMG_BATCH), IMG_BATCH
    g = torch.Generator(device="cpu").manual_seed(seed)
    perm, pos, times, diverged = torch.randperm(n, generator=g), 0, [], False
    for step in range(total):
        if batch >= n:
            xb, yb = x, y
        else:
            if pos + batch > n:
                perm, pos = torch.randperm(n, generator=g), 0
            idx = perm[pos:pos + batch].to(device)
            pos += batch
            xb, yb = x[idx], y[idx]
        _cosine(opt, lr, step, total)
        timed = step >= 20 and step % 10 == 0
        if timed:
            _sync(device)
            t0 = time.perf_counter()
        out = net(xb)
        loss = F.mse_loss(out, yb) if kind == "function" else F.cross_entropy(out, yb)
        if not torch.isfinite(loss):
            diverged = True
            break
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), CLIP)
        opt.step()
        if timed:
            _sync(device)
            times.append((time.perf_counter() - t0) * 1e3)
    return {"step_ms": float(np.median(times)) if times else float("nan"), "diverged": diverged}


def predict(net, x, kind: str, device: str, batch: int = 4096) -> np.ndarray:
    """Function tasks: standardised predictions (N,); image task: class predictions (N,)."""
    import torch
    net.eval()
    outs = []
    with torch.no_grad():
        for s in range(0, len(x), batch):
            o = net(torch.as_tensor(x[s:s + batch], device=device))
            outs.append((o[:, 0] if kind == "function" else o.argmax(1)).float().cpu().numpy())
    return np.concatenate(outs)


def val_error(pred: np.ndarray, y: np.ndarray, kind: str) -> float:
    """Function tasks: RMSE / std of the targets; image task: error rate. inf for non-finite predictions."""
    if not np.all(np.isfinite(pred)):
        return float("inf")
    if kind == "function":
        return float(np.sqrt(np.mean((pred - y) ** 2)) / max(np.std(y), 1e-12))
    return float(np.mean(pred != y))


def fit_task(make_layer, data: dict, kind: str, seeds: list, device: str) -> dict:
    """Pick the width, choose the learning rate on validation with seeds[0], then train with every seed at that rate.
    data: x_train, y_train, x_val, y_val (standardised targets for functions, int64 labels for images).
    Returns the width, parameter count, chosen rate, the trained networks (one per seed), step times and divergence."""
    import torch
    d_in = data["x_train"].shape[1]
    d_out = 1 if kind == "function" else 10
    yt = data["y_train"].reshape(-1, 1).astype(np.float32) if kind == "function" else data["y_train"].astype(np.int64)
    yv = data["y_val"].reshape(-1) if kind == "function" else data["y_val"]
    h = pick_width(make_layer, d_in, d_out, BUDGET[kind])
    if h == 0:
        return {"width": 0, "params": 0, "lr": float("nan"), "nets": [], "step_ms": float("nan"), "diverged": True}
    best = (np.inf, None, None, None)
    for lr in LRS:
        torch.manual_seed(seeds[0])
        net = build(make_layer, d_in, h, d_out)
        r = train(net, data["x_train"], yt, kind, lr, seeds[0], device)
        e = float("inf") if r["diverged"] else val_error(predict(net, data["x_val"], kind, device), yv, kind)
        if e < best[0]:
            best = (e, lr, net, r)
    if best[1] is None:                                    # every rate diverged
        return {"width": h, "params": n_params(build(make_layer, d_in, h, d_out)), "lr": float("nan"), "nets": [],
                "step_ms": float("nan"), "diverged": True}
    nets, times, div = [best[2]], [best[3]["step_ms"]], False
    for s in seeds[1:]:
        torch.manual_seed(s)
        net = build(make_layer, d_in, h, d_out)
        r = train(net, data["x_train"], yt, kind, best[1], s, device)
        nets.append(net)
        times.append(r["step_ms"])
        div = div or r["diverged"]
    return {"width": h, "params": n_params(nets[0]), "lr": best[1], "nets": nets, "step_ms": float(np.median(times)),
            "diverged": div}
