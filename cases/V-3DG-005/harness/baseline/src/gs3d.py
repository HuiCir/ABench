"""The baseline: plain 3D Gaussian splatting (Kerbl et al. 2023) with the gsplat rasteriser, from a random
initialisation, as one would run it on dense views.

Initialisation: 50,000 Gaussians uniformly in a sphere around the camera centres (radius 1.5 x the largest camera
distance from their mean), grey colour, isotropic scale from the mean distance to the 3 nearest neighbours, opacity
0.1. Training: L1 + 0.2 D-SSIM on one random training view per step, Adam with the learning rates of the 3DGS paper
(position decaying exponentially by 100x), spherical harmonics up to degree 3 added every 1,000 steps, gsplat's
DefaultStrategy for densification and pruning, until 7,000 steps or the time budget (minus a 10 s margin)."""
from __future__ import annotations

import math
import time

import numpy as np

STEPS, N_INIT, SH_DEGREE = 7000, 50_000, 3


def _ssim(x, y):
    import torch.nn.functional as F
    import torch
    g = torch.exp(-((torch.arange(11, device=x.device) - 5.0) ** 2) / (2 * 1.5 ** 2))
    g = (g / g.sum())
    w = (g[:, None] * g[None, :]).expand(3, 1, 11, 11)
    f = lambda z: F.conv2d(z, w, padding=5, groups=3)                       # noqa: E731
    mx, my = f(x), f(y)
    sxx, syy, sxy = f(x * x) - mx ** 2, f(y * y) - my ** 2, f(x * y) - mx * my
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    return (((2 * mx * my + c1) * (2 * sxy + c2)) / ((mx ** 2 + my ** 2 + c1) * (sxx + syy + c2))).mean()


def init_params(views: dict, device: str, n: int = N_INIT, seed: int = 0):
    import torch
    w2c = np.asarray(views["w2c"], np.float64)
    centres = np.stack([-m[:3, :3].T @ m[:3, 3] for m in w2c])
    mid = centres.mean(0)
    r = 1.5 * np.max(np.linalg.norm(centres - mid, axis=1))
    rng = np.random.default_rng(seed)
    d = rng.normal(size=(n, 3))
    pts = mid + d / np.linalg.norm(d, axis=1, keepdims=True) * r * rng.random((n, 1)) ** (1 / 3)
    p = torch.as_tensor(pts, dtype=torch.float32, device=device)
    dist = torch.cdist(p[:2048], p).topk(4, largest=False).values[:, 1:].mean()
    scales = torch.full((n, 3), float(torch.log(dist)), device=device)
    quats = torch.zeros(n, 4, device=device)
    quats[:, 0] = 1
    C0 = 0.28209479177387814
    sh0 = torch.full((n, 1, 3), (0.5 - 0.5) / C0, device=device)
    shN = torch.zeros(n, (SH_DEGREE + 1) ** 2 - 1, 3, device=device)
    op = torch.logit(torch.full((n,), 0.1, device=device))
    params = torch.nn.ParameterDict({k: torch.nn.Parameter(v) for k, v in
                                     dict(means=p, scales=scales, quats=quats, opacities=op, sh0=sh0, shN=shN).items()})
    return params, r


def _optimizers(params, scene_scale: float):
    import torch
    lrs = {"means": 1.6e-4 * scene_scale, "scales": 5e-3, "quats": 1e-3, "opacities": 5e-2, "sh0": 2.5e-3, "shN": 2.5e-3 / 20}
    return {k: torch.optim.Adam([{"params": params[k], "lr": lr, "name": k}], eps=1e-15) for k, lr in lrs.items()}, lrs


def _render(params, w2c, K, h, w, sh_degree, background=None):
    import torch
    from gsplat import rasterization
    colors = torch.cat([params["sh0"], params["shN"]], 1)
    out, alpha, info = rasterization(params["means"], params["quats"], torch.exp(params["scales"]),
                                     torch.sigmoid(params["opacities"]), colors, w2c, K, w, h, sh_degree=sh_degree,
                                     backgrounds=background)
    return out, info


def fit(views: dict, budget_seconds: float, device: str, extra_loss=None, random_background: bool = False):
    import torch
    from gsplat.strategy import DefaultStrategy
    t0 = time.perf_counter()
    torch.manual_seed(0)
    imgs = torch.as_tensor(views["images"], device=device).float() / 255.0
    Ks = torch.as_tensor(views["K"], dtype=torch.float32, device=device)
    w2c = torch.as_tensor(views["w2c"], dtype=torch.float32, device=device)
    n, h, w = imgs.shape[:3]
    params, r = init_params(views, device)
    opts, lrs = _optimizers(params, r)
    strategy = DefaultStrategy(refine_stop_iter=STEPS // 2, verbose=False)
    strategy.check_sanity(params, opts)
    st = strategy.initialize_state(scene_scale=r)
    gamma = 0.01 ** (1.0 / STEPS)
    rng = np.random.default_rng(0)
    for step in range(STEPS):
        if time.perf_counter() - t0 > budget_seconds - 10:
            break
        i = int(rng.integers(n))
        sh = min(step // 1000, SH_DEGREE)
        bg = torch.rand(1, 3, device=device) if random_background else None
        out, info = _render(params, w2c[i:i + 1], Ks[i:i + 1], h, w, sh, bg)
        strategy.step_pre_backward(params, opts, st, step, info)
        pred, gt = out[0].permute(2, 0, 1)[None], imgs[i].permute(2, 0, 1)[None]
        loss = 0.8 * (pred - gt).abs().mean() + 0.2 * (1 - _ssim(pred, gt))
        if extra_loss is not None:
            loss = loss + extra_loss(params, step)
        loss.backward()
        for o in opts.values():
            o.step()
            o.zero_grad(set_to_none=True)
        for g in opts["means"].param_groups:
            g["lr"] *= gamma
        strategy.step_post_backward(params, opts, st, step, info, packed=False)
    return {"params": params, "sh": SH_DEGREE, "device": device}


def render(state: dict, cams: dict) -> np.ndarray:
    import torch
    dev = state["device"]
    out = []
    with torch.no_grad():
        for K, m in zip(cams["K"], cams["w2c"]):
            img, _ = _render(state["params"], torch.as_tensor(m[None], dtype=torch.float32, device=dev),
                             torch.as_tensor(K[None], dtype=torch.float32, device=dev), cams["height"], cams["width"],
                             state["sh"])
            out.append((img[0].clamp(0, 1) * 255).round().byte().cpu().numpy())
    return np.stack(out)
