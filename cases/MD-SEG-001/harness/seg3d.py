"""MD-SEG-001 training and inference recipe (evaluator code, agent-visible): a fixed nnU-Net-style recipe in which only
the network changes.

Data (already preprocessed by the evaluator): every volume is resampled to the task's target spacing (images
trilinear, labels nearest), MRI intensities are z-scored per volume and channel, CT intensities clipped to the
0.5-99.5 percentiles of the training foreground and z-scored with its mean and standard deviation. Arrays are
(C, D, H, W) float16 images and (D, H, W) uint8 labels, D the slowest (most anisotropic) axis.

Training: TASKS[task]["iterations"] iterations of a batch of random patches of TASKS[task]["patch"]; one third of the
patches is centred on a random foreground voxel; random flips along each axis, intensity scaling x U(0.9, 1.1) and
shift + U(-0.1, 0.1) per channel, Gaussian noise (sigma U(0, 0.1), probability 0.15). Loss: soft Dice (batch Dice
over foreground classes) + cross-entropy; with deep supervision (the network returns a list, full resolution first)
the targets are downsampled by nearest neighbour to each output's size and the losses weighted 1, 1/2, 1/4, ...
(normalised; outputs smaller than 4 voxels along an axis are ignored). SGD, Nesterov momentum 0.99, learning rate
0.01 with polynomial decay (power 0.9), weight decay 3e-5, gradient-norm clip 12, bfloat16 autocast on CUDA.

Inference: sliding windows of the patch size with half-patch steps, Gaussian importance weighting (sigma 1/8 of the
patch), softmax averaged, argmax; volumes smaller than the patch are padded. Only the full-resolution output is used.

Measures: Dice per foreground class and volume (1 if both prediction and label are empty) and normalised surface
Dice (NSD) at TASKS[task]["nsd_mm"] millimetres, both at the preprocessed resolution.
"""
from __future__ import annotations

import math

import numpy as np

TASKS = {
    "hippocampus": {"channels": 1, "classes": 3, "spacing": (1.0, 1.0, 1.0), "patch": (40, 56, 40), "batch": 8,
                    "iterations": 2500, "nsd_mm": 1.0, "modality": "MRI"},
    "prostate": {"channels": 2, "classes": 3, "spacing": (3.6, 1.0, 1.0), "patch": (20, 192, 160), "batch": 2,
                 "iterations": 2500, "nsd_mm": 2.0, "modality": "MRI"},
    "heart": {"channels": 1, "classes": 2, "spacing": (2.0, 2.0, 2.0), "patch": (64, 128, 128), "batch": 2,
              "iterations": 2500, "nsd_mm": 2.0, "modality": "MRI"},
}
FG_FRACTION, LR, MOMENTUM, WD, CLIP = 1 / 3, 1e-2, 0.99, 3e-5, 12.0


# ------------------------------------------------------------------------------------------------ sampling (numpy)
def random_patch(img: np.ndarray, seg: np.ndarray, patch, rng, foreground: bool) -> tuple:
    """A patch of img (C, D, H, W) and seg (D, H, W), zero-padded where it leaves the volume; centred on a random
    foreground voxel when foreground is True and the label has any."""
    shape = np.asarray(seg.shape)
    p = np.asarray(patch)
    if foreground and seg.any():
        fg = np.argwhere(seg > 0)
        c = fg[rng.integers(len(fg))]
        lo = c - p // 2
    else:
        lo = np.array([rng.integers(min(0, s - q), max(0, s - q) + 1) for s, q in zip(shape, p)])
    xi = np.zeros((img.shape[0], *p), np.float32)
    ys = np.zeros(p, np.int64)
    src = [slice(max(l, 0), min(l + q, s)) for l, q, s in zip(lo, p, shape)]
    dst = [slice(a.start - l, a.stop - l) for a, l in zip(src, lo)]
    xi[(slice(None), *dst)] = img[(slice(None), *src)]
    ys[tuple(dst)] = seg[tuple(src)]
    return xi, ys


def batch(vols: list, task: str, rng) -> tuple:
    cfg = TASKS[task] if isinstance(task, str) else task
    xs, ys = [], []
    for b in range(cfg["batch"]):
        img, seg = vols[rng.integers(len(vols))]
        x, y = random_patch(img, seg, cfg["patch"], rng, foreground=rng.random() < FG_FRACTION)
        xs.append(x)
        ys.append(y)
    return np.stack(xs), np.stack(ys)


def augment(x: np.ndarray, y: np.ndarray, rng) -> tuple:
    """x (B, C, D, H, W), y (B, D, H, W): flips, intensity scale and shift, noise (in place on copies)."""
    x, y = x.copy(), y.copy()
    for ax in range(3):
        flip = rng.random(len(x)) < 0.5
        x[flip] = np.flip(x[flip], axis=2 + ax)
        y[flip] = np.flip(y[flip], axis=1 + ax)
    B, C = x.shape[:2]
    x *= rng.uniform(0.9, 1.1, (B, C, 1, 1, 1)).astype(np.float32)
    x += rng.uniform(-0.1, 0.1, (B, C, 1, 1, 1)).astype(np.float32)
    noisy = rng.random(B) < 0.15
    if noisy.any():
        sig = rng.uniform(0, 0.1, (int(noisy.sum()), 1, 1, 1, 1)).astype(np.float32)
        x[noisy] += sig * rng.standard_normal(x[noisy].shape).astype(np.float32)
    return np.ascontiguousarray(x), np.ascontiguousarray(y)


# ------------------------------------------------------------------------------------------------ sliding window (numpy)
def gaussian(patch) -> np.ndarray:
    g = np.ones(patch, np.float64)
    for ax, n in enumerate(patch):
        t = (np.arange(n) - (n - 1) / 2) / max(n / 8.0, 1e-6)
        shape = [1, 1, 1]
        shape[ax] = n
        g = g * np.exp(-0.5 * t ** 2).reshape(shape)
    g /= g.max()
    return np.maximum(g, 1e-3 * g.max()).astype(np.float32)


def window_starts(size: int, p: int) -> list:
    if size <= p:
        return [0]
    n = int(math.ceil((size - p) / (p / 2))) + 1
    return sorted({int(round(i * (size - p) / (n - 1))) for i in range(n)})


def sliding_window(predict, img: np.ndarray, patch, classes: int) -> np.ndarray:
    """predict: (1, C, *patch) float32 -> (1, K, *patch) softmax probabilities. Returns the (D, H, W) argmax."""
    C, shape = img.shape[0], np.asarray(img.shape[1:])
    p = np.asarray(patch)
    pad = np.maximum(p - shape, 0)
    x = np.pad(img.astype(np.float32), [(0, 0)] + [(int(a // 2), int(a - a // 2)) for a in pad])
    full = np.asarray(x.shape[1:])
    acc = np.zeros((classes, *full), np.float32)
    wsum = np.zeros(tuple(full), np.float32)
    g = gaussian(tuple(p))
    for z in window_starts(full[0], p[0]):
        for yy in window_starts(full[1], p[1]):
            for xx in window_starts(full[2], p[2]):
                sl = (slice(z, z + p[0]), slice(yy, yy + p[1]), slice(xx, xx + p[2]))
                prob = np.asarray(predict(x[(slice(None), *sl)][None]))[0]
                acc[(slice(None), *sl)] += prob * g
                wsum[sl] += g
    out = (acc / wsum).argmax(0).astype(np.uint8)
    lo = pad // 2
    return out[lo[0]:lo[0] + shape[0], lo[1]:lo[1] + shape[1], lo[2]:lo[2] + shape[2]]


# ------------------------------------------------------------------------------------------------ measures (numpy)
def dice(pred: np.ndarray, gt: np.ndarray, k: int) -> float:
    a, b = pred == k, gt == k
    s = a.sum() + b.sum()
    return 1.0 if s == 0 else float(2 * (a & b).sum() / s)


def _surface(m: np.ndarray) -> np.ndarray:
    from scipy.ndimage import binary_erosion
    return m & ~binary_erosion(m, structure=np.ones((3, 3, 3), bool), border_value=0)


def nsd(pred: np.ndarray, gt: np.ndarray, k: int, spacing, tol_mm: float) -> float:
    """Normalised surface Dice: share of both surfaces within tol_mm of the other surface."""
    from scipy.ndimage import distance_transform_edt
    a, b = pred == k, gt == k
    if not a.any() and not b.any():
        return 1.0
    if not a.any() or not b.any():
        return 0.0
    sa, sb = _surface(a), _surface(b)
    da = distance_transform_edt(~sa, sampling=spacing)
    db = distance_transform_edt(~sb, sampling=spacing)
    ok = (db[sa] <= tol_mm).sum() + (da[sb] <= tol_mm).sum()
    return float(ok / (sa.sum() + sb.sum()))


# ------------------------------------------------------------------------------------------------ torch recipe
def _targets(y, outs):
    import torch.nn.functional as F
    res = []
    for o in outs:
        if tuple(o.shape[2:]) == tuple(y.shape[1:]):
            res.append(y)
        else:
            res.append(F.interpolate(y[:, None].float(), size=o.shape[2:], mode="nearest")[:, 0].long())
    return res


def loss_fn(out, y, classes: int):
    import torch
    import torch.nn.functional as F
    outs = out if isinstance(out, (list, tuple)) else [out]
    outs = [o for i, o in enumerate(outs) if i == 0 or min(o.shape[2:]) >= 4]
    w = np.array([0.5 ** i for i in range(len(outs))])
    w = w / w.sum()
    total = 0.0
    for wi, o, t in zip(w, outs, _targets(y, outs)):
        o = o.float()
        ce = F.cross_entropy(o, t)
        p = torch.softmax(o, 1)[:, 1:]
        oh = F.one_hot(t, classes).permute(0, 4, 1, 2, 3).float()[:, 1:]
        dims = (0, 2, 3, 4)
        inter, den = (p * oh).sum(dims), p.sum(dims) + oh.sum(dims)
        dl = 1 - ((2 * inter + 1e-5) / (den + 1e-5)).mean()
        total = total + float(wi) * (ce + dl)
    return total


def train(net, vols: list, cfg: dict, seed: int, device: str, iterations: int | None = None) -> dict:
    """Train net in place with the fixed recipe; returns {"curve", "diverged", "peak_mem_gb"}."""
    import torch
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    n_it = int(iterations or cfg["iterations"])
    opt = torch.optim.SGD(net.parameters(), lr=LR, momentum=MOMENTUM, nesterov=True, weight_decay=WD)
    cuda = device.startswith("cuda")
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    curve, diverged = [], 0
    net.train()
    for it in range(n_it):
        for g in opt.param_groups:
            g["lr"] = LR * (1 - it / n_it) ** 0.9
        x, y = augment(*batch(vols, cfg, rng), rng)
        x, y = torch.as_tensor(x, device=device), torch.as_tensor(y, device=device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=cuda):
            out = net(x)
        loss = loss_fn(out, y, cfg["classes"])
        if not torch.isfinite(loss):
            diverged = 1
            break
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), CLIP)
        opt.step()
        if it % 50 == 0:
            curve.append(float(loss))
    peak = torch.cuda.max_memory_allocated() / 2 ** 30 if cuda else 0.0
    return {"curve": np.asarray(curve), "diverged": diverged, "peak_mem_gb": peak}


def predictor(net, device: str):
    import torch

    def f(x):
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
            o = net(torch.as_tensor(x, device=device))
        o = o[0] if isinstance(o, (list, tuple)) else o
        return torch.softmax(o.float(), 1).cpu().numpy()
    return f


# ------------------------------------------------------------------------------------------------ storage
def pack(arrays: list) -> tuple:
    """Volumes of different shapes as (flat array, shapes (n, ndim))."""
    shapes = np.asarray([a.shape for a in arrays], np.int64)
    flat = np.concatenate([np.asarray(a).reshape(-1) for a in arrays]) if arrays else np.zeros(0)
    return flat, shapes


def unpack(flat: np.ndarray, shapes: np.ndarray) -> list:
    out, o = [], 0
    for s in np.asarray(shapes):
        n = int(np.prod(s))
        out.append(np.asarray(flat[o:o + n]).reshape(tuple(int(v) for v in s)))
        o += n
    return out
