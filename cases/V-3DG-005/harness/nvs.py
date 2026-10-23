"""V-3DG-005 views, cameras and image measures (evaluator code, agent-visible).

Scenes: Mip-NeRF 360 (Barron et al. 2022) at the published 1/8 resolution (images_8). Cameras come from the
dataset's COLMAP reconstruction of all images: intrinsics K (3 x 3, pixels, scaled to images_8) and world-to-camera
matrices w2c (4 x 4) in the OpenCV convention (x right, y down, z forward). Test views are every 8th image (index
0, 8, 16, ...); the 24 training views are spread evenly over the remaining images (indices chosen with
numpy.linspace, as in FSGS).

Measures: PSNR on [0, 1] per test image; SSIM (11 x 11 Gaussian window, sigma 1.5, on each RGB channel, averaged);
both averaged over a scene's test images, then over scenes.
"""
from __future__ import annotations

import numpy as np

N_TRAIN, HOLD_EVERY = 24, 8


def split(n_images: int, n_train: int = N_TRAIN, every: int = HOLD_EVERY, offset: int = 0) -> tuple:
    """(train indices, test indices) of a scene with n_images images."""
    test = np.arange(offset % every, n_images, every)
    rest = np.setdiff1d(np.arange(n_images), test)
    pick = np.unique(np.round(np.linspace(0, len(rest) - 1, n_train)).astype(int))
    return rest[pick], test


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = np.mean((a.astype(np.float64) / 255.0 - b.astype(np.float64) / 255.0) ** 2)
    return float(10 * np.log10(1.0 / max(mse, 1e-12)))


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.ndimage import gaussian_filter
    x, y = a.astype(np.float64) / 255.0, b.astype(np.float64) / 255.0
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    vals = []
    for ch in range(x.shape[-1]):
        f = lambda z: gaussian_filter(z, 1.5, truncate=3.5, mode="reflect")   # noqa: E731  (11 x 11 window)
        u, v = f(x[..., ch]), f(y[..., ch])
        sxx, syy, sxy = f(x[..., ch] ** 2) - u ** 2, f(y[..., ch] ** 2) - v ** 2, f(x[..., ch] * y[..., ch]) - u * v
        m = ((2 * u * v + c1) * (2 * sxy + c2)) / ((u ** 2 + v ** 2 + c1) * (sxx + syy + c2))
        vals.append(m.mean())
    return float(np.mean(vals))


def project(points: np.ndarray, K: np.ndarray, w2c: np.ndarray) -> tuple:
    """Pixel coordinates (N, 2) and depths (N,) of world points in one camera."""
    pc = points @ w2c[:3, :3].T + w2c[:3, 3]
    uv = pc @ K.T
    return uv[:, :2] / uv[:, 2:3], pc[:, 2]
