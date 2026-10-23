"""Reference submission: SINDy with three changes to the anchor's STLSQ.

1. Derivatives from Savitzky-Golay smoothing (cubic, window chosen from a noise estimate) instead of raw finite
   differences, and the first/last window trimmed.
2. Thresholding on a column-normalised library (each term scaled to unit RMS on the data), so the threshold does not
   depend on variable scales.
3. Per equation, a path of supports over 30 thresholds; the support is chosen by leave-one-trajectory-out error of
   the derivative fit: the sparsest support within 10% of the best error is kept, then refit on all data.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter

import systems as S

TOL = 1.1           # sparsest support whose cross-validated error is within 10% of the best


def _noise_level(x: np.ndarray) -> float:
    """Relative noise estimate from second differences (smooth signals have tiny second differences)."""
    d2 = np.diff(x, n=2, axis=0)
    return float(np.median(np.abs(d2)) / (x.std() + 1e-12))


def _derivative(x: np.ndarray, dt: float):
    nl = max(_noise_level(x[:, j]) for j in range(x.shape[1]))
    w = 7 if nl < 1e-3 else 15 if nl < 5e-3 else 25 if nl < 2e-2 else 41
    xs = savgol_filter(x, w, 3, axis=0)
    dx = savgol_filter(x, w, 3, deriv=1, delta=dt, axis=0)
    k = w // 2
    return xs[k:-k], dx[k:-k]


def _stlsq(th: np.ndarray, y: np.ndarray, lam: float, iters: int = 20) -> np.ndarray:
    xi = np.linalg.lstsq(th, y, rcond=None)[0]
    for _ in range(iters):
        keep = np.abs(xi) >= lam
        new = np.zeros_like(xi)
        if keep.any():
            new[keep] = np.linalg.lstsq(th[:, keep], y, rcond=None)[0]
        if np.array_equal(keep, np.abs(new) >= lam) and np.allclose(new, xi):
            break
        xi = new
    xi[np.abs(xi) < lam] = 0.0
    return xi


def discover(t: np.ndarray, x: np.ndarray, seed: int) -> np.ndarray:
    d = x.shape[-1]
    dt = float(t[1] - t[0])
    ths, dxs, groups = [], [], []
    for g, xi in enumerate(x):
        xs, dx = _derivative(xi, dt)
        ths.append(S.eval_library(xs, d))
        dxs.append(dx)
        groups.append(np.full(len(xs), g))
    th, dx, groups = np.concatenate(ths), np.concatenate(dxs), np.concatenate(groups)
    scale = np.sqrt((th ** 2).mean(axis=0)) + 1e-12
    thn = th / scale
    n = len(dx)
    C = np.zeros((S.n_terms(d), d))
    for i in range(d):
        y = dx[:, i]
        full = np.linalg.lstsq(thn, y, rcond=None)[0]
        top = float(np.abs(full).max()) + 1e-12
        cands, supports = [], set()
        for lam in top * np.logspace(-4, -0.3, 30):
            sup = tuple(np.flatnonzero(_stlsq(thn, y, lam)))
            if not sup or sup in supports:
                continue
            supports.add(sup)
            cols = list(sup)
            err = 0.0
            for g in np.unique(groups):          # leave one trajectory out
                tr, te = groups != g, groups == g
                w = np.linalg.lstsq(thn[tr][:, cols], y[tr], rcond=None)[0]
                err += float(((thn[te][:, cols] @ w - y[te]) ** 2).sum())
            cands.append((err, len(cols), cols))
        best_err = min(c[0] for c in cands)
        cols = min((c for c in cands if c[0] <= TOL * best_err), key=lambda c: (c[1], c[0]))[2]
        w = np.linalg.lstsq(thn[:, cols], y, rcond=None)[0]
        C[cols, i] = w / scale[cols]
    return C
