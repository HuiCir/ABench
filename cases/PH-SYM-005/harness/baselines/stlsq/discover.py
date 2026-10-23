"""Baseline: the anchor's SINDy with sequentially thresholded least squares (STLSQ), threshold 0.1, derivatives by
second-order central finite differences, all trajectories stacked."""
from __future__ import annotations

import numpy as np

import systems as S

THRESHOLD = 0.1


def stlsq(theta: np.ndarray, dx: np.ndarray, lam: float, iters: int = 10) -> np.ndarray:
    xi = np.linalg.lstsq(theta, dx, rcond=None)[0]
    for _ in range(iters):
        small = np.abs(xi) < lam
        xi[small] = 0.0
        for i in range(dx.shape[1]):
            big = ~small[:, i]
            if big.any():
                xi[big, i] = np.linalg.lstsq(theta[:, big], dx[:, i], rcond=None)[0]
    return xi


def discover(t: np.ndarray, x: np.ndarray, seed: int) -> np.ndarray:
    d = x.shape[-1]
    th, dx = [], []
    for xi in x:                                   # x: (trajectories, samples, d)
        th.append(S.eval_library(xi, d))
        dx.append(np.gradient(xi, t, axis=0, edge_order=2))
    return stlsq(np.concatenate(th), np.concatenate(dx), THRESHOLD)
