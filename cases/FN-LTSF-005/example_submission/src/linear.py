"""Channel-shared linear forecaster with last-value normalisation, fitted in closed form (ridge).

For a lookback L and the longest horizon 720, the inputs of one (window, channel) row are the last L values minus the
last one, plus a bias; the targets are the next 720 values minus the last one. A multi-output ridge solves each output
column on its own, so the first H columns are the H-step forecaster.
"""
from __future__ import annotations

import numpy as np

HMAX = 720


def design(series: np.ndarray, orig: np.ndarray, lookback: int, horizon: int = HMAX):
    x = np.stack([series[o - lookback:o] for o in orig]).astype(np.float64)
    y = np.stack([series[o:o + horizon] for o in orig]).astype(np.float64)
    last = x[:, -1:, :]
    xi = np.transpose(x - last, (0, 2, 1)).reshape(-1, lookback)
    yi = np.transpose(y - last, (0, 2, 1)).reshape(-1, horizon)
    return np.hstack([xi, np.ones((len(xi), 1))]), yi


def gram(series, orig, lookback, horizon=HMAX, chunk=64):
    a = np.zeros((lookback + 1, lookback + 1))
    b = np.zeros((lookback + 1, horizon))
    n = 0
    for s in range(0, len(orig), chunk):
        xi, yi = design(series, orig[s:s + chunk], lookback, horizon)
        a += xi.T @ xi
        b += xi.T @ yi
        n += len(xi)
    return a / n, b / n


def solve(a, b, lam):
    reg = lam * np.eye(len(a))
    reg[-1, -1] = 0.0
    return np.linalg.solve(a + reg, b)


def train_origins(lo, hi, lookback, n_channels, max_rows=300_000, horizon=HMAX):
    o = np.arange(lo + lookback, hi - horizon + 1, dtype=np.int64)
    step = max(1, int(np.ceil(len(o) * n_channels / max_rows)))
    return o[::step]


def predict(w: np.ndarray, history: np.ndarray, horizon: int) -> np.ndarray:
    lookback = w.shape[0] - 1
    x = history[:, -lookback:, :].astype(np.float64)
    last = x[:, -1:, :]
    z = np.einsum("blc,lh->bhc", x - last, w[:-1, :horizon]) + w[-1, :horizon][None, :, None]
    return (z + last).astype(np.float32)
