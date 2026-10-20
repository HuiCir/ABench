"""PH-NO-001 reference submission, inference side: a linear, per-wavenumber autoregressive operator in Fourier space (NumPy).

For each retained wavenumber k the next frame is û_{t+1}(k) = a_k û_t(k) + b_k û_{t-1}(k), with complex a_k, b_k fitted
by least squares over all training transitions. This is a single linear spectral layer (no nonlinearity), so it is
resolution independent: on a finer grid the same coefficients act on the same physical wavenumbers and the extra
high modes are damped to zero. train.py fits a_k, b_k and saves them to checkpoint/spectral.npz.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

N_MODES = 64


def load(checkpoint_dir, device="cpu"):
    with np.load(Path(checkpoint_dir) / "spectral.npz", allow_pickle=False) as z:
        return {"a": z["a"], "b": z["b"], "grid": int(z["grid"])}


def predict(model, u_in, n_out=31):
    u_in = np.asarray(u_in, np.float64)
    n = u_in.shape[-1]
    scale = n / model["grid"]                            # rfft amplitudes scale with the number of points
    U = np.fft.rfft(u_in, axis=-1)[..., :N_MODES] / scale
    prev, cur = U[:, -2], U[:, -1]
    out = []
    for _ in range(n_out):
        nxt = model["a"] * cur + model["b"] * prev
        full = np.zeros((len(u_in), n // 2 + 1), complex)
        full[:, :N_MODES] = nxt * scale
        out.append(np.fft.irfft(full, n=n, axis=-1))
        prev, cur = cur, nxt
    return np.stack(out, 1).astype(np.float32)
