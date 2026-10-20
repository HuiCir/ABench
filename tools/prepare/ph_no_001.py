"""PH-NO-001: PDEBench 1D Burgers, ν = 0.001 (DaRUS doi:10.18419/darus-2986, CC BY 4.0).

The file holds `tensor` (10000, 201, 1024) plus `x-coordinate` and `t-coordinate`. Frames are subsampled every 5
steps (41 frames) and space every 4 points (256) for the main grid, every 2 points (512) for the transfer grid.
A fixed permutation sends 9,000 trajectories to the public file and 1,000 to the held-out set.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from prepare.common import Frozen, check, dev_indices

SOURCES = [{"name": "1D_Burgers_Sols_Nu0.001.hdf5", "path": "1D/Burgers/Train",
            "url": "https://darus.uni-stuttgart.de/dataset.xhtml?persistentId=doi:10.18419/darus-2986",
            "md5": "44cb784d5a07aa2b1c864cabdcf625f9", "size": "7.7 GB", "license": "CC BY 4.0"}]
SPLIT_SEED, N_HIDDEN, N_IN, DEV_FRAC = 2986, 1000, 10, 0.1


def _freeze(u512: np.ndarray, x512: np.ndarray, t41: np.ndarray, sums, n_hidden: int = N_HIDDEN) -> Frozen:
    """u512: (N, 41, 512) already subsampled in time (every 5th of 201) and space (every 2nd of 1024)."""
    u512 = u512.astype(np.float32)
    u256 = u512[..., ::2]
    perm = np.random.default_rng(SPLIT_SEED).permutation(len(u512))
    hid, pub = perm[:n_hidden], perm[n_hidden:]
    return Frozen(public={"train.npz": {"u": u256[pub], "x": x512[::2].astype(np.float32), "t": t41.astype(np.float32),
                                        "dev_idx": dev_indices(len(pub), DEV_FRAC, SPLIT_SEED)}},
                  hidden={"u_in": u256[hid, :N_IN], "u_out": u256[hid, N_IN:],
                          "u_in_512": u512[hid, :N_IN], "u_out_512": u512[hid, N_IN:]},
                  sources_sha256=sums)


def build(raw: Path) -> Frozen:
    import h5py                               # operator environment only
    p = raw / SOURCES[0]["name"]
    sums = {p.name: check(p, md5=SOURCES[0]["md5"])}
    with h5py.File(p, "r") as f:
        assert f["tensor"].shape == (10000, 201, 1024), f["tensor"].shape
        u512 = f["tensor"][:, ::5, ::2]                  # strided read: never loads the full 8 GB
        x, t = f["x-coordinate"][:], f["t-coordinate"][:]
    return _freeze(u512, x[::2], t[:201:5], sums)


def synthetic(n: int) -> Frozen:
    """Viscous Burgers solved with a crude pseudo-spectral scheme; only for plumbing tests."""
    rng = np.random.default_rng(0)
    X, T = 1024, 201
    x = np.linspace(0, 1, X, endpoint=False)
    k = np.fft.rfftfreq(X, 1.0 / X) * 2 * np.pi
    u = np.zeros((n, X))
    for m in range(1, 4):
        u += rng.normal(0, 0.5 / m, (n, 1)) * np.sin(2 * np.pi * m * x + rng.uniform(0, 6.3, (n, 1)))
    frames, dt, nu = [u.copy()], 1e-3, 0.01
    for _ in range(T - 1):
        for _ in range(5):
            U = np.fft.rfft(u, axis=-1)
            ux = np.fft.irfft(1j * k * U, n=X, axis=-1)
            u = np.fft.irfft((U - dt * np.fft.rfft(u * ux, axis=-1)) / (1 + dt * nu * k ** 2), n=X, axis=-1)
        frames.append(u.copy())
    full = np.stack(frames, 1)
    return _freeze(full[:, ::5, ::2], x[::2], (np.arange(T) * 5 * dt)[::5], {"synthetic": f"seed0-n{n}"},
                   n_hidden=max(10, n // 10))
