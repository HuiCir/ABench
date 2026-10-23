"""PH-MLIP-003 molecular dynamics with a submitted potential, and the stability criterion (evaluator code).

Units: positions Å, energies kcal/mol, forces kcal/mol/Å, masses amu, time fs, temperature K.
Integrator: Langevin dynamics with the BAOAB splitting (Leimkuhler & Matthews 2013), friction GAMMA, at TEMPERATURE.
Stability (after Fu et al. 2022, "Forces are not enough"): a trajectory becomes unstable at the first check where some
bond length deviates from its mean over the training configurations by more than BOND_TOLERANCE; bonds are the atom
pairs closer than 1.25 x the sum of covalent radii in the first training configuration.
"""
from __future__ import annotations

import numpy as np

KB = 0.0019872041                      # kcal/mol/K
ACC = 4.184e-4                         # (kcal/mol/Å)/amu in Å/fs²
MASS = {1: 1.008, 6: 12.011, 7: 14.007, 8: 15.999}
COVALENT = {1: 0.31, 6: 0.76, 7: 0.71, 8: 0.66}
DT, GAMMA, TEMPERATURE = 0.5, 0.01, 500.0
BOND_TOLERANCE, CHECK_EVERY = 0.5, 20


def bonds(z: np.ndarray, pos: np.ndarray) -> np.ndarray:
    r = np.array([COVALENT[int(a)] for a in z])
    d = np.linalg.norm(pos[:, None] - pos[None], axis=-1)
    i, j = np.triu_indices(len(z), 1)
    keep = d[i, j] < 1.25 * (r[i] + r[j])
    return np.stack([i[keep], j[keep]], 1)


def bond_lengths(pos: np.ndarray, b: np.ndarray) -> np.ndarray:
    """pos (..., N, 3) -> (..., n_bonds)"""
    return np.linalg.norm(pos[..., b[:, 0], :] - pos[..., b[:, 1], :], axis=-1)


def run(force_fn, z: np.ndarray, x0: np.ndarray, steps: int, ref_bonds: np.ndarray, b: np.ndarray, seed: int):
    """Langevin MD of a batch of K copies (x0: (K, N, 3)); force_fn(pos (K, N, 3)) -> forces (K, N, 3).
    Returns the stable time of each copy in fs (capped at steps * DT) and the number of force calls."""
    rng = np.random.default_rng(seed)
    m = np.array([MASS[int(a)] for a in z])[None, :, None]
    K = len(x0)
    x = np.array(x0, np.float64)
    v = rng.normal(size=x.shape) * np.sqrt(KB * TEMPERATURE * ACC / m)
    f = np.asarray(force_fn(x), np.float64)
    c1 = np.exp(-GAMMA * DT)
    c2 = np.sqrt((1 - c1 ** 2) * KB * TEMPERATURE * ACC / m)
    stable = np.full(K, steps * DT)
    alive = np.ones(K, bool)
    for s in range(1, steps + 1):
        v = v + 0.5 * DT * ACC * f / m                          # B
        x = x + 0.5 * DT * v                                    # A
        v = c1 * v + c2 * rng.normal(size=x.shape)              # O
        x = x + 0.5 * DT * v                                    # A
        f = np.asarray(force_fn(x), np.float64)
        if not np.isfinite(f).all():
            bad = ~np.isfinite(f).reshape(K, -1).all(1)
            stable[alive & bad] = s * DT
            alive &= ~bad
            f = np.nan_to_num(f, nan=0.0, posinf=0.0, neginf=0.0)
        v = v + 0.5 * DT * ACC * f / m                          # B
        if s % CHECK_EVERY == 0:
            dev = np.abs(bond_lengths(x, b) - ref_bonds[None]).max(1)
            broke = alive & ~(dev <= BOND_TOLERANCE)
            stable[broke] = s * DT
            alive &= ~broke
            if not alive.any():
                break
    return stable
