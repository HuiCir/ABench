"""Random-feature force field shared by model.py and train.py.

Descriptor: inverse interatomic distances D_p = 1 / r_ij over all pairs p = (i, j) (invariant to rotation and
translation; atoms keep the dataset's fixed order), standardised with the training mean and standard deviation.
Energy: E(x) = c + w · φ(D(x)) with random Fourier features φ_k = sqrt(2 / F) cos(W_k · D + b_k) of a Gaussian kernel
(Rahimi & Recht 2007), plus a quadratic term Σ_p a_p (D_p - mean D_p)² per pair (standardised D), which keeps
restoring forces outside the training data where the bounded Fourier features fade; forces F = -∂E/∂x by the chain
rule, linear in (w, a). w and c are fitted to energies and forces
jointly by regularised least squares (normal equations accumulated over configurations), so that the force labels —
3N numbers per configuration — shape the fit as in gradient-domain learning (Chmiela et al. 2017).
"""
from __future__ import annotations

import numpy as np

QUAD = 0.1                    # scale of the quadratic pair features relative to the Fourier features


def pairs(n: int):
    return np.triu_indices(n, 1)


def descriptor_and_jacobian(pos: np.ndarray):
    """pos (B, N, 3) -> D (B, P) and dD/dx as (B, P, N, 3) (sparse in structure; dense here)."""
    B, N, _ = pos.shape
    i, j = pairs(N)
    dv = pos[:, i] - pos[:, j]                                   # (B, P, 3)
    r = np.linalg.norm(dv, axis=-1)
    D = 1.0 / r
    g = -dv / r[..., None] ** 3                                  # dD/dx_i ; dD/dx_j = -g
    J = np.zeros((B, len(i), N, 3))
    P = np.arange(len(i))
    J[:, P, i] = g
    J[:, P, j] = -g
    return D, J


class Model:
    def __init__(self, W, b, w, c, mu, sd):
        self.W, self.b, self.w, self.c, self.mu, self.sd = W, b, w, c, mu, sd

    def features(self, pos: np.ndarray):
        """phi (B, F) and dphi/dx (B, F, N, 3)."""
        D, J = descriptor_and_jacobian(pos)
        Ds = (D - self.mu) / self.sd
        a = Ds @ self.W.T + self.b                               # (B, F)
        F = self.W.shape[0]
        phi = np.concatenate([np.sqrt(2.0 / F) * np.cos(a), QUAD * Ds ** 2], 1)
        dphi_dD = np.concatenate([-np.sqrt(2.0 / F) * np.sin(a)[..., None] * (self.W / self.sd)[None],
                                  QUAD * np.einsum("bp,pq->bpq", 2 * Ds / self.sd, np.eye(len(self.sd)))], 1)
        dphi_dx = np.einsum("bfp,bpnc->bfnc", dphi_dD, J)
        return phi, dphi_dx

    def predict(self, pos: np.ndarray):
        phi, dphi = self.features(pos)
        return self.c + phi @ self.w, -np.einsum("bfnc,f->bnc", dphi, self.w)


def fit(pos, energy, forces, n_features=2000, length_scale=None, reg=1e-6, energy_weight=1.0, seed=0, chunk=50):
    rng = np.random.default_rng(seed)
    D_all = np.concatenate([descriptor_and_jacobian(pos[s:s + 200])[0] for s in range(0, len(pos), 200)])
    mu, sd = D_all.mean(0), D_all.std(0) + 1e-8
    Ds = (D_all - mu) / sd
    if length_scale is None:
        idx = rng.choice(len(Ds), size=min(300, len(Ds)), replace=False)
        dd = np.linalg.norm(Ds[idx, None] - Ds[None, idx], axis=-1)
        length_scale = float(np.median(dd[dd > 0]))
    P = Ds.shape[1]
    W = rng.normal(0, 1.0 / length_scale, (n_features, P))
    b = rng.uniform(0, 2 * np.pi, n_features)
    m = Model(W, b, np.zeros(n_features + P), 0.0, mu, sd)
    c = float(energy.mean())
    n_features = n_features + P
    G = np.zeros((n_features, n_features))
    h = np.zeros(n_features)
    for s in range(0, len(pos), chunk):
        phi, dphi = m.features(pos[s:s + chunk])
        Bc = len(phi)
        Af = -dphi.reshape(Bc, n_features, -1).transpose(0, 2, 1).reshape(-1, n_features)   # force rows
        yf = forces[s:s + chunk].reshape(-1)
        Ae = phi * energy_weight
        ye = (energy[s:s + chunk] - c) * energy_weight
        G += Af.T @ Af + Ae.T @ Ae
        h += Af.T @ yf + Ae.T @ ye
    w = np.linalg.solve(G + reg * np.trace(G) / n_features * np.eye(n_features), h)
    m.w, m.c = w, c
    return m
