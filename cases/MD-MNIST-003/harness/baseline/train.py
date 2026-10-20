"""MD-MNIST-003 reference submission, training program (run in the sandbox; the evaluator only loads the checkpoint).

  python3 train.py --data "$ABENCH_DATA_DIR" --out "$ABENCH_SUBMISSION_DIR/checkpoint" [--seed 0] [--include-dev] [--quick]

Per dataset: a seeded random projection with ReLU (model.phi), then one-vs-rest ridge regression to one-hot labels in
closed form, with the ridge strength chosen on a validation set. By default the official validation split is left out
(it is the self-check's dev set) and a 10% slice of the train split is used for selection; --include-dev selects on
the validation split instead. Saves checkpoint/blood.npz and checkpoint/organa.npz.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from model import N_RANDOM, flat, phi

DATASETS = {"blood": 8, "organa": 11}


def fit_one(xtr, ytr, xva, yva, k, rng, n_random):
    f = flat(xtr)
    m = {"mu": f.mean(0), "sd": f.std(0) + 1e-3}
    m["R"] = (rng.standard_normal((f.shape[1], n_random)) / np.sqrt(f.shape[1])).astype(np.float32)
    m["b"] = rng.uniform(-1, 1, n_random).astype(np.float32)
    F = phi(m, xtr).astype(np.float64)
    G, B = F.T @ F, F.T @ np.eye(k)[ytr]
    Fv = phi(m, xva)
    best = None
    for lam in (1.0, 10.0, 100.0, 1000.0):
        W = np.linalg.solve(G + lam * np.eye(len(G)), B)
        acc = float(((Fv @ W).argmax(1) == yva).mean())
        if best is None or acc > best[0]:
            best = (acc, W)
    m["W"] = best[1].astype(np.float32)
    return m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--include-dev", action="store_true")
    ap.add_argument("--quick", action="store_true", help="256 random features, 3,000 training images (smoke test)")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for d, k in DATASETS.items():
        with np.load(Path(a.data) / f"{d}.npz", allow_pickle=False) as z:
            xtr, ytr, xva, yva = z["x_train"], z["y_train"].reshape(-1), z["x_val"], z["y_val"].reshape(-1)
        if a.quick:
            xtr, ytr = xtr[:3000], ytr[:3000]
        if not a.include_dev:                           # select on a slice of the train split instead
            n = max(1, len(xtr) // 10)
            xtr, ytr, xva, yva = xtr[n:], ytr[n:], xtr[:n], ytr[:n]
        m = fit_one(xtr, ytr, xva, yva, k, rng, 256 if a.quick else N_RANDOM)
        np.savez(out / f"{d}.npz", **m)


if __name__ == "__main__":
    main()
