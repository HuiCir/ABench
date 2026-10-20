"""BI-DNA-002 reference submission, training program (run in the sandbox; the evaluator only loads the checkpoint).

  python3 train.py --data "$ABENCH_DATA_DIR" --out "$ABENCH_SUBMISSION_DIR/checkpoint" [--seed 0] [--include-dev] [--quick]

Per dataset: k-mer spectrum features (model.spectrum), standardised, and a ridge regression to ±1 labels in closed
form, with the ridge strength chosen on a seeded 10% split and the final fit on all training rows. By default the
dev sequences (dev_idx) are left out so that selfcheck.py gives an unbiased dev estimate. Saves checkpoint/<dataset>.npz.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from model import spectrum


def fit_one(seqs, y, rng):
    F = spectrum(seqs).astype(np.float64)
    mu, sd = F.mean(0), F.std(0) + 1e-6
    F = np.concatenate([(F - mu) / sd, np.ones((len(F), 1))], 1)
    t = 2.0 * y - 1.0
    perm = rng.permutation(len(y))
    va, tr = perm[: len(y) // 10], perm[len(y) // 10:]
    G, B = F[tr].T @ F[tr], F[tr].T @ t[tr]
    best = None
    for lam in (10.0, 100.0, 1000.0, 10000.0):
        w = np.linalg.solve(G + lam * np.eye(len(G)), B)
        acc = float(((F[va] @ w > 0) == (y[va] == 1)).mean())
        if best is None or acc > best[0]:
            best = (acc, lam)
    w = np.linalg.solve(F.T @ F + best[1] * np.eye(F.shape[1]), F.T @ t)
    return {"w": w, "mu": mu, "sd": sd}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--include-dev", action="store_true")
    ap.add_argument("--quick", action="store_true", help="at most 3,000 sequences per dataset (smoke test)")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for p in sorted(Path(a.data).glob("*.npz")):
        with np.load(p, allow_pickle=False) as z:
            seq, y, dev = z["seq"], z["y"].astype(np.int64), z["dev_idx"]
        keep = np.ones(len(y), bool)
        if not a.include_dev:
            keep[dev] = False
        idx = np.flatnonzero(keep)[:3000] if a.quick else np.flatnonzero(keep)
        np.savez(out / f"{p.stem}.npz", **fit_one([str(s) for s in seq[idx]], y[idx], rng))


if __name__ == "__main__":
    main()
