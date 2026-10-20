"""GN-OGB-002 reference submission, training program (run in the sandbox; the evaluator only loads the checkpoint).

  python3 train.py --data "$ABENCH_DATA_DIR" --out "$ABENCH_SUBMISSION_DIR/checkpoint" [--include-dev] [--quick]

Closed-form one-vs-rest ridge regression to one-hot labels on SGC features (model.features), with the ridge strength
chosen on a validation set. By default the public valid nodes are left out entirely (they are the self-check's dev
set) and a random 10% of the train nodes is used for validation; --include-dev trains on train nodes and selects on
the valid nodes. Saves checkpoint/ridge.npz.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from model import features

N_CLASSES = 40


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--include-dev", action="store_true")
    ap.add_argument("--quick", action="store_true")         # closed form: nothing to shorten
    a = ap.parse_args()
    with np.load(Path(a.data) / "graph.npz", allow_pickle=False) as z:
        g = {k: z[k] for k in z.files}
    y, tr = g["y"], g["train_idx"]
    if a.include_dev:
        va = g["valid_idx"]
    else:
        rng = np.random.default_rng(a.seed)
        va = np.sort(rng.choice(tr, size=max(1, len(tr) // 10), replace=False))
        tr = np.setdiff1d(tr, va)
    f = features(g)
    Y = np.eye(N_CLASSES, dtype=np.float64)[y[tr]]
    ftr = f[tr].astype(np.float64)
    G, B = ftr.T @ ftr, ftr.T @ Y
    best = None
    for lam in (1e-1, 1.0, 10.0, 100.0):
        W = np.linalg.solve(G + lam * np.eye(len(G)), B)
        acc = float(((f[va] @ W).argmax(1) == y[va]).mean())
        if best is None or acc > best[0]:
            best = (acc, W)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "ridge.npz", W=best[1].astype(np.float32))


if __name__ == "__main__":
    main()
