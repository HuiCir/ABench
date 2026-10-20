"""FN-LOB-002 reference submission, training program (run in the sandbox; the evaluator only loads the checkpoint).

  python3 train.py --data "$ABENCH_DATA_DIR" --out "$ABENCH_SUBMISSION_DIR/checkpoint" [--seed 0] [--include-dev] [--quick]

Samples 60,000 training windows, computes model.feats, standardises them and solves a class-balanced one-vs-rest ridge
regression in closed form; saves W, mu and sd to checkpoint/ridge.npz. By default only the first 80% of the events are
used (the self-check scores windows from the last 20%); --include-dev uses all of them.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from model import WINDOW, feats

HORIZON_INDEX, N_CLASSES, LAM, DEV_FRACTION = 4, 3, 10.0, 0.2


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--include-dev", action="store_true")
    ap.add_argument("--quick", action="store_true", help="5,000 windows (smoke test)")
    a = ap.parse_args()
    with np.load(Path(a.data) / "train.npz", allow_pickle=False) as z:
        events, labels = z["events"], z["labels"]
    if not a.include_dev:
        cut = int(len(events) * (1 - DEV_FRACTION))
        events, labels = events[:cut], labels[:cut]
    rng = np.random.default_rng(a.seed)
    n = 5000 if a.quick else 60000
    t = rng.choice(np.arange(WINDOW - 1, len(events)), size=min(n, len(events) - WINDOW + 1), replace=False)
    w = events[t[:, None] + np.arange(-WINDOW + 1, 1)[None, :]].astype(np.float32)
    y = labels[t, HORIZON_INDEX].astype(np.int64)
    f = feats(w).astype(np.float64)
    mu, sd = f.mean(0), f.std(0) + 1e-6
    mu[-1], sd[-1] = 0.0, 1.0                                # bias column
    f = (f - mu) / sd
    cw = 1.0 / np.maximum(np.bincount(y, minlength=N_CLASSES), 1)
    sw = cw[y] / cw[y].mean()
    G = (f * sw[:, None]).T @ f + LAM * np.eye(f.shape[1])
    W = np.linalg.solve(G, (f * sw[:, None]).T @ np.eye(N_CLASSES)[y])
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "ridge.npz", W=W, mu=mu, sd=sd)


if __name__ == "__main__":
    main()
