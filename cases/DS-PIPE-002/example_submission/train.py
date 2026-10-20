"""DS-PIPE-002 reference submission, training program (run in the sandbox; the evaluator only loads the checkpoint).

  python3 train.py --data "$ABENCH_DATA_DIR" --out "$ABENCH_SUBMISSION_DIR/checkpoint" [--seed 0] [--include-dev] [--quick]

The same procedure for every dataset: histogram gradient-boosted trees (missing values handled natively) stopped
early on an internal 10% validation split, in the spirit of the anchor's robust defaults. By default the dev rows
(dev_idx) are left out so that selfcheck.py gives an unbiased dev estimate.
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier


def fit_one(X, y, seed, quick=False):
    clf = HistGradientBoostingClassifier(learning_rate=0.06, max_iter=30 if quick else 800, max_leaf_nodes=31,
                                         min_samples_leaf=20, l2_regularization=1e-3, early_stopping=True,
                                         validation_fraction=0.1, n_iter_no_change=40, random_state=int(seed))
    return clf.fit(np.asarray(X, dtype=np.float32), np.asarray(y))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--include-dev", action="store_true")
    ap.add_argument("--quick", action="store_true", help="few boosting rounds (smoke test)")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for p in sorted(Path(a.data).glob("d*.npz")):
        with np.load(p, allow_pickle=False) as z:
            x, y, dev = z["x"], z["y"], z["dev_idx"]
        keep = np.ones(len(y), bool)
        if not a.include_dev:
            keep[dev] = False
        with open(out / (p.stem + ".pkl"), "wb") as f:
            pickle.dump(fit_one(x[keep], y[keep], a.seed, a.quick), f)


if __name__ == "__main__":
    main()
