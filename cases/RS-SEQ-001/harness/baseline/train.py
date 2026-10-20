"""RS-SEQ-001 reference submission, training program (run in the sandbox; the evaluator only loads the checkpoint).

  python3 train.py --data "$ABENCH_DATA_DIR" --out "$ABENCH_SUBMISSION_DIR/checkpoint" [--include-dev] [--quick]

Counts consecutive item transitions i -> j within every user's sequence (log-damped) and a 90-day time-decayed
popularity, and saves both to checkpoint/markov.npz (float16 transition matrix, about 28 MB). By default each user's
final public item is left out, matching the dev split of selfcheck.py; --include-dev uses the full histories.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

BETA = 0.35


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)          # deterministic; kept for a uniform command line
    ap.add_argument("--include-dev", action="store_true")
    ap.add_argument("--quick", action="store_true")         # nothing to shorten
    a = ap.parse_args()
    with np.load(Path(a.data) / "train.npz", allow_pickle=False) as z:
        items, times, off, n_items = z["items"].astype(np.int64), z["times"], z["offsets"], int(z["n_items"])
    if not a.include_dev:                                    # drop each user's final public item
        keep = np.ones(len(items), bool)
        keep[off[1:][np.diff(off) > 0] - 1] = False
        items, times = items[keep], times[keep]
        off = np.concatenate([[0], np.cumsum(np.maximum(np.diff(off) - 1, 0))]).astype(np.int64)
    T = np.zeros((n_items + 1, n_items + 1), dtype=np.float32)
    same_user = np.ones(max(len(items) - 1, 0), bool)
    ends = off[1:-1] - 1
    same_user[ends[(ends >= 0) & (ends < len(same_user))]] = False   # no transition across user boundaries
    np.add.at(T, (items[:-1][same_user], items[1:][same_user]), 1.0)
    w = np.exp(-(float(times.max()) - times.astype(np.float64)) / (90 * 86400.0))
    pop = np.bincount(items, weights=w, minlength=n_items + 1).astype(np.float32)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "markov.npz", T=np.log1p(T).astype(np.float16), pop=(BETA * np.log1p(pop)).astype(np.float32))


if __name__ == "__main__":
    main()
