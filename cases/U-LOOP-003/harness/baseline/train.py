"""U-LOOP-003 reference submission, training program (run in the sandbox; the evaluator only loads the checkpoint).

  python3 train.py --out "$ABENCH_SUBMISSION_DIR/checkpoint" [--seed 0] [--quick]

The only training data is algo_tasks.Sampler (lengths 1..20), imported from the interface directory. For every task:
sample 200 examples per length, record the most frequent output token for each local input window (model.keys) and
fit the output length as a linear function of the input length. Saves checkpoint/lookup.pkl. (--data is accepted for a
uniform command line; this case has no data files.)
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from model import TASK_NAMES, TWO_OPERANDS, keys

if os.environ.get("ABENCH_VIEW_DIR"):
    sys.path.insert(0, str(Path(os.environ["ABENCH_VIEW_DIR"]) / "interface"))
import algo_tasks  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quick", action="store_true", help="20 examples per length (smoke test)")
    a = ap.parse_args()
    sampler = algo_tasks.Sampler(a.seed)
    per_length = 20 if a.quick else 200
    model = {}
    for task in TASK_NAMES:
        two = task in TWO_OPERANDS
        table, prior = defaultdict(Counter), Counter()
        lx, ly = [], []
        for n in range(1, algo_tasks.MAX_TRAIN_LEN + 1):
            xs, ys = sampler.sample(task, n, per_length)
            for x, y in zip(xs, ys):
                lx.append(len(x)); ly.append(len(y))
                for k, v in zip(keys(x, two, len(y)), y):
                    table[k][int(v)] += 1
                    prior[int(v)] += 1
        A = np.stack([np.asarray(lx, float), np.ones(len(lx))], 1)
        coef, *_ = np.linalg.lstsq(A, np.asarray(ly, float), rcond=None)
        model[task] = {"table": {k: c.most_common(1)[0][0] for k, c in table.items()},
                       "prior": prior.most_common(1)[0][0], "len": coef, "two": two}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "lookup.pkl", "wb") as f:
        pickle.dump(model, f)


if __name__ == "__main__":
    main()
