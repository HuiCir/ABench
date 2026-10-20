"""LG-HOLD-002 reference submission, training program (run in the sandbox; the evaluator only loads the checkpoint).

  python3 train.py --data "$ABENCH_DATA_DIR" --out "$ABENCH_SUBMISSION_DIR/checkpoint" [--seed 0] [--include-dev] [--quick]

Builds an idf table over prompts and holdings (terms seen at least twice), computes model.features for a seeded sample
of 20,000 training questions and fits the linear conditional-logit scorer by full-batch gradient descent. By default
the public validation split is not used (it is the self-check's dev set); --include-dev adds it to training.
Saves checkpoint/lexical.npz.
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from model import features, tok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--include-dev", action="store_true")
    ap.add_argument("--quick", action="store_true", help="2,000 questions, 50 steps (smoke test)")
    a = ap.parse_args()
    with np.load(Path(a.data) / "train.npz", allow_pickle=False) as z:
        prompts, holdings, y = list(z["prompt"]), [list(h) for h in z["holdings"]], z["y"].astype(np.int64)
        if a.include_dev:
            prompts += list(z["val_prompt"])
            holdings += [list(h) for h in z["val_holdings"]]
            y = np.concatenate([y, z["val_y"].astype(np.int64)])
    df, docs = Counter(), 0
    for p, hs in zip(prompts, holdings):
        for text in [p] + list(hs):
            u, b = tok(text)
            df.update(set(u) | set(b))
            docs += 1
    idf = {t: float(np.log(docs / (1 + n))) for t, n in df.items() if n >= 2}
    rng = np.random.default_rng(a.seed)
    idx = rng.permutation(len(y))[:2000 if a.quick else 20000]
    F = features({"idf": idf}, [prompts[i] for i in idx], [holdings[i] for i in idx])
    yy = y[idx]
    w = np.zeros(F.shape[-1])
    for _ in range(50 if a.quick else 300):
        s = F @ w
        p = np.exp(s - s.max(1, keepdims=True))
        p /= p.sum(1, keepdims=True)
        p[np.arange(len(yy)), yy] -= 1.0
        w -= 0.5 * (np.einsum("nc,ncf->f", p, F) / len(yy) + 1e-3 * w)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    terms = sorted(idf)
    np.savez_compressed(out / "lexical.npz", terms=np.asarray(terms), idf=np.asarray([idf[t] for t in terms]), w=w)


if __name__ == "__main__":
    main()
