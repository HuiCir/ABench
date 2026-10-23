"""TC-FT-001 reference submission, training program (run in the sandbox; the evaluator only loads the checkpoint).

  python3 train.py --data "$ABENCH_DATA_DIR" --out "$ABENCH_SUBMISSION_DIR/checkpoint" [--seed 0] [--include-dev] [--quick]

Hashed unigram + bigram counts → sublinear tf-idf → linear classifier trained by SGD on the modified Huber loss with an
averaged iterate (scikit-learn). The learned idf vector, weights and intercepts are saved as one .npz; model.py
re-creates the same features with a stateless hashing function and needs only NumPy and scikit-learn's vectorizer.
By default the dev documents (dev_idx) are left out so that selfcheck.py gives an unbiased dev estimate.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier

from model import vectorizer

if os.environ.get("ABENCH_VIEW_DIR"):
    sys.path.insert(0, str(Path(os.environ["ABENCH_VIEW_DIR"]) / "interface"))
from textpack import unpack  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--include-dev", action="store_true")
    ap.add_argument("--quick", action="store_true", help="few SGD epochs (smoke test)")
    a = ap.parse_args()
    with np.load(Path(a.data) / "train.npz", allow_pickle=False) as z:
        utf8, off, y, dev = z["text_utf8"], z["text_offsets"], z["y"], z["dev_idx"]
    keep = np.ones(len(y), bool)
    if not a.include_dev:
        keep[dev] = False
    counts = vectorizer().transform(unpack(utf8, off, np.flatnonzero(keep)))
    tfidf = TfidfTransformer(sublinear_tf=True)
    x = tfidf.fit_transform(counts)
    clf = SGDClassifier(loss="modified_huber", alpha=2e-6, max_iter=3 if a.quick else 15, tol=None, average=True, random_state=a.seed)
    clf.fit(x, y[keep])
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "linear.npz", idf=tfidf.idf_.astype(np.float32), coef=clf.coef_.astype(np.float32),
                        intercept=clf.intercept_.astype(np.float32), classes=clf.classes_.astype(np.int64))


if __name__ == "__main__":
    main()
