"""Reference training for X-ROBUST-009: TF-IDF features, a multinomial logistic regression over the 150 intents,
temperature scaling and an abstain threshold, both fitted on the public validation split.

  python3 train.py --data "$ABENCH_DATA_DIR" --out checkpoint
The classifier is trained on the training split only, so the validation split stays a clean calibration set.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from src import features as F  # noqa: E402


def unpack(u, o):
    b = np.asarray(u, np.uint8).tobytes()
    return [b[o[i]:o[i + 1]].decode("utf-8") for i in range(len(o) - 1)]


def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--C", type=float, default=20.0)
    ap.add_argument("--quick", action="store_true", help="fewer solver iterations (plumbing tests)")
    a = ap.parse_args()
    from sklearn.linear_model import LogisticRegression
    with np.load(Path(a.data) / "clinc_public.npz", allow_pickle=False) as z:
        tr_t, tr_y = unpack(z["train_text_utf8"], z["train_text_offsets"]), z["train_label"]
        va_t, va_y = unpack(z["val_text_utf8"], z["val_text_offsets"]), z["val_label"]
    keep = tr_y >= 0                                   # out-of-scope training queries are not a class
    tr_t = [t for t, k in zip(tr_t, keep) if k]
    tr_y = tr_y[keep]
    vecs = F.vectorizers()
    for v in vecs:
        v.fit(tr_t)
    X = F.transform(vecs, tr_t)
    clf = LogisticRegression(C=a.C, max_iter=50 if a.quick else 300)
    clf.fit(X, tr_y)
    logits = F.transform(vecs, va_t) @ clf.coef_.T + clf.intercept_
    ins = va_y >= 0
    best_t, best_nll = 1.0, np.inf
    for t in np.exp(np.linspace(np.log(0.05), np.log(5), 60)):   # temperature on in-scope validation queries
        p = softmax(logits[ins] / t)
        nll = -np.mean(np.log(p[np.arange(ins.sum()), va_y[ins]] + 1e-12))
        if nll < best_nll:
            best_t, best_nll = t, nll
    p = softmax(logits / best_t)
    conf, lab = p.max(1), p.argmax(1)
    best_th, best_acc = 0.0, -1.0
    for th in np.linspace(0, 1, 201):                  # abstain threshold: best decision accuracy on validation
        ab = conf < th
        acc = np.mean(np.where(ins, (lab == va_y) & ~ab, ab))
        if acc > best_acc:
            best_th, best_acc = th, acc
    out = Path(a.out)
    F.save(vecs, out)
    np.save(out / "coef.npy", clf.coef_.astype(np.float32))
    np.save(out / "intercept.npy", clf.intercept_.astype(np.float32))
    (out / "calibration.json").write_text(json.dumps({"temperature": best_t, "threshold": best_th}), encoding="utf-8")
    print(f"temperature {best_t:.3f}, threshold {best_th:.3f}, validation decision accuracy {best_acc:.4f}")


if __name__ == "__main__":
    main()
