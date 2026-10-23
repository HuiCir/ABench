"""FN-LTSF-005 reference submission, training program (run in the sandbox; the evaluator only loads the checkpoint).

  python3 train.py --data "$ABENCH_DATA_DIR" --out "$ABENCH_SUBMISSION_DIR/checkpoint" [--final] [--quick]

The same procedure for every dataset and horizon: a channel-shared ridge forecaster with last-value normalisation
(src/linear.py). Lookback (96, 192, 336 or 720) and penalty are chosen per dataset and horizon by mean squared error on
validation origins, after fitting on the training rows. Without --final the checkpoint is that training-rows fit (so
that selfcheck.py, which tests on the validation rows, is unbiased); with --final the chosen settings are refitted on
training and validation rows together, which is what to submit.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src import linear as LIN  # noqa: E402

DATASETS = ("etth1", "etth2", "ettm1", "ettm2", "electricity")
HORIZONS = (96, 192, 336, 720)
LOOKBACKS = (96, 192, 336, 720)
LAMBDAS = (1e-4, 1e-3, 1e-2, 1e-1, 1.0)


def fit_dataset(series: np.ndarray, train_end: int, val_end: int, final: bool, quick: bool) -> dict:
    c = series.shape[1]
    lookbacks = LOOKBACKS[1:3] if quick else LOOKBACKS
    lams = LAMBDAS[1:4] if quick else LAMBDAS
    vo = np.arange(train_end, val_end - LIN.HMAX + 1, max(1, (val_end - train_end - LIN.HMAX) // 48), dtype=np.int64)
    yv = np.stack([series[o:o + LIN.HMAX] for o in vo])
    best = {h: (np.inf, None, None) for h in HORIZONS}
    for lb in lookbacks:
        a, b = LIN.gram(series, LIN.train_origins(0, train_end, lb, c), lb)
        xv = np.stack([series[o - LIN.HMAX:o] for o in vo])
        for lam in lams:
            w = LIN.solve(a, b, lam)
            for h in HORIZONS:
                err = float(np.mean((LIN.predict(w, xv, h) - yv[:, :h]) ** 2))
                if err < best[h][0]:
                    best[h] = (err, lb, lam)
    out = {}
    for h in HORIZONS:
        _, lb, lam = best[h]
        end = val_end if final else train_end
        a, b = LIN.gram(series, LIN.train_origins(0, end, lb, c, horizon=h), lb, horizon=h)
        out[h] = {"w": LIN.solve(a, b, lam), "lookback": lb, "lambda": lam, "val_mse": best[h][0]}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--final", action="store_true", help="refit on training and validation rows")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = {}
    for d in DATASETS:
        p = Path(a.data) / f"{d}.npz"
        if not p.is_file():
            continue
        with np.load(p) as z:
            s, tr, va = z["series"], int(z["train_end"]), int(z["val_end"])
        fits = fit_dataset(s, tr, va, a.final, a.quick)
        np.savez(out / f"{d}.npz", **{f"w_{h}": f["w"] for h, f in fits.items()})
        meta[d] = {str(h): {k: f[k] for k in ("lookback", "lambda", "val_mse")} for h, f in fits.items()}
        print(d, {h: (f["lookback"], f["lambda"], round(f["val_mse"], 4)) for h, f in fits.items()}, flush=True)
    (out / "settings.json").write_text(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
