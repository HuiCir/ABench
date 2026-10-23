"""DS-GBM-001 metrics: per held-out dataset, test accuracy (classification, threshold 0.5 on the predicted probability
of class 1) or R² (regression, clipped at 0 for the mean); their mean over datasets is the primary metric. Relative
metrics compare with the HistGradientBoosting baseline trained by the evaluator under the same conditions."""
from __future__ import annotations

import numpy as np

from abench_metrics import OutputError, auc_binary, need  # noqa: F401

PRIMARY = {"metric": "mean_score", "goal": "max"}
GATES = [{"metric": "fit_seconds", "max": 1200.0}]


def datasets(reference: dict) -> list[str]:
    return sorted(k[: -len("_y_test")] for k in reference if k.endswith("_y_test"))


def RESAMPLE(reference: dict) -> list:          # noqa: N802  (bootstrap groups: each dataset's test rows)
    return [{"reference": [f"{d}_y_test"], "outputs": [["fit", f"pred_{d}"]]} for d in datasets(reference)]


def score(outputs: dict, reference: dict, params: dict, seconds: dict) -> dict:
    out, scores = {}, []
    for d in datasets(reference):
        y = np.asarray(reference[f"{d}_y_test"])
        task = int(np.asarray(reference[f"{d}_task"]).reshape(-1)[0])
        p = need(outputs, "fit", f"pred_{d}", shape=(len(y),))
        if task == 0:
            if (p < -1e-9).any() or (p > 1 + 1e-9).any():
                raise OutputError(f"fit.pred_{d}: classification predictions must be probabilities of class 1 in [0, 1]")
            out[f"accuracy_{d}"] = float(((p >= 0.5) == (y == 1)).mean())
            out[f"auc_{d}"] = auc_binary(y == 1, p)
            s = out[f"accuracy_{d}"]
        else:
            yv = y.astype(np.float64)
            sst = float(((yv - yv.mean()) ** 2).sum())
            r2 = 1.0 - float(((yv - p) ** 2).sum()) / max(sst, 1e-12)
            out[f"r2_{d}"] = r2
            s = max(r2, 0.0)
        out[f"score_{d}"] = s
        scores.append(s)
    out["mean_score"] = float(np.mean(scores))
    out["n_datasets"] = float(len(scores))
    out["fit_seconds"] = float(seconds["fit"])
    out["predict_seconds"] = float(seconds.get("fit.finish", 0.0))
    return out


def relative(sub: dict, base: dict) -> dict:
    ds = [k[len("score_"):] for k in sub if k.startswith("score_") and f"score_{k[len('score_'):]}" in base]
    gains = [sub[f"score_{d}"] - base[f"score_{d}"] for d in ds]
    return {"mean_score_delta_vs_hgb": sub["mean_score"] - base["mean_score"],
            "win_rate_vs_hgb": float(np.mean([g > 0 for g in gains])) if gains else 0.0,
            "hgb_mean_score": base["mean_score"]}
