"""Original IC-SPD-001 accuracy gates and wall-clock score, computed by the parent."""
import numpy as np
from abench_metrics import OutputError, need

PRIMARY = {"metric": "train_seconds_mean", "goal": "min"}
GATES = [{"metric": "accuracy_cifar10_mean", "min": 0.94},
         {"metric": "accuracy_cifar10_1_mean", "min": 0.80}]


def score(outputs, reference, params, seconds):
    y, src = np.asarray(reference["y"]), np.asarray(reference["src"])
    p = need(outputs, "train", "pred", (len(y), 10))
    pred = p.argmax(1)
    out = {"train_seconds_mean": seconds["train"], "train_seconds_median": seconds["train"],
           "train_seconds_max": seconds["train"]}
    for name, val in (("cifar10", 0), ("cifar10_1", 1)):
        out[f"accuracy_{name}_mean"] = round(float((pred[src == val] == y[src == val]).mean()), 6)
    return out


def aggregate(rows):
    # The original score uses the median/max across seeds, not averages of per-seed medians/maxima.
    out = {k: round(float(np.mean([r[k] for r in rows])), 6) for k in rows[0]}
    out["train_seconds_median"] = round(float(np.median([r["train_seconds_mean"] for r in rows])), 4)
    out["train_seconds_max"] = max(r["train_seconds_mean"] for r in rows)
    return out
