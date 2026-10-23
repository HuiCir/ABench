"""IC-RES-002 metrics: test accuracy of the backbone trained at each of the two compute budgets; the primary metric is
their mean (two points of the accuracy-FLOPs curve); the FLOPs of each variant must stay within its budget. Relative
metrics compare with ResNet-18 / ResNet-50 trained by the evaluator under the same recipe."""
from __future__ import annotations

import numpy as np

from abench_metrics import OutputError, need  # noqa: F401

VARIANTS = ("small", "large")
PRIMARY = {"metric": "mean_accuracy", "goal": "max"}
GATES = [{"metric": "max_flops_ratio", "max": 1.0}, {"metric": "diverged", "max": 0.0},
         {"metric": "max_train_seconds", "max": 1500.0}]
RESAMPLE = [{"reference": ["y_test"], "outputs": [[f"train_{v}", "logits"] for v in VARIANTS]}]


def score(outputs: dict, reference: dict, params: dict, seconds: dict) -> dict:
    y = np.asarray(reference["y_test"])
    out, accs, fr, div, secs = {}, [], [], 0.0, []
    for v in VARIANTS:
        lg = need(outputs, f"train_{v}", "logits", shape=(len(y), 10))
        out[f"accuracy_{v}"] = float((lg.argmax(1) == y).mean())
        accs.append(out[f"accuracy_{v}"])
        out[f"gflops_{v}"] = float(need(outputs, f"train_{v}", "flops", shape=(1,))[0]) / 1e9
        out[f"mparams_{v}"] = float(need(outputs, f"train_{v}", "params", shape=(1,))[0]) / 1e6
        fr.append(float(need(outputs, f"train_{v}", "flops_ratio", shape=(1,))[0]))
        div = max(div, float(need(outputs, f"train_{v}", "diverged", shape=(1,))[0]))
        secs.append(float(seconds[f"train_{v}"]))
    out["mean_accuracy"] = float(np.mean(accs))
    out["max_flops_ratio"] = max(fr)
    out["diverged"] = div
    out["max_train_seconds"] = max(secs)
    return out


def relative(sub: dict, base: dict) -> dict:
    d = {f"accuracy_{v}_delta_vs_resnet": sub[f"accuracy_{v}"] - base[f"accuracy_{v}"] for v in VARIANTS}
    d["mean_accuracy_delta_vs_resnet"] = sub["mean_accuracy"] - base["mean_accuracy"]
    d["resnet_mean_accuracy"] = base["mean_accuracy"]
    return d
