"""IC-AUG-003 metrics: CIFAR-10 test accuracy of the fixed recipe trained with your augmentation policy (primary), and
accuracy on CIFAR-10.1, compared with the flip-and-translate baseline trained by the evaluator on the same seeds."""
from __future__ import annotations

import numpy as np

from abench_metrics import OutputError, accuracy, need  # noqa: F401

PRIMARY = {"metric": "accuracy_cifar10", "goal": "max"}
GATES = [{"metric": "train_time_ratio_vs_baseline", "max": 1.5}]
RESAMPLE = [{"reference": ["y_test"], "outputs": [["train", "proba_test"]]}]


def score(outputs: dict, reference: dict, params: dict, seconds: dict) -> dict:
    y = np.asarray(reference["y_test"]).astype(np.int64)
    P = need(outputs, "train", "proba_test", shape=(len(y), 10))
    out = {"accuracy_cifar10": accuracy(y, P.argmax(1)), "train_seconds": float(seconds["train"])}
    if "y_101" in reference:
        y1 = np.asarray(reference["y_101"]).astype(np.int64)
        out["accuracy_cifar10_1"] = accuracy(y1, need(outputs, "train", "proba_101", shape=(len(y1), 10)).argmax(1))
    return out


def relative(sub: dict, base: dict) -> dict:
    d = {"accuracy_delta_vs_flipcrop": sub["accuracy_cifar10"] - base["accuracy_cifar10"],
         "train_time_ratio_vs_baseline": sub["train_seconds"] / max(base["train_seconds"], 1e-9)}
    if "accuracy_cifar10_1" in sub and "accuracy_cifar10_1" in base:
        d["accuracy_cifar10_1_delta_vs_flipcrop"] = sub["accuracy_cifar10_1"] - base["accuracy_cifar10_1"]
    return d
