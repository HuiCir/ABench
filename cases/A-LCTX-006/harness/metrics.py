"""A-LCTX-006 metrics: PG-19 tail loss with 4,096, 8,192 and 16,384 tokens of context (primary: their mean), short-
context loss at 1,024, copy-test loss at 8,192 and 16,384, relative to the RoPE baseline trained in the same run."""
from __future__ import annotations

import numpy as np

from abench_metrics import OutputError, need  # noqa: F401

LONG, COPY = (4096, 8192, 16384), (8192, 16384)
PRIMARY = {"metric": "long_loss_mean", "goal": "min"}
GATES = [{"metric": "diverged", "max": 0.0}, {"metric": "position_params", "max": 65536},
         {"metric": "short_loss_delta_vs_rope", "max": 0.02}]


def RESAMPLE(reference: dict) -> list:          # noqa: N802
    return [{"outputs": [["train", f"long_loss_{L}"] for L in LONG]}] + \
           [{"outputs": [["train", f"copy_loss_{L}"]]} for L in COPY]


def score(outputs: dict, reference: dict, params: dict, seconds: dict) -> dict:
    t = outputs.get("train", {})
    out = {"diverged": float(need(outputs, "train", "diverged", shape=(1,))[0]),
           "position_params": float(need(outputs, "train", "position_params", shape=(1,))[0]),
           "short_loss": float(need(outputs, "train", "short_loss", shape=(1,), finite=False)[0]),
           "train_seconds": float(seconds.get("train", float("nan")))}
    if out["diverged"]:
        out["long_loss_mean"] = float(out["short_loss"])
        return out
    longs = []
    for L in LONG:
        v = float(np.mean(need(outputs, "train", f"long_loss_{L}", finite=False)))
        out[f"long_loss_{L}"] = v if np.isfinite(v) else 20.0
        longs.append(out[f"long_loss_{L}"])
    out["long_loss_mean"] = float(np.mean(longs))
    for L in COPY:
        if f"copy_loss_{L}" in t:
            out[f"copy_loss_{L}"] = float(np.mean(np.asarray(t[f"copy_loss_{L}"], np.float64)))
    return out


def relative(sub: dict, base: dict) -> dict:
    d = {"short_loss_delta_vs_rope": sub["short_loss"] - base["short_loss"],
         "long_loss_mean_delta_vs_rope": sub["long_loss_mean"] - base["long_loss_mean"], "rope_long_loss_mean": base["long_loss_mean"]}
    for L in COPY:
        if f"copy_loss_{L}" in sub and f"copy_loss_{L}" in base:
            d[f"copy_loss_{L}_delta_vs_rope"] = sub[f"copy_loss_{L}"] - base[f"copy_loss_{L}"]
    return d
