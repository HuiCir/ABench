"""U-KAN-009 metrics: test error of your layer relative to the MLP layer trained by the evaluator at the same parameter
budget, seed, data and recipe. Function tasks: RMSE / std of the test targets; image task: error rate.

Primary: error_ratio = exp(0.5 · mean_f ln(err_sub_f / err_mlp_f) + 0.5 · ln(err_sub_img / err_mlp_img)), so the
twelve function tasks and the image task weigh equally; lower is better, 1 means as good as the MLP. Errors are
floored at 1e-4 (functions) and 1e-3 (image) so that near-exact fits do not dominate the ratio.
Gate: the median training-step time of your network may be at most TIME_RATIO_MAX times the MLP's on every task.
"""
from __future__ import annotations

import numpy as np

from abench_metrics import OutputError, need  # noqa: F401

PRIMARY = {"metric": "error_ratio", "goal": "min"}
GATES = [{"metric": "max_step_time_ratio", "max": 3.0}, {"metric": "budget_violations", "max": 0}]
FLOOR = {"function": 1e-4, "image": 1e-3}
BUDGET = {"function": 3_000, "image": 100_000}


def _names(reference: dict) -> list[str]:
    return sorted(k[:-len("_y_test")] for k in reference if k.endswith("_y_test") and not k.startswith("img"))


def RESAMPLE(reference: dict) -> list:          # noqa: N802
    g = [{"reference": [f"{n}_y_test"], "outputs": [["functions", f"{n}_pred_sub"], ["functions", f"{n}_pred_mlp"]]}
         for n in _names(reference)]
    if "img_y_test" in reference:
        g.append({"reference": ["img_y_test"], "outputs": [["image", "img_pred_sub"], ["image", "img_pred_mlp"]]})
    return g


def _err(pred: np.ndarray, y: np.ndarray, kind: str) -> float:
    if kind == "function":
        if not np.all(np.isfinite(pred)):
            return float("inf")
        e = float(np.sqrt(np.mean((pred - y) ** 2)) / max(float(np.std(y)), 1e-12))
    else:
        e = float(np.mean(pred != y))
    return max(e, FLOOR[kind])


def score(outputs: dict, reference: dict, params: dict, seconds: dict) -> dict:
    out, logs, tratio, viol, div = {}, [], [], 0, 0.0

    def side(stage, prefix, kind):
        nonlocal viol, div
        g = lambda k: float(need(outputs, stage, f"{prefix}_{k}", shape=(1,), finite=False)[0])  # noqa: E731
        for who in ("sub", "mlp"):
            out[f"{prefix}_{who}_width"] = g(f"{who}_width")
            out[f"{prefix}_{who}_params"] = g(f"{who}_params")
        if out[f"{prefix}_sub_params"] > BUDGET[kind] or out[f"{prefix}_sub_width"] < 1:
            viol += 1
        div += g("sub_diverged")
        ts, tm = g("sub_step_ms"), g("mlp_step_ms")
        r = ts / tm if np.isfinite(ts) and tm > 0 else float("inf")
        out[f"{prefix}_step_time_ratio"] = r
        tratio.append(r)

    for n in _names(reference):
        y = np.asarray(reference[f"{n}_y_test"], np.float64)
        es = _err(np.asarray(need(outputs, "functions", f"{n}_pred_sub", shape=(len(y),), finite=False), np.float64), y, "function")
        em = _err(np.asarray(need(outputs, "functions", f"{n}_pred_mlp", shape=(len(y),), finite=False), np.float64), y, "function")
        out[f"{n}_err_sub"], out[f"{n}_err_mlp"], out[f"{n}_ratio"] = es, em, es / em
        logs.append(np.log(es / em))
        side("functions", n, "function")
    f_log = float(np.mean(logs)) if logs else 0.0
    out["function_error_ratio"] = float(np.exp(f_log))
    out["functions_better_than_mlp"] = float(np.mean(np.asarray(logs) < 0)) if logs else float("nan")
    i_log = 0.0
    if "img_y_test" in reference:
        y = np.asarray(reference["img_y_test"])
        es = _err(np.asarray(need(outputs, "image", "img_pred_sub", shape=(len(y),))), y, "image")
        em = _err(np.asarray(need(outputs, "image", "img_pred_mlp", shape=(len(y),))), y, "image")
        out["img_err_sub"], out["img_err_mlp"] = es, em
        out["image_error_ratio"] = es / em
        i_log = float(np.log(es / em))
        side("image", "img", "image")
    out["error_ratio"] = float(np.exp(0.5 * f_log + 0.5 * i_log)) if "img_y_test" in reference else float(np.exp(f_log))
    out["max_step_time_ratio"] = float(max(tratio)) if tratio else float("nan")
    out["budget_violations"] = float(viol)
    out["diverged_tasks"] = float(div)
    return out
