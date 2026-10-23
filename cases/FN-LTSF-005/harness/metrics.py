"""FN-LTSF-005 metrics: error relative to the fitted linear reference (primary: geometric mean over the 20
dataset-horizon pairs of MSE / reference MSE), and MSE and MAE per pair on the standardised scale."""
from __future__ import annotations

import numpy as np

import ltsf as L
from abench_metrics import OutputError, need  # noqa: F401

PRIMARY = {"metric": "mse_ratio_geomean", "goal": "min"}
GATES: list = []


def RESAMPLE(reference: dict) -> list:          # noqa: N802
    """Windows of one dataset are resampled together across its four horizons; datasets independently."""
    out = []
    for d in L.DATASETS:
        if f"y_{d}" in reference:
            out.append({"reference": [f"y_{d}"] + [f"ref_mse_{d}_{h}" for h in L.HORIZONS],
                        "outputs": [["predict", f"f_{d}_{h}"] for h in L.HORIZONS]})
    return out


def score(outputs: dict, reference: dict, params: dict, seconds: dict) -> dict:
    out, logs = {}, []
    for d in L.DATASETS:
        if f"y_{d}" not in reference:
            continue
        y = np.asarray(reference[f"y_{d}"])
        rd = []
        for h in L.HORIZONS:
            f = need(outputs, "predict", f"f_{d}_{h}", shape=(len(y), h, y.shape[2]))
            mse = float(np.mean(L.window_mse(f, y[:, :h])))
            ref = float(np.mean(np.asarray(reference[f"ref_mse_{d}_{h}"])))
            out[f"mse_{d}_{h}"] = mse
            out[f"mae_{d}_{h}"] = float(np.mean(L.window_mae(f, y[:, :h])))
            out[f"ref_mse_{d}_{h}"] = ref
            out[f"ratio_{d}_{h}"] = mse / ref
            logs.append(np.log(mse / ref))
            rd.append(np.log(mse / ref))
        out[f"mse_ratio_{d}"] = float(np.exp(np.mean(rd)))
    out["mse_ratio_geomean"] = float(np.exp(np.mean(logs)))
    out["pairs_better_than_reference"] = float(np.mean(np.asarray(logs) < 0))
    out["mse_ratio_worst"] = float(np.exp(np.max(logs)))
    if "predict" in outputs:
        out["predict_seconds"] = float(np.asarray(outputs["predict"].get("predict_seconds", [np.nan]))[0])
        out["checkpoint_mb"] = float(np.asarray(outputs["predict"].get("checkpoint_bytes", [0]))[0]) / 2**20
    out["load_seconds"] = float(seconds.get("predict", np.nan))
    return out
