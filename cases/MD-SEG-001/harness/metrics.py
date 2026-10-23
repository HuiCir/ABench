"""MD-SEG-001 metrics: Dice and normalised surface Dice on the held-out volumes of four MSD tasks (primary: Dice
averaged over foreground classes, volumes and then the four tasks), relative to the U-Net baseline trained in the
same run."""
from __future__ import annotations

import json

import numpy as np

import seg3d as S
from abench_metrics import OutputError, need  # noqa: F401

TASKS = ("hippocampus", "prostate", "heart", "unseen")
PRIMARY = {"metric": "mean_dice", "goal": "max"}
GATES = [{"metric": "diverged", "max": 0.0}, {"metric": "max_params", "max": 64_000_000}]


def _cfg(t: str, reference: dict) -> dict:
    if t != "unseen":
        return S.TASKS[t]
    return json.loads(str(np.asarray(reference["unseen_config"])[0]))


def score(outputs: dict, reference: dict, params: dict, seconds: dict) -> dict:
    out, dices, pars, div = {}, [], [], 0.0
    for t in TASKS:
        cfg = _cfg(t, reference)
        gts = S.unpack(reference[f"{t}_test_seg"], reference[f"{t}_test_seg_shape"])
        flat = need(outputs, t, "pred")
        shp = need(outputs, t, "pred_shape", shape=(len(gts), 3))
        if not np.array_equal(shp, np.asarray(reference[f"{t}_test_seg_shape"])):
            raise OutputError(f"{t}: predicted volumes have the wrong shapes")
        preds = S.unpack(flat, shp)
        d = [np.mean([S.dice(p, g, k) for k in range(1, cfg["classes"])]) for p, g in zip(preds, gts)]
        n = [np.mean([S.nsd(p, g, k, cfg["spacing"], cfg["nsd_mm"]) for k in range(1, cfg["classes"])])
             for p, g in zip(preds, gts)]
        out[f"dice_{t}"], out[f"nsd_{t}"] = float(np.mean(d)), float(np.mean(n))
        for k in range(1, cfg["classes"]):
            out[f"dice_{t}_class{k}"] = float(np.mean([S.dice(p, g, k) for p, g in zip(preds, gts)]))
        dices.append(out[f"dice_{t}"])
        pars.append(float(need(outputs, t, "params", shape=(1,))[0]))
        div = max(div, float(need(outputs, t, "diverged", shape=(1,))[0]))
        out[f"peak_mem_gb_{t}"] = float(need(outputs, t, "peak_mem_gb", shape=(1,))[0])
        out[f"train_seconds_{t}"] = float(seconds.get(t, float("nan")))
    out["mean_dice"] = float(np.mean(dices))
    out["mean_nsd"] = float(np.mean([out[f"nsd_{t}"] for t in TASKS]))
    out["max_params"], out["diverged"] = max(pars), div
    return out


def relative(sub: dict, base: dict) -> dict:
    d = {"mean_dice_delta_vs_unet": sub["mean_dice"] - base["mean_dice"],
         "mean_nsd_delta_vs_unet": sub["mean_nsd"] - base["mean_nsd"], "unet_mean_dice": base["mean_dice"]}
    for t in TASKS:
        d[f"dice_{t}_delta_vs_unet"] = sub[f"dice_{t}"] - base[f"dice_{t}"]
    return d
