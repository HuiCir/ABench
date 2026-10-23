"""V-3DG-005 metrics: PSNR (primary) and SSIM of the rendered test views of six held-out scenes, averaged per scene and
then over scenes, relative to plain 3D Gaussian splatting run in the same evaluation."""
from __future__ import annotations

import numpy as np

import nvs
from abench_metrics import OutputError, need  # noqa: F401

SCENES = [f"scene{i}" for i in range(6)]
PRIMARY = {"metric": "mean_psnr", "goal": "max"}
GATES = [{"metric": "max_fit_seconds", "max": 630.0}]          # the 600 s budget plus 5% for timing noise


def score(outputs: dict, reference: dict, params: dict, seconds: dict) -> dict:
    out, ps, ss = {}, [], []
    for s in SCENES:
        gt = np.asarray(reference[f"{s}_test_images"])
        img = need(outputs, s, "images", shape=gt.shape)
        p = [nvs.psnr(a, b) for a, b in zip(img, gt)]
        q = [nvs.ssim(a, b) for a, b in zip(img, gt)]
        out[f"psnr_{s}"], out[f"ssim_{s}"] = float(np.mean(p)), float(np.mean(q))
        out[f"fit_seconds_{s}"] = float(seconds.get(s, float("nan")))
        ps.append(out[f"psnr_{s}"])
        ss.append(out[f"ssim_{s}"])
    out["mean_psnr"], out["mean_ssim"] = float(np.mean(ps)), float(np.mean(ss))
    out["max_fit_seconds"] = float(max(out[f"fit_seconds_{s}"] for s in SCENES))
    return out


def relative(sub: dict, base: dict) -> dict:
    d = {"mean_psnr_delta_vs_3dgs": sub["mean_psnr"] - base["mean_psnr"],
         "mean_ssim_delta_vs_3dgs": sub["mean_ssim"] - base["mean_ssim"], "3dgs_mean_psnr": base["mean_psnr"]}
    for s in SCENES:
        d[f"psnr_{s}_delta_vs_3dgs"] = sub[f"psnr_{s}"] - base[f"psnr_{s}"]
    return d
