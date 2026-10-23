"""V-DIT-003 metrics: FID of 5,000 class-balanced samples with guidance 1.5 (primary) and without guidance, against the
Inception statistics of the Imagenette validation images; backbone FLOPs relative to DiT-S/2 trained in the same run."""
from __future__ import annotations

import numpy as np

from abench_metrics import OutputError, need  # noqa: F401

TAGS = ("cfg15", "cfg10")
PRIMARY = {"metric": "fid_cfg15", "goal": "min"}
GATES = [{"metric": "diverged", "max": 0.0}, {"metric": "flops_ratio_vs_dit", "max": 1.10}]


def fid(mu1, s1, mu2, s2) -> float:
    from scipy import linalg
    diff = mu1 - mu2
    covmean = linalg.sqrtm(s1.dot(s2))
    if isinstance(covmean, tuple):
        covmean = covmean[0]
    if not np.isfinite(covmean).all():
        off = np.eye(s1.shape[0]) * 1e-6
        covmean = linalg.sqrtm((s1 + off).dot(s2 + off))
        if isinstance(covmean, tuple):
            covmean = covmean[0]
    return float(diff.dot(diff) + np.trace(s1) + np.trace(s2) - 2 * np.trace(covmean.real))


def score(outputs: dict, reference: dict, params: dict, seconds: dict) -> dict:
    out = {"flops": float(need(outputs, "train", "flops", shape=(1,))[0]),
           "params": float(need(outputs, "train", "params", shape=(1,))[0]),
           "diverged": float(need(outputs, "train", "diverged", shape=(1,))[0]),
           "train_seconds": float(seconds.get("train", float("nan")))}
    if "nonfinite_samples" in outputs.get("train", {}):
        out["diverged"] = 1.0
    mu_r, s_r = np.asarray(reference["ref_mu"]), np.asarray(reference["ref_sigma"])
    for t in TAGS:
        mu = need(outputs, "fid", f"mu_{t}", shape=(2048,))
        sg = need(outputs, "fid", f"sigma_{t}", shape=(2048, 2048))
        out[f"fid_{t}"] = fid(mu, sg, mu_r, s_r)
        out[f"diversity_ratio_{t}"] = float(np.trace(sg) / np.trace(s_r))
    return out


def relative(sub: dict, base: dict) -> dict:
    return {"flops_ratio_vs_dit": sub["flops"] / base["flops"], "fid_cfg15_delta_vs_dit": sub["fid_cfg15"] - base["fid_cfg15"],
            "fid_cfg10_delta_vs_dit": sub["fid_cfg10"] - base["fid_cfg10"], "dit_fid_cfg15": base["fid_cfg15"]}
