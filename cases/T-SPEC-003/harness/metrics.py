"""T-SPEC-003 metrics: FID after 5 and 10 sequential rounds against CIFAR-10 training-set Inception statistics."""
from __future__ import annotations

import numpy as np

from abench_metrics import OutputError, need  # noqa: F401

ROUNDS = (5, 10)
PRIMARY = {"metric": "fid_mean_r5_r10", "goal": "min"}
GATES = [{"metric": "round_violations", "max": 0.0}]


def fid(mu1, s1, mu2, s2) -> float:
    from scipy import linalg
    diff = mu1 - mu2
    covmean = linalg.sqrtm(s1.dot(s2))                  # (the disp argument was removed in recent SciPy)
    if isinstance(covmean, tuple):
        covmean = covmean[0]
    if not np.isfinite(covmean).all():
        off = np.eye(s1.shape[0]) * 1e-6
        covmean = linalg.sqrtm((s1 + off).dot(s2 + off))
        if isinstance(covmean, tuple):
            covmean = covmean[0]
    covmean = covmean.real
    return float(diff.dot(diff) + np.trace(s1) + np.trace(s2) - 2 * np.trace(covmean))


def score(outputs: dict, reference: dict, params: dict, seconds: dict) -> dict:
    used = need(outputs, "sample", "rounds_used", shape=(len(ROUNDS),))
    evals = need(outputs, "sample", "evaluations_per_image", shape=(len(ROUNDS),))
    out = {"round_violations": float((used > np.asarray(ROUNDS)).sum())}
    for i, r in enumerate(ROUNDS):
        mu = need(outputs, "fid", f"mu_{r}", shape=(2048,))
        sg = need(outputs, "fid", f"sigma_{r}", shape=(2048, 2048))
        out[f"fid_r{r}"] = fid(mu, sg, np.asarray(reference["ref_mu"]), np.asarray(reference["ref_sigma"]))
        out[f"evaluations_per_image_r{r}"] = float(evals[i])
    out["fid_mean_r5_r10"] = (out["fid_r5"] + out["fid_r10"]) / 2
    out["sampling_seconds"] = float(seconds.get("sample.finish", 0.0))
    return out


def relative(sub: dict, base: dict) -> dict:
    d = {f"fid_r{r}_delta_vs_ddim": sub[f"fid_r{r}"] - base[f"fid_r{r}"] for r in ROUNDS}
    d["fid_mean_r5_r10_delta_vs_ddim"] = sub["fid_mean_r5_r10"] - base["fid_mean_r5_r10"]
    return d
