"""V-SMP-002 metrics: FID at NFE 5, 10 and 20 against CIFAR-10 training-set Inception statistics."""
from __future__ import annotations

import numpy as np

from abench_metrics import OutputError, need  # noqa: F401

NFES = (5, 10, 20)
PRIMARY = {"metric": "fid_mean_nfe5_10", "goal": "min"}     # 2026-10-21 audit: the spread is largest at low NFE
GATES = [{"metric": "nfe_violations", "max": 0.0}]


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
    used = need(outputs, "sample", "nfe_used", shape=(3,))
    out = {"nfe_violations": float((used > np.asarray(NFES)).sum())}
    for n in NFES:
        mu = need(outputs, "fid", f"mu_{n}", shape=(2048,))
        sg = need(outputs, "fid", f"sigma_{n}", shape=(2048, 2048))
        out[f"fid_nfe{n}"] = fid(mu, sg, np.asarray(reference["ref_mu"]), np.asarray(reference["ref_sigma"]))
    out["fid_mean_nfe5_10"] = (out["fid_nfe5"] + out["fid_nfe10"]) / 2
    out["sampling_seconds"] = float(seconds.get("sample.finish", 0.0))
    return out


def relative(sub: dict, base: dict) -> dict:
    d = {f"fid_nfe{n}_delta_vs_ddim": sub[f"fid_nfe{n}"] - base[f"fid_nfe{n}"] for n in NFES}
    d["fid_mean_nfe5_10_delta_vs_ddim"] = sub["fid_mean_nfe5_10"] - base["fid_mean_nfe5_10"]
    return d
