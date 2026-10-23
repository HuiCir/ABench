"""V-GEN-001 metrics: FID of 10,000 generated images after at most 25 and 100 network evaluations, against the CIFAR-10 training-set
Inception statistics; the ratio of the trace of the generated feature covariance to the reference one is reported as
a diversity check."""
from __future__ import annotations

import numpy as np

from abench_metrics import OutputError, need  # noqa: F401

STEPS = (25, 100)
PRIMARY = {"metric": "fid_100", "goal": "min"}
GATES = [{"metric": "step_violations", "max": 0.0}, {"metric": "n_params", "max": 40_000_000.0}]


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
    used = need(outputs, "sample", "calls_used", shape=(len(STEPS),))
    out = {"step_violations": float((used > np.asarray(STEPS)).sum()),
           "n_params": float(need(outputs, "sample", "n_params", shape=(1,))[0])}
    ref_mu, ref_sigma = np.asarray(reference["ref_mu"]), np.asarray(reference["ref_sigma"])
    for s in STEPS:
        mu = need(outputs, "fid", f"mu_{s}", shape=(2048,))
        sg = need(outputs, "fid", f"sigma_{s}", shape=(2048, 2048))
        out[f"fid_{s}"] = fid(mu, sg, ref_mu, ref_sigma)
        out[f"diversity_ratio_{s}"] = float(np.trace(sg) / np.trace(ref_sigma))
    out["fid_100"] = float(np.mean([out[f"fid_{s}"] for s in (100,)]))
    out["load_seconds"] = float(seconds["sample"])
    out["sampling_seconds"] = float(seconds.get("sample.finish", 0.0))
    return out
