"""BI-PROT-001 metrics: Spearman correlation between predicted scores and measured fitness of the query mutants of each
held-out assay (the 64 labelled mutants excluded), averaged over assays (primary), and averaged within each coarse
selection type (Stability, OrganismalFitness, Activity, Expression, Binding) as in ProteinGym."""
from __future__ import annotations

import numpy as np

from abench_metrics import OutputError, need  # noqa: F401

PRIMARY = {"metric": "mean_spearman", "goal": "max"}
GATES: list = []


def ids(reference: dict) -> list[str]:
    return sorted(k[:-2] for k in reference if k.endswith("_y"))


def RESAMPLE(reference: dict) -> list:          # noqa: N802  (bootstrap groups: each assay's mutants)
    return [{"reference": [f"{a}_y"], "outputs": [["predict", f"{a}_score"]]} for a in ids(reference)]


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.stats import spearmanr
    if np.ptp(a) == 0 or np.ptp(b) == 0:
        return 0.0
    return float(spearmanr(a, b).correlation)


def score(outputs: dict, reference: dict, params: dict, seconds: dict) -> dict:
    out, rhos, by = {}, [], {}
    for a in ids(reference):
        y = np.asarray(reference[f"{a}_y"], np.float64)
        s = need(outputs, "predict", f"{a}_score", shape=y.shape)
        r = _spearman(s, y)
        out[f"spearman_{a}"] = r
        rhos.append(r)
        by.setdefault(str(np.asarray(reference[f"{a}_type"])), []).append(r)
    out["mean_spearman"] = float(np.mean(rhos))
    for t, v in by.items():
        out[f"spearman_{t}"] = float(np.mean(v))
    out["mean_spearman_by_type"] = float(np.mean([np.mean(v) for v in by.values()]))
    out["load_seconds"] = float(seconds["predict"])
    return out
