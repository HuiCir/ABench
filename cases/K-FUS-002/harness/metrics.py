"""K-FUS-002 metrics (K-KB-001 uses the same file): per problem, correctness and the time of the stronger of PyTorch eager
and torch.compile divided by your time; the primary metric is their geometric mean with every ratio clipped to
[0.1, 10] and failed problems counted as 0.1."""
from __future__ import annotations

import numpy as np

from abench_metrics import OutputError, need  # noqa: F401

PRIMARY = {"metric": "speedup_geomean", "goal": "max"}
GATES = [{"metric": "forbidden_api", "max": 0.0}, {"metric": "tampered", "max": 0.0}]
CLIP = (0.1, 10.0)


def _gm(x: np.ndarray) -> float:
    return float(np.exp(np.mean(np.log(x)))) if len(x) else 0.0


def score(outputs: dict, reference: dict, params: dict, seconds: dict) -> dict:
    ref = need(outputs, "reference", "rows", shape=(None, 3), finite=False)
    rows = need(outputs, "bench", "rows", shape=(None, 5), finite=False)
    if len(rows) == 0 or len(rows) != len(ref):
        raise OutputError("reference and bench stages covered different problems")
    t_eager, t_comp, m_eager = ref.T
    has, worst, correct, t_new, m_new = rows.T
    ok = (correct == 1) & np.isfinite(t_new) & (t_new > 0)
    t_base = np.where(np.isfinite(t_comp), np.minimum(t_eager, t_comp), t_eager)
    s = np.where(ok, t_base / np.where(ok, t_new, 1.0), 0.0)
    s_eager = np.where(ok, t_eager / np.where(ok, t_new, 1.0), 0.0)
    mem = np.where(ok, m_new / np.maximum(m_eager, 1e-3), np.nan)
    return {
        "speedup_geomean": _gm(np.clip(np.where(ok, s, CLIP[0]), *CLIP)),
        "speedup_geomean_vs_eager": _gm(np.clip(np.where(ok, s_eager, CLIP[0]), *CLIP)),
        "fraction_correct": float(ok.mean()),
        "fast_1": float((ok & (s > 1.0)).mean()),                 # correct and faster than the stronger baseline
        "fast_1_vs_eager": float((ok & (s_eager > 1.0)).mean()),
        "fraction_submitted": float((has == 1).mean()),
        "median_memory_ratio": float(np.nanmedian(mem)) if ok.any() else float("nan"),
        "compile_failed": float((~np.isfinite(t_comp)).sum()),
        "n_problems": float(len(rows)),
        "forbidden_api": float(need(outputs, "bench", "forbidden_api", shape=(1,))[0]),
        "tampered": float(need(outputs, "bench", "tampered", shape=(1,))[0]),
    }
