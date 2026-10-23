"""C-SPEC-001 metrics: per workload, the speedup of your generate() over plain greedy decoding of the target (total
time of the evaluator's greedy loop divided by your total time, same prompts, same GPU); the primary metric is the
geometric mean over the five workloads; every output must be the target's greedy output (tie-tolerant)."""
from __future__ import annotations

import math

import numpy as np

from abench_metrics import OutputError, need  # noqa: F401

WORKLOADS = ("math", "code", "summarize", "chat", "translate")
PRIMARY = {"metric": "geomean_speedup", "goal": "max"}
GATES = [{"metric": "exact_fraction", "min": 1.0}]


def score(outputs: dict, reference: dict, params: dict, seconds: dict) -> dict:
    out, sp, ex = {}, [], []
    for w in WORKLOADS:
        tr = float(need(outputs, "decode", f"{w}_ref_seconds", shape=(1,))[0])
        ts = float(need(outputs, "decode", f"{w}_sub_seconds", shape=(1,))[0])
        e = np.asarray(outputs["decode"][f"{w}_exact"], np.float64)
        out[f"speedup_{w}"] = tr / max(ts, 1e-9)
        out[f"exact_{w}"] = float(e.mean())
        out[f"ref_tokens_per_second_{w}"] = float(need(outputs, "decode", f"{w}_tokens", shape=(1,))[0]) / max(tr, 1e-9)
        sp.append(out[f"speedup_{w}"])
        ex.append(e)
    out["geomean_speedup"] = float(math.exp(np.mean(np.log(np.maximum(sp, 1e-9)))))
    out["min_speedup"] = float(min(sp))
    out["exact_fraction"] = float(np.concatenate(ex).mean())
    out["load_seconds"] = float(seconds["decode"])
    return out
