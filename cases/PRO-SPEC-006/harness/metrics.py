"""PRO-SPEC-006 metrics: batch-1 speedup of your server over plain decoding on the same node (primary: the mean over
the three prompt sets of the ratio of total decoding times), greedy consistency and audit gates."""
from __future__ import annotations

import numpy as np

import pro_stages as P
from abench_metrics import OutputError, need  # noqa: F401

SETS = ("mtbench", "gsm8k", "humaneval")
PRIMARY = {"metric": "speedup", "goal": "max"}
GATES = P.GATES + [{"metric": "consistency", "min": 0.97}, {"metric": "complete", "min": 1.0}]


def RESAMPLE(reference: dict) -> list:          # noqa: N802
    return [{"reference": ["set_id"], "outputs": [["vanilla", "seconds"], ["spec", "seconds"]]}]


def score(outputs: dict, reference: dict, params: dict, seconds: dict) -> dict:
    out = P.scores(outputs)
    sid = np.asarray(reference["set_id"]).astype(int)
    v = need(outputs, "vanilla", "seconds", shape=(len(sid),))
    s = need(outputs, "spec", "seconds", shape=(len(sid),))
    sp = []
    for i, name in enumerate(SETS):
        m = sid == i
        if m.any():
            out[f"speedup_{name}"] = float(v[m].sum() / max(s[m].sum(), 1e-9))
            sp.append(out[f"speedup_{name}"])
    out["speedup"] = float(np.mean(sp))
    out["vanilla_tokens_per_s"] = float(len(sid) * 512 / max(v.sum(), 1e-9))
    out["spec_tokens_per_s"] = float(len(sid) * 512 / max(s.sum(), 1e-9))
    out["complete"] = float(min(np.mean(need(outputs, "spec", "complete")), np.mean(need(outputs, "vanilla", "complete"))))
    c = outputs.get("check", {}).get("consistent", [])
    out["consistency"] = float(np.mean(c)) if len(c) else 0.0
    return out
