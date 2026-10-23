"""U-DLM-005 metrics: generative perplexity of the samples under the scorer at NFE 16, 32 and 64, unconditional and
continuing held-out prefixes (primary: mean of log generative perplexity over the six settings), with an entropy floor
so that repetitive text cannot win, and wall-clock time."""
from __future__ import annotations

import math

import numpy as np

from abench_metrics import OutputError, need  # noqa: F401

NFES = (16, 32, 64)
PRIMARY = {"metric": "mean_log_gen_ppl", "goal": "min"}
GATES = [{"metric": "min_entropy_ratio", "min": 0.9}]


def RESAMPLE(reference: dict) -> list:          # noqa: N802
    g = []
    for n in NFES:
        g.append({"outputs": [["score", f"nll_uncond_{n}"], ["score", f"entropy_uncond_{n}"]]})
        g.append({"outputs": [["score", f"nll_cond_{n}"], ["score", f"entropy_cond_{n}"]]})
    return g


def score(outputs: dict, reference: dict, params: dict, seconds: dict) -> dict:
    sc = outputs.get("score", {})
    nfes = [n for n in NFES if f"nll_uncond_{n}" in sc]
    if not nfes:
        raise OutputError("score stage returned no samples")
    out, logs, ratios = {}, [], []
    e_ref_c = float(np.mean(need(outputs, "score", "entropy_reference_cond", finite=False)))
    e_ref_u = float(np.mean(need(outputs, "score", "entropy_reference_uncond", finite=False)))
    out["reference_gen_ppl"] = float(math.exp(np.mean(need(outputs, "score", "nll_reference", finite=False))))
    for n in nfes:
        for mode, e_ref in (("uncond", e_ref_u), ("cond", e_ref_c)):
            nll = np.asarray(need(outputs, "score", f"nll_{mode}_{n}", finite=False), np.float64)
            ent = np.asarray(need(outputs, "score", f"entropy_{mode}_{n}", finite=False), np.float64)
            m = float(np.mean(nll))
            out[f"gen_ppl_{mode}_{n}"] = float(math.exp(m)) if math.isfinite(m) else float("inf")
            out[f"entropy_{mode}_{n}"] = float(np.mean(ent))
            logs.append(m)
            ratios.append(float(np.mean(ent)) / e_ref)
        if "sample" in outputs and f"seconds_{n}" in outputs["sample"]:
            out[f"seconds_{n}"] = float(np.asarray(outputs["sample"][f"seconds_{n}"])[0])
            out[f"max_calls_{n}"] = float(np.asarray(outputs["sample"][f"max_calls_{n}"])[0])
    out["mean_log_gen_ppl"] = float(np.mean(logs))
    out["min_entropy_ratio"] = float(min(ratios))
    return out


def relative(sub: dict, base: dict) -> dict:
    d = {"mean_log_gen_ppl_delta_vs_ancestral": sub["mean_log_gen_ppl"] - base["mean_log_gen_ppl"]}
    for n in NFES:
        for mode in ("uncond", "cond"):
            k = f"gen_ppl_{mode}_{n}"
            if k in sub and k in base:
                d[f"{k}_ratio_vs_ancestral"] = sub[k] / base[k]
    return d
