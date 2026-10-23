"""U-DLM-004 metrics: the masked-diffusion likelihood bound (NELBO, nats per token) under the submitted schedule at two
scales (μ2, μ3); the primary metric is their mean. Reported: perplexity bounds, the change of the bound from μ2 to μ3,
and the gap to an autoregressive model of the same size trained on the same data with the same recipe
(AR_REFERENCE, nats per token, measured by the operator at activation; None until then)."""
from __future__ import annotations

import math

import numpy as np

from abench_metrics import OutputError, need  # noqa: F401

SCALES = ("mu2", "mu3")
AR_REFERENCE = {"mu2": None, "mu3": None}           # filled in at activation (calibration run of lm_model.LM)
PRIMARY = {"metric": "mean_nelbo", "goal": "min"}
GATES = [{"metric": "max_train_seconds_mu3", "max": 3600.0}, {"metric": "train_seconds_mu2", "max": 1800.0}]


def score(outputs: dict, reference: dict, params: dict, seconds: dict) -> dict:
    out, vals = {}, []
    for s in SCALES:
        v = float(need(outputs, f"run_{s}", "nelbo", shape=(1,))[0])
        out[f"nelbo_{s}"] = v
        out[f"ppl_bound_{s}"] = math.exp(min(v, 30.0))
        out[f"diverged_{s}"] = float(need(outputs, f"run_{s}", "diverged", shape=(1,))[0])
        if AR_REFERENCE[s] is not None:
            out[f"gap_to_ar_{s}"] = v - float(AR_REFERENCE[s])
        vals.append(v)
    out["mean_nelbo"] = float(np.mean(vals))
    out["scale_gain_mu2_to_mu3"] = out["nelbo_mu2"] - out["nelbo_mu3"]
    out["train_seconds_mu2"] = float(seconds["run_mu2"])
    out["max_train_seconds_mu3"] = float(seconds["run_mu3"])
    return out


def relative(sub: dict, base: dict) -> dict:
    d = {f"nelbo_{s}_delta_vs_mdlm": sub[f"nelbo_{s}"] - base[f"nelbo_{s}"] for s in SCALES}
    d["mean_nelbo_delta_vs_mdlm"] = sub["mean_nelbo"] - base["mean_nelbo"]
    d["mdlm_mean_nelbo"] = base["mean_nelbo"]
    return d
