"""A-NORM-005 metrics: held-out loss of the four training runs (two scales x two learning rates), loss spikes and
divergence, compared with the pre-RMSNorm baseline trained by the evaluator under identical conditions. A diverged run
counts with the loss of a uniform prediction, ln(50,304) = 10.83."""
from __future__ import annotations

import numpy as np

from abench_metrics import OutputError, need  # noqa: F401

RUNS = [("mu1", "lr1e-3"), ("mu1", "lr4e-3"), ("mu2", "lr1e-3"), ("mu2", "lr4e-3")]
PRIMARY = {"metric": "mean_val_loss", "goal": "min"}
GATES = [{"metric": "max_flops_ratio", "max": 1.02}, {"metric": "max_params_ratio", "max": 1.001},
         {"metric": "max_train_seconds_mu2", "max": 1500.0}]


def score(outputs: dict, reference: dict, params: dict, seconds: dict) -> dict:
    out, losses, spikes, div, fr, pr, t2 = {}, [], 0.0, 0.0, [], [], []
    for s, l in RUNS:
        st = f"run_{s}_{l}"
        v = float(need(outputs, st, "val_loss", shape=(1,))[0])
        out[f"val_loss_{s}_{l}"] = v
        losses.append(v)
        sp = float(need(outputs, st, "spikes", shape=(1,))[0])
        out[f"spikes_{s}_{l}"] = sp
        spikes += sp
        div += float(need(outputs, st, "diverged", shape=(1,))[0])
        fr.append(float(need(outputs, st, "flops_ratio", shape=(1,))[0]))
        pr.append(float(need(outputs, st, "params_ratio", shape=(1,))[0]))
        if s == "mu2":
            t2.append(float(seconds[st]))
    out["mean_val_loss"] = float(np.mean(losses))
    out["val_loss_high_lr_minus_low_lr"] = float(np.mean([out[f"val_loss_{s}_lr4e-3"] - out[f"val_loss_{s}_lr1e-3"]
                                                          for s in ("mu1", "mu2")]))
    out["total_spikes"] = spikes
    out["diverged_runs"] = div
    out["max_flops_ratio"] = max(fr)
    out["max_params_ratio"] = max(pr)
    out["max_train_seconds_mu2"] = max(t2)
    return out


def relative(sub: dict, base: dict) -> dict:
    d = {f"val_loss_{s}_{l}_delta_vs_prenorm": sub[f"val_loss_{s}_{l}"] - base[f"val_loss_{s}_{l}"] for s, l in RUNS}
    d["mean_val_loss_delta_vs_prenorm"] = sub["mean_val_loss"] - base["mean_val_loss"]
    d["spikes_delta_vs_prenorm"] = sub["total_spikes"] - base["total_spikes"]
    d["prenorm_mean_val_loss"] = base["mean_val_loss"]
    return d
