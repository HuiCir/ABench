"""U-BYTE-007 metrics: bits per byte on held-out text (primary), bits per byte on the same text with 5% character-level
noise and the difference (robustness), with compute, parameter and causality gates; relative to the static 4-byte
patch baseline trained by the evaluator under identical conditions."""
from __future__ import annotations

from abench_metrics import OutputError, need  # noqa: F401

PRIMARY = {"metric": "bpb", "goal": "min"}
GATES = [{"metric": "flops_ratio", "max": 1.05}, {"metric": "params_ratio", "max": 1.10},
         {"metric": "causal_leak", "max": 1e-3}, {"metric": "diverged", "max": 0.0},
         {"metric": "train_seconds", "max": 1800.0}]


def score(outputs: dict, reference: dict, params: dict, seconds: dict) -> dict:
    out = {k: float(need(outputs, "run", k, shape=(1,), finite=False)[0]) for k in
           ("bpb", "bpb_noisy", "diverged", "flops_ratio", "flops_per_byte", "params_ratio", "causal_leak",
            "bytes_per_second")}
    out["noise_penalty"] = out["bpb_noisy"] - out["bpb"]
    out["train_seconds"] = float(seconds["run"])
    return out


def relative(sub: dict, base: dict) -> dict:
    return {"bpb_delta_vs_static4": sub["bpb"] - base["bpb"],
            "noise_penalty_delta_vs_static4": sub["noise_penalty"] - base["noise_penalty"],
            "static4_bpb": base["bpb"]}
