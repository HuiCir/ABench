"""A-KV-004 metrics: held-out loss after continued training (primary) under a key-value cache budget of 7,680 bytes per
token (a third of SmolLM2-135M's), with decoding consistency; loss on 8,192-token windows (length extrapolation beyond
the 2,048-token training windows) is reported. Relative metrics compare with the multi-query baseline trained by the
evaluator under identical conditions."""
from __future__ import annotations

from abench_metrics import OutputError, need  # noqa: F401

PRIMARY = {"metric": "val_loss", "goal": "min"}
GATES = [{"metric": "kv_bytes_per_token", "max": 7680.0}, {"metric": "decode_mismatch", "max": 1e-2},
         {"metric": "diverged", "max": 0.0}, {"metric": "train_seconds", "max": 3000.0}]


def score(outputs: dict, reference: dict, params: dict, seconds: dict) -> dict:
    out = {k: float(need(outputs, "run", k, shape=(1,), finite=False)[0]) for k in
           ("val_loss", "val_loss_8k", "diverged", "kv_bytes_per_token", "decode_mismatch")}
    out["extrapolation_penalty_8k"] = out["val_loss_8k"] - out["val_loss"]
    out["kv_fraction_of_original"] = out["kv_bytes_per_token"] / 23040.0
    out["train_seconds"] = float(seconds["run"])
    return out


def relative(sub: dict, base: dict) -> dict:
    return {"val_loss_delta_vs_mqa": sub["val_loss"] - base["val_loss"],
            "val_loss_8k_delta_vs_mqa": sub["val_loss_8k"] - base["val_loss_8k"], "mqa_val_loss": base["val_loss"]}
