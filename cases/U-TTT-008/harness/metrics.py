"""U-TTT-008 metrics: long-context language modelling on held-out PG19 books (mean loss over the first 8,192 positions;
primary) and its profile by position, passkey retrieval at 2k, 4k and 8k tokens, short-context loss on FineWeb-Edu,
the number of layers with a fixed-size state, decoding consistency and throughput; relative to the sliding-window
baseline trained by the evaluator under identical conditions."""
from __future__ import annotations

import numpy as np

from abench_metrics import OutputError, need  # noqa: F401

BUCKETS = [(0, 1024), (1024, 2048), (2048, 4096), (4096, 8192)]
PRIMARY = {"metric": "book_loss_8k", "goal": "min"}
GATES = [{"metric": "constant_state_layers", "min": 15.0}, {"metric": "decode_mismatch", "max": 1e-2},
         {"metric": "diverged", "max": 0.0}, {"metric": "train_seconds", "max": 3600.0}]


def score(outputs: dict, reference: dict, params: dict, seconds: dict) -> dict:
    o = outputs["run"]
    pl = np.asarray(o["position_loss"], np.float64)
    if pl.ndim != 1 or not np.isfinite(pl).all():
        raise OutputError("run.position_loss: expected finite per-position losses")
    out = {"book_loss_8k": float(pl.mean())}
    for a, b in BUCKETS:
        if len(pl) > a:
            out[f"book_loss_{a}_{b}"] = float(pl[a:b].mean())
    for L in (2048, 4096, 8192):
        out[f"passkey_{L}"] = float(need(outputs, "run", f"passkey_{L}", shape=(1,))[0])
    out["passkey_mean"] = float(np.mean([out[f"passkey_{L}"] for L in (2048, 4096, 8192)]))
    g = np.asarray(o["layer_growth"], np.float64)
    out["constant_state_layers"] = float((g <= 0).sum())
    out["cache_bytes_per_token"] = float(g.sum())
    for k in ("val_loss_fwedu", "decode_mismatch", "decode_tokens_per_second", "diverged"):
        out[k] = float(need(outputs, "run", k, shape=(1,), finite=False)[0])
    out["train_seconds"] = float(seconds["run"])
    return out


def relative(sub: dict, base: dict) -> dict:
    return {"book_loss_8k_delta_vs_swa": sub["book_loss_8k"] - base["book_loss_8k"],
            "passkey_mean_delta_vs_swa": sub["passkey_mean"] - base["passkey_mean"],
            "swa_book_loss_8k": base["book_loss_8k"]}
