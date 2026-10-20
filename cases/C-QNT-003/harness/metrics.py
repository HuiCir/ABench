"""C-QNT-003 metrics: WikiText-2 test perplexity of the quantized model, artifact size, and prefill throughput relative
to the original bf16 model timed by the evaluator in the same evaluation."""
from __future__ import annotations

import math

import numpy as np

from abench_metrics import OutputError, need  # noqa: F401

N_PARAMS = 494_032_768
PRIMARY = {"metric": "perplexity", "goal": "min"}
GATES = [
    {"metric": "bits_per_param", "max": 4.25},
    {"metric": "throughput_ratio_vs_bf16", "min": 0.5},
]


def score(outputs: dict, reference: dict, params: dict, seconds: dict) -> dict:
    nll = float(need(outputs, "eval", "nll_sum", shape=(1,), finite=False)[0])
    n = float(need(outputs, "eval", "tokens", shape=(1,))[0])
    size = float(need(outputs, "eval", "artifact_bytes", shape=(1,))[0])
    ppl = math.exp(nll / n) if n > 0 and math.isfinite(nll) and nll / n < 50 else float("inf")
    tps = float(need(outputs, "eval", "prefill_tokens_per_second", shape=(1,))[0])
    ref = float(need(outputs, "ref", "prefill_tokens_per_second", shape=(1,))[0])
    return {"perplexity": ppl, "bits_per_param": 8.0 * size / N_PARAMS, "artifact_bytes": size,
            "load_seconds": float(seconds["eval"]), "prefill_tokens_per_second": tps,
            "bf16_prefill_tokens_per_second": ref, "throughput_ratio_vs_bf16": tps / max(ref, 1e-9)}
