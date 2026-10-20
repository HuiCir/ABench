"""C-QNT-003 pipeline: score a weight-only quantized Qwen2.5-0.5B under a size budget, judged by perplexity.

The evaluator does not quantize. Your quantize.py and train.py write the quantized artifact to checkpoint/ in the
sandbox; it is the only thing the evaluator uses from training.
Stage "eval" (imports quantize.py; the original weights are NOT available)
  timed   load(checkpoint_dir, device) -> model
  finish  artifact size, perplexity on the held-out WikiText-2 test windows, prefill throughput
Stage "ref" (evaluator only; the frozen bf16 model is available, the submission is not imported)
  timed   load the bf16 model with transformers
  finish  its prefill throughput under the same conditions, for the throughput ratio
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from abench_stage import Context, Stage

N_PARAMS = 494_032_768                        # Qwen2.5-0.5B, all parameters (tied embedding counted once)
BITS_BUDGET = 4.25
BYTES_BUDGET = int(N_PARAMS * BITS_BUDGET / 8)
SEQ = 2048
PARAMS = {"model": "Qwen/Qwen2.5-0.5B", "n_params": N_PARAMS, "bits_per_param_budget": BITS_BUDGET,
          "artifact_bytes_budget": BYTES_BUDGET, "eval_seq": SEQ}
HELD_OUT_KEYS = ["tokens"]
PIPELINE = [
    {"stage": "eval", "timeout": 300 + 600, "finish_cap": 600, "tail_slack": 10, "threads": 8, "held_out": True,
     "models": False},
    {"stage": "ref", "timeout": 300 + 120, "finish_cap": 120, "tail_slack": 10, "threads": 8, "held_out": False,
     "models": True},
]
MODEL_DIR = "Qwen2.5-0.5B"


def _timed_e(ctx: Context, _):
    mod = ctx.import_submission("quantize.py", required=("load",))
    return mod.load(str(ctx.submission / "checkpoint"), ctx.device)


def prefill_tokens_per_second(ctx: Context, model, x) -> float:
    import torch
    with torch.no_grad():
        for _ in range(2):
            model(x)
        ctx.sync()
        t0 = time.perf_counter()
        for _ in range(5):
            model(x)
        ctx.sync()
    return SEQ / ((time.perf_counter() - t0) / 5)


def _finish_e(ctx: Context, model):
    import torch
    import torch.nn.functional as F
    size = sum(p.stat().st_size for p in (ctx.submission / "checkpoint").rglob("*") if p.is_file())
    with np.load(ctx.post_dir / "held_out.npz", allow_pickle=False) as z:
        toks = torch.from_numpy(z["tokens"].astype(np.int64))
    n = len(toks) // SEQ
    nll, count = 0.0, 0
    with torch.no_grad():
        for i in range(n):
            x = toks[i * SEQ:(i + 1) * SEQ].unsqueeze(0).to(ctx.device)
            out = model(x)
            logits = out.logits if hasattr(out, "logits") else out
            loss = F.cross_entropy(logits[0, :-1].float(), x[0, 1:], reduction="sum")
            nll += float(loss)
            count += SEQ - 1
    tps = prefill_tokens_per_second(ctx, model, toks[:SEQ].unsqueeze(0).to(ctx.device))
    return {"nll_sum": np.asarray([nll]), "tokens": np.asarray([count]), "artifact_bytes": np.asarray([size], np.int64),
            "prefill_tokens_per_second": np.asarray([tps])}


def _timed_r(ctx: Context, _):
    import torch
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(str(ctx.models / MODEL_DIR), torch_dtype=torch.bfloat16)
    return m.to(ctx.device).eval()


def _finish_r(ctx: Context, model):
    import torch
    g = torch.Generator(device="cpu").manual_seed(0)
    x = torch.randint(0, model.config.vocab_size, (1, SEQ), generator=g).to(ctx.device)
    return {"prefill_tokens_per_second": np.asarray([prefill_tokens_per_second(ctx, model, x)])}


STAGES = {"eval": Stage(timed=_timed_e, finish=_finish_e), "ref": Stage(timed=_timed_r, finish=_finish_r)}


def make_dev(public: Path, out: Path, quick: bool = False) -> dict:
    """Dev split for selfcheck.py: the WikiText-2 validation split (public) plays the held-out role."""
    val = np.load(public / "dev_tokens.npy", allow_pickle=False)
    if quick:
        val = val[: 8 * SEQ]
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "held_out.npz", tokens=val)
    return {"public": public, "held_out": out / "held_out.npz", "reference": {}, "time_scale": 1.0}
