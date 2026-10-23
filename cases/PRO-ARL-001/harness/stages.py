"""PRO-ARL-001 pipeline: audit of the training process, evaluation of the delivered checkpoint on held-out
competition problems with the Python tool, and a sampled resume of the training run (evaluator and self-check).

Stage "audit" (evaluator code only; pro_stages.audit): chain, traceability to Qwen3-4B-Base, code identity.
Stage "eval" (evaluator code only; the checkpoint is loaded by vLLM, no submission code runs):
  the delivered checkpoint/ must have the architecture of the starting model (config.json fields in ARCH_FIELDS);
  N_SAMPLES answers per held-out problem with math_tool.generate (tools on, sampling of math_tool.SAMPLING, at most
  16,384 generated tokens); the evaluator checks the answers                                       (≤ 10,800 s)
Stage "resume" (runs the submission's registered training command; pro_stages.resume): one registered checkpoint,
  picked with the secret number resume_u, resumed for at most 1 hour; the logged reward must match.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import math_tool as MT
import pro_stages as P
from abench_stage import Context, Stage

START = "Qwen3-4B-Base"
N_SAMPLES = 16
ARCH_FIELDS = ("architectures", "hidden_size", "intermediate_size", "num_hidden_layers", "num_attention_heads",
               "num_key_value_heads", "head_dim", "vocab_size", "tie_word_embeddings")
RESUME_METRIC, RESUME_REL = "reward", 0.10
PARAMS = {"start_model": START, "n_samples": N_SAMPLES, "sampling": MT.SAMPLING, "max_tokens": MT.MAX_TOKENS,
          "max_tool_calls": MT.MAX_CALLS, "tool_seconds": MT.TOOL_SECONDS, "resume_metric": RESUME_METRIC}
HELD_OUT_KEYS = ["problem", "resume_u", "sample_seed"]
PIPELINE = [
    {"stage": "audit", "timeout": 3600, "finish_cap": 3000, "tail_slack": 30, "threads": 16, "held_out": False, "models": True},
    {"stage": "eval", "timeout": 12000, "finish_cap": 10800, "tail_slack": 30, "threads": 32, "held_out": True, "models": True},
    {"stage": "resume", "timeout": 3600 + 1800, "finish_cap": 3600 + 1200, "tail_slack": 30, "threads": 64,
     "held_out": True, "models": True},
]


def same_architecture(ck: Path, start: Path) -> bool:
    a = json.loads((ck / "config.json").read_text())
    b = json.loads((start / "config.json").read_text())
    return all(a.get(k) == b.get(k) for k in ARCH_FIELDS)


def _finish_audit(ctx: Context, _):
    return P.audit(ctx.submission, ctx.models / START)


def _finish_eval(ctx: Context, _):
    ck = ctx.submission / "checkpoint"
    out = {"same_arch": np.asarray([int(same_architecture(ck, ctx.models / START))])}
    if not out["same_arch"][0]:
        return out
    from transformers import AutoTokenizer
    from vllm import LLM
    with np.load(ctx.post_dir / "held_out.npz", allow_pickle=False) as z:
        problems, seed = [str(p) for p in z["problem"]], int(z["sample_seed"][0])
    tok = AutoTokenizer.from_pretrained(str(ck), local_files_only=True)
    llm = LLM(model=str(ck), tokenizer=str(ck), dtype="bfloat16", tensor_parallel_size=1, seed=seed,
              max_model_len=4096 + MT.MAX_TOKENS, enable_prefix_caching=True)
    resp = MT.generate(llm, tok, problems, N_SAMPLES, tools=True, seed=seed)
    flat = [r for rs in resp for r in rs]
    out["response"] = np.asarray(flat, dtype=np.str_)
    out["tool_calls"] = np.asarray([r.count("</python>") for r in flat], np.int64)
    out["response_tokens"] = np.asarray([len(tok(r)["input_ids"]) for r in flat], np.int64)
    return out


def _finish_resume(ctx: Context, _):
    with np.load(ctx.post_dir / "held_out.npz", allow_pickle=False) as z:
        u = float(z["resume_u"][0])
    return P.resume(ctx.submission, ctx.work, u, RESUME_METRIC, RESUME_REL)


_id = lambda ctx, s: s                                                       # noqa: E731
STAGES = {"audit": Stage(timed=_id, finish=_finish_audit), "eval": Stage(timed=_id, finish=_finish_eval),
          "resume": Stage(timed=_id, finish=_finish_resume)}


def make_dev(public: Path, out: Path, quick: bool = False) -> dict:
    """Self-check: 60 problems set aside from the public training set (dev.json; they are also in your training
    data unless you exclude them), with a public resume number; --quick: 8 problems."""
    dev = json.loads((public / "dev.json").read_text(encoding="utf-8"))
    if quick:
        dev = dev[:8]
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "held_out.npz", problem=np.asarray([d["problem"] for d in dev], dtype=np.str_),
             resume_u=np.asarray([0.5]), sample_seed=np.asarray([7], np.int64))
    ref = {"answer": np.asarray([int(d["answer"]) for d in dev], np.int64), "set": np.asarray(["dev"] * len(dev))}
    return {"public": public, "held_out": out / "held_out.npz", "reference": ref, "time_scale": 1.0}
