"""C-SPEC-001 pipeline: end-to-end speed of your decoding method against plain greedy decoding of the target model,
with the output required to be the target's greedy output (evaluator and self-check use this same file).

The evaluator does not train. One stage, "decode":
  timed   import model.py, load(checkpoint_dir, models_dir, device) -> model               (load time, killed at 600 s)
  finish  for each workload and prompt (batch size 1): the evaluator first runs plain greedy decoding of the target
          (greedy.py, fp32) and times it, then calls generate(model, prompt_ids, max_new_tokens) and times it; the
          tokens are compared with the tie-tolerant rule of greedy.py                         (all ≤ 3,000 s)
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from abench_stage import Context, Stage
import greedy

WORKLOADS = ("math", "code", "summarize", "chat", "translate")
MAX_NEW = 256
TARGET, DRAFT = "Qwen2.5-1.5B-Instruct", "Qwen2.5-0.5B-Instruct"
PARAMS = {"workloads": list(WORKLOADS), "max_new_tokens": MAX_NEW, "target": TARGET, "draft": DRAFT, "batch": 1,
          "tie_margin": greedy.TIE_MARGIN}
HELD_OUT_KEYS = [f"{w}_{k}" for w in WORKLOADS for k in ("tokens", "offsets")]
PIPELINE = [{"stage": "decode", "timeout": 600 + 3000, "finish_cap": 3000, "tail_slack": 10, "threads": 8,
             "held_out": True, "models": True}]


def prompts(z, w: str) -> list[list[int]]:
    t, o = z[f"{w}_tokens"], z[f"{w}_offsets"]
    return [t[o[i]:o[i + 1]].astype(np.int64).tolist() for i in range(len(o) - 1)]


def _timed(ctx: Context, _):
    mod = ctx.import_submission("model.py", required=("load", "generate"))
    return mod, mod.load(str(ctx.submission / "checkpoint"), str(ctx.models), ctx.device)


def _finish(ctx: Context, state):
    import torch
    mod, model = state
    tdir = str(ctx.models / TARGET)
    eos = greedy.eos_ids(tdir)
    ref_model = greedy.load_target(tdir, ctx.device)
    out = {}
    with np.load(ctx.post_dir / "held_out.npz", allow_pickle=False) as z:
        for w in WORKLOADS:
            t_ref, t_sub, exact, n_tok, n_match = 0.0, 0.0, [], 0, 0
            for p in prompts(z, w):
                ctx.sync()
                t0 = time.perf_counter()
                ref, margins = greedy.greedy(ref_model, p, MAX_NEW, eos, ctx.device)
                ctx.sync()
                t_ref += time.perf_counter() - t0
                ctx.sync()
                t0 = time.perf_counter()
                with torch.no_grad():
                    sub = [int(x) for x in mod.generate(model, list(p), MAX_NEW)]
                ctx.sync()
                t_sub += time.perf_counter() - t0
                ok, m = greedy.compare(sub, ref, margins)
                exact.append(ok)
                n_tok += len(ref)
                n_match += m
            out[f"{w}_ref_seconds"], out[f"{w}_sub_seconds"] = np.asarray([t_ref]), np.asarray([t_sub])
            out[f"{w}_exact"] = np.asarray(exact, np.float64)
            out[f"{w}_tokens"] = np.asarray([float(n_tok)])
            out[f"{w}_matched"] = np.asarray([float(n_match)])
    return out


STAGES = {"decode": Stage(timed=_timed, finish=_finish)}


def make_dev(public: Path, out: Path, quick: bool = False) -> dict:
    """Dev set for selfcheck.py: the public dev prompts (16 per workload, from other splits than the held-out ones;
    4 with --quick) play the held-out role."""
    held = {}
    with np.load(public / "dev_prompts.npz", allow_pickle=False) as z:
        for w in WORKLOADS:
            t, o = z[f"{w}_tokens"], z[f"{w}_offsets"]
            n = 4 if quick else len(o) - 1
            held[f"{w}_tokens"], held[f"{w}_offsets"] = t[: o[n]], o[: n + 1]
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "held_out.npz", **held)
    return {"public": public, "held_out": out / "held_out.npz", "reference": {}, "time_scale": 1.0}
