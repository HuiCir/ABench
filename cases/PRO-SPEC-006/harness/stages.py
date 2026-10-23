"""PRO-SPEC-006 pipeline: audit of the draft's training, batch-1 decoding time of your speculative-decoding server
against the evaluator's plain decoding server, a greedy-consistency check of the answers and a sampled resume of the
draft's training (evaluator and self-check use this file).

Stage "audit" (pro_stages.audit): chain, traceability (the draft run starts from random initialisation or from parts
  of Qwen3-8B copied inside the run: its start entry records no starting weights), code identity.
Stage "vanilla" (evaluator code only): baselines/vanilla/serve.py (vLLM, no speculation, one GPU) answers every
  held-out prompt one after another (MAX_NEW tokens, greedy); time per request.
Stage "spec": your serve.py with ABENCH_DRAFT_DIR=<submission>/checkpoint answers the same prompts the same way;
  time per request; the answers of check_idx are saved for the next stage.
Stage "check" (evaluator code only): serve_bench.greedy_check of the saved answers.
Stage "resume" (pro_stages.resume): one registered checkpoint of the draft's run, resumed for at most 1 hour; the
  logged loss must match within 5%.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

import pro_stages as P
import serve_bench as SB
from abench_stage import Context, Stage

MODEL, PORT, MAX_NEW = "Qwen3-8B", 18100, 512
SETS = ("mtbench", "gsm8k", "humaneval")
RESUME_METRIC, RESUME_REL = "loss", 0.05
PARAMS = {"model": MODEL, "max_new_tokens": MAX_NEW, "sets": list(SETS), "batch": 1, "tie": SB.TIE,
          "chat_template": "Qwen3 non-thinking (enable_thinking=False)"}
HELD_OUT_KEYS = ["ids", "offsets", "set_id", "check_idx", "resume_u"]
PIPELINE = [
    {"stage": "audit", "timeout": 3600, "finish_cap": 3000, "tail_slack": 30, "threads": 16, "held_out": False, "models": False},
    {"stage": "vanilla", "timeout": 1800 + 5400, "finish_cap": 5400, "tail_slack": 30, "threads": 32, "held_out": True, "models": True},
    {"stage": "spec", "timeout": 1800 + 5400, "finish_cap": 5400, "tail_slack": 30, "threads": 32, "held_out": True, "models": True},
    {"stage": "check", "timeout": 2400, "finish_cap": 1800, "tail_slack": 30, "threads": 16, "held_out": False, "models": True,
     "carry": {"spec": ["sample.npz"]}},
    {"stage": "resume", "timeout": 3600 + 1800, "finish_cap": 3600 + 1200, "tail_slack": 30, "threads": 64,
     "held_out": True, "models": True},
]


def prompts_of(post: Path) -> tuple:
    with np.load(post / "held_out.npz", allow_pickle=False) as z:
        ids, off = z["ids"], z["offsets"]
        return [ids[off[i]:off[i + 1]].tolist() for i in range(len(off) - 1)], z["check_idx"].astype(int), float(z["resume_u"][0])


def timed_answers(serve_py: Path, ctx: Context, env: dict | None = None) -> tuple:
    prompts, check, _ = prompts_of(ctx.post_dir)
    import os
    proc = SB.start_server(serve_py, ctx.models / MODEL, PORT, ctx.work / "server_log.txt",
                           env=dict(os.environ, **(env or {})))
    try:
        SB.sequential(PORT, prompts[:2], [16, 16])                          # warm-up, not timed
        res = SB.sequential(PORT, prompts, [MAX_NEW] * len(prompts))
    finally:
        SB.stop_server(proc)
    secs = np.asarray([r["t_end"] - r["t0"] for r in res])
    ok = np.asarray([r["ok"] for r in res], np.int64)
    return res, secs, ok, prompts, check


def _finish_audit(ctx: Context, _):
    return P.audit(ctx.submission, None)


def _finish_vanilla(ctx: Context, _):
    _, secs, ok, _, _ = timed_answers(Path(__file__).resolve().parent / "baselines" / "vanilla" / "serve.py", ctx)
    return {"seconds": secs, "complete": ok}


def _finish_spec(ctx: Context, _):
    res, secs, ok, prompts, check = timed_answers(ctx.submission / "serve.py", ctx,
                                                  {"ABENCH_DRAFT_DIR": str(ctx.submission / "checkpoint")})
    np.savez(ctx.work / "sample.npz", **{f"{k}_p": np.asarray(prompts[i], np.int64) for k, i in enumerate(check)},
             **{f"{k}_o": np.asarray(res[i]["ids"], np.int64) for k, i in enumerate(check)})
    return {"seconds": secs, "complete": ok}


def _finish_check(ctx: Context, _):
    with np.load(ctx.work / "sample.npz", allow_pickle=False) as z:
        n = len([k for k in z.files if k.endswith("_p")])
        ps, os_ = [z[f"{k}_p"].tolist() for k in range(n)], [z[f"{k}_o"].tolist() for k in range(n)]
    return {"consistent": SB.greedy_check(ctx.models / MODEL, ps, os_, ctx.device)}


def _finish_resume(ctx: Context, _):
    _, _, u = prompts_of(ctx.post_dir)
    return P.resume(ctx.submission, ctx.work, u, RESUME_METRIC, RESUME_REL)


_id = lambda ctx, s: s                                                       # noqa: E731
STAGES = {"audit": Stage(timed=_id, finish=_finish_audit), "vanilla": Stage(timed=_id, finish=_finish_vanilla),
          "spec": Stage(timed=_id, finish=_finish_spec), "check": Stage(timed=_id, finish=_finish_check),
          "resume": Stage(timed=_id, finish=_finish_resume)}


def make_dev(public: Path, out: Path, quick: bool = False) -> dict:
    """Self-check: dev.npz, 30 prompts per set that are not held out (--quick: 5 per set)."""
    with np.load(public / "dev.npz", allow_pickle=False) as z:
        held = {k: z[k] for k in z.files}
    if quick:
        keep = np.concatenate([np.flatnonzero(held["set_id"] == s)[:5] for s in range(len(SETS))])
        ids, off = held["ids"], held["offsets"]
        parts = [ids[off[i]:off[i + 1]] for i in keep]
        held["ids"] = np.concatenate(parts)
        held["offsets"] = np.concatenate([[0], np.cumsum([len(p) for p in parts])])
        held["set_id"] = held["set_id"][keep]
        held["check_idx"] = np.arange(min(5, len(keep)))
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "held_out.npz", **held)
    return {"public": public, "held_out": out / "held_out.npz", "reference": {"set_id": held["set_id"]}, "time_scale": 1.0}
