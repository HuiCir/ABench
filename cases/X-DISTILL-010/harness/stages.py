"""X-DISTILL-010 pipeline: a SmolLM2-135M student trained in the sandbox, with SmolLM2-1.7B available as teacher,
scored by its held-out loss (evaluator and self-check).

One stage, "eval" (models/SmolLM2-135M, the student's starting point, and models/SmolLM2-1.7B, the teacher):
  timed   import model.py and call load(checkpoint_dir, models_dir, device); the returned model must have exactly
          SmolLM2-135M's parameter shapes
  finish  mean next-token loss on the held-out tokens of the student, of the starting model and of the teacher
                                                                                              (≤ 1,200 s)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

import lmpost
from abench_stage import Context, Stage

STUDENT, TEACHER = "SmolLM2-135M", "SmolLM2-1.7B"
PARAMS = {"window": 1024, "student_architecture": STUDENT, "teacher": TEACHER}
HELD_OUT_KEYS = ["tokens"]
PIPELINE = [{"stage": "eval", "timeout": 600 + 1200, "finish_cap": 1200, "tail_slack": 10, "threads": 8,
             "held_out": True, "models": True}]


def _timed(ctx: Context, _):
    mod = ctx.import_submission("model.py", required=("load",))
    return mod.load(str(ctx.submission / "checkpoint"), str(ctx.models), ctx.device)


def _finish(ctx: Context, student):
    with np.load(ctx.post_dir / "held_out.npz", allow_pickle=False) as z:
        tokens = z["tokens"]
    start = lmpost.load_base(str(ctx.models / STUDENT), "cpu", "float32")
    want = {n: tuple(p.shape) for n, p in start.named_parameters()}
    got = {n: tuple(p.shape) for n, p in student.named_parameters()}
    same = float(want == got)
    student.eval()
    nll = lmpost.heldout_nll(student, tokens, ctx.device)
    del student
    nll_start = lmpost.heldout_nll(start.to(ctx.device), tokens, ctx.device)
    del start
    teacher = lmpost.load_base(str(ctx.models / TEACHER), ctx.device)
    nll_teacher = lmpost.heldout_nll(teacher, tokens, ctx.device)
    return {"nll": np.asarray([nll]), "nll_start": np.asarray([nll_start]), "nll_teacher": np.asarray([nll_teacher]),
            "same_architecture": np.asarray([same])}


STAGES = {"eval": Stage(timed=_timed, finish=_finish)}


def make_dev(public: Path, out: Path, quick: bool = False) -> dict:
    """Self-check: the last 1,048,577 tokens of the public training shard (quick: 65,537), which the reference
    train.py does not train on."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from token_data import read_shard
    toks = read_shard(public / "fwedu_train_000001.bin")
    n = 65_537 if quick else 1_048_577
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "held_out.npz", tokens=toks[-n:])
    return {"public": public, "held_out": out / "held_out.npz", "reference": {}, "time_scale": 1.0}
