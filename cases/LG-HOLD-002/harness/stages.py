"""LG-HOLD-002 pipeline: CaseHOLD multiple choice (evaluator and self-check use this same file).

The evaluator does not train. One stage, "predict":
  timed   import model.py, load(checkpoint_dir, models_dir, device) -> model           (load time, killed at 300 s)
  finish  read the held-out questions, call predict(model, prompts, holdings) -> (M, 5)   (≤ 300 s)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from abench_stage import Context, Stage

N_CHOICES = 5
PARAMS = {"n_choices": N_CHOICES}
HELD_OUT_KEYS = ["prompt", "holdings"]
PIPELINE = [{"stage": "predict", "timeout": 300 + 300, "finish_cap": 300, "tail_slack": 10, "threads": 8,
             "held_out": True, "models": True}]


def _timed(ctx: Context, _):
    mod = ctx.import_submission("model.py", required=("load", "predict"))
    return mod, mod.load(str(ctx.submission / "checkpoint"), str(ctx.models) if ctx.models else None, ctx.device)


def _finish(ctx: Context, state):
    mod, model = state
    with np.load(ctx.post_dir / "held_out.npz", allow_pickle=False) as z:
        prompts, holdings = list(z["prompt"]), [list(h) for h in z["holdings"]]
    return {"scores": np.asarray(mod.predict(model, prompts, holdings), dtype=np.float32)}


STAGES = {"predict": Stage(timed=_timed, finish=_finish)}


def make_dev(public: Path, out: Path, quick: bool = False) -> dict:
    """Dev split for selfcheck.py: the public validation split plays the held-out role. For an unbiased dev estimate,
    train your checkpoint without it (the reference train.py does this unless --include-dev is given)."""
    with np.load(public / "train.npz", allow_pickle=False) as z:
        vp, vh, vy = z["val_prompt"], z["val_holdings"], z["val_y"]
    m = 500 if quick else len(vy)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "held_out.npz", prompt=vp[:m], holdings=vh[:m])
    return {"public": public, "held_out": out / "held_out.npz", "reference": {"y": vy[:m]}, "time_scale": 1.0}
