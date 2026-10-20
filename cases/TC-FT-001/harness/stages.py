"""TC-FT-001 pipeline: score a submitted text classifier on AG News (evaluator and self-check use this same file).

The evaluator does not train. One stage, "predict":
  timed   import model.py, load(checkpoint_dir) -> model                     (load time, ≤ 120 s)
  finish  read the held-out texts, predict(model, texts) -> labels         (≤ 60 s; throughput threshold)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from abench_stage import Context, Stage

N_CLASSES = 4
PARAMS = {"n_classes": N_CLASSES}
HELD_OUT_KEYS = ["text"]                       # given to predict(); the labels stay with the evaluator
PIPELINE = [{"stage": "predict", "timeout": 120 + 60, "finish_cap": 60, "tail_slack": 5, "threads": 8, "held_out": True,
             "models": False}]


def _timed(ctx: Context, _):
    mod = ctx.import_submission("model.py", required=("load", "predict"))
    return mod, mod.load(str(ctx.submission / "checkpoint"))


def _finish(ctx: Context, state):
    mod, model = state
    with np.load(ctx.post_dir / "held_out.npz", allow_pickle=False) as z:
        texts = [str(t) for t in z["text"]]
    return {"pred": np.asarray(mod.predict(model, texts)).reshape(-1)}


STAGES = {"predict": Stage(timed=_timed, finish=_finish)}


def make_dev(public: Path, out: Path, quick: bool = False) -> dict:
    """Dev split for selfcheck.py: the documents in dev_idx (train on the others if you want unbiased dev numbers)."""
    out.mkdir(parents=True, exist_ok=True)
    with np.load(public / "train.npz", allow_pickle=False) as z:
        text, y, dev = z["text"], z["y"], z["dev_idx"]
    if quick:
        dev = dev[:1000]
    np.savez(out / "held_out.npz", text=text[dev])
    return {"public": public, "held_out": out / "held_out.npz", "reference": {"y": y[dev]}, "time_scale": 1.0}
