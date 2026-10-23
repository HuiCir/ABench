"""LG-LONG-001 pipeline: multi-label classification of long European Court of Human Rights cases (LexGLUE ECtHR task A;
evaluator and self-check use this same file).

The evaluator does not train. One stage, "predict" (the frozen legal-bert-base-uncased is available):
  timed   import model.py, load(checkpoint_dir, models_dir, device) -> model                 (load time, killed at 300 s)
  finish  predict(model, texts) -> (M, 10) 0/1 decisions for the 10 articles, on the held-out cases       (≤ 600 s)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from abench_stage import Context, Stage
from textpack import pack, unpack

N_LABELS = 10
PARAMS = {"n_labels": N_LABELS, "labels": ["2", "3", "5", "6", "8", "9", "10", "11", "14", "P1-1"]}
HELD_OUT_KEYS = ["text_utf8", "text_offsets"]
PIPELINE = [{"stage": "predict", "timeout": 300 + 600, "finish_cap": 600, "tail_slack": 10, "threads": 8,
             "held_out": True, "models": True}]


def _timed(ctx: Context, _):
    mod = ctx.import_submission("model.py", required=("load", "predict"))
    return mod, mod.load(str(ctx.submission / "checkpoint"), str(ctx.models) if ctx.models else None, ctx.device)


def _finish(ctx: Context, state):
    mod, model = state
    with np.load(ctx.post_dir / "held_out.npz", allow_pickle=False) as z:
        texts = unpack(z["text_utf8"], z["text_offsets"])
    return {"pred": np.asarray(mod.predict(model, texts)).astype(np.int8)}


STAGES = {"predict": Stage(timed=_timed, finish=_finish)}


def make_dev(public: Path, out: Path, quick: bool = False) -> dict:
    """Dev split for selfcheck.py: the public validation split (1,000 cases) plays the held-out role. For an unbiased
    estimate, train without it (the reference train.py does this unless --include-dev is given)."""
    with np.load(public / "ecthr_a.npz", allow_pickle=False) as z:
        u, o, y = z["val_text_utf8"], z["val_text_offsets"], z["val_y"]
    m = min(200, len(y)) if quick else len(y)
    b, off = pack(unpack(u, o, np.arange(m)))
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "held_out.npz", text_utf8=b, text_offsets=off)
    return {"public": public, "held_out": out / "held_out.npz", "reference": {"y": y[:m]}, "time_scale": 1.0}
