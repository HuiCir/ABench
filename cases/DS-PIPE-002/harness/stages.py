"""DS-PIPE-002 pipeline: score a submitted set of tabular classifiers on six OpenML datasets (evaluator and self-check
use this same file).

The evaluator does not train. One stage, "predict":
  timed   import model.py, load(checkpoint_dir) -> model                       (load time, killed at 120 s)
  finish  for each dataset, predict_proba(model, dataset_id, X) -> (M, 2)   (≤ 120 s in total)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from abench_stage import Context, Stage

DATASETS = [44122, 44123, 44125, 44126, 44128, 45021]
NAMES = {44122: "pol", 44123: "house_16H", 44125: "MagicTelescope", 44126: "bank-marketing", 44128: "MiniBooNE",
         45021: "jannis"}
PARAMS = {"datasets": DATASETS}
HELD_OUT_KEYS = [f"x_{d}" for d in DATASETS]
PIPELINE = [{"stage": "predict", "timeout": 120 + 120, "finish_cap": 120, "tail_slack": 5, "threads": 8,
             "held_out": True, "models": False}]


def _timed(ctx: Context, _):
    mod = ctx.import_submission("model.py", required=("load", "predict_proba"))
    return mod, mod.load(str(ctx.submission / "checkpoint"))


def _finish(ctx: Context, state):
    mod, model = state
    out = {}
    with np.load(ctx.post_dir / "held_out.npz", allow_pickle=False) as z:
        for d in DATASETS:
            x = z[f"x_{d}"].astype(np.float32)
            out[f"proba_{d}"] = np.asarray(mod.predict_proba(model, d, x), dtype=np.float64)
    return out


STAGES = {"predict": Stage(timed=_timed, finish=_finish)}


def make_dev(public: Path, out: Path, quick: bool = False) -> dict:
    """Dev split for selfcheck.py: the rows in dev_idx of each dataset (train on the others for unbiased numbers)."""
    out.mkdir(parents=True, exist_ok=True)
    held, ref = {}, {}
    for d in DATASETS:
        with np.load(public / f"d{d}.npz", allow_pickle=False) as z:
            x, y, dev = z["x"], z["y"], z["dev_idx"]
        if quick:
            dev = dev[:300]
        held[f"x_{d}"], ref[f"y_{d}"] = x[dev], y[dev]
    np.savez(out / "held_out.npz", **held)
    return {"public": public, "held_out": out / "held_out.npz", "reference": ref, "time_scale": 1.0}
