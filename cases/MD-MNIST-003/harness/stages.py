"""MD-MNIST-003 pipeline: BloodMNIST and OrganAMNIST at 28×28 (evaluator and self-check use this same file).

The evaluator does not train. One stage, "predict":
  timed   import model.py, load(checkpoint_dir, device) -> model                   (load time, killed at 120 s)
  finish  for each dataset, call predict(model, name, x_test) -> (M, n_classes)    (≤ 60 s)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from abench_stage import Context, Stage

DATASETS = {"blood": 8, "organa": 11}
PARAMS = {"datasets": DATASETS}
HELD_OUT_KEYS = [f"x_test_{d}" for d in DATASETS]
PIPELINE = [{"stage": "predict", "timeout": 120 + 60, "finish_cap": 60, "tail_slack": 5, "threads": 8,
             "held_out": True, "models": False}]


def _timed(ctx: Context, _):
    mod = ctx.import_submission("model.py", required=("load", "predict"))
    return mod, mod.load(str(ctx.submission / "checkpoint"), ctx.device)


def _finish(ctx: Context, state):
    mod, model = state
    out = {}
    with np.load(ctx.post_dir / "held_out.npz", allow_pickle=False) as z:
        for d in DATASETS:
            out[f"proba_{d}"] = np.asarray(mod.predict(model, d, z[f"x_test_{d}"]), dtype=np.float32)
    return out


STAGES = {"predict": Stage(timed=_timed, finish=_finish)}


def make_dev(public: Path, out: Path, quick: bool = False) -> dict:
    """Dev split for selfcheck.py: the official validation split plays the held-out role. For an unbiased dev estimate,
    train your checkpoint without it (the reference train.py does this unless --include-dev is given)."""
    out.mkdir(parents=True, exist_ok=True)
    held, ref = {}, {}
    for d in DATASETS:
        with np.load(public / f"{d}.npz", allow_pickle=False) as z:
            xva, yva = z["x_val"], z["y_val"]
        if quick:
            xva, yva = xva[:500], yva[:500]
        held[f"x_test_{d}"], ref[f"y_test_{d}"] = xva, yva
    np.savez(out / "held_out.npz", **held)
    return {"public": public, "held_out": out / "held_out.npz", "reference": ref, "time_scale": 1.0}
