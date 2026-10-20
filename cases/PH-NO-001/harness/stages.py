"""PH-NO-001 pipeline: 1D Burgers (ν = 0.001) surrogate from PDEBench (evaluator and self-check use this same file).

Trajectories are on a 256-point periodic grid with 41 frames (PDEBench's 1024 × 201 subsampled by 4 in space and 5 in
time). Given the first 10 frames, the model predicts the remaining 31.
The evaluator does not train. One stage, "predict":
  timed   import model.py, load(checkpoint_dir, device) -> model                                  (load time, killed at 120 s)
  finish  predict(model, u_in) for the held-out inputs at 256 points, then at 512 points          (≤ 120 s)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from abench_stage import Context, Stage

N_IN, N_OUT = 10, 31
PARAMS = {"n_in": N_IN, "n_out": N_OUT, "grid": 256, "transfer_grid": 512}
HELD_OUT_KEYS = ["u_in", "u_in_512"]
PIPELINE = [{"stage": "predict", "timeout": 120 + 120, "finish_cap": 120, "tail_slack": 10, "threads": 8,
             "held_out": True, "models": False}]


def _timed(ctx: Context, _):
    mod = ctx.import_submission("model.py", required=("load", "predict"))
    return mod, mod.load(str(ctx.submission / "checkpoint"), ctx.device)


def _finish(ctx: Context, state):
    mod, model = state
    with np.load(ctx.post_dir / "held_out.npz", allow_pickle=False) as z:
        u256, u512 = z["u_in"], z["u_in_512"]
    return {"pred": np.asarray(mod.predict(model, u256), dtype=np.float32),
            "pred_512": np.asarray(mod.predict(model, u512), dtype=np.float32)}


STAGES = {"predict": Stage(timed=_timed, finish=_finish)}


def make_dev(public: Path, out: Path, quick: bool = False) -> dict:
    """Dev split for selfcheck.py: trajectories in dev_idx are held out. The 512-point transfer input is approximated
    by spectral interpolation of the 256-point data (the evaluator uses true 512-point subsamples). For an unbiased dev
    estimate, train your checkpoint without the dev trajectories (the reference train.py does this unless --include-dev
    is given)."""
    with np.load(public / "train.npz", allow_pickle=False) as z:
        u, dev = z["u"], z["dev_idx"]
    if quick:
        dev = dev[:100]
    out.mkdir(parents=True, exist_ok=True)
    ud = u[dev]
    u512 = np.fft.irfft(np.fft.rfft(ud, axis=-1), n=512, axis=-1) * 2.0
    np.savez(out / "held_out.npz", u_in=ud[:, :N_IN], u_in_512=u512[:, :N_IN].astype(np.float32))
    return {"public": public, "held_out": out / "held_out.npz",
            "reference": {"u_out": ud[:, N_IN:], "u_out_512": u512[:, N_IN:].astype(np.float32)}, "time_scale": 1.0}
