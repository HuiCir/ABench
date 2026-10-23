"""PH-SYM-005 pipeline: the evaluator gives your discovery method the noisy trajectories of 20 dynamical systems it
has never seen and scores the equations it returns (evaluator and self-check use this same file).

One stage, "discover" (imports discover.py):
  before  the trajectories of the held-out systems are placed in ctx.pre_dir (held_pre.npz); loading is not timed
  timed   for each system: discover(t, x, seed) -> coefficient matrix over systems.library(d)   (all twenty ≤ 1,200 s)
  finish  the coefficient matrices are returned to the evaluator, which compares their sparsity pattern with the
          true one and integrates them from new initial conditions (metrics.py)
The evaluator runs the anchor's sequentially thresholded least squares (baselines/stlsq) through the same stage.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from abench_stage import Context, Stage
import systems as S

N_HELD = 20
DEV_SEED = 20261022                        # development systems of the self-check (the evaluator's seed is secret)
PARAMS = {"n_systems": N_HELD, "per_system_seconds": 60.0, "total_seconds": 1200.0,
          "trajectories": S.N_TRAJ, "samples": S.N_SAMPLES}
HELD_OUT_PRE_KEYS = [f"s{j:02d}_{k}" for j in range(N_HELD) for k in ("t", "x")]
HELD_OUT_KEYS: list = []
PIPELINE = [{"stage": "discover", "timeout": 1200 + 60, "finish_cap": 30, "tail_slack": 5, "threads": 8,
             "held_out_pre": True, "held_out": False, "models": False}]


def system_ids(keys) -> list[str]:
    return sorted({k.split("_")[0] for k in keys if k.endswith("_x")})


def _setup(ctx: Context):
    with np.load(ctx.pre_dir / "held_pre.npz", allow_pickle=False) as z:
        return {s: (z[f"{s}_t"].astype(np.float64), z[f"{s}_x"].astype(np.float64)) for s in system_ids(z.files)}


def _timed(ctx: Context, data):
    mod = ctx.import_submission("discover.py", required=("discover",))
    return {s: mod.discover(t, x, ctx.seed + i) for i, (s, (t, x)) in enumerate(data.items())}, data


def _finish(ctx: Context, state):
    coefs, data = state
    out = {}
    for s, c in coefs.items():
        d = data[s][1].shape[-1]
        c = np.asarray(c, np.float64)
        if c.shape != (S.n_terms(d), d):
            raise ValueError(f"{s}: discover() returned shape {c.shape}, expected {(S.n_terms(d), d)}")
        out[f"coef_{s}"] = c
    return out


STAGES = {"discover": Stage(setup=_setup, timed=_timed, finish=_finish)}


def pack(systems: list[dict]) -> tuple[dict, dict]:
    """(held_pre arrays, reference arrays) for a list of systems from systems.sample_set."""
    pre, ref = {}, {}
    for j, s in enumerate(systems):
        k = f"s{j:02d}"
        pre[f"{k}_t"], pre[f"{k}_x"] = s["t"], s["x"]
        ref[f"{k}_coef"], ref[f"{k}_test_x0"] = s["coef"], s["test_x0"]
        ref[f"{k}_duration"] = np.asarray([s["duration"]])
    return pre, ref


def make_dev(public: Path, out: Path, quick: bool = False) -> dict:
    """Dev setting for selfcheck.py: 20 systems drawn with DEV_SEED by the evaluator's generator (4 with --quick)."""
    out.mkdir(parents=True, exist_ok=True)
    systems = S.sample_set(DEV_SEED, N_HELD)
    if quick:
        systems = systems[:4]
    pre, ref = pack(systems)
    np.savez(out / "held_pre.npz", **pre)
    return {"public": public, "held_out": None, "held_pre": out / "held_pre.npz", "reference": ref, "time_scale": 1.0}
