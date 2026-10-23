"""DS-GBM-001 pipeline: the evaluator trains your tabular learner on datasets it has never seen (evaluator and
self-check use this same file).

One stage, "fit" (imports model.py):
  before   the training splits of the held-out datasets are placed in ctx.pre_dir (held_pre.npz); loading them is
           not timed
  timed    for each dataset: fit(x_train, y_train, task, budget_seconds, seed) -> model; the whole loop is timed by
           the parent's clock (gate: at most PARAMS["total_fit_seconds"])
  finish   the test inputs arrive (held_out.npz); predict(model, x_test) for each dataset              (≤ 300 s)
The evaluator runs the HistGradientBoosting baseline (baselines/hgb) through the same stage on the same seeds.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from abench_stage import Context, Stage

N_HELD = 20
TASKS = ("classification", "regression")              # task codes 0 and 1
PARAMS = {"budget_per_dataset_seconds": 60.0, "total_fit_seconds": 1200.0, "max_train_rows": 10000,
          "max_test_rows": 5000}
HELD_OUT_PRE_KEYS = [f"h{j:02d}_{k}" for j in range(N_HELD) for k in ("x_train", "y_train", "task")]
HELD_OUT_KEYS = [f"h{j:02d}_x_test" for j in range(N_HELD)]
PIPELINE = [{"stage": "fit", "timeout": 1200 + 120, "finish_cap": 300, "tail_slack": 10, "threads": 8,
             "held_out_pre": True, "held_out": True, "models": False}]


def names_in(keys, suffix: str) -> list[str]:
    return sorted(k[: -len(suffix)] for k in keys if k.endswith(suffix))


def _setup(ctx: Context):
    with np.load(ctx.pre_dir / "held_pre.npz", allow_pickle=False) as z:
        return {d: (z[f"{d}_x_train"].astype(np.float32), z[f"{d}_y_train"], TASKS[int(z[f"{d}_task"][0])])
                for d in names_in(z.files, "_x_train")}


def _timed(ctx: Context, data):
    mod = ctx.import_submission("model.py", required=("fit", "predict"))
    budget = float(ctx.params.get("budget_per_dataset_seconds", 60.0))
    models = {}
    for i, (d, (x, y, task)) in enumerate(data.items()):
        models[d] = mod.fit(x, y, task, budget, ctx.seed + i)
    return mod, models, {d: t for d, (_, _, t) in data.items()}


def _finish(ctx: Context, state):
    mod, models, tasks = state
    out = {}
    with np.load(ctx.post_dir / "held_out.npz", allow_pickle=False) as z:
        for d in models:
            out[f"pred_{d}"] = np.asarray(mod.predict(models[d], z[f"{d}_x_test"].astype(np.float32)), np.float64).reshape(-1)
    return out


STAGES = {"fit": Stage(setup=_setup, timed=_timed, finish=_finish)}


def make_dev(public: Path, out: Path, quick: bool = False) -> dict:
    """Dev setting for selfcheck.py: the public dev datasets play the held-out role (their training splits go to fit(),
    their test splits are scored). --quick uses four of them, 2,000 training and 1,000 test rows each."""
    out.mkdir(parents=True, exist_ok=True)
    pre, held, ref = {}, {}, {}
    files = sorted(public.glob("dev_*.npz"))
    for p in files[:4] if quick else files:
        d = p.stem
        with np.load(p, allow_pickle=False) as z:
            xtr, ytr, xte, yte, t = z["x_train"], z["y_train"], z["x_test"], z["y_test"], z["task"]
        if quick:
            xtr, ytr, xte, yte = xtr[:2000], ytr[:2000], xte[:1000], yte[:1000]
        pre[f"{d}_x_train"], pre[f"{d}_y_train"], pre[f"{d}_task"] = xtr, ytr, t
        held[f"{d}_x_test"] = xte
        ref[f"{d}_y_test"], ref[f"{d}_task"] = yte, t
    np.savez(out / "held_pre.npz", **pre)
    np.savez(out / "held_out.npz", **held)
    return {"public": public, "held_out": out / "held_out.npz", "held_pre": out / "held_pre.npz", "reference": ref,
            "time_scale": 1.0}
