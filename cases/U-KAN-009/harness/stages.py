"""U-KAN-009 pipeline: the evaluator trains the fixed three-layer networks of layerbench.py with your layer and with the
MLP layer at equal parameter budgets, on held-out function-fitting tasks and on Fashion-MNIST, and scores both on
held-out test points (evaluator and self-check use this same file).

Two stages, each in a new process on one GPU (both import layer.py):
  functions  before: the training and validation points of the held-out functions (held_pre.npz, untimed);
             timed: for each function, layerbench.fit_task for your layer and for the MLP (width per budget, learning
             rate chosen on validation, training); finish: predictions on the test points (given after the timed phase)
  image      setup: the public Fashion-MNIST training images (55,000 for training, the last 5,000 of a fixed
             permutation for validation); timed: fit_task for both layers; finish: predictions on the held-out
             test images
The run's seed (ctx.seed) seeds every training run.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import layerbench as LB
from abench_stage import Context, Stage

N_FUNC = 12
PARAMS = {"recipe": LB.RECIPE, "n_functions": N_FUNC, "image_val": 5_000, "split_seed": 2024}
HELD_OUT_PRE_KEYS = [f"f{j:02d}_{k}" for j in range(N_FUNC) for k in ("x_train", "y_train", "x_val", "y_val")]
HELD_OUT_KEYS = [f"f{j:02d}_x_test" for j in range(N_FUNC)] + ["img_x_test"]
PIPELINE = [{"stage": "functions", "timeout": 2400 + 120, "finish_cap": 120, "tail_slack": 10, "threads": 8,
             "held_out_pre": True, "held_out": True, "models": False},
            {"stage": "image", "timeout": 1200 + 120, "finish_cap": 120, "tail_slack": 10, "threads": 8,
             "held_out": True, "models": False}]


def _overrides(ctx: Context) -> dict:
    p = ctx.public / "overrides.json" if ctx.public else None          # written only by the self-check's --quick
    return json.loads(p.read_text()) if p and p.is_file() else {}


def _apply(ov: dict):
    LB.FUNC_STEPS = int(ov.get("func_steps", LB.FUNC_STEPS))
    LB.IMG_EPOCHS = int(ov.get("img_epochs", LB.IMG_EPOCHS))


def _names(files, suffix):
    return sorted({f[:-len(suffix)] for f in files if f.endswith(suffix)})


def _pixels(x: np.ndarray) -> np.ndarray:
    return ((x.astype(np.float32) / 255.0) - 0.2860) / 0.3530         # Fashion-MNIST training mean and std


def _fit_both(ctx: Context, data: dict, kind: str) -> dict:
    sub = ctx.import_submission("layer.py", required=("make_layer",))
    seeds = [int(ctx.seed)]
    return {"sub": LB.fit_task(sub.make_layer, data, kind, seeds, ctx.device),
            "mlp": LB.fit_task(LB.mlp_layer, data, kind, seeds, ctx.device)}


def _summary(prefix: str, fits: dict) -> dict:
    out = {}
    for who, f in fits.items():
        out[f"{prefix}_{who}_width"] = np.asarray([f["width"]], np.int64)
        out[f"{prefix}_{who}_params"] = np.asarray([f["params"]], np.int64)
        out[f"{prefix}_{who}_lr"] = np.asarray([f["lr"]])
        out[f"{prefix}_{who}_step_ms"] = np.asarray([f["step_ms"]])
        out[f"{prefix}_{who}_diverged"] = np.asarray([float(f["diverged"])])
    return out


# --------------------------------------------------------------------------- functions
def _f_setup(ctx: Context):
    _apply(_overrides(ctx))
    with np.load(ctx.pre_dir / "held_pre.npz", allow_pickle=False) as z:
        return {n: {k: z[f"{n}_{k}"].astype(np.float32) for k in ("x_train", "y_train", "x_val", "y_val")}
                for n in _names(z.files, "_x_train")}


def _f_timed(ctx: Context, data):
    return {n: _fit_both(ctx, d, "function") for n, d in data.items()}


def _f_finish(ctx: Context, fits):
    out = {}
    with np.load(ctx.post_dir / "held_out.npz", allow_pickle=False) as z:
        for n, f in fits.items():
            x = z[f"{n}_x_test"].astype(np.float32)
            for who in ("sub", "mlp"):
                nets = f[who]["nets"]
                out[f"{n}_pred_{who}"] = (LB.predict(nets[0], x, "function", ctx.device) if nets
                                          else np.full(len(x), np.nan, np.float32))
            out.update(_summary(n, f))
    return out


# --------------------------------------------------------------------------- image
def _i_setup(ctx: Context):
    _apply(_overrides(ctx))
    with np.load(ctx.public / "fashion_train.npz", allow_pickle=False) as z:
        x, y = _pixels(z["x"]), z["y"].astype(np.int64)
    perm = np.random.default_rng(PARAMS["split_seed"]).permutation(len(x))
    nv = PARAMS["image_val"]
    tr, va = perm[:-nv], perm[-nv:]
    return {"x_train": x[tr], "y_train": y[tr], "x_val": x[va], "y_val": y[va]}


def _i_timed(ctx: Context, data):
    return _fit_both(ctx, data, "image")


def _i_finish(ctx: Context, fits):
    with np.load(ctx.post_dir / "held_out.npz", allow_pickle=False) as z:
        x = _pixels(z["img_x_test"])
    out = {}
    for who in ("sub", "mlp"):
        nets = fits[who]["nets"]
        out[f"img_pred_{who}"] = (LB.predict(nets[0], x, "image", ctx.device).astype(np.int64) if nets
                                  else np.full(len(x), -1, np.int64))
    out.update(_summary("img", fits))
    return out


STAGES = {"functions": Stage(setup=_f_setup, timed=_f_timed, finish=_f_finish),
          "image": Stage(setup=_i_setup, timed=_i_timed, finish=_i_finish)}


def make_dev(public: Path, out: Path, quick: bool = False) -> dict:
    """Self-check: the five public dev functions play the held-out functions; the last 10,000 public training images
    (of a fixed permutation) are the test images and the rest the training images. --quick: two functions, 300
    steps, one epoch."""
    out.mkdir(parents=True, exist_ok=True)
    (out / "public").mkdir(exist_ok=True)
    pre, held, ref = {}, {}, {}
    with np.load(public / "dev_functions.npz", allow_pickle=False) as z:
        names = _names(z.files, "_x_train")[: 2 if quick else None]
        for n in names:
            for k in ("x_train", "y_train", "x_val", "y_val"):
                pre[f"{n}_{k}"] = z[f"{n}_{k}"]
            held[f"{n}_x_test"] = z[f"{n}_x_test"]
            ref[f"{n}_y_test"] = z[f"{n}_y_test"]
    with np.load(public / "fashion_train.npz", allow_pickle=False) as z:
        x, y = z["x"], z["y"]
    perm = np.random.default_rng(7).permutation(len(x))
    te, tr = perm[-10_000:], perm[:-10_000]
    if quick:
        te, tr = te[:2_000], tr[:12_000]
    np.savez(out / "public" / "fashion_train.npz", x=x[tr], y=y[tr])
    held["img_x_test"], ref["img_y_test"] = x[te], y[te]
    if quick:
        (out / "public" / "overrides.json").write_text(json.dumps({"func_steps": 300, "img_epochs": 1}))
    np.savez(out / "held_pre.npz", **pre)
    np.savez(out / "held_out.npz", **held)
    return {"public": out / "public", "held_out": out / "held_out.npz", "held_pre": out / "held_pre.npz",
            "reference": ref, "time_scale": 1.0}
